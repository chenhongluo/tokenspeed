# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Linear-attention parallel-mapping plumbing tests (cheap, no GPU).

``mapping.linear_attn`` defaults to the attention TP width on existing
deployment shapes, while ``--linear-attn-tp-size`` exposes the independent
KDA TP domain needed by MLA-DP and MLA-CP deployments.
"""

import argparse
import unittest
from unittest.mock import patch

from tokenspeed.runtime.distributed.mapping import (
    LinearAttnLayerMapping,
    Mapping,
)


class TestLinearAttnMappingDefaults(unittest.TestCase):
    def test_defaults_follow_attn_tp_tp8(self):
        m = Mapping(rank=0, world_size=8, attn_tp_size=8)
        self.assertEqual(m.linear_attn.tp_size, m.attn.tp_size)
        self.assertEqual(m.linear_attn.tp_group, m.attn.tp_group)
        self.assertEqual(m.linear_attn.dp_size, 1)

    def test_defaults_follow_attn_tp_dp8(self):
        # attention-DP8: attn tp=1, so linear-attn defaults to tp=1 per rank.
        m = Mapping(rank=3, world_size=8, attn_tp_size=1, attn_dp_size=8)
        self.assertEqual(m.linear_attn.tp_size, 1)
        self.assertEqual(m.linear_attn.dp_size, 8)
        self.assertEqual(m.linear_attn.tp_rank, 0)

    def test_hybrid_shape_linear_attn_tp_spans_dp_ranks(self):
        # The target hybrid: MLA-DP8 with KDA-TP8 over the same ranks.
        m = Mapping(
            rank=3, world_size=8, attn_tp_size=1, attn_dp_size=8, linear_attn_tp_size=8
        )
        self.assertEqual(m.linear_attn.tp_size, 8)
        self.assertEqual(m.linear_attn.tp_rank, 3)
        self.assertEqual(m.linear_attn.tp_group, tuple(range(8)))
        self.assertEqual(m.linear_attn.dp_size, 1)
        # MLA stays data-parallel.
        self.assertEqual(m.attn.dp_size, 8)

    def test_rank_propagates_to_linear_attn(self):
        m = Mapping(world_size=8, attn_tp_size=1, attn_dp_size=8, linear_attn_tp_size=8)
        m.rank = 5
        self.assertEqual(m.linear_attn.tp_rank, 5)

    def test_pp_stage_world(self):
        # KDA mapping resolves inside one pipeline stage.
        m = Mapping(rank=0, world_size=16, attn_tp_size=8, pp_size=2)
        self.assertEqual(m.linear_attn.tp_size, 8)
        self.assertEqual(m.linear_attn.dp_size, 1)

    def test_standalone_linear_attn_mapping_complement(self):
        km = LinearAttnLayerMapping(rank=6, world_size=8, tp_size=4)
        self.assertEqual((km.tp_size, km.dp_size), (4, 2))
        self.assertEqual(km.tp_rank, 2)
        self.assertEqual(km.tp_group, (4, 5, 6, 7))
        self.assertEqual(km.dp_rank, 1)


class TestLinearAttnServerArgs(unittest.TestCase):
    def test_server_args_exposes_only_the_generic_linear_attn_field(self):
        import dataclasses

        from tokenspeed.runtime.utils.server_args import ServerArgs

        names = {f.name for f in dataclasses.fields(ServerArgs)}
        self.assertIn("linear_attn_tp_size", names)
        self.assertNotIn("kda_tp_size", names)

    @staticmethod
    def _resolve(argv, *, enable_cp=False):
        import tokenspeed.runtime.utils.server_args as server_args_module
        from tokenspeed.runtime.utils.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        with (
            patch.object(ServerArgs, "__post_init__"),
            patch.object(server_args_module, "ENABLE_CP", enable_cp),
        ):
            server_args = ServerArgs.from_cli_args(parser.parse_args(argv))
            server_args.resolve_basic_defaults()
            server_args.resolve_parallelism()
        return server_args.mapping

    def test_prefill_cp8_keeps_linear_attention_tp8(self):
        mapping = self._resolve(
            [
                "--model",
                "test/model",
                "--world-size",
                "8",
                "--attn-tp-size",
                "8",
                "--linear-attn-tp-size",
                "8",
                "--dense-tp-size",
                "1",
                "--ep-size",
                "8",
            ],
            enable_cp=True,
        )

        self.assertEqual((mapping.attn.tp_size, mapping.attn.cp_size), (1, 8))
        self.assertEqual(mapping.linear_attn.tp_size, 8)
        self.assertEqual(mapping.dense.tp_size, 1)
        self.assertEqual(mapping.moe.ep_size, 8)

    def test_decode_dp8_keeps_linear_attention_tp8(self):
        mapping = self._resolve(
            [
                "--model",
                "test/model",
                "--world-size",
                "8",
                "--attn-tp-size",
                "1",
                "--data-parallel-size",
                "8",
                "--linear-attn-tp-size",
                "8",
                "--dense-tp-size",
                "8",
                "--ep-size",
                "8",
            ]
        )

        self.assertEqual((mapping.attn.tp_size, mapping.attn.dp_size), (1, 8))
        self.assertEqual(mapping.linear_attn.tp_size, 8)
        self.assertEqual(mapping.dense.tp_size, 8)
        self.assertEqual(mapping.moe.ep_size, 8)

    def test_bounded_service_uses_one_lockstep_tp8_world(self):
        from tokenspeed.runtime.configs.lite_config import (
            is_lite_replicated_mla_mapping,
        )

        mapping = self._resolve(
            [
                "--model",
                "test/model",
                "--world-size",
                "8",
                "--attn-tp-size",
                "8",
                "--linear-attn-tp-size",
                "8",
                "--dense-tp-size",
                "8",
                "--ep-size",
                "8",
            ]
        )

        mapping.rank = 0
        self.assertTrue(is_lite_replicated_mla_mapping(mapping))
        self.assertEqual(mapping.attn.tp_group, tuple(range(8)))
        self.assertEqual(mapping.linear_attn.tp_group, tuple(range(8)))
        self.assertEqual(mapping.dense.tp_group, tuple(range(8)))
        self.assertEqual(mapping.moe.ep_group, tuple(range(8)))

        from tokenspeed.runtime.pd.topology import PDParallelTopology

        topology = PDParallelTopology.from_mapping(mapping)
        topology.require_cache_pd_supported()
        self.assertEqual((topology.tp_size, topology.cp_size), (8, 1))


if __name__ == "__main__":
    unittest.main()
