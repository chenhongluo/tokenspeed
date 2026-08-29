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

import json


def test_flash_kda_config_resolves_hybrid_layer_pattern() -> None:
    from tokenspeed.runtime.configs.flash_kda_config import FLASHLocalConfig

    config = FLASHLocalConfig(num_hidden_layers=8, fa_interval=4, use_mla=True)

    assert config.linear_layer_ids == [0, 1, 2, 4, 5, 6]
    assert config.full_attention_layer_ids == [3, 7]


def test_get_config_loads_flash_lite_without_model_type(tmp_path) -> None:
    from tokenspeed.runtime.configs.flash_kda_config import FLASHLocalConfig
    from tokenspeed.runtime.utils.hf_transformers_utils import get_config

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "architectures": ["FLASHLocalForCausalLM"],
                "vocab_size": 163840,
                "hidden_size": 3072,
                "num_layers": 28,
                "num_attention_heads": 32,
                "n_routed_experts": 384,
                "moe_topk": 12,
                "ngram_vocab_size_ratio": 28.8,
                "emb_split_num": 4,
                "emb_neighbor_num": 4,
                "ngram_exclude_sp_token": True,
                "special_token_scope": "0:4,36:55",
                "fa_interval": 4,
                "use_mla": 1,
            }
        )
    )

    config = get_config(str(tmp_path), trust_remote_code=False)

    assert isinstance(config, FLASHLocalConfig)
    assert config.num_hidden_layers == 28
    assert config.num_experts == 384
    assert config.num_experts_per_token == 12
    assert config.over_embedding_m == 4_718_592
    assert config.oe_ignore_tokens == list(range(4)) + list(range(36, 55))
