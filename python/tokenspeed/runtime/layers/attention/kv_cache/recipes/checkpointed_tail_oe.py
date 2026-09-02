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

"""Lite's Kimi-K3 cache geometry plus the three-token OE snapshot."""

from __future__ import annotations

from collections.abc import Mapping

from typing_extensions import override

from tokenspeed.runtime.layers.attention.kv_cache.recipes.base import (
    CacheGroupDeclaration,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.kimi_k3 import (
    KimiK3Recipe,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    CacheLayout,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    CacheGroupSpec,
)

OE_TAIL_GROUP = "lite_oe"
OE_TAIL_FIELD = "layer.0.lite.oe.context"


class CheckpointedTailOERecipe(KimiK3Recipe):
    """Add Checkpointed-tail OE state without publishing a second cache family."""

    @property
    @override
    def max_padding_fraction(self) -> float:
        return float("inf")

    @override
    def groups(self) -> tuple[CacheGroupDeclaration, ...]:
        return super().groups() + (
            (
                CacheGroupSpec(
                    group_id=OE_TAIL_GROUP,
                    retention="full_history",
                    family="state",
                    transfer_policy=(
                        "latest_snapshot" if self.pd_disaggregation_enabled else None
                    ),
                    checkpoint_granularity=self.prefix_granularity,
                ),
                (
                    CacheFieldSpec(
                        OE_TAIL_FIELD,
                        "slot.0",
                        (3,),
                        "int32",
                    ),
                ),
            ),
        )

    @override
    def packing(self, groups: tuple[CacheGroupDeclaration, ...]) -> Mapping[str, int]:
        packing = dict(super().packing(groups))
        fields = {spec.group_id: declared for spec, declared in groups}
        plane_bytes = (
            sum(
                field.payload_bytes
                for field in fields[FULL_ATTENTION]
                if field.plane_id == "slot.0"
            )
            * packing[FULL_ATTENTION]
        )
        oe_bytes = sum(field.payload_bytes for field in fields[OE_TAIL_GROUP])
        if plane_bytes % oe_bytes:
            raise ValueError(
                "Checkpointed-tail OE context must divide the existing slot.0 plane"
            )
        packing[OE_TAIL_GROUP] = plane_bytes // oe_bytes
        return packing

    @override
    def num_lcm_blocks(self, layout: CacheLayout) -> int:
        budgeted = (
            self.cache_budget_bytes - self.workspace_bytes()
        ) // layout.lcm_block_bytes - 1
        if budgeted < 1:
            raise ValueError(
                "Lite cache budget must hold a null parent and one usable LCM parent"
            )
        if self.token_limit is None:
            return budgeted
        return min(budgeted, self.parents_needed(layout, self.token_limit))
