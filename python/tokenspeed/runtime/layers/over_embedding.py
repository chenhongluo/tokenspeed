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
"""LongCat branch-local over-embedding runtime layer."""

from __future__ import annotations

import math
import re
from dataclasses import replace

import torch
from tokenspeed_kernel.ops.over_embedding import (
    OverEmbeddingSpec,
    append_packed_lookup_,
    longcat_lite_tp4_spec,
    longcat_lite_tp8_spec,
    longcat_pro_tp8_spec,
    project_add_word_,
)
from torch import nn
from torch.nn.parameter import Parameter

from tokenspeed.runtime.distributed.comm_ops import all_reduce
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding


def resolve_longcat_oe_hyperparameters(config) -> tuple[int, int]:
    """Return ``(max_ngram_order, hashes_per_order)`` from HF field aliases."""
    max_ngram_order = getattr(config, "emb_neighbor_num", None)
    if max_ngram_order is None:
        max_ngram_order = getattr(config, "oe_neighbor_num", None)
    hashes_per_order = getattr(config, "emb_split_num", None)
    if hashes_per_order is None:
        hashes_per_order = getattr(config, "oe_split_num", None)
    if max_ngram_order is None or hashes_per_order is None:
        raise ValueError(
            "LongCat OE requires neighbor_num (maximum n-gram order) and "
            "split_num (hash branches per order)"
        )
    return int(max_ngram_order), int(hashes_per_order)


def resolve_longcat_oe_spec(
    *,
    vocab_size: int,
    hidden_size: int,
    max_ngram_order: int,
    hashes_per_order: int,
    modulus0: int,
    tp_size: int,
    tp_rank: int,
) -> OverEmbeddingSpec:
    """Resolve and validate one supported LongCat OE ownership profile."""
    if (vocab_size, hidden_size, tp_size) == (163840, 8192, 8):
        spec = longcat_pro_tp8_spec(tp_rank)
    elif (vocab_size, hidden_size, tp_size) == (163840, 3072, 4):
        spec = longcat_lite_tp4_spec(tp_rank)
    elif (vocab_size, hidden_size, tp_size) == (163840, 3072, 8):
        spec = longcat_lite_tp8_spec(tp_rank)
    else:
        raise ValueError(
            "unsupported LongCat OE profile: "
            f"vocab_size={vocab_size}, hidden_size={hidden_size}, TP={tp_size}"
        )

    branch_count = (max_ngram_order - 1) * hashes_per_order
    expected = {
        "max_ngram_order": (max_ngram_order, spec.max_ngram_order),
        "branch_count": (branch_count, spec.branch_count),
        "modulus0": (
            modulus0,
            min(
                fragment.modulus - 2 * fragment.branch_id for fragment in spec.fragments
            ),
        ),
    }
    mismatches = [
        f"{name}={actual} (expected {wanted})"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if mismatches:
        raise ValueError(
            f"checkpoint does not match {spec.profile} TP{tp_size}: "
            + ", ".join(mismatches)
        )
    return spec


def _branch_index(weight_name: str, *, projection: bool) -> int | None:
    patterns = (
        (
            r"(?:^|\.)oe_embed_proj(\d+)\.weight$",
            r"(?:^|\.)ngram_embeddings\.post_projs\.(\d+)\.weight$",
        )
        if projection
        else (
            r"(?:^|\.)oe_embed_tokens(\d+)\.weight$",
            r"(?:^|\.)ngram_embeddings\.embedders\.(\d+)\.weight$",
        )
    )
    for pattern in patterns:
        match = re.search(pattern, weight_name)
        if match is not None:
            return int(match.group(1))
    return None


class LongCatOverEmbedding(nn.Module):
    """Word embedding plus TP-local OE branches and one final TP reduction."""

    requires_request_prefix_tokens = True

    def __init__(
        self,
        *,
        num_embeddings: int,
        embedding_dim: int,
        over_embedding_m: int,
        hashes_per_order: int,
        max_ngram_order: int,
        tp_rank: int,
        tp_size: int,
        tp_group: tuple[int, ...] | None,
        params_dtype: torch.dtype | None = None,
        ignored_token_ids: tuple[int, ...] = (),
        eos_token_id: int | None = None,
        fix_normalize_factor: bool = False,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.tp_group = tp_group
        dtype = params_dtype or torch.get_default_dtype()
        self.spec = resolve_longcat_oe_spec(
            vocab_size=num_embeddings,
            hidden_size=embedding_dim,
            max_ngram_order=max_ngram_order,
            hashes_per_order=hashes_per_order,
            modulus0=over_embedding_m + 1,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        self.spec = replace(
            self.spec,
            ignored_token_ids=tuple(dict.fromkeys(ignored_token_ids)),
            eos_token_id=eos_token_id,
        )
        self.normalize_scale = (
            math.sqrt(self.spec.scale)
            if fix_normalize_factor
            else float(self.spec.scale)
        )

        self.word_embedding = VocabParallelEmbedding(
            num_embeddings,
            embedding_dim,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            params_dtype=params_dtype,
        )
        self.oe_tables = nn.ParameterList(
            [
                Parameter(
                    torch.empty(
                        (fragment.modulus, fragment.feature_width),
                        dtype=dtype,
                    ),
                    requires_grad=False,
                )
                for fragment in self.spec.fragments
            ]
        )
        self.projection = Parameter(
            torch.empty(
                (self.spec.local_width, embedding_dim),
                dtype=dtype,
            ),
            requires_grad=False,
        )
        self._table_loaded = [False] * len(self.spec.fragments)
        self._projection_loaded = [False] * len(self.spec.fragments)
        self._padding_req_pool_index = -1
        self.register_buffer(
            "_req_pool_indices", torch.empty(0, dtype=torch.int64), persistent=False
        )
        self.register_buffer(
            "_input_lengths", torch.empty(0, dtype=torch.int32), persistent=False
        )
        self.register_buffer(
            "history_token_ids", torch.empty(0, dtype=torch.int32), persistent=False
        )
        self.register_buffer(
            "committed_lengths", torch.empty(0, dtype=torch.int32), persistent=False
        )
        self.register_buffer(
            "staged_prefix_ids", torch.empty(0, dtype=torch.int32), persistent=False
        )
        self.register_buffer(
            "pending_prefix_lengths",
            torch.empty(0, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "pending_prefix_counts",
            torch.empty(0, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "_input_start_offsets",
            torch.empty(0, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "_active_request_mask",
            torch.empty(0, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "_effective_input_lengths",
            torch.empty(0, dtype=torch.int32),
            persistent=False,
        )

    @property
    def request_prefix_lookback(self) -> int:
        """Return the number of prefix tokens required by the first lookup."""
        return self.spec.history_lookback

    def bind_runtime_inputs(
        self,
        *,
        req_pool_indices: torch.Tensor,
        input_lengths: torch.Tensor,
        max_request_slots: int,
        history_capacity: int,
        padding_req_pool_index: int,
    ) -> None:
        """Bind graph-stable batch inputs and allocate OE-owned history.

        Args:
            req_pool_indices: Per-request slots with a dedicated padding value.
            input_lengths: Per-request packed token counts.
            max_request_slots: Exclusive upper bound for real request slots.
            history_capacity: Maximum absolute token position stored per request.
            padding_req_pool_index: Slot used by graph-padding rows.
        """
        if req_pool_indices.device != input_lengths.device:
            raise ValueError("OE runtime inputs must be on the same device")
        if padding_req_pool_index != max_request_slots:
            raise ValueError(
                "OE padding slot must immediately follow the real request slots"
            )
        if history_capacity <= 0:
            raise ValueError("OE history_capacity must be positive")

        device = req_pool_indices.device
        slot_count = max_request_slots + 1
        max_batch_size = req_pool_indices.numel()
        self._padding_req_pool_index = padding_req_pool_index
        self._req_pool_indices = req_pool_indices
        self._input_lengths = input_lengths
        self.history_token_ids = torch.zeros(
            (slot_count, history_capacity), dtype=torch.int32, device=device
        )
        self.committed_lengths = torch.zeros(
            slot_count, dtype=torch.int32, device=device
        )
        self.staged_prefix_ids = torch.zeros(
            (slot_count, self.spec.history_lookback),
            dtype=torch.int32,
            device=device,
        )
        self.pending_prefix_lengths = torch.full(
            (slot_count,), -1, dtype=torch.int32, device=device
        )
        self.pending_prefix_counts = torch.zeros(
            slot_count, dtype=torch.int32, device=device
        )
        self._input_start_offsets = torch.zeros(
            max_batch_size + 1, dtype=torch.int32, device=device
        )
        self._active_request_mask = torch.zeros(
            max_batch_size, dtype=torch.bool, device=device
        )
        self._effective_input_lengths = torch.zeros(
            max_batch_size, dtype=torch.int32, device=device
        )

    def stage_prefixes(
        self,
        *,
        req_pool_indices,
        prefix_lengths,
        request_token_ids,
    ) -> None:
        """Stage the bounded prefix tails consumed by the next forward.

        ``request_token_ids`` contains at most ``request_prefix_lookback``
        tokens ending at each corresponding prefix boundary. This method only
        fills forward inputs; authoritative history is changed in ``forward``.
        """
        if not (len(req_pool_indices) == len(prefix_lengths) == len(request_token_ids)):
            raise ValueError("OE prefix staging inputs must have matching lengths")
        if self.pending_prefix_lengths.numel() == 0:
            raise RuntimeError("OE runtime inputs must be bound before staging")

        lookback = self.spec.history_lookback
        capacity = self.history_token_ids.shape[1]
        for row, prefix_length, token_ids in zip(
            req_pool_indices, prefix_lengths, request_token_ids
        ):
            row = int(row)
            prefix_length = int(prefix_length)
            tail = tuple(int(token_id) for token_id in token_ids)
            if not 0 <= row < self._padding_req_pool_index:
                raise ValueError(f"OE request slot {row} is out of range")
            if not 0 <= prefix_length <= capacity:
                raise ValueError(
                    f"OE prefix length {prefix_length} exceeds capacity {capacity}"
                )
            if len(tail) > lookback or len(tail) > prefix_length:
                raise ValueError(
                    f"OE prefix tail has {len(tail)} tokens for boundary "
                    f"{prefix_length} and lookback {lookback}"
                )
            if tail:
                self.staged_prefix_ids[row, : len(tail)].copy_(
                    torch.as_tensor(
                        tail,
                        dtype=torch.int32,
                        device=self.staged_prefix_ids.device,
                    )
                )
            self.pending_prefix_counts[row] = len(tail)
            self.pending_prefix_lengths[row] = prefix_length

    def _prepare_history(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.history_token_ids.numel() == 0:
            raise RuntimeError("OE runtime inputs must be bound before forward")
        if batch_size <= 0:
            raise ValueError("OE forward requires a non-empty request batch")

        rows = self._req_pool_indices[:batch_size]
        active = self._active_request_mask[:batch_size]
        torch.ne(rows, self._padding_req_pool_index, out=active)
        lengths = self._effective_input_lengths[:batch_size]
        lengths.copy_(self._input_lengths[:batch_size])
        lengths.masked_fill_(~active, 0)
        lengths[-1].add_(input_ids.numel() - lengths.sum())
        offsets = self._input_start_offsets[: batch_size + 1]
        offsets[0].zero_()
        torch.cumsum(lengths, dim=0, out=offsets[1:])

        pending_lengths = self.pending_prefix_lengths.index_select(0, rows)
        pending_counts = self.pending_prefix_counts.index_select(0, rows)
        staged = self.staged_prefix_ids.index_select(0, rows)
        capacity = self.history_token_ids.shape[1]
        for column in range(self.spec.history_lookback):
            history_positions = pending_lengths - pending_counts + column
            valid = (
                active
                & (pending_lengths >= 0)
                & (pending_counts > column)
                & (history_positions >= 0)
            )
            history_positions.clamp_(0, capacity - 1)
            previous = self.history_token_ids[rows, history_positions]
            self.history_token_ids[rows, history_positions] = torch.where(
                valid, staged[:, column], previous
            )

        first_token_offsets = offsets[:-1].clamp_max(input_ids.numel() - 1).long()
        first_positions = positions.index_select(0, first_token_offsets).to(torch.int32)
        previous_lengths = self.committed_lengths.index_select(0, rows)
        self.committed_lengths[rows] = torch.where(
            active, first_positions, previous_lengths
        )
        self.pending_prefix_lengths.index_fill_(0, rows, -1)
        self.pending_prefix_counts.index_fill_(0, rows, 0)
        return offsets, rows, active

    def load_weight(self, weight_name: str, loaded_weight: torch.Tensor) -> bool:
        """Load one word, OE table, or projection checkpoint tensor.

        Returns:
            ``True`` when this layer owns the weight name. Non-local OE branches
            also return ``True`` because skipping them is intentional.
        """
        if weight_name.endswith("embed_tokens.weight") and not (
            "oe_embed_tokens" in weight_name
            or "ngram_embeddings.embedders" in weight_name
        ):
            self.word_embedding.weight_loader(
                self.word_embedding.weight,
                loaded_weight,
            )
            return True

        table_branch = _branch_index(weight_name, projection=False)
        if table_branch is not None:
            for local_index, fragment in enumerate(self.spec.fragments):
                if fragment.branch_id != table_branch:
                    continue
                expected = (fragment.modulus, self.spec.branch_width)
                if tuple(loaded_weight.shape) != expected:
                    raise ValueError(
                        f"OE table branch {table_branch} has shape "
                        f"{tuple(loaded_weight.shape)}, expected {expected}"
                    )
                self.oe_tables[local_index].data.copy_(
                    loaded_weight[
                        :,
                        fragment.feature_begin : fragment.feature_end,
                    ]
                )
                self._table_loaded[local_index] = True
            return True

        projection_branch = _branch_index(weight_name, projection=True)
        if projection_branch is not None:
            if tuple(loaded_weight.shape) == (
                self.embedding_dim,
                self.spec.branch_width,
            ):
                branch_projection = loaded_weight.t()
            elif tuple(loaded_weight.shape) == (
                self.spec.branch_width,
                self.embedding_dim,
            ):
                branch_projection = loaded_weight
            else:
                raise ValueError(
                    f"OE projection branch {projection_branch} has shape "
                    f"{tuple(loaded_weight.shape)}, expected "
                    f"({self.embedding_dim}, {self.spec.branch_width}) or "
                    f"({self.spec.branch_width}, {self.embedding_dim})"
                )
            output_begin = 0
            for local_index, fragment in enumerate(self.spec.fragments):
                output_end = output_begin + fragment.feature_width
                if fragment.branch_id == projection_branch:
                    self.projection.data[output_begin:output_end].copy_(
                        branch_projection[fragment.feature_begin : fragment.feature_end]
                    )
                    self._projection_loaded[local_index] = True
                output_begin = output_end
            return True
        return False

    def validate_loaded_weights(self) -> None:
        """Fail startup when any rank-local projection fragment is missing."""
        missing_tables = [
            fragment.branch_id
            for fragment, loaded in zip(
                self.spec.fragments,
                self._table_loaded,
            )
            if not loaded
        ]
        missing_projections = [
            fragment.branch_id
            for fragment, loaded in zip(
                self.spec.fragments,
                self._projection_loaded,
            )
            if not loaded
        ]
        if missing_tables or missing_projections:
            raise RuntimeError(
                "missing LongCat OE weights: "
                f"tables={missing_tables}, projections={missing_projections}"
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx,
    ) -> torch.Tensor:
        if input_ids.numel() == 0:
            return self.word_embedding(input_ids)
        offsets, req_pool_indices, active_request_mask = self._prepare_history(
            input_ids, positions, int(ctx.bs)
        )
        word_partial = self.word_embedding(input_ids, reduce_results=False)
        activation = append_packed_lookup_(
            input_ids,
            offsets,
            req_pool_indices,
            active_request_mask,
            self.history_token_ids,
            self.committed_lengths,
            tuple(self.oe_tables),
            spec=self.spec,
            enable_pdl=True,
        )
        local_partial = project_add_word_(
            word_partial,
            activation,
            self.projection,
            scale=self.normalize_scale,
        )
        if self.tp_size > 1:
            local_partial = all_reduce(local_partial, self.tp_group)
        return local_partial

    @property
    def weight(self) -> Parameter:
        """Expose the tied regular word-embedding weight."""
        return self.word_embedding.weight

    @weight.setter
    def weight(self, value: Parameter) -> None:
        self.word_embedding.weight = value


__all__ = [
    "LongCatOverEmbedding",
    "resolve_longcat_oe_hyperparameters",
    "resolve_longcat_oe_spec",
]
