# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batched GPU prep for Disagg-DFlash draft speculate (colocated-style Triton)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _prepare_disagg_dflash_inputs_kernel(
    # Outputs
    out_input_ids_ptr,
    out_query_positions_ptr,
    out_query_start_loc_ptr,
    out_seq_lens_ptr,
    out_query_slot_mapping_ptr,
    out_context_slot_mapping_ptr,
    # Inputs
    ctx_pos_ptr,
    qsl_ptr,
    bonus_ptr,
    last_valid_pos_ptr,
    block_table_ptr,
    block_table_stride,
    # Scalars
    parallel_drafting_token_id,
    block_size,
    num_query_per_req,
    max_model_len,
    BLOCK_SIZE: tl.constexpr,
):
    """Fill context slots + query buffers for one request (dense packed BT rows)."""
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    ctx_start = tl.load(qsl_ptr + req_idx)
    ctx_end = tl.load(qsl_ptr + req_idx + 1)
    num_ctx = ctx_end - ctx_start
    # last_valid_pos is precomputed on host (handles empty valid context).
    last_valid_pos = tl.load(last_valid_pos_ptr + req_idx)
    bonus_token = tl.load(bonus_ptr + req_idx).to(tl.int32)
    query_base = req_idx * num_query_per_req

    j = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    is_ctx = j < num_ctx
    is_query = (j >= num_ctx) & (j < num_ctx + num_query_per_req)
    query_off = j - num_ctx

    # --- Context slots ---
    ctx_pos_idx = ctx_start + tl.where(is_ctx, j, 0)
    ctx_pos = tl.load(ctx_pos_ptr + ctx_pos_idx, mask=is_ctx, other=0)
    ctx_block_num = ctx_pos // block_size
    ctx_block_num = tl.minimum(ctx_block_num, block_table_stride - 1)
    ctx_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + ctx_block_num,
        mask=is_ctx,
        other=0,
    ).to(tl.int64)
    ctx_slot = ctx_block_id * block_size + (ctx_pos % block_size)
    tl.store(out_context_slot_mapping_ptr + ctx_start + j, ctx_slot, mask=is_ctx)

    # --- Query positions / input_ids / slots ---
    query_pos = last_valid_pos + 1 + query_off
    query_idx = query_base + query_off
    is_bonus = is_query & (query_off == 0)
    input_id = tl.where(is_bonus, bonus_token, parallel_drafting_token_id)

    q_block_num = query_pos // block_size
    q_block_num = tl.minimum(q_block_num, block_table_stride - 1)
    q_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + q_block_num,
        mask=is_query,
        other=0,
    ).to(tl.int64)
    q_slot = q_block_id * block_size + (query_pos % block_size)

    tl.store(out_input_ids_ptr + query_idx, input_id, mask=is_query)
    clamped_query_pos = tl.minimum(query_pos, max_model_len - 1)
    tl.store(out_query_positions_ptr + query_idx, clamped_query_pos, mask=is_query)
    tl.store(out_query_slot_mapping_ptr + query_idx, q_slot, mask=is_query)

    if block_idx == 0:
        tl.store(out_query_start_loc_ptr + req_idx, query_base)
        tl.store(
            out_seq_lens_ptr + req_idx, last_valid_pos + 1 + num_query_per_req
        )


def prepare_disagg_dflash_inputs(
    *,
    input_ids: torch.Tensor,
    query_positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    query_slot_mapping: torch.Tensor,
    context_slot_mapping: torch.Tensor,
    ctx_pos: torch.Tensor,
    qsl: torch.Tensor,
    bonus: torch.Tensor,
    last_valid_pos: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    parallel_drafting_token_id: int,
    num_reqs: int,
    num_query_per_req: int,
    max_model_len: int,
    max_tokens_per_req: int,
) -> None:
    """One Triton launch: context slots + query ids/pos/slots/seq_lens.

    ``block_table`` must be dense packed rows ``[num_reqs, max_blocks]``
    (arena slots already gathered), matching CUDA-graph packed BTs.
    """
    if num_reqs <= 0:
        return
    BLOCK_SIZE = min(256, triton.next_power_of_2(max(1, max_tokens_per_req)))
    num_blocks = triton.cdiv(max_tokens_per_req, BLOCK_SIZE)
    _prepare_disagg_dflash_inputs_kernel[(num_reqs, num_blocks)](
        input_ids,
        query_positions,
        query_start_loc,
        seq_lens,
        query_slot_mapping,
        context_slot_mapping,
        ctx_pos,
        qsl,
        bonus,
        last_valid_pos,
        block_table,
        block_table.stride(0),
        parallel_drafting_token_id,
        block_size,
        num_query_per_req,
        max_model_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
