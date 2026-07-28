# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared verify-side packing helpers for Disagg-DFlash."""

from __future__ import annotations

import torch

from vllm.v1.spec_decode.disagg_dflash.protocol import DisaggDFlashSpeculateRequest


def pack_speculate_request(
    *,
    req_ids: list[str],
    reduced_hiddens: torch.Tensor,
    positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_rejected: torch.Tensor,
    num_sampled: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    num_speculative_tokens: int,
    num_reqs: int,
) -> DisaggDFlashSpeculateRequest:
    """Pack one incremental speculate RPC.

    ``reduced_hiddens`` / ``positions`` cover only this step's scheduled
    context tokens (already L*H→H reduced). Full-prefix history lives in the
    draft server's KV.
    """
    return DisaggDFlashSpeculateRequest(
        req_ids=req_ids,
        context_hiddens=reduced_hiddens,
        context_positions=positions,
        query_start_loc=query_start_loc[: num_reqs + 1].to(dtype=torch.int32),
        num_rejected=num_rejected[:num_reqs].to(dtype=torch.int32),
        num_sampled=num_sampled[:num_reqs].to(dtype=torch.int32),
        # Always 1D [num_reqs] — never ship [num_reqs, 1].
        last_sampled=last_sampled.reshape(-1)[:num_reqs].to(dtype=torch.int64),
        next_prefill_tokens=next_prefill_tokens.reshape(-1)[:num_reqs].to(
            dtype=torch.int64
        ),
        temperature=temperature.reshape(-1)[:num_reqs].to(dtype=torch.float32),
        seeds=seeds.reshape(-1)[:num_reqs].to(dtype=torch.int64),
        num_speculative_tokens=num_speculative_tokens,
    )
