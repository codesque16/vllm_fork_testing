# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disaggregated DFlash speculative decoding (experimental).

Gated by ``SpeculativeConfig.disagg_dflash_address``. When unset, colocated
DFlash is unchanged.
"""

from vllm.v1.spec_decode.disagg_dflash.protocol import (
    CMD_FREE,
    CMD_SPECULATE,
    DisaggDFlashFreeRequest,
    DisaggDFlashSpeculateRequest,
    DisaggDFlashSpeculateResponse,
)

__all__ = [
    "CMD_FREE",
    "CMD_SPECULATE",
    "DisaggDFlashFreeRequest",
    "DisaggDFlashSpeculateRequest",
    "DisaggDFlashSpeculateResponse",
]
