# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Process-level Disagg/DFlash debug logging flags (CLI-configured only).

Call :func:`configure_disagg_debug_logging` once at worker / draft-server startup
from ObservabilityConfig or draft CLI. No environment-variable fallbacks.
"""

from __future__ import annotations

from vllm.v1.spec_decode.disagg_dflash.timing_model import configure_timing_model

# Verify-side Disagg profile (proxy / transport wall clocks).
_disagg_profile: bool = False
_disagg_profile_every: int = 50

# Draft / colocated DFlash forward profile (CUDA sync).
_draft_profile: bool = False
_draft_profile_every: int = 20

# NIXL/ZMQ transfer INFO spam control: -1 off, 0 every xfer, N every Nth.
_nixl_log_every: int = -1


def configure_disagg_debug_logging(
    *,
    enable_sd_timing_model: bool = False,
    sd_timing_model_log_every: int = 20,
    enable_disagg_dflash_profile: bool = False,
    disagg_dflash_profile_log_every: int = 50,
    enable_dflash_draft_profile: bool = False,
    dflash_draft_profile_log_every: int = 20,
    disagg_dflash_nixl_log_every: int = -1,
) -> None:
    """Apply CLI / ObservabilityConfig debug logging settings for this process."""
    global _disagg_profile, _disagg_profile_every
    global _draft_profile, _draft_profile_every, _nixl_log_every

    configure_timing_model(
        enabled=bool(enable_sd_timing_model),
        log_every=int(sd_timing_model_log_every),
    )
    _disagg_profile = bool(enable_disagg_dflash_profile)
    _disagg_profile_every = max(1, int(disagg_dflash_profile_log_every))
    _draft_profile = bool(enable_dflash_draft_profile)
    _draft_profile_every = max(1, int(dflash_draft_profile_log_every))
    _nixl_log_every = int(disagg_dflash_nixl_log_every)


def disagg_profile_enabled() -> bool:
    return _disagg_profile


def disagg_profile_log_every() -> int:
    return _disagg_profile_every


def draft_profile_enabled() -> bool:
    return _draft_profile


def draft_profile_log_every() -> int:
    return _draft_profile_every


def nixl_log_every() -> int:
    """-1 disable, 0 every transfer, N>0 every Nth transfer."""
    return _nixl_log_every
