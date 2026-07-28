# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared Tpv / Tad timing model for colocated and Disagg-DFlash SD.

Per decode speculate step::

    Tad = Ttransfer + Td + T_tok
    Ideal overlap: Tpv ≈ Tad
    slack = Tpv - Tad
    step_ms ≈ max(Tpv, Tad) + await_ms
    tpot_est_ms ≈ step_ms / accepted

Handoff / TTFT inputs (separate log line)::

    TTFT ≈ prefill_ms + p2d_kv_ms + first_step_ms

Enable via ``--enable-sd-timing-model`` (CLI only; no env fallback).
"""

from __future__ import annotations

from prometheus_client import Gauge

from vllm.logger import init_logger

logger = init_logger(__name__)

_LABELS = ("mode",)

# Configured by EngineCore / draft server CLI (default off).
_enabled: bool = False
_every: int = 20

_steps = 0
_handoff_steps = 0
# Stash of recent P→D KV xfer samples (ms) for optional drain/logging.
_p2d_kv_samples_ms: list[float] = []


def configure_timing_model(
    *,
    enabled: bool,
    log_every: int = 20,
) -> None:
    """Apply CLI / ObservabilityConfig settings for this process."""
    global _enabled, _every
    _enabled = bool(enabled)
    _every = max(1, int(log_every))


def timing_model_enabled() -> bool:
    return _enabled


def _timing_every() -> int:
    return _every


gauge_tpv_ms = Gauge(
    "vllm:sd_timing_tpv_ms",
    "Verifier step time Tpv (target forward + sample), milliseconds.",
    labelnames=_LABELS,
)
gauge_tad_ms = Gauge(
    "vllm:sd_timing_tad_ms",
    "Async drafting Tad = Ttransfer + Td + T_tok, milliseconds.",
    labelnames=_LABELS,
)
gauge_ttransfer_ms = Gauge(
    "vllm:sd_timing_ttransfer_ms",
    "Hiddens transfer time (NIXL e2e; 0 if colocated), milliseconds.",
    labelnames=_LABELS,
)
gauge_td_ms = Gauge(
    "vllm:sd_timing_td_ms",
    "Draft forward time Td, milliseconds.",
    labelnames=_LABELS,
)
gauge_tzmq_ms = Gauge(
    "vllm:sd_timing_tzmq_ms",
    "Draft token return time T_tok (control-plane recv; 0 if colocated), "
    "milliseconds.",
    labelnames=_LABELS,
)
gauge_slack_ms = Gauge(
    "vllm:sd_timing_slack_ms",
    "Overlap slack Tpv - Tad, milliseconds (positive => draft finishes first).",
    labelnames=_LABELS,
)
gauge_await_ms = Gauge(
    "vllm:sd_timing_await_ms",
    "Residual verify-side wait after draft fire (not part of Tad), milliseconds.",
    labelnames=_LABELS,
)
gauge_accepted = Gauge(
    "vllm:sd_timing_accepted",
    "Mean accepted draft tokens per request this step.",
    labelnames=_LABELS,
)
gauge_tpot_est_ms = Gauge(
    "vllm:sd_timing_tpot_est_ms",
    "Estimated TPOT = step_ms / accepted, milliseconds.",
    labelnames=_LABELS,
)
gauge_prefill_ms = Gauge(
    "vllm:sd_timing_prefill_ms",
    "Prefill GPU forward+sample chunk time, milliseconds.",
    labelnames=_LABELS,
)
gauge_p2d_kv_ms = Gauge(
    "vllm:sd_timing_p2d_kv_ms",
    "P→D NIXL KV transfer duration, milliseconds.",
    labelnames=_LABELS,
)


def note_p2d_kv_ms(xfer_ms: float) -> None:
    """Record one completed P→D KV transfer sample (called from NIXL worker)."""
    if not timing_model_enabled():
        return
    _p2d_kv_samples_ms.append(float(xfer_ms))


def drain_p2d_kv_ms() -> float | None:
    """Mean of stashed P→D samples since last drain, or None if empty."""
    global _p2d_kv_samples_ms
    if not _p2d_kv_samples_ms:
        return None
    mean_ms = sum(_p2d_kv_samples_ms) / len(_p2d_kv_samples_ms)
    _p2d_kv_samples_ms = []
    return mean_ms


def record_sd_timing(
    *,
    mode: str,
    num_reqs: int,
    tpv_ms: float,
    ttransfer_ms: float,
    td_ms: float,
    tzmq_ms: float,
    await_ms: float = 0.0,
    accepted: float = 0.0,
    stall_ms: float = 0.0,
) -> None:
    """Log / export one SDTiming sample. No-op unless timing model enabled."""
    if not timing_model_enabled():
        return

    # tzmq_ms is the token-return control-plane time (logged as T_tok).
    t_tok_ms = float(tzmq_ms)
    tad_ms = float(ttransfer_ms) + float(td_ms) + t_tok_ms
    tpv = float(tpv_ms)
    await_v = float(await_ms)
    accepted_v = float(accepted)
    stall_v = float(stall_ms)
    slack_ms = tpv - tad_ms
    ratio = (tad_ms / tpv) if tpv > 1e-9 else float("inf")
    step_ms = max(tpv, tad_ms) + await_v
    tpot_est_ms = step_ms / max(accepted_v, 1e-9)

    labels = {"mode": mode}
    gauge_tpv_ms.labels(**labels).set(tpv)
    gauge_tad_ms.labels(**labels).set(tad_ms)
    gauge_ttransfer_ms.labels(**labels).set(ttransfer_ms)
    gauge_td_ms.labels(**labels).set(td_ms)
    gauge_tzmq_ms.labels(**labels).set(t_tok_ms)
    gauge_slack_ms.labels(**labels).set(slack_ms)
    gauge_await_ms.labels(**labels).set(await_v)
    gauge_accepted.labels(**labels).set(accepted_v)
    gauge_tpot_est_ms.labels(**labels).set(tpot_est_ms)

    global _steps
    _steps += 1
    every = max(1, _timing_every())
    if _steps % every == 0:
        # T_hs maps to context/aux hiddens transfer (ttransfer).
        logger.info(
            "[DisaggDFlash][timing] mode=%s Tpv_ms=%.2f T_hs_ms=%.2f "
            "Td_ms=%.2f T_tok_ms=%.2f Tad_ms=%.2f slack_ms=%.2f "
            "await_ms=%.2f accepted=%.2f stall_ms=%.2f "
            "(n=%d step_ms=%.2f tpot_est_ms=%.2f ratio=%.3f)",
            mode,
            tpv,
            ttransfer_ms,
            td_ms,
            t_tok_ms,
            tad_ms,
            slack_ms,
            await_v,
            accepted_v,
            stall_v,
            num_reqs,
            step_ms,
            tpot_est_ms,
            ratio,
        )


def record_sd_handoff(
    *,
    mode: str,
    num_reqs: int,
    prefill_ms: float = 0.0,
    p2d_kv_ms: float = 0.0,
) -> None:
    """Log / export TTFT handoff pieces. No-op unless timing model enabled."""
    if not timing_model_enabled():
        return
    if prefill_ms <= 0.0 and p2d_kv_ms <= 0.0:
        return

    labels = {"mode": mode}
    if prefill_ms > 0.0:
        gauge_prefill_ms.labels(**labels).set(prefill_ms)
    if p2d_kv_ms > 0.0:
        gauge_p2d_kv_ms.labels(**labels).set(p2d_kv_ms)

    global _handoff_steps
    _handoff_steps += 1
    every = max(1, _timing_every())
    if _handoff_steps % every == 0:
        logger.info(
            "SDTimingHandoff mode=%s n=%d: prefill_ms=%.2f p2d_kv_ms=%.2f",
            mode,
            num_reqs,
            prefill_ms,
            p2d_kv_ms,
        )
