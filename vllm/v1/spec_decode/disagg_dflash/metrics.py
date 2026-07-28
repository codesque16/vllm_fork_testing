# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prometheus metrics for the Disagg-DFlash draft server."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, start_http_server

from vllm.logger import init_logger

logger = init_logger(__name__)

_LABELS = ("model_name",)

# --- KV / capacity (gauges) ---
# Disagg-prefixed (compat) + standard names for scrapers shared with verify.
gauge_kv_cache_usage = Gauge(
    "vllm:disagg_dflash_kv_cache_usage_perc",
    "Disagg-DFlash draft KV-cache usage. 1 means 100 percent usage.",
    labelnames=_LABELS,
)
gauge_kv_cache_free_blocks = Gauge(
    "vllm:disagg_dflash_kv_cache_free_blocks",
    "Disagg-DFlash draft free KV blocks (block 0 reserved).",
    labelnames=_LABELS,
)
gauge_kv_cache_total_blocks = Gauge(
    "vllm:disagg_dflash_kv_cache_total_blocks",
    "Disagg-DFlash draft usable KV blocks (excludes reserved block 0).",
    labelnames=_LABELS,
)
gauge_kv_cache_usage_std = Gauge(
    "vllm:kv_cache_usage_perc",
    "Draft KV-cache usage fraction (0..1). Same semantics as EngineCore.",
    labelnames=_LABELS,
)
gauge_kv_cache_usage_bytes = Gauge(
    "vllm:kv_cache_usage_bytes",
    "Exact draft KV bytes currently in use (used_blocks × bytes_per_block).",
    labelnames=_LABELS,
)
gauge_kv_cache_total_bytes = Gauge(
    "vllm:kv_cache_total_bytes",
    "Exact draft KV pool bytes allocated (usable_blocks × bytes_per_block).",
    labelnames=_LABELS,
)
gauge_kv_cache_usage_gib = Gauge(
    "vllm:kv_cache_usage_gib",
    "Draft KV currently in use, gibibytes.",
    labelnames=_LABELS,
)
gauge_kv_cache_total_gib = Gauge(
    "vllm:kv_cache_total_gib",
    "Draft KV pool allocated, gibibytes.",
    labelnames=_LABELS,
)
gauge_num_seqs = Gauge(
    "vllm:disagg_dflash_num_seqs",
    "Disagg-DFlash draft live sequences currently holding KV.",
    labelnames=_LABELS,
)
gauge_free_slots = Gauge(
    "vllm:disagg_dflash_free_slots",
    "Disagg-DFlash draft free request slots (max_num_seqs - live).",
    labelnames=_LABELS,
)
gauge_last_batch_reqs = Gauge(
    "vllm:disagg_dflash_last_batch_num_reqs",
    "Number of requests in the most recent draft speculate batch.",
    labelnames=_LABELS,
)
gauge_last_batch_ctx_tokens = Gauge(
    "vllm:disagg_dflash_last_batch_num_ctx_tokens",
    "Context tokens in the most recent draft speculate batch.",
    labelnames=_LABELS,
)
gauge_last_cg_mode = Gauge(
    "vllm:disagg_dflash_last_cg_mode",
    "1 if the last speculate used FULL CUDA graph, else 0 (eager).",
    labelnames=_LABELS,
)

# --- Throughput / health (counters) ---
counter_speculate_batches = Counter(
    "vllm:disagg_dflash_speculate_batches_total",
    "Total draft speculate RPC batches handled.",
    labelnames=_LABELS,
)
counter_speculate_reqs = Counter(
    "vllm:disagg_dflash_speculate_requests_total",
    "Total request-rows across draft speculate batches.",
    labelnames=_LABELS,
)
counter_speculate_errors = Counter(
    "vllm:disagg_dflash_speculate_errors_total",
    "Total draft speculate RPCs that raised (e.g. OOM).",
    labelnames=_LABELS,
)
counter_free_reqs = Counter(
    "vllm:disagg_dflash_free_requests_total",
    "Total request ids freed on the draft (RPC + reclaim).",
    labelnames=_LABELS,
)
counter_reclaim_reqs = Counter(
    "vllm:disagg_dflash_reclaim_requests_total",
    "Sequences force-reclaimed due to block pressure (FREE delayed/dropped).",
    labelnames=_LABELS,
)

# --- Latency ---
histogram_speculate_seconds = Histogram(
    "vllm:disagg_dflash_speculate_seconds",
    "Wall time of one draft speculate RPC (seconds).",
    labelnames=_LABELS,
    buckets=(
        0.001,
        0.002,
        0.003,
        0.005,
        0.0075,
        0.01,
        0.015,
        0.02,
        0.03,
        0.05,
        0.075,
        0.1,
        0.2,
        0.5,
        1.0,
    ),
)
histogram_batch_num_reqs = Histogram(
    "vllm:disagg_dflash_batch_num_reqs",
    "Request count per draft speculate batch.",
    labelnames=_LABELS,
    buckets=(1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024),
)

# Same names as EngineCore PerfMetricsProm so shared scrapers work.
# Rates: rate(estimated_flops_per_gpu_total) / 1e12 → TF/s/GPU;
#        rate(read+write bytes) / 1e9 → GB/s/GPU.
counter_estimated_flops = Counter(
    "vllm:estimated_flops_per_gpu_total",
    "Estimated floating point operations per GPU (draft query forward).",
    labelnames=_LABELS,
)
counter_estimated_read_bytes = Counter(
    "vllm:estimated_read_bytes_per_gpu_total",
    "Estimated bytes read from memory per GPU (draft query forward).",
    labelnames=_LABELS,
)
counter_estimated_write_bytes = Counter(
    "vllm:estimated_write_bytes_per_gpu_total",
    "Estimated bytes written to memory per GPU (draft query forward).",
    labelnames=_LABELS,
)

_metrics_started = False


def start_draft_metrics_server(
    port: int,
    host: str = "0.0.0.0",
    *,
    model_name: str = "",
) -> None:
    """Expose draft metrics at ``http://{host}:{port}/metrics``."""
    global _metrics_started
    if port <= 0:
        return
    if _metrics_started:
        return
    start_http_server(port, addr=host)
    _metrics_started = True
    observe_draft_kv(
        model_name=model_name,
        usage=0.0,
        free_blocks=0,
        total_blocks=0,
        num_seqs=0,
        free_slots=0,
        used_bytes=0,
        total_bytes=0,
    )
    logger.info(
        "Disagg-DFlash draft metrics listening on http://%s:%d/metrics",
        host,
        port,
    )


def _labels(model_name: str) -> dict[str, str]:
    return {"model_name": model_name or "unknown"}


def observe_draft_kv(
    *,
    model_name: str,
    usage: float,
    free_blocks: int,
    total_blocks: int,
    num_seqs: int,
    free_slots: int = 0,
    used_bytes: int = 0,
    total_bytes: int = 0,
) -> None:
    labels = _labels(model_name)
    gauge_kv_cache_usage.labels(**labels).set(usage)
    gauge_kv_cache_free_blocks.labels(**labels).set(free_blocks)
    gauge_kv_cache_total_blocks.labels(**labels).set(total_blocks)
    gauge_kv_cache_usage_std.labels(**labels).set(usage)
    gauge_kv_cache_usage_bytes.labels(**labels).set(used_bytes)
    gauge_kv_cache_total_bytes.labels(**labels).set(total_bytes)
    gib = 1024.0**3
    gauge_kv_cache_usage_gib.labels(**labels).set(used_bytes / gib)
    gauge_kv_cache_total_gib.labels(**labels).set(total_bytes / gib)
    gauge_num_seqs.labels(**labels).set(num_seqs)
    gauge_free_slots.labels(**labels).set(free_slots)


def observe_speculate(
    *,
    model_name: str,
    num_reqs: int,
    num_ctx_tokens: int,
    elapsed_s: float,
    cg_full: bool,
) -> None:
    labels = _labels(model_name)
    counter_speculate_batches.labels(**labels).inc()
    counter_speculate_reqs.labels(**labels).inc(num_reqs)
    gauge_last_batch_reqs.labels(**labels).set(num_reqs)
    gauge_last_batch_ctx_tokens.labels(**labels).set(num_ctx_tokens)
    gauge_last_cg_mode.labels(**labels).set(1.0 if cg_full else 0.0)
    histogram_speculate_seconds.labels(**labels).observe(elapsed_s)
    histogram_batch_num_reqs.labels(**labels).observe(num_reqs)


def observe_speculate_error(*, model_name: str) -> None:
    counter_speculate_errors.labels(**_labels(model_name)).inc()


def observe_estimated_mfu(
    *,
    model_name: str,
    num_flops_per_gpu: int,
    num_read_bytes_per_gpu: int,
    num_write_bytes_per_gpu: int,
) -> None:
    """Increment EngineCore-compatible MFU counters for one draft speculate."""
    if not (
        num_flops_per_gpu or num_read_bytes_per_gpu or num_write_bytes_per_gpu
    ):
        return
    labels = _labels(model_name)
    if num_flops_per_gpu:
        counter_estimated_flops.labels(**labels).inc(num_flops_per_gpu)
    if num_read_bytes_per_gpu:
        counter_estimated_read_bytes.labels(**labels).inc(num_read_bytes_per_gpu)
    if num_write_bytes_per_gpu:
        counter_estimated_write_bytes.labels(**labels).inc(num_write_bytes_per_gpu)


def observe_free(*, model_name: str, num_reqs: int) -> None:
    if num_reqs > 0:
        counter_free_reqs.labels(**_labels(model_name)).inc(num_reqs)


def observe_reclaim(*, model_name: str, num_reqs: int = 1) -> None:
    if num_reqs > 0:
        counter_reclaim_reqs.labels(**_labels(model_name)).inc(num_reqs)
