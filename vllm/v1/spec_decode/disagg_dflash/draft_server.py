# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CLI / library entry for the Disagg-DFlash draft server."""

from __future__ import annotations

import argparse

from msgspec import msgpack

from vllm.logger import init_logger
from vllm.v1.spec_decode.disagg_dflash.connector import DisaggDFlashServerSocket
from vllm.v1.spec_decode.disagg_dflash.draft_engine import DisaggDFlashDraftEngine
from vllm.v1.spec_decode.disagg_dflash.protocol import (
    PAYLOAD_IPC,
    PAYLOAD_NIXL,
    DisaggDFlashSpeculateRequest,
)

logger = init_logger(__name__)


def run_draft_server(
    *,
    draft_model: str,
    target_model: str,
    num_speculative_tokens: int,
    bind: str = "tcp://0.0.0.0:50051",
    max_model_len: int = 8192,
    max_num_seqs: int = 64,
    gpu_memory_utilization: float = 0.8,
    block_size: int = 16,
    num_gpu_blocks: int | None = None,
    attention_backend: str | None = "FLASH_ATTN",
    transport: str = "zmq",
    ipc_max_num_tokens: int = 16384,
    enable_logging_iteration_details: bool = False,
    enable_sd_timing_model: bool = False,
    sd_timing_model_log_every: int = 20,
    enable_dflash_draft_profile: bool = False,
    dflash_draft_profile_log_every: int = 20,
    disagg_dflash_nixl_log_every: int = -1,
    enable_cudagraph: bool = True,
    cudagraph_max_reqs: int | None = None,
    metrics_port: int = 9101,
    metrics_host: str = "0.0.0.0",
    enable_mfu_metrics: bool = True,
) -> None:
    from vllm.config import set_current_vllm_config
    from vllm.v1.spec_decode.disagg_dflash.metrics import start_draft_metrics_server

    engine = DisaggDFlashDraftEngine(
        draft_model=draft_model,
        target_model=target_model,
        num_speculative_tokens=num_speculative_tokens,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        block_size=block_size,
        num_gpu_blocks=num_gpu_blocks,
        attention_backend=attention_backend,
        ipc_max_num_tokens=ipc_max_num_tokens,
        enable_logging_iteration_details=enable_logging_iteration_details,
        enable_sd_timing_model=enable_sd_timing_model,
        sd_timing_model_log_every=sd_timing_model_log_every,
        enable_dflash_draft_profile=enable_dflash_draft_profile,
        dflash_draft_profile_log_every=dflash_draft_profile_log_every,
        disagg_dflash_nixl_log_every=disagg_dflash_nixl_log_every,
        enable_cudagraph=enable_cudagraph,
        cudagraph_max_reqs=cudagraph_max_reqs,
        enable_mfu_metrics=enable_mfu_metrics,
    )
    if metrics_port > 0:
        start_draft_metrics_server(
            metrics_port,
            host=metrics_host,
            model_name=draft_model,
        )
        engine._record_kv_metrics()
    server = DisaggDFlashServerSocket(bind)
    transport = (transport or "zmq").lower()

    if transport == "cuda_ipc":
        # Eagerly allocate / export IPC handles before clients connect.
        engine.ensure_ipc_staging()
    elif transport == "nixl":
        # Eagerly register draft VRAM staging + NIXL agent before clients connect.
        engine.ensure_nixl_staging()
    elif transport == "nccl":
        raise NotImplementedError(
            "--transport nccl is reserved but not implemented. "
            "Use zmq, cuda_ipc, or nixl."
        )
    elif transport != "zmq":
        raise ValueError(f"Unknown --transport {transport!r}")

    def speculate_decoder(frames: list[bytes]) -> DisaggDFlashSpeculateRequest:
        meta = msgpack.decode(frames[0])
        mode = meta.get("payload_mode")
        if mode == PAYLOAD_IPC:
            staging = engine.ensure_ipc_staging()
            # Pull peer-written IPC cudaMalloc buffer into torch staging first.
            n_ctx = int(meta["num_ctx_tokens"])
            hiddens = staging.pull_hiddens_from_ipc(n_ctx)
            return DisaggDFlashSpeculateRequest.decode(
                frames, context_hiddens=hiddens
            )
        if mode == PAYLOAD_NIXL:
            staging = engine.ensure_nixl_staging()
            # Verify already NIXL-WRITEs into the ping-pong slot named in meta.
            n_ctx = int(meta["num_ctx_tokens"])
            slot = int(meta.get("staging_slot", 0))
            hiddens = staging.take_hiddens(n_ctx, staging_slot=slot)
            return DisaggDFlashSpeculateRequest.decode(
                frames, context_hiddens=hiddens
            )
        return DisaggDFlashSpeculateRequest.decode(frames)

    def handler(cmd, payload):
        return engine.handle(cmd, payload)

    logger.info(
        "Disagg-DFlash draft server serving on %s (transport=%s)", bind, transport
    )
    try:
        # Keep VllmConfig active for attention / CustomOp during serve.
        with set_current_vllm_config(engine.vllm_config):
            engine.warmup()
            server.serve(handler, speculate_decoder=speculate_decoder)
    finally:
        server.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Disagg-DFlash draft server")
    parser.add_argument("--draft-model", required=True, help="DFlash draft model path")
    parser.add_argument(
        "--target-model",
        required=True,
        help="Target model path (for embed_tokens / lm_head weights)",
    )
    parser.add_argument("--num-speculative-tokens", type=int, required=True)
    parser.add_argument("--bind", default="tcp://0.0.0.0:50051")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help=(
            "Fraction of remaining GPU memory (after weights + CG/activation "
            "reserve) used for the draft KV pool. Same role as vllm serve."
        ),
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help=(
            "Draft KV page size in tokens (default 16). Same meaning as "
            "vllm serve --block-size. Must be a positive power-of-two that "
            "the attention backend supports (typically 16 or 32)."
        ),
    )
    parser.add_argument(
        "--num-gpu-blocks",
        type=int,
        default=None,
        help=(
            "Draft KV block pool size (token pages). Default: auto from "
            "remaining HBM × gpu_memory_utilization. Each block holds "
            "--block-size tokens."
        ),
    )
    parser.add_argument(
        "--attention-backend",
        default="FLASH_ATTN",
        help="Attention backend for non-causal DFlash (e.g. FLASH_ATTN)",
    )
    parser.add_argument(
        "--transport",
        default="zmq",
        choices=["zmq", "cuda_ipc", "nccl", "nixl"],
        help=(
            "Data-plane transport. zmq=tensors over ZMQ (default); "
            "cuda_ipc=same-node GPU staging via CUDA IPC; "
            "nixl=NIXL WRITE into draft VRAM staging (P/D-style; control on ZMQ). "
            "nccl is reserved."
        ),
    )
    parser.add_argument(
        "--ipc-max-num-tokens",
        type=int,
        default=16384,
        help=(
            "Hidden staging capacity in tokens (cuda_ipc and nixl)."
        ),
    )
    parser.add_argument(
        "--enable-logging-iteration-details",
        action="store_true",
        help=(
            "Log each draft speculate pack (EngineCore-style): "
            "context vs generation reqs, context/query tokens, live seqs, "
            "free slots/blocks, and elapsed ms."
        ),
    )
    parser.add_argument(
        "--enable-sd-timing-model",
        action="store_true",
        help=(
            "Measure draft forward_ms for SDTiming Tad (piggybacked on "
            "speculate response). Mirror of EngineCore --enable-sd-timing-model. "
            "Uses CUDA synchronize — debug only."
        ),
    )
    parser.add_argument(
        "--sd-timing-model-log-every",
        type=int,
        default=20,
        help="Log every N SDTiming samples (default 20).",
    )
    parser.add_argument(
        "--enable-dflash-draft-profile",
        action="store_true",
        help=(
            "CUDA-sync draft forward phase breakdown logs. Debug only."
        ),
    )
    parser.add_argument(
        "--dflash-draft-profile-log-every",
        type=int,
        default=20,
        help="Log every N draft profile samples (default 20).",
    )
    parser.add_argument(
        "--disagg-dflash-nixl-log-every",
        type=int,
        default=-1,
        help=(
            "NIXL/ZMQ transfer INFO log rate: -1 off (default), "
            "0 every transfer, N every Nth. Verbose — leave off for benches."
        ),
    )
    parser.add_argument(
        "--enable-cudagraph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Capture FULL CUDA graphs for the draft query forward (default on). "
            "Use --no-enable-cudagraph to force eager (debug / capture OOM)."
        ),
    )
    parser.add_argument(
        "--cudagraph-max-reqs",
        type=int,
        default=None,
        help=(
            "Cap FULL CUDA-graph capture batch size (requests). "
            "Default: --max-num-seqs. Lower if capture is slow or OOMs."
        ),
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=9101,
        help=(
            "Prometheus HTTP port for draft metrics "
            "(KV usage %% / GiB, MFU flops/bytes, speculate latency/batch size, "
            "reclaim/errors). Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--metrics-host",
        default="0.0.0.0",
        help="Bind address for the Prometheus metrics HTTP server.",
    )
    parser.add_argument(
        "--enable-mfu-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Expose EngineCore-compatible estimated flops/bytes counters on "
            "/metrics for draft query forwards (default on). "
            "Use --no-enable-mfu-metrics to disable."
        ),
    )
    args = parser.parse_args(argv)
    if args.block_size <= 0:
        parser.error("--block-size must be a positive integer")
    run_draft_server(
        draft_model=args.draft_model,
        target_model=args.target_model,
        num_speculative_tokens=args.num_speculative_tokens,
        bind=args.bind,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        block_size=args.block_size,
        num_gpu_blocks=args.num_gpu_blocks,
        attention_backend=args.attention_backend,
        transport=args.transport,
        ipc_max_num_tokens=args.ipc_max_num_tokens,
        enable_logging_iteration_details=args.enable_logging_iteration_details,
        enable_sd_timing_model=args.enable_sd_timing_model,
        sd_timing_model_log_every=args.sd_timing_model_log_every,
        enable_dflash_draft_profile=args.enable_dflash_draft_profile,
        dflash_draft_profile_log_every=args.dflash_draft_profile_log_every,
        disagg_dflash_nixl_log_every=args.disagg_dflash_nixl_log_every,
        enable_cudagraph=args.enable_cudagraph,
        cudagraph_max_reqs=args.cudagraph_max_reqs,
        metrics_port=args.metrics_port,
        metrics_host=args.metrics_host,
        enable_mfu_metrics=args.enable_mfu_metrics,
    )


if __name__ == "__main__":
    main()
