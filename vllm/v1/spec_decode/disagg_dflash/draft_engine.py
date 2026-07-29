# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone DFlash draft engine for the Disagg-DFlash server process."""

from __future__ import annotations

import gc
import os
import time
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    download_weights_from_hf,
    safetensors_weights_iterator,
)
from vllm.platforms import current_platform
from vllm.utils.mem_utils import DeviceMemoryProfiler, MemorySnapshot, format_gib
from vllm.utils.mem_utils import memory_profiling
from vllm.utils.network_utils import get_distributed_init_method, get_ip, get_open_port
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.utils import request_memory
from vllm.v1.spec_decode.disagg_dflash.protocol import (
    CMD_HELLO,
    DisaggDFlashFreeRequest,
    DisaggDFlashSpeculateRequest,
    DisaggDFlashSpeculateResponse,
    encode_hello_reply,
    encode_pong,
)
from vllm.v1.spec_decode.disagg_dflash.prep import prepare_disagg_dflash_inputs
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
    get_kv_cache_spec,
    init_attn_backend,
    init_kv_cache,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.spec_decode.dflash.utils import get_dflash_causal
from vllm.v1.worker.gpu.spec_decode.utils import get_parallel_drafting_token_id
from vllm.v1.worker.utils import prepare_kernel_block_sizes

logger = init_logger(__name__)

def _draft_profile_enabled() -> bool:
    from vllm.v1.spec_decode.disagg_dflash.debug_logging import draft_profile_enabled

    return draft_profile_enabled()


def _measure_draft_forward() -> bool:
    """CUDA-sync draft forward timing for profile and/or Tpv/Tad model."""
    if _draft_profile_enabled():
        return True
    from vllm.v1.spec_decode.disagg_dflash.timing_model import timing_model_enabled

    return timing_model_enabled()



def _iter_weights(model_path: str, revision: str | None = None):
    try:
        folder = download_weights_from_hf(
            model_path,
            cache_dir=None,
            allow_patterns=["*.safetensors", "*.bin"],
            revision=revision,
        )
    except Exception:
        folder = model_path
    st = []
    bins = []
    for root, _, files in os.walk(folder):
        for f in files:
            p = os.path.join(root, f)
            if f.endswith(".safetensors"):
                st.append(p)
            elif f.endswith(".bin"):
                bins.append(p)
    if st:
        yield from safetensors_weights_iterator(st, use_tqdm_on_load=False)
    else:
        for p in bins:
            sd = torch.load(p, map_location="cpu", weights_only=True)
            yield from sd.items()


def _copy_embedding_and_lm_head_from_target(
    draft_model: nn.Module, target_model: str, dtype: torch.dtype
) -> None:
    """Load target embed_tokens / lm_head into the draft model by value."""
    draft_inner = draft_model.model if hasattr(draft_model, "model") else draft_model
    embed = getattr(draft_inner, "embed_tokens", None)
    lm_head = getattr(draft_model, "lm_head", None)

    need_embed = embed is not None
    need_lm = lm_head is not None
    if not need_embed and not need_lm:
        return

    for name, tensor in _iter_weights(target_model):
        if need_embed and (
            name == "model.embed_tokens.weight" or name.endswith("embed_tokens.weight")
        ):
            weight_loader = getattr(embed.weight, "weight_loader", default_weight_loader)
            weight_loader(embed.weight, tensor.to(dtype=dtype))
            need_embed = False
            logger.info("Disagg-DFlash draft: loaded embed_tokens from target")
        if need_lm and (name == "lm_head.weight" or name.endswith("lm_head.weight")):
            weight_loader = getattr(
                lm_head.weight, "weight_loader", default_weight_loader
            )
            weight_loader(lm_head.weight, tensor.to(dtype=dtype))
            need_lm = False
            logger.info("Disagg-DFlash draft: loaded lm_head from target")
        if not need_embed and not need_lm:
            break

    if need_embed:
        logger.warning("Disagg-DFlash draft: embed_tokens not found in target weights")
    if need_lm:
        logger.warning("Disagg-DFlash draft: lm_head not found in target weights")


class _BlockPool:
    def __init__(self, num_blocks: int):
        # Block 0 reserved as null.
        # Free list as a stack: allocate/free touch only the end (O(n_alloc)).
        # The old `self._free = self._free[n:]` copied the entire remaining
        # pool on every alloc (~2M blocks → ~10ms spikes in prep_ensure_ms).
        self.num_blocks = num_blocks
        self._free = list(range(1, num_blocks))

    def num_free(self) -> int:
        return len(self._free)

    def allocate(self, n: int) -> list[int]:
        if n > len(self._free):
            raise RuntimeError(
                f"Disagg-DFlash draft OOM: need {n} blocks, have {len(self._free)} "
                f"(total={self.num_blocks - 1} usable). "
                f"Raise --num-gpu-blocks or check FREE RPCs are reaching the draft."
            )
        # Pop from the end so we never rewrite the remaining free list.
        start = len(self._free) - n
        out = self._free[start:]
        del self._free[start:]
        return out

    def free(self, blocks: list[int]) -> None:
        self._free.extend(blocks)


class DisaggDFlashDraftEngine:
    """Owns DFlash weights + KV on the draft GPU(s)."""

    def __init__(
        self,
        draft_model: str,
        target_model: str,
        num_speculative_tokens: int,
        max_model_len: int = 8192,
        max_num_seqs: int = 64,
        gpu_memory_utilization: float = 0.8,
        dtype: str = "auto",
        block_size: int = 16,
        num_gpu_blocks: int | None = None,
        attention_backend: str | None = None,
        ipc_max_num_tokens: int = 16384,
        enable_logging_iteration_details: bool = False,
        enable_sd_timing_model: bool = False,
        sd_timing_model_log_every: int = 20,
        enable_dflash_draft_profile: bool = False,
        dflash_draft_profile_log_every: int = 20,
        disagg_dflash_nixl_log_every: int = -1,
        enable_cudagraph: bool = True,
        cudagraph_max_reqs: int | None = None,
        enable_mfu_metrics: bool = True,
    ):
        self.num_speculative_tokens = num_speculative_tokens
        self.num_query_per_req = 1 + num_speculative_tokens
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.block_size = block_size
        self.ipc_max_num_tokens = ipc_max_num_tokens
        self._gpu_memory_utilization = float(gpu_memory_utilization)
        self._ipc_staging = None
        self._nixl_staging = None
        self.draft_model_name = draft_model
        self._enable_mfu_metrics = bool(enable_mfu_metrics)
        self._perf_metrics = None
        self._init_snapshot: MemorySnapshot | None = None
        self._weights_memory: int = 0
        # Mirror EngineCore --enable-logging-iteration-details for the draft
        # RPC loop (no SchedulerOutput; log the verify-sent pack instead).
        self.enable_logging_iteration_details = bool(enable_logging_iteration_details)
        from vllm.v1.spec_decode.disagg_dflash.debug_logging import (
            configure_disagg_debug_logging,
        )

        configure_disagg_debug_logging(
            enable_sd_timing_model=bool(enable_sd_timing_model),
            sd_timing_model_log_every=int(sd_timing_model_log_every),
            enable_dflash_draft_profile=bool(enable_dflash_draft_profile),
            dflash_draft_profile_log_every=int(dflash_draft_profile_log_every),
            disagg_dflash_nixl_log_every=int(disagg_dflash_nixl_log_every),
        )
        self._enable_cudagraph = bool(enable_cudagraph)
        self._cudagraph_max_reqs = (
            int(cudagraph_max_reqs) if cudagraph_max_reqs is not None else None
        )
        self._bytes_per_block: int = 0
        self._iteration_index = 0
        # Tick bumped on every speculate; used for LRU reclaim when the pool
        # is empty (FREE RPC dropped / delayed under load).
        self._tick = 0

        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        # Resolved after draft weights load (fits into free HBM). CLI None → auto.
        self._num_gpu_blocks_arg = num_gpu_blocks

        # Target config is required by DFlashQwen3ForCausalLM (layer count / vocab).
        target_model_config = ModelConfig(
            model=target_model,
            runner="generate",
            max_model_len=max_model_len,
            dtype=dtype,
            trust_remote_code=True,
        )
        from vllm.config import SpeculativeConfig
        from vllm.config.attention import AttentionConfig
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        attn_backend = None
        if attention_backend is not None:
            attn_backend = AttentionBackendEnum[attention_backend.upper()]

        speculative_config = SpeculativeConfig(
            target_model_config=target_model_config,
            target_parallel_config=ParallelConfig(tensor_parallel_size=1),
            model=draft_model,
            method="dflash",
            num_speculative_tokens=num_speculative_tokens,
            attention_backend=attn_backend,
        )
        draft_model_config = speculative_config.draft_model_config
        dflash_cfg = getattr(draft_model_config.hf_config, "dflash_config", None) or {}
        causal = bool(dflash_cfg.get("causal", False))

        max_num_batched_tokens = min(
            max_num_seqs * self.num_query_per_req * 4,
            32768,
        )
        self.vllm_config = VllmConfig(
            model_config=target_model_config,
            cache_config=CacheConfig(
                block_size=block_size,
                gpu_memory_utilization=gpu_memory_utilization,
                cache_dtype="auto",
                enable_prefix_caching=False,
            ),
            parallel_config=ParallelConfig(tensor_parallel_size=1),
            scheduler_config=SchedulerConfig(
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
                max_model_len=max_model_len,
                is_encoder_decoder=target_model_config.is_encoder_decoder,
            ),
            device_config=DeviceConfig(device="cuda"),
            load_config=LoadConfig(),
            speculative_config=speculative_config,
            attention_config=AttentionConfig(
                use_non_causal=not causal,
                backend=attn_backend,
            ),
        )
        if self._enable_mfu_metrics:
            self._init_perf_metrics(draft_model_config)

        # Model-parallel init requires the current VllmConfig to be set.
        with set_current_vllm_config(self.vllm_config):
            self._init_distributed()
            # Same as Worker.init_device: snapshot free memory *before* weights
            # so request_memory() / available-KV math matches EngineCore.
            gc.collect()
            torch.accelerator.empty_cache()
            self._init_snapshot = MemorySnapshot(device=self.device)
            self.vllm_config.cache_config.gpu_memory_utilization = (
                self._gpu_memory_utilization
            )

            # Load DFlash weights as the draft model_config (same as colocated path).
            with DeviceMemoryProfiler(self.device) as m:
                self.model = get_model(
                    vllm_config=self.vllm_config,
                    model_config=draft_model_config,
                )
                _copy_embedding_and_lm_head_from_target(
                    self.model, target_model, self.vllm_config.model_config.dtype
                )
                if hasattr(self.model, "model") and hasattr(
                    self.model.model, "_build_fused_kv_buffers"
                ):
                    self.model.model._build_fused_kv_buffers()
            self._weights_memory = int(m.consumed_memory)
            logger.info(
                "Disagg-DFlash draft: model loading took %s GiB",
                format_gib(self._weights_memory),
            )

            self.parallel_drafting_token_id = get_parallel_drafting_token_id(
                draft_model_config.hf_config
            )
            self.dflash_causal = get_dflash_causal(draft_model_config)
            self.hidden_size = draft_model_config.get_hidden_size()
            self.dtype = self.vllm_config.model_config.dtype
            draft_inner = (
                self.model.model if hasattr(self.model, "model") else self.model
            )
            embed = getattr(draft_inner, "embed_tokens", None)
            if embed is None and hasattr(self.model, "get_input_embeddings"):
                embed = self.model.get_input_embeddings()
            if embed is None:
                raise RuntimeError(
                    "Disagg-DFlash draft: could not resolve embed_tokens for vocab size"
                )
            self.embed_vocab_size = int(embed.weight.shape[0])
            if (
                self.parallel_drafting_token_id < 0
                or self.parallel_drafting_token_id >= self.embed_vocab_size
            ):
                raise RuntimeError(
                    "Disagg-DFlash mask_token_id="
                    f"{self.parallel_drafting_token_id} out of range for "
                    f"embed vocab_size={self.embed_vocab_size}"
                )

            # KV / attention setup (must stay inside config context).
            # Use draft model layers for KV specs: temporarily swap model_config
            # for get_kv_cache_spec discovery against the loaded draft module.
            kv_cache_spec = get_kv_cache_spec(self.vllm_config)
            # If target has no draft layers in forward context yet, fall back to
            # inspecting the loaded draft module's attention layers.
            if not kv_cache_spec:
                from vllm.config import get_layers_from_vllm_config
                from vllm.model_executor.layers.attention_layer_base import (
                    AttentionLayerBase,
                )

                # Ensure draft attentions are registered in static_forward_context
                # by re-running get_kv_cache_spec after model init (layers register
                # themselves during construction under set_current_vllm_config).
                kv_cache_spec = get_kv_cache_spec(self.vllm_config)

            from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
            from vllm.v1.kv_cache_interface import KVCacheTensor

            layer_names = list(kv_cache_spec.keys())
            assert layer_names, "DFlash draft model produced no KV cache specs"
            kv_cache_groups = get_kv_cache_groups(self.vllm_config, kv_cache_spec)
            assert kv_cache_groups, "DFlash draft produced no KV cache groups"

            # Bytes for one logical block id across all independent arenas.
            bytes_per_block = sum(
                int(g.kv_cache_spec.page_size_bytes) for g in kv_cache_groups
            )
            self._bytes_per_block = int(bytes_per_block)
            num_gpu_blocks = self._compute_num_gpu_blocks(
                bytes_per_block=bytes_per_block,
                num_kv_groups=len(kv_cache_groups),
            )

            # One arena PER KV group. Qwen3.5-DFlash produces 6 groups
            # (5× SlidingWindowSpec + 1× FullAttentionSpec). Sharing a single
            # raw tensor across those groups while mirroring the same block IDs
            # made every layer overwrite the same physical pages → ~0% accept.
            # Independent arenas let us safely mirror block IDs across groups.
            kv_cache_tensors = [
                KVCacheTensor(
                    size=int(group.kv_cache_spec.page_size_bytes) * num_gpu_blocks,
                    shared_by=list(group.layer_names),
                )
                for group in kv_cache_groups
            ]
            self.kv_cache_config = KVCacheConfig(
                num_blocks=num_gpu_blocks,
                kv_cache_tensors=kv_cache_tensors,
                kv_cache_groups=kv_cache_groups,
            )
            self.vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
            self._log_kv_cache_capacity(int(num_gpu_blocks) * int(bytes_per_block))

            self.attn_groups, self.attn_cg_support, _ = init_attn_backend(
                self.kv_cache_config,
                self.vllm_config,
                self.device,
                active_layer_names=set(layer_names),
            )
            self.kernel_block_sizes = prepare_kernel_block_sizes(
                self.kv_cache_config, self.attn_groups
            )
            max_blocks = (max_model_len + block_size - 1) // block_size + 2
            n_groups = len(self.kv_cache_config.kv_cache_groups)
            if len(set(self.kernel_block_sizes)) > 1:
                raise NotImplementedError(
                    "Disagg-DFlash draft currently requires uniform "
                    f"kernel_block_sizes across KV groups; got {self.kernel_block_sizes}"
                )
            self.block_tables = BlockTables(
                block_sizes=[block_size] * n_groups,
                max_num_reqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_blocks_per_group=[max_blocks] * n_groups,
                device=self.device,
                kernel_block_sizes=self.kernel_block_sizes,
            )
            logger.info(
                "Disagg-DFlash draft KV groups: %d (%s); "
                "allocated %d independent KV arenas × %d blocks",
                n_groups,
                [
                    (len(g.layer_names), type(g.kv_cache_spec).__name__)
                    for g in kv_cache_groups
                ],
                len(kv_cache_tensors),
                num_gpu_blocks,
            )
            self._runner_kv_caches: list = []
            init_kv_cache(
                self._runner_kv_caches,
                self.vllm_config.compilation_config.static_forward_context,
                self.kv_cache_config,
                self.attn_groups,
                self.device,
                self.vllm_config.cache_config.cache_dtype,
                self.kernel_block_sizes,
                self.vllm_config,
            )

        self.block_pool = _BlockPool(num_gpu_blocks)
        # req_id -> state
        self._seqs: dict[str, dict[str, Any]] = {}
        self._req_id_to_slot: dict[str, int] = {}
        # Pop from the front so early requests land in low slot rows. Still
        # must gather block-table rows by *actual* slot below — never assume
        # the batch is packed into rows [0, num_reqs).
        self._free_slots: deque[int] = deque(range(max_num_seqs))

        self.input_buffers = InputBuffers(
            max_num_reqs=max_num_seqs,
            max_num_tokens=max_num_seqs * self.num_query_per_req,
            device=self.device,
        )
        self.draft_tokens = torch.zeros(
            max_num_seqs,
            num_speculative_tokens,
            dtype=torch.int64,
            device=self.device,
        )
        # Precompute sample indices for mask positions (skip bonus at offset 0).
        # Layout matches colocated DFlash; reused by CUDA-graph and eager paths.
        max_query = max_num_seqs * self.num_query_per_req
        self._sample_indices = torch.empty(
            max_num_seqs * self.num_speculative_tokens,
            dtype=torch.long,
            device=self.device,
        )
        for i in range(max_num_seqs):
            base = i * self.num_query_per_req
            for j in range(self.num_speculative_tokens):
                self._sample_indices[i * self.num_speculative_tokens + j] = base + 1 + j

        # Dense BT rows for attention / CUDA graphs. Arena slots are packed here
        # so FA metadata always reads the same persistent addresses (required for
        # FULL CG replay). See BlockTables.get_dummy_block_tables.
        self._packed_block_tables = [
            torch.zeros_like(bt) for bt in self.block_tables.input_block_tables
        ]
        # Persistent prep buffers (avoid per-step torch.zeros / Python slot loops).
        self._context_slot_mapping = torch.zeros(
            self.ipc_max_num_tokens, dtype=torch.int64, device=self.device
        )
        self._query_slot_mapping = torch.zeros(
            max_query, dtype=torch.int64, device=self.device
        )
        self._bonus_gpu = torch.zeros(
            max_num_seqs, dtype=torch.int32, device=self.device
        )
        self._last_valid_pos_gpu = torch.zeros(
            max_num_seqs, dtype=torch.int64, device=self.device
        )
        self._qsl_gpu = torch.zeros(
            max_num_seqs + 1, dtype=torch.int32, device=self.device
        )
        # Pinned host staging for prep meta / BT appends (avoid per-step
        # torch.as_tensor(..., device=cuda) syncs that serialize ctx H2D).
        self._slots_gpu = torch.zeros(
            max_num_seqs, dtype=torch.long, device=self.device
        )
        pin = self.device.type == "cuda"
        self._slots_host = torch.zeros(
            max_num_seqs, dtype=torch.long, pin_memory=pin
        )
        self._qsl_host = torch.zeros(
            max_num_seqs + 1, dtype=torch.int32, pin_memory=pin
        )
        self._bonus_host = torch.zeros(
            max_num_seqs, dtype=torch.int32, pin_memory=pin
        )
        self._last_valid_pos_host = torch.zeros(
            max_num_seqs, dtype=torch.int64, pin_memory=pin
        )
        # Worst case: every seq grows by many blocks in one step (rare);
        # size for one full row rewrite as a safety valve.
        max_bt_cols = int(self.block_tables.input_block_tables[0].shape[1])
        self._bt_append_host = torch.empty(
            max_num_seqs * max_bt_cols, dtype=torch.int32, pin_memory=pin
        )
        self._bt_append_gpu = torch.empty(
            max_num_seqs * max_bt_cols, dtype=torch.int32, device=self.device
        )
        self.query_cudagraph_manager: DFlashCudaGraphManager | None = None
        self._cudagraph_enabled = False
        self._init_cudagraph_manager()

        logger.info(
            "Disagg-DFlash draft engine ready: draft=%s target_embed_from=%s K=%d "
            "cudagraph=%s",
            draft_model,
            target_model,
            num_speculative_tokens,
            self._cudagraph_enabled,
        )

    def _max_cg_reqs(self) -> int:
        cg_cap = (
            self._cudagraph_max_reqs
            if self._cudagraph_max_reqs is not None
            else self.max_num_seqs
        )
        return min(self.max_num_seqs, max(1, int(cg_cap)))

    def _planned_capture_token_sizes(self) -> list[int]:
        """Token counts that FULL CG capture will retain (same as manager init)."""
        query_len = self.num_query_per_req
        max_capture = self._max_cg_reqs() * query_len
        capture_sizes = [i for i in (1, 2, 4) if i <= max_capture]
        if max_capture >= 8:
            capture_sizes += list(range(8, min(max_capture + 1, 256), 8))
        if max_capture >= 256:
            capture_sizes += list(range(256, max_capture + 1, 16))
        if max_capture not in capture_sizes:
            capture_sizes.append(max_capture)
        return sorted(set(capture_sizes))

    def _profile_peak_activation_scratch(self) -> None:
        """Drive a torch peak without KV (draft has no skip_attn profile_run).

        Allocates a working set sized like the largest FULL CG batch so
        ``memory_profiling`` records a realistic activation headroom.
        """
        n = max(self._max_cg_reqs() * self.num_query_per_req, self.num_query_per_req)
        h = int(self.hidden_size)
        # Concurrent activations roughly similar to a small transformer step.
        a = torch.empty((n, h), dtype=self.dtype, device=self.device)
        b = torch.empty((n, h), dtype=self.dtype, device=self.device)
        c = torch.empty((n, h * 4), dtype=self.dtype, device=self.device)
        torch.accelerator.synchronize()
        del a, b, c

    def _profile_cudagraph_memory_like_enginecore(self) -> int:
        """Same estimator shape as GPUModelRunner.profile_cudagraph_memory.

        Capture 2 sample CUDA graphs (largest + smallest token count), measure
        free-memory deltas, then:
            estimate = first_capture + (n_graphs - 1) * per_graph
        with ``per_graph = max(second_delta, 1 MiB)``.

        Done before real draft KV/attention exist, so the captured body is a
        draft-sized activation workspace (not the full FA path). That matches
        EngineCore's *method*; absolute bytes are a slight underestimate vs
        full ``capture_cudagraphs()``, with ``gpu_memory_utilization`` as the
        remaining safety margin.
        """
        if not self._enable_cudagraph:
            return 0
        if not envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:
            return 0

        sizes = self._planned_capture_token_sizes()
        if not sizes:
            return 0
        n_graphs = len(sizes)
        # EngineCore profiles descs[:2] with largest first.
        sample_token_counts = [sizes[-1]] + ([sizes[0]] if n_graphs > 1 else [])
        hidden = int(self.hidden_size)
        dtype = self.dtype

        def _make_body(num_tokens: int):
            # Static buffers retained by the graph (draft-scale workspace).
            bufs = [
                torch.zeros(
                    (num_tokens, hidden), dtype=dtype, device=self.device
                )
                for _ in range(3)
            ]

            def body():
                x = bufs[0]
                x = x + bufs[1]
                x = x * bufs[2]
                bufs[0].copy_(x)

            return body, bufs

        mem_samples: list[int] = []
        keep_alive: list[Any] = []
        try:
            torch.accelerator.synchronize()
            torch.accelerator.empty_cache()
            for nt in sample_token_counts:
                body, bufs = _make_body(nt)
                # Warmup outside the graph (allocator + kernels).
                body()
                torch.accelerator.synchronize()
                free_before = torch.accelerator.get_memory_info()[0]
                g = torch.cuda.CUDAGraph()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.graph(g, stream=s):
                    body()
                torch.cuda.current_stream().wait_stream(s)
                torch.accelerator.synchronize()
                free_after = torch.accelerator.get_memory_info()[0]
                mem_samples.append(max(free_before - free_after, 0))
                keep_alive.append((g, bufs))

            first_capture = mem_samples[0]
            per_graph = max(
                mem_samples[1] if len(mem_samples) > 1 else 0, 1 << 20
            )
            estimate = int(first_capture + per_graph * (n_graphs - 1))
            logger.info(
                "Disagg-DFlash draft: estimated CUDA graph memory %.2f GiB "
                "(%d graphs; first=%.2f MiB, per_graph=%.2f MiB × %d)",
                estimate / (1024**3),
                n_graphs,
                first_capture / (1 << 20),
                per_graph / (1 << 20),
                n_graphs - 1,
            )
            return estimate
        finally:
            keep_alive.clear()
            gc.collect()
            torch.accelerator.empty_cache()
            torch.accelerator.synchronize()

    def _determine_available_kv_memory_bytes(self) -> int:
        """Mirror Worker.determine_available_memory (no MM IPC carve-out).

        available = request_memory(init_snapshot)
                    - non_kv_cache_memory
                    - cudagraph_memory_estimate
        """
        assert self._init_snapshot is not None
        requested = request_memory(
            self._init_snapshot, self.vllm_config.cache_config
        )

        cudagraph_memory_estimate = 0
        with memory_profiling(
            self._init_snapshot, weights_memory=self._weights_memory
        ) as profile_result:
            self._profile_peak_activation_scratch()
            profile_torch_peak = torch.accelerator.memory_stats(self.device).get(
                "allocated_bytes.all.peak", 0
            )
            if self._enable_cudagraph and envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:
                cudagraph_memory_estimate = (
                    self._profile_cudagraph_memory_like_enginecore()
                )

        # Use pre-cudagraph torch peak (same as gpu_worker).
        profile_result.torch_peak_increase = (
            profile_torch_peak - profile_result.before_profile.torch_peak
        )
        profile_result.non_kv_cache_memory = (
            profile_result.non_torch_increase
            + profile_result.torch_peak_increase
            + profile_result.weights_memory
        )
        cg_applied = (
            cudagraph_memory_estimate
            if envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS
            else 0
        )
        available = int(
            requested - profile_result.non_kv_cache_memory - cg_applied
        )
        logger.info(
            "Available KV cache memory: %s GiB",
            format_gib(max(available, 0)),
        )
        logger.info(
            "Disagg-DFlash draft memory profile: requested=%s GiB "
            "weights=%s GiB activation_peak=%s GiB non_torch=%s GiB "
            "cudagraph_est=%s GiB (applied=%s)",
            format_gib(requested),
            format_gib(self._weights_memory),
            format_gib(profile_result.torch_peak_increase),
            format_gib(profile_result.non_torch_increase),
            format_gib(cudagraph_memory_estimate),
            format_gib(cg_applied),
        )
        return max(available, 0)

    def _compute_num_gpu_blocks(
        self, *, bytes_per_block: int, num_kv_groups: int
    ) -> int:
        """Size KV blocks from EngineCore-style available memory."""
        bpb = max(int(bytes_per_block), 1)
        floor_blocks = max(self.max_num_seqs + 1, 1024)

        if self._num_gpu_blocks_arg is not None:
            num_gpu_blocks = max(int(self._num_gpu_blocks_arg), 2)
            logger.info(
                "Disagg-DFlash draft: --num-gpu-blocks=%d (%.2f GiB KV)",
                num_gpu_blocks,
                (num_gpu_blocks * bpb) / (1024**3),
            )
            return num_gpu_blocks

        available = self._determine_available_kv_memory_bytes()
        # No artificial block cap — EngineCore available_kv already bounds HBM.
        # (Old hard_cap=262144 × 32KiB/block was clamping a 66 GiB budget to 8 GiB.)
        mem_blocks = max(available // bpb, 0)
        num_gpu_blocks = max(floor_blocks, mem_blocks)
        if mem_blocks < floor_blocks:
            logger.warning(
                "Disagg-DFlash draft: profiled KV budget only fits %d blocks "
                "(floor=%d); using floor — risk of OOM under load.",
                mem_blocks,
                floor_blocks,
            )
        logger.info(
            "Disagg-DFlash draft: auto --num-gpu-blocks=%d "
            "(%.2f GiB KV @ %.1f KiB/block×%d groups; "
            "available_kv=%s GiB gpu_memory_utilization=%.2f)",
            num_gpu_blocks,
            (num_gpu_blocks * bpb) / (1024**3),
            bpb / 1024,
            num_kv_groups,
            format_gib(available),
            self._gpu_memory_utilization,
        )
        return num_gpu_blocks

    def _log_kv_cache_capacity(self, available_kv_bytes: int) -> None:
        """Mirror EngineCore's size / concurrency INFO lines."""
        from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity

        # Available line already logged in _determine_available_kv_memory_bytes
        # for the auto path; still print when using --num-gpu-blocks override.
        if self._num_gpu_blocks_arg is not None:
            logger.info(
                "Available KV cache memory: %s GiB",
                format_gib(int(available_kv_bytes)),
            )
        num_tokens, max_concurrency = get_kv_cache_capacity(
            self.vllm_config, self.kv_cache_config
        )
        logger.info("GPU KV cache size: %s tokens", f"{num_tokens:,}")
        logger.info(
            "Maximum concurrency for %s tokens per request: %.2fx",
            f"{self.max_model_len:,}",
            max_concurrency,
        )

    def _init_distributed(self) -> None:
        if not torch.distributed.is_initialized():
            init_method = get_distributed_init_method(get_ip(), get_open_port())
            # Use gloo for the single-process draft server so we do not
            # interfere with the verify fleet's NCCL process groups.
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=init_method,
                local_rank=0,
                backend="gloo",
            )
            ensure_model_parallel_initialized(1, 1)

    def kv_cache_usage(self) -> float:
        """Fraction of usable draft KV blocks in use (0..1), like target usage."""
        usable = max(self.block_pool.num_blocks - 1, 1)
        return 1.0 - self.block_pool.num_free() / usable

    def _init_perf_metrics(self, draft_model_config) -> None:
        """Build EngineCore-compatible analytic MFU estimators for the draft."""
        from vllm.v1.metrics.perf import ModelMetrics

        # Engine VllmConfig.model_config is the *target*; MFU must use draft dims.
        saved = self.vllm_config.model_config
        try:
            self.vllm_config.model_config = draft_model_config
            metrics = ModelMetrics(self.vllm_config)
        except Exception:
            logger.exception(
                "Disagg-DFlash: failed to init draft MFU metrics; counters disabled"
            )
            return
        finally:
            self.vllm_config.model_config = saved
        if not metrics.is_enabled():
            logger.warning(
                "Disagg-DFlash: no ComponentMetrics for draft model; "
                "estimated_flops/bytes counters will stay at 0"
            )
            return
        self._perf_metrics = metrics
        logger.info(
            "Disagg-DFlash draft MFU metrics enabled (%d components)",
            len(metrics.metrics),
        )

    def _observe_mfu(self, *, num_reqs: int, seq_lens_np) -> None:
        """Estimate flops/bytes for the draft query forward and bump counters."""
        if self._perf_metrics is None or num_reqs <= 0:
            return
        from vllm.v1.metrics.perf import ExecutionContext
        from vllm.v1.spec_decode.disagg_dflash.metrics import observe_estimated_mfu

        ctx = ExecutionContext()
        qpr = int(self.num_query_per_req)
        is_prefill = qpr > 1
        for i in range(num_reqs):
            ctx.add(qpr, int(seq_lens_np[i]), is_prefill=is_prefill)
        observe_estimated_mfu(
            model_name=self.draft_model_name,
            num_flops_per_gpu=self._perf_metrics.get_num_flops(ctx, True),
            num_read_bytes_per_gpu=self._perf_metrics.get_read_bytes(ctx, True),
            num_write_bytes_per_gpu=self._perf_metrics.get_write_bytes(ctx, True),
        )

    def _record_kv_metrics(self) -> None:
        from vllm.v1.spec_decode.disagg_dflash.metrics import observe_draft_kv

        usable = max(self.block_pool.num_blocks - 1, 0)
        free = self.block_pool.num_free()
        used_blocks = max(usable - free, 0)
        bpb = max(int(self._bytes_per_block), 0)
        observe_draft_kv(
            model_name=self.draft_model_name,
            usage=self.kv_cache_usage(),
            free_blocks=free,
            total_blocks=usable,
            num_seqs=len(self._seqs),
            free_slots=len(self._free_slots),
            used_bytes=used_blocks * bpb,
            total_bytes=usable * bpb,
        )

    def free(self, request: DisaggDFlashFreeRequest) -> list[bytes]:
        from vllm.v1.spec_decode.disagg_dflash.metrics import observe_free
        from vllm.v1.spec_decode.disagg_dflash.protocol import encode_pong

        n_freed = 0
        for req_id in request.req_ids:
            state = self._seqs.pop(req_id, None)
            slot = self._req_id_to_slot.pop(req_id, None)
            if state is not None:
                self.block_pool.free(state["blocks"])
                n_freed += 1
            if slot is not None:
                self._free_slots.append(slot)
                # Clear block table row (gpu + CPU mirror used for slot mapping).
                for bt in self.block_tables.block_tables:
                    bt.gpu[slot].zero_()
                for ibt in self.block_tables.input_block_tables:
                    ibt[slot].zero_()
                self.block_tables.num_blocks.gpu[:, slot] = 0
        observe_free(model_name=self.draft_model_name, num_reqs=n_freed)
        self._record_kv_metrics()
        return encode_pong()

    def _ensure_seq(self, req_id: str) -> int:
        if req_id in self._req_id_to_slot:
            return self._req_id_to_slot[req_id]
        if not self._free_slots:
            # Last-resort reclaim: drop the oldest tracked request.
            # This should not happen if FREE RPCs are delivered; log loudly.
            victim = next(iter(self._req_id_to_slot))
            logger.warning(
                "Disagg-DFlash draft: no free slots (%d in use); "
                "force-freeing oldest req_id=%s (FREE RPC likely dropped)",
                len(self._req_id_to_slot),
                victim,
            )
            self.free(DisaggDFlashFreeRequest(req_ids=[victim]))
        if not self._free_slots:
            raise RuntimeError("Disagg-DFlash draft: no free request slots")
        slot = self._free_slots.popleft()
        self._req_id_to_slot[req_id] = slot
        self._seqs[req_id] = {
            "slot": slot,
            "blocks": [],
            "ctx_len": 0,
            "bt_synced_n": 0,
            "last_used": self._tick,
        }
        return slot

    def _reclaim_blocks(self, need: int, protect: set[str]) -> None:
        """Free LRU sequences not in ``protect`` until ``need`` blocks are free."""
        if self.block_pool.num_free() >= need:
            return
        victims = sorted(
            (
                (rid, state)
                for rid, state in self._seqs.items()
                if rid not in protect
            ),
            key=lambda x: x[1].get("last_used", 0),
        )
        for rid, state in victims:
            if self.block_pool.num_free() >= need:
                break
            n_blk = len(state["blocks"])
            logger.warning(
                "Disagg-DFlash draft: reclaiming req_id=%s (%d blocks, "
                "free=%d need=%d, %d seqs live) — FREE likely dropped/delayed",
                rid,
                n_blk,
                self.block_pool.num_free(),
                need,
                len(self._seqs),
            )
            from vllm.v1.spec_decode.disagg_dflash.metrics import observe_reclaim

            observe_reclaim(model_name=self.draft_model_name, num_reqs=1)
            self.free(DisaggDFlashFreeRequest(req_ids=[rid]))

    def _sync_bt_row_full(self, slot: int, blocks: list[int]) -> None:
        """Rewrite an entire arena BT row (fallback / first sync)."""
        n = len(blocks)
        host = self._bt_append_host
        host[:n] = torch.as_tensor(blocks, dtype=torch.int32)
        gpu = self._bt_append_gpu
        gpu[:n].copy_(host[:n], non_blocking=True)
        n_groups = len(self.block_tables.block_tables)
        for gid in range(n_groups):
            bt = self.block_tables.block_tables[gid].gpu
            bt[slot, :n].copy_(gpu[:n], non_blocking=True)
            if n < bt.shape[1]:
                bt[slot, n:].zero_()
            self.block_tables.num_blocks.gpu[gid, slot] = n
            ibt = self.block_tables.input_block_tables[gid]
            ibt[slot, :n].copy_(gpu[:n], non_blocking=True)
            if n < ibt.shape[1]:
                ibt[slot, n:].zero_()

    def _append_bt_rows(
        self, updates: list[tuple[int, int, list[int]]]
    ) -> None:
        """Append newly allocated block ids into arena BT rows.

        ``updates``: list of (slot, start_col, new_block_ids).
        """
        if not updates:
            return
        host = self._bt_append_host
        gpu = self._bt_append_gpu
        flat: list[int] = []
        spans: list[tuple[int, int, int]] = []  # slot, start_col, length
        for slot, start_col, new_blocks in updates:
            n = len(new_blocks)
            if n == 0:
                continue
            flat.extend(new_blocks)
            spans.append((slot, start_col, n))
        cursor = len(flat)
        if cursor == 0:
            return
        host[:cursor] = torch.as_tensor(flat, dtype=torch.int32)
        gpu[:cursor].copy_(host[:cursor], non_blocking=True)
        n_groups = len(self.block_tables.block_tables)
        off = 0
        for slot, start_col, n in spans:
            end_col = start_col + n
            chunk = gpu[off : off + n]
            for gid in range(n_groups):
                bt = self.block_tables.block_tables[gid].gpu
                bt[slot, start_col:end_col].copy_(chunk, non_blocking=True)
                self.block_tables.num_blocks.gpu[gid, slot] = end_col
                ibt = self.block_tables.input_block_tables[gid]
                ibt[slot, start_col:end_col].copy_(chunk, non_blocking=True)
            off += n

    def _ensure_blocks(
        self, req_id: str, needed_tokens: int, protect: set[str] | None = None
    ) -> None:
        """Ensure one req has enough blocks (used by reclaim / non-batch paths)."""
        self._ensure_blocks_batch(
            [req_id], [needed_tokens], protect=protect if protect is not None else {req_id}
        )

    def _ensure_blocks_batch(
        self,
        req_ids: list[str],
        needed_tokens: list[int],
        protect: set[str] | None = None,
    ) -> None:
        """Batch-allocate blocks and append-only sync BT rows to GPU.

        Steady-state decode usually grows by 0–1 blocks/req; rewriting full
        rows via ``torch.as_tensor(..., device=cuda)`` was a major prep tax.
        """
        protected = protect if protect is not None else set(req_ids)
        extras: list[int] = []
        haves: list[int] = []
        slots: list[int] = []
        need_full_sync: list[bool] = []
        bs = self.block_size
        max_len = self.max_model_len
        for req_id, need_tok in zip(req_ids, needed_tokens):
            state = self._seqs[req_id]
            need_tok = min(int(need_tok), max_len)
            need_blocks = (need_tok + bs - 1) // bs
            have = len(state["blocks"])
            synced = int(state.get("bt_synced_n", 0))
            extra = need_blocks - have if need_blocks > have else 0
            extras.append(extra)
            haves.append(have)
            slots.append(int(state["slot"]))
            # Full rewrite if GPU row was never synced or drifted.
            need_full_sync.append(synced != have and extra == 0)

        total_extra = int(sum(extras))
        if total_extra == 0 and not any(need_full_sync):
            return

        append_updates: list[tuple[int, int, list[int]]] = []

        def _apply_extra(i: int, chunk: list[int]) -> None:
            state = self._seqs[req_ids[i]]
            have = haves[i]
            slot = slots[i]
            state["blocks"].extend(chunk)
            n_tot = have + len(chunk)
            if int(state.get("bt_synced_n", 0)) != have:
                self._sync_bt_row_full(slot, state["blocks"])
            else:
                append_updates.append((slot, have, chunk))
            state["bt_synced_n"] = n_tot

        if total_extra > 0:
            try:
                new_blocks = self.block_pool.allocate(total_extra)
                cursor = 0
                for i, extra in enumerate(extras):
                    if extra > 0:
                        _apply_extra(i, new_blocks[cursor : cursor + extra])
                        cursor += extra
            except RuntimeError:
                # Bulk alloc failed (pool smaller than spike, or fragmented by
                # stale seqs). Reclaim then allocate per-req so FREE victims
                # can replenish the pool between requests.
                self._reclaim_blocks(total_extra, protect=protected)
                for i, extra in enumerate(extras):
                    if extra <= 0:
                        continue
                    try:
                        chunk = self.block_pool.allocate(extra)
                    except RuntimeError:
                        self._reclaim_blocks(extra, protect=protected)
                        chunk = self.block_pool.allocate(extra)
                    _apply_extra(i, chunk)

        for i, req_id in enumerate(req_ids):
            if extras[i] == 0 and need_full_sync[i]:
                state = self._seqs[req_id]
                self._sync_bt_row_full(slots[i], state["blocks"])
                state["bt_synced_n"] = len(state["blocks"])

        self._append_bt_rows(append_updates)

    def _slot_mapping_for_positions(
        self, slot: int, positions: torch.Tensor, group_id: int = 0
    ) -> torch.Tensor:
        """Compute paged slot ids for absolute positions of one request."""
        bs = int(self.block_tables.block_sizes[group_id])
        blocks = self.block_tables.input_block_tables[group_id][slot]
        block_ids = blocks[positions // bs]
        offsets = positions % bs
        return (block_ids.to(dtype=torch.int64) * bs + offsets).to(dtype=torch.int64)

    def _init_cudagraph_manager(self) -> None:
        """Mirror colocated DFlashSpeculator FULL CUDA graphs for the query forward."""
        if not self._enable_cudagraph:
            logger.info("Disagg-DFlash draft: CUDA graphs disabled via --no-enable-cudagraph")
            return
        supports_full = (
            self.attn_cg_support.min_cg_support.value
            >= AttentionCGSupport.UNIFORM_BATCH.value
        )
        if not supports_full:
            logger.warning(
                "Disagg-DFlash draft attention (%s) does not support FULL CUDA "
                "graphs; running eagerly.",
                self.attn_cg_support.min_cg_attn_backend,
            )
            return

        query_len = self.num_query_per_req
        max_cg_reqs = self._max_cg_reqs()
        capture_sizes = self._planned_capture_token_sizes()
        max_capture = max(capture_sizes) if capture_sizes else query_len

        cc = self.vllm_config.compilation_config
        cc.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        cc.cudagraph_capture_sizes = capture_sizes
        cc.max_cudagraph_capture_size = max_capture

        self.query_cudagraph_manager = DFlashCudaGraphManager(
            self.vllm_config,
            self.device,
            CUDAGraphMode.FULL_DECODE_ONLY,
            decode_query_len=query_len,
        )
        self._cudagraph_enabled = bool(
            self.query_cudagraph_manager.needs_capture()
        )
        if self._cudagraph_enabled:
            logger.info(
                "Disagg-DFlash draft: CUDA graphs enabled "
                "(max_reqs=%d capture_sizes=%d token counts, query_len=%d)",
                max_cg_reqs,
                len(capture_sizes),
                query_len,
            )

    def _pack_block_tables(
        self, slot_idx: torch.Tensor, num_reqs: int, num_reqs_padded: int
    ) -> list[torch.Tensor]:
        """Copy arena BT rows into persistent dense buffers for attention/CG."""
        for gid, ibt in enumerate(self.block_tables.input_block_tables):
            dst = self._packed_block_tables[gid]
            dst[:num_reqs].copy_(ibt.index_select(0, slot_idx))
            if num_reqs_padded > num_reqs:
                dst[num_reqs:num_reqs_padded].zero_()
        return [bt[:num_reqs_padded] for bt in self._packed_block_tables]

    def _group_causal(self) -> bool | dict[int, bool]:
        causal: bool | dict[int, bool] = self.dflash_causal
        if hasattr(self.model, "get_draft_attn_causal") and hasattr(
            self.model, "get_draft_kv_cache_layer_names"
        ):
            layer_names = self.model.get_draft_kv_cache_layer_names()
            layer_causal = self.model.get_draft_attn_causal()
            name_to_gid = {
                ln: gid
                for gid, group in enumerate(self.kv_cache_config.kv_cache_groups)
                for ln in group.layer_names
            }
            causal = {
                name_to_gid[name]: layer_causal[i]
                for i, name in enumerate(layer_names)
                if name in name_to_gid
            }
        return causal

    @torch.inference_mode()
    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        """Query forward + argmax sample into ``draft_tokens`` (CG-captured)."""
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens_padded)
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens_padded,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=batch_descriptor,
        ):
            hidden = self.model(
                input_ids=self.input_buffers.input_ids[:num_tokens_padded],
                positions=self.input_buffers.positions[:num_tokens_padded],
                inputs_embeds=None,
            )
        num_sample = num_reqs * self.num_speculative_tokens
        logits = self.model.compute_logits(hidden[self._sample_indices[:num_sample]])
        tokens = logits.argmax(dim=-1).view(num_reqs, self.num_speculative_tokens)
        self.draft_tokens[:num_reqs].copy_(tokens)

    def capture_cudagraphs(self) -> None:
        """Capture FULL CUDA graphs for the draft query forward (colocated path)."""
        if not self._cudagraph_enabled or self.query_cudagraph_manager is None:
            return
        if self.query_cudagraph_manager._graphs_captured:
            return
        logger.info("Disagg-DFlash draft: capturing CUDA graphs...")
        # Point BlockTables.input_block_tables at packed buffers so
        # get_dummy_block_tables returns the same addresses used at runtime.
        original_ibt = self.block_tables.input_block_tables
        self.block_tables.input_block_tables = self._packed_block_tables
        try:
            self.query_cudagraph_manager.capture(
                self._generate_draft,
                self.input_buffers,
                self.block_tables,
                self.attn_groups,
                self.kv_cache_config,
                self.max_model_len,
                causal=self._group_causal(),
                progress_bar_desc="Capturing Disagg-DFlash draft CUDA graphs",
            )
        finally:
            self.block_tables.input_block_tables = original_ibt
        torch.cuda.synchronize(self.device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        logger.info(
            "Disagg-DFlash draft: captured %d CUDA graphs "
            "(free after capture=%.2f/%.2f GiB)",
            len(self.query_cudagraph_manager.graphs),
            free_bytes / (1024**3),
            total_bytes / (1024**3),
        )

    @staticmethod
    def _as_numpy_i64(t: torch.Tensor) -> np.ndarray:
        """Host numpy view of a meta tensor (never .item() a CUDA tensor)."""
        if t.is_cuda:
            t = t.detach().cpu()
        return np.asarray(t.numpy(), dtype=np.int64)

    @staticmethod
    def _as_numpy_i32(t: torch.Tensor) -> np.ndarray:
        if t.is_cuda:
            t = t.detach().cpu()
        return np.asarray(t.numpy(), dtype=np.int32)

    def _log_iteration_details(
        self,
        request: DisaggDFlashSpeculateRequest,
        *,
        num_reqs: int,
        num_ctx_tokens: int,
        num_new_reqs: int,
        num_rejected_tokens: int,
        elapsed_ms: float,
    ) -> None:
        """EngineCore-style pack log for one draft speculate round."""
        if not self.enable_logging_iteration_details:
            return
        if any(rid.startswith("__warmup__") for rid in request.req_ids):
            return
        num_query_tokens = num_reqs * self.num_query_per_req
        # Context-phase = first time we see the req (no prior draft KV).
        # Generation-phase = already tracked; this round appends context + drafts.
        num_ctx_reqs = num_new_reqs
        num_gen_reqs = num_reqs - num_new_reqs
        free_blocks = self.block_pool.num_free()
        kv_usage_pct = self.kv_cache_usage() * 100.0
        logger.info(
            "DraftIteration(%d): %d context requests, %d context tokens, "
            "%d generation requests, %d query tokens (K=%d), "
            "%d rejected tokens, live_seqs=%d free_slots=%d free_blocks=%d, "
            "iteration elapsed time: %.2f ms, GPU KV cache usage: %.1f%%",
            self._iteration_index,
            num_ctx_reqs,
            num_ctx_tokens,
            num_gen_reqs,
            num_query_tokens,
            self.num_speculative_tokens,
            num_rejected_tokens,
            len(self._seqs),
            len(self._free_slots),
            free_blocks,
            elapsed_ms,
            kv_usage_pct,
        )
        self._iteration_index += 1

    @torch.inference_mode()
    def speculate(
        self, request: DisaggDFlashSpeculateRequest
    ) -> DisaggDFlashSpeculateResponse:
        num_reqs = len(request.req_ids)
        device = self.device
        profile = _draft_profile_enabled()
        log_iter = self.enable_logging_iteration_details
        t0 = time.perf_counter()
        t_q = t0
        phase: dict[str, float] = {}
        cg_full = False
        # Count new vs already-tracked before _ensure_seq allocates slots.
        num_new_reqs = sum(1 for rid in request.req_ids if rid not in self._seqs)

        # Hiddens / positions needed on GPU for KV write. Meta tensors stay on
        # host — decoding already produced CPU tensors; uploading then .item()
        # was the main draft-side latency tax at n≈64.
        ctx_h = request.context_hiddens
        if not ctx_h.is_cuda:
            ctx_h = ctx_h.to(device=device, dtype=self.dtype, non_blocking=True)
        else:
            ctx_h = ctx_h.to(dtype=self.dtype)
        ctx_pos = request.context_positions
        if not ctx_pos.is_cuda:
            ctx_pos_gpu = ctx_pos.to(device=device, dtype=torch.int64, non_blocking=True)
            ctx_pos_np = self._as_numpy_i64(ctx_pos)
        else:
            ctx_pos_gpu = ctx_pos.to(dtype=torch.int64)
            ctx_pos_np = self._as_numpy_i64(ctx_pos)

        qsl_np = self._as_numpy_i32(request.query_start_loc)
        num_rejected_np = self._as_numpy_i32(request.num_rejected)
        num_sampled_np = self._as_numpy_i32(request.num_sampled)
        last_sampled_np = self._as_numpy_i64(request.last_sampled)
        next_prefill_np = self._as_numpy_i64(request.next_prefill_tokens)

        slots = [self._ensure_seq(rid) for rid in request.req_ids]
        self._tick += 1
        protect = set(request.req_ids)
        for rid in request.req_ids:
            self._seqs[rid]["last_used"] = self._tick

        # Mirror colocated prepare_dflash_inputs:
        # - write KV for all scheduled context tokens (incl. rejected tail slots)
        # - place query after the last *valid* context position
        # - do NOT roll back ctx_len by num_rejected (rejected tokens were never
        #   committed as durable context length; they arrive in this batch)
        num_query = num_reqs * self.num_query_per_req
        n_ctx = int(ctx_h.shape[0])
        if n_ctx > self._context_slot_mapping.numel():
            # Staging / HELLO preferred_max_tokens should cover this; grow once.
            self._context_slot_mapping = torch.zeros(
                n_ctx, dtype=torch.int64, device=device
            )
        vocab_size = int(self.embed_vocab_size)
        mask_id = int(self.parallel_drafting_token_id)
        if mask_id < 0 or mask_id >= vocab_size:
            raise RuntimeError(
                f"Disagg-DFlash mask_token_id={mask_id} out of range for "
                f"embed vocab_size={vocab_size}"
            )

        # Bonus selection on host (meta is already CPU).
        bonus_np = np.where(
            num_sampled_np[:num_reqs] > 0,
            last_sampled_np[:num_reqs],
            next_prefill_np[:num_reqs],
        ).astype(np.int64, copy=False)
        oob = (bonus_np < 0) | (bonus_np >= vocab_size)
        if np.any(oob):
            for i in np.nonzero(oob)[0].tolist():
                logger.warning(
                    "Disagg-DFlash draft: OOB bonus token %d (req=%s, vocab=%d); "
                    "clamping into [0, %d]",
                    int(bonus_np[i]),
                    request.req_ids[i],
                    vocab_size,
                    vocab_size - 1,
                )
                bonus_np[i] = 0 if bonus_np[i] < 0 else vocab_size - 1

        # Vectorized host pass: last_valid_pos + needs; one batched ensure.
        qpr = self.num_query_per_req
        t_ensure = time.perf_counter() if profile else 0.0
        last_valid_pos_np = np.zeros(num_reqs, dtype=np.int64)
        seq_lens_np = np.zeros(num_reqs, dtype=np.int32)
        needs_np = np.zeros(num_reqs, dtype=np.int64)
        max_tokens_per_req = qpr
        if num_reqs > 0:
            s_arr = qsl_np[:num_reqs].astype(np.int64, copy=False)
            e_arr = qsl_np[1 : num_reqs + 1].astype(np.int64, copy=False)
            n_rej = num_rejected_np[:num_reqs].astype(np.int64, copy=False)
            n_req_ctx = e_arr - s_arr
            valid_e = e_arr - n_rej
            max_tokens_per_req = int(n_req_ctx.max()) + qpr

            ctx_lens = np.fromiter(
                (self._seqs[rid]["ctx_len"] for rid in request.req_ids),
                dtype=np.int64,
                count=num_reqs,
            )
            has_valid = valid_e > s_arr
            last_valid_pos_np[has_valid] = ctx_pos_np[valid_e[has_valid] - 1]
            last_valid_pos_np[~has_valid] = np.maximum(0, ctx_lens[~has_valid] - 1)

            # Per-req max context position (decode is usually uniform K+1).
            max_ctx_pos = np.zeros(num_reqs, dtype=np.int64)
            if n_ctx > 0:
                if (
                    int(n_req_ctx.min()) == int(n_req_ctx.max())
                    and int(n_req_ctx[0]) > 0
                    and int(s_arr[0]) == 0
                ):
                    tpr = int(n_req_ctx[0])
                    max_ctx_pos = (
                        ctx_pos_np[: num_reqs * tpr]
                        .reshape(num_reqs, tpr)
                        .max(axis=1)
                    )
                else:
                    for i in range(num_reqs):
                        s_i, e_i = int(s_arr[i]), int(e_arr[i])
                        if e_i > s_i:
                            max_ctx_pos[i] = int(ctx_pos_np[s_i:e_i].max())

            need_from_lv = last_valid_pos_np + 1 + qpr
            need_from_ctx = max_ctx_pos + 1 + qpr
            needs_np = np.where(
                n_req_ctx > 0, np.maximum(need_from_lv, need_from_ctx), need_from_lv
            )
            seq_lens_np[:] = (last_valid_pos_np + 1 + qpr).astype(np.int32)
            new_ctx = np.maximum(ctx_lens, last_valid_pos_np + 1)
            for i, rid in enumerate(request.req_ids):
                self._seqs[rid]["ctx_len"] = int(new_ctx[i])

            self._ensure_blocks_batch(
                request.req_ids, needs_np.tolist(), protect=protect
            )
        if profile:
            phase["prep_ensure_ms"] = (time.perf_counter() - t_ensure) * 1000.0
            t_pack = time.perf_counter()

        # Pack arena BT → dense rows, then one Triton launch for slots + query.
        if num_reqs > 0:
            self._slots_host[:num_reqs] = torch.as_tensor(slots, dtype=torch.long)
            self._slots_gpu[:num_reqs].copy_(
                self._slots_host[:num_reqs], non_blocking=True
            )
            slot_idx = self._slots_gpu[:num_reqs]
            self._pack_block_tables(slot_idx, num_reqs, num_reqs)
            self._qsl_host[: num_reqs + 1].copy_(
                torch.from_numpy(np.asarray(qsl_np[: num_reqs + 1], dtype=np.int32))
            )
            self._qsl_gpu[: num_reqs + 1].copy_(
                self._qsl_host[: num_reqs + 1], non_blocking=True
            )
            self._bonus_host[:num_reqs].copy_(
                torch.from_numpy(np.asarray(bonus_np[:num_reqs], dtype=np.int32))
            )
            self._bonus_gpu[:num_reqs].copy_(
                self._bonus_host[:num_reqs], non_blocking=True
            )
            self._last_valid_pos_host[:num_reqs].copy_(
                torch.from_numpy(last_valid_pos_np[:num_reqs])
            )
            self._last_valid_pos_gpu[:num_reqs].copy_(
                self._last_valid_pos_host[:num_reqs], non_blocking=True
            )
            if profile:
                phase["prep_pack_ms"] = (time.perf_counter() - t_pack) * 1000.0
                t_kern = time.perf_counter()
            prepare_disagg_dflash_inputs(
                input_ids=self.input_buffers.input_ids,
                query_positions=self.input_buffers.positions,
                query_start_loc=self.input_buffers.query_start_loc,
                seq_lens=self.input_buffers.seq_lens,
                query_slot_mapping=self._query_slot_mapping,
                context_slot_mapping=self._context_slot_mapping,
                ctx_pos=ctx_pos_gpu,
                qsl=self._qsl_gpu,
                bonus=self._bonus_gpu,
                last_valid_pos=self._last_valid_pos_gpu,
                block_table=self._packed_block_tables[0],
                block_size=int(self.block_tables.block_sizes[0]),
                parallel_drafting_token_id=mask_id,
                num_reqs=num_reqs,
                num_query_per_req=self.num_query_per_req,
                max_model_len=self.max_model_len,
                max_tokens_per_req=max_tokens_per_req,
            )
            for gid in range(self.block_tables.slot_mappings.shape[0]):
                self.block_tables.slot_mappings[gid, :num_query].copy_(
                    self._query_slot_mapping[:num_query]
                )
            if profile:
                # Include kernel launch queueing; wall time folds into prep_ms
                # via the synchronize below.
                phase["prep_kernel_ms"] = (time.perf_counter() - t_kern) * 1000.0
        elif profile:
            phase["prep_pack_ms"] = 0.0
            phase["prep_kernel_ms"] = 0.0

        # Precompute context K/V for all scheduled context tokens.
        if n_ctx > 0:
            context_slot_mapping = self._context_slot_mapping[:n_ctx]
            n_groups = self.block_tables.slot_mappings.shape[0]
            same_bs = len({int(b) for b in self.block_tables.block_sizes}) == 1
            if n_groups <= 1 or same_bs:
                layer_names: list[str] = []
                if hasattr(self.model, "get_draft_kv_cache_layer_names"):
                    layer_names = list(self.model.get_draft_kv_cache_layer_names())
                if layer_names:
                    context_slots: torch.Tensor | list[torch.Tensor | None] = [
                        context_slot_mapping for _ in layer_names
                    ]
                else:
                    context_slots = (
                        context_slot_mapping
                        if n_groups <= 1
                        else [context_slot_mapping] * n_groups
                    )
            else:
                group_slots = []
                for gid in range(n_groups):
                    sm = torch.zeros(n_ctx, dtype=torch.int64, device=device)
                    for i, slot in enumerate(slots):
                        s = int(qsl_np[i])
                        e = int(qsl_np[i + 1])
                        if e > s:
                            sm[s:e] = self._slot_mapping_for_positions(
                                slot, ctx_pos_gpu[s:e], group_id=gid
                            )
                    group_slots.append(sm)
                layer_names = []
                if hasattr(self.model, "get_draft_kv_cache_layer_names"):
                    layer_names = list(self.model.get_draft_kv_cache_layer_names())
                if not layer_names:
                    layer_names = [
                        ln
                        for g in self.kv_cache_config.kv_cache_groups
                        for ln in g.layer_names
                    ]
                name_to_gid = {
                    ln: gid
                    for gid, group in enumerate(self.kv_cache_config.kv_cache_groups)
                    for ln in group.layer_names
                }
                context_slots = [
                    group_slots[name_to_gid[name]]
                    if name in name_to_gid
                    else group_slots[0]
                    for name in layer_names
                ]
            if profile:
                torch.cuda.synchronize()
                phase["prep_ms"] = (time.perf_counter() - t0) * 1000.0
                t_kv = time.perf_counter()
            self.model.precompute_and_store_context_kv(
                ctx_h,
                ctx_pos_gpu,
                context_slots,
            )
            if profile:
                torch.cuda.synchronize()
                phase["kv_ms"] = (time.perf_counter() - t_kv) * 1000.0
                t_q = time.perf_counter()
        elif profile:
            torch.cuda.synchronize()
            phase["prep_ms"] = (time.perf_counter() - t0) * 1000.0
            phase["kv_ms"] = 0.0
            t_q = time.perf_counter()

        max_seq_len = int(seq_lens_np.max()) if num_reqs > 0 else 1
        causal = self._group_causal()

        batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
            self.query_cudagraph_manager,
            num_reqs,
            num_query,
            uniform_token_count=self.num_query_per_req if num_reqs > 0 else None,
            dp_size=1,
            dp_rank=0,
            need_eager=(
                not self._cudagraph_enabled or self.query_cudagraph_manager is None
            ),
        )
        num_reqs_padded = batch_desc.num_reqs or num_reqs
        num_tokens_padded = batch_desc.num_tokens

        # Match colocated prepare_dflash_inputs + BaseSpeculator._build_draft_attn_metadata:
        # pad slots with PAD_SLOT_ID, seq_lens=0, and *clamp* query_start_loc so
        # padded requests have zero query length (not phantom token ranges).
        last_query_end = num_query
        if num_tokens_padded > num_query:
            self.input_buffers.input_ids[num_query:num_tokens_padded] = int(
                self.parallel_drafting_token_id
            )
            self.input_buffers.positions[num_query:num_tokens_padded] = 0
        if num_reqs_padded > num_reqs or num_tokens_padded > num_query:
            self.block_tables.slot_mappings[:, last_query_end:num_tokens_padded] = (
                PAD_SLOT_ID
            )
            self._query_slot_mapping[last_query_end:num_tokens_padded] = PAD_SLOT_ID
            self.input_buffers.seq_lens[num_reqs:num_reqs_padded] = 0
        # Uniform query with clamp: qsl[i] = min(i, num_reqs) * query_len.
        query_start_loc_cpu = (
            torch.clamp(
                torch.arange(num_reqs_padded + 1, dtype=torch.int32), max=num_reqs
            )
            * self.num_query_per_req
        )
        self.input_buffers.query_start_loc[: num_reqs_padded + 1].copy_(
            query_start_loc_cpu.to(device=device, non_blocking=True)
        )

        attn_block_tables = self._pack_block_tables(
            slot_idx, num_reqs, num_reqs_padded
        )
        # Refresh FA metadata / scheduler_metadata into persistent builder
        # buffers (required before FULL CG replay; same as colocated).
        attn_metadata = build_attn_metadata(
            attn_groups=self.attn_groups,
            num_reqs=num_reqs_padded,
            num_tokens=num_tokens_padded,
            query_start_loc_gpu=self.input_buffers.query_start_loc[
                : num_reqs_padded + 1
            ],
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=self.num_query_per_req,
            seq_lens=self.input_buffers.seq_lens[:num_reqs_padded],
            max_seq_len=max(max_seq_len, 1),
            block_tables=attn_block_tables,
            slot_mappings=self.block_tables.slot_mappings[:, :num_tokens_padded],
            kv_cache_config=self.kv_cache_config,
            causal=causal,
        )
        slot_mappings = build_slot_mappings_by_layer(
            self.block_tables.slot_mappings[:, :num_tokens_padded],
            self.kv_cache_config,
        )

        measure_fwd = profile or _measure_draft_forward()
        if profile:
            torch.cuda.synchronize()
            phase["query_prep_ms"] = (time.perf_counter() - t_q) * 1000.0
            t_fwd = time.perf_counter()
        elif measure_fwd:
            torch.cuda.synchronize()
            t_fwd = time.perf_counter()
        else:
            t_fwd = 0.0

        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            assert self.query_cudagraph_manager is not None
            self.query_cudagraph_manager.run_fullgraph(batch_desc)
            cg_full = True
            if profile:
                phase["cg_mode"] = 1.0
        else:
            self._generate_draft(
                num_reqs,
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode=batch_desc.cg_mode,
            )
            if profile:
                phase["cg_mode"] = 0.0

        draft_forward_ms: float | None = None
        if measure_fwd:
            torch.cuda.synchronize()
            draft_forward_ms = (time.perf_counter() - t_fwd) * 1000.0
            if profile:
                phase["forward_ms"] = draft_forward_ms
                phase["sample_ms"] = 0.0  # folded into _generate_draft / CG
                phase["total_ms"] = (time.perf_counter() - t0) * 1000.0
                if not hasattr(self, "_draft_profile_steps"):
                    self._draft_profile_steps = 0
                self._draft_profile_steps += 1
                from vllm.v1.spec_decode.disagg_dflash.debug_logging import (
                    draft_profile_log_every,
                )

                every = draft_profile_log_every()
                if self._draft_profile_steps % max(1, every) == 0:
                    logger.info(
                        "Disagg-DFlash draft profile (n=%d): %s",
                        num_reqs,
                        {k: round(v, 2) for k, v in phase.items()},
                    )

        num_ctx_tokens = int(qsl_np[num_reqs]) if num_reqs > 0 else 0
        elapsed_s = time.perf_counter() - t0
        if log_iter:
            num_rejected_tokens = int(num_rejected_np[:num_reqs].sum())
            self._log_iteration_details(
                request,
                num_reqs=num_reqs,
                num_ctx_tokens=num_ctx_tokens,
                num_new_reqs=num_new_reqs,
                num_rejected_tokens=num_rejected_tokens,
                elapsed_ms=elapsed_s * 1000.0,
            )

        if not any(rid.startswith("__warmup__") for rid in request.req_ids):
            from vllm.v1.spec_decode.disagg_dflash.metrics import observe_speculate

            observe_speculate(
                model_name=self.draft_model_name,
                num_reqs=num_reqs,
                num_ctx_tokens=num_ctx_tokens,
                elapsed_s=elapsed_s,
                cg_full=cg_full,
            )
            self._observe_mfu(num_reqs=num_reqs, seq_lens_np=seq_lens_np)
        self._record_kv_metrics()
        return DisaggDFlashSpeculateResponse(
            draft_tokens=self.draft_tokens[:num_reqs].detach().cpu(),
            draft_forward_ms=draft_forward_ms,
        )

    def _maybe_raise_staging_capacity(self, preferred: object) -> None:
        if preferred is None:
            return
        try:
            pref = int(preferred)
        except (TypeError, ValueError):
            return
        if pref <= self.ipc_max_num_tokens:
            return
        logger.info(
            "Disagg-DFlash draft: raising ipc_max_num_tokens %d → %d from verify HELLO",
            self.ipc_max_num_tokens,
            pref,
        )
        self.ipc_max_num_tokens = pref
        if self._ipc_staging is not None and self._ipc_staging.max_tokens < pref:
            self._ipc_staging = None
        if self._nixl_staging is not None and self._nixl_staging.max_tokens < pref:
            try:
                self._nixl_staging.close()
            except Exception:
                pass
            self._nixl_staging = None

    def ensure_ipc_staging(self):
        """Allocate (once) draft-owned CUDA IPC staging buffers."""
        if self._ipc_staging is not None:
            return self._ipc_staging
        from vllm.v1.spec_decode.disagg_dflash.transport_cuda_ipc import (
            CudaIpcStaging,
        )

        h = self.vllm_config.model_config.get_hidden_size()
        self._ipc_staging = CudaIpcStaging(
            max_tokens=self.ipc_max_num_tokens,
            max_num_seqs=self.max_num_seqs,
            hidden_size=h,
            num_speculative_tokens=self.num_speculative_tokens,
            dtype=self.dtype,
            device=self.device,
        )
        logger.info(
            "Disagg-DFlash draft: CUDA IPC staging ready "
            "(max_tokens=%d max_seqs=%d H=%d K=%d dtype=%s)",
            self.ipc_max_num_tokens,
            self.max_num_seqs,
            h,
            self.num_speculative_tokens,
            self.dtype,
        )
        return self._ipc_staging

    def ensure_nixl_staging(self):
        """Allocate (once) draft-owned NIXL-registered VRAM staging."""
        if self._nixl_staging is not None:
            return self._nixl_staging
        from vllm.v1.spec_decode.disagg_dflash.transport_nixl import NixlStaging

        h = self.vllm_config.model_config.get_hidden_size()
        self._nixl_staging = NixlStaging(
            max_tokens=self.ipc_max_num_tokens,
            max_num_seqs=self.max_num_seqs,
            hidden_size=h,
            num_speculative_tokens=self.num_speculative_tokens,
            dtype=self.dtype,
            device=self.device,
        )
        logger.info(
            "Disagg-DFlash draft: NIXL staging ready "
            "(max_tokens=%d max_seqs=%d H=%d K=%d dtype=%s)",
            self.ipc_max_num_tokens,
            self.max_num_seqs,
            h,
            self.num_speculative_tokens,
            self.dtype,
        )
        return self._nixl_staging

    def handle_hello(self, hello_meta: dict) -> list[bytes]:
        transport = str(hello_meta.get("transport", "zmq")).lower()
        if transport == "zmq":
            return encode_hello_reply({"transport": "zmq"})
        if transport == "cuda_ipc":
            self._maybe_raise_staging_capacity(hello_meta.get("preferred_max_tokens"))
            staging = self.ensure_ipc_staging()
            return encode_hello_reply(staging.hello_payload())
        if transport == "nixl":
            self._maybe_raise_staging_capacity(hello_meta.get("preferred_max_tokens"))
            staging = self.ensure_nixl_staging()
            verify_meta = hello_meta.get("agent_metadata")
            if verify_meta:
                try:
                    import base64

                    staging.add_remote_verify(base64.b64decode(verify_meta))
                except Exception as e:
                    return encode_hello_reply(
                        {
                            "transport": "zmq",
                            "error": f"nixl add_remote_agent failed: {e}",
                        }
                    )
            return encode_hello_reply(staging.hello_payload())
        if transport == "nccl":
            return encode_hello_reply(
                {
                    "transport": "zmq",
                    "error": f"transport {transport!r} not implemented",
                }
            )
        return encode_hello_reply(
            {"transport": "zmq", "error": f"unknown transport {transport!r}"}
        )

    def warmup(self) -> None:
        """Run one dummy speculate so torch.compile / cudagraph happen before serve."""
        from vllm.v1.spec_decode.disagg_dflash.protocol import (
            DisaggDFlashFreeRequest,
            DisaggDFlashSpeculateRequest,
        )

        logger.info("Disagg-DFlash draft: running compile/warmup speculate...")
        h = self.vllm_config.model_config.get_hidden_size()
        req = DisaggDFlashSpeculateRequest(
            req_ids=["__warmup__"],
            context_hiddens=torch.zeros(1, h, dtype=self.dtype),
            context_positions=torch.zeros(1, dtype=torch.int64),
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            num_rejected=torch.zeros(1, dtype=torch.int32),
            num_sampled=torch.ones(1, dtype=torch.int32),
            last_sampled=torch.zeros(1, dtype=torch.int64),
            next_prefill_tokens=torch.zeros(1, dtype=torch.int64),
            temperature=torch.zeros(1, dtype=torch.float32),
            seeds=torch.zeros(1, dtype=torch.int64),
            num_speculative_tokens=self.num_speculative_tokens,
        )
        self.speculate(req)
        self.free(DisaggDFlashFreeRequest(req_ids=["__warmup__"]))
        # Capture after one eager step so FA builders are initialized.
        self.capture_cudagraphs()
        logger.info("Disagg-DFlash draft: warmup complete")

    def handle(self, cmd: str, payload) -> list[bytes]:
        if cmd == CMD_HELLO:
            return self.handle_hello(payload)
        if cmd == "speculate":
            # IPC pull already synchronized inside pull_hiddens_from_ipc /
            # memcpy_d2d when using cuda_ipc; ZMQ path needs no extra sync.
            try:
                return self.speculate(payload).encode()
            except Exception:
                from vllm.v1.spec_decode.disagg_dflash.metrics import (
                    observe_speculate_error,
                )

                observe_speculate_error(model_name=self.draft_model_name)
                raise
        if cmd == "free":
            return self.free(payload)
        return encode_pong()
