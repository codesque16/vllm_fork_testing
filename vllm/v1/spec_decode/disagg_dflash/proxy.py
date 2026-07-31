# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify-side Disagg-DFlash proxy (Model Runner V2 / BaseSpeculator)."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.spec_decode.disagg_dflash.common import pack_speculate_request
from vllm.v1.spec_decode.disagg_dflash.projector import (
    build_and_load_projector,
    reduce_aux_hidden_states,
)
from vllm.v1.spec_decode.disagg_dflash.protocol import DisaggDFlashFreeRequest
from vllm.v1.spec_decode.disagg_dflash.transport import (
    DisaggDFlashClientTransport,
    create_client_transport,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

logger = init_logger(__name__)


def _profile_enabled() -> bool:
    from vllm.v1.spec_decode.disagg_dflash.debug_logging import disagg_profile_enabled

    return disagg_profile_enabled()


@dataclass
class DisaggProposePending:
    """In-flight Disagg-DFlash propose between fire and wait."""

    num_reqs: int
    early_done: bool = False
    profile: bool = False
    t_proj0: float = 0.0
    t_proj1: float = 0.0
    t_pack0: float = 0.0
    t_pack1: float = 0.0
    t_fire: float = 0.0
    # Verifier forward+sample window for the step that fired this draft (Tpv).
    tpv_ms: float = 0.0
    # Mean accepted draft tokens at fire time (bonus excluded).
    accepted: float = 0.0
    speculate_ms: dict[str, float] = field(default_factory=dict)
    # Fire-time request-state indices — write drafts here at finish even if
    # the next execute_model batch order differs (cross-step pipeline).
    idx_mapping: torch.Tensor | None = None
    req_ids: list[str] = field(default_factory=list)
    # Preloaded ZMQ response from the background wait (send-and-forget).
    response: Any | None = None


class DisaggDFlashProxy(BaseSpeculator):
    """Remote DFlash drafter. Does not load draft transformer weights."""

    supports_mm_inputs = False

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.use_disagg_dflash()
        self.vllm_config = vllm_config
        self.device = device
        self.speculative_config = vllm_config.speculative_config
        self.num_speculative_steps = self.speculative_config.num_speculative_tokens
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs

        self._tp_rank = get_tp_group().rank_in_group
        self._tp_size = get_tp_group().world_size
        self._client: DisaggDFlashClientTransport | None = None
        self._projector: nn.Module | None = None
        self._known_req_ids: set[str] = set()
        self._transport_name = self.speculative_config.disagg_dflash_transport
        self._profile_steps = 0
        from vllm.v1.spec_decode.disagg_dflash.debug_logging import (
            disagg_profile_log_every,
        )

        self._profile_every = disagg_profile_log_every()
        # Default ON via SpeculativeConfig.disagg_dflash_cross_step.
        self._cross_step = bool(self.speculative_config.disagg_dflash_cross_step)
        self._async_complete = bool(
            self._cross_step
            and getattr(self.speculative_config, "disagg_dflash_async_complete", True)
        )
        # Cross-step: one in-flight speculate whose finish is deferred into the
        # next execute_model (before FREE / prepare_inputs).
        self._deferred_pending: DisaggProposePending | None = None
        # Completed bg wait stashed for send-and-forget apply (pending, response).
        self._completed: tuple[DisaggProposePending, Any] | None = None
        # Background ZMQ wait so draft RTT advances while the worker runs
        # post-sample work / DP sync. Socket access is serialized by the lock.
        self._client_lock = threading.Lock()
        self._wait_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="disagg-dflash-wait"
        )
        self._wait_future: Future[Any] | None = None
        # Req ids whose remote draft was applied since the last status poll
        # (consumed by DraftTokensHandler / scheduler).
        self._ready_req_ids: list[str] = []
        # Side-stream projector so reduce_aux can run under rejection sampling.
        self._proj_stream = torch.cuda.Stream(device=device)
        self._proj_ready = torch.cuda.Event(enable_timing=False)
        self._proj_reduced: torch.Tensor | None = None
        self._proj_num_tokens: int = 0
        self._proj_t0: float = 0.0
        self._proj_inflight: bool = False
        # Last verify forward+sample window (ms), set by model_runner when
        # --enable-sd-timing-model is on.
        self._last_tpv_ms: float = 0.0
        self._last_accepted: float = 0.0
        # Upper-bound A/B: skip L*H→H projector compute / TP collectives.
        self._noop_reduce = bool(
            getattr(self.speculative_config, "disagg_dflash_noop_reduce", False)
        )

        self.draft_tokens = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )
        # Token-only wire protocol: rejection sampling uses target logits only.
        self.draft_logits: torch.Tensor | None = None
        # Dummy attribute so isinstance(..., DraftModelSpeculator) paths skip us.
        self.model = None

    def set_tpv_ms(self, tpv_ms: float) -> None:
        """Record verifier Tpv for the step about to fire a remote draft."""
        self._last_tpv_ms = float(tpv_ms)

    def set_acceptance_stats(self, accepted: float) -> None:
        """Mean accepted draft tokens (bonus excluded) for the fire step."""
        self._last_accepted = float(accepted)

    def load_model(self, target_model: nn.Module) -> None:
        if self._noop_reduce:
            self._projector = None
            logger.warning(
                "Disagg-DFlash: disagg_dflash_noop_reduce=True — skipping fc "
                "projector; wire uses last_hidden_states (timing A/B only)."
            )
        else:
            self._projector = build_and_load_projector(self.vllm_config, self.device)
        if self._tp_rank == 0:
            address = self.speculative_config.disagg_dflash_address
            assert address is not None
            self._client = create_client_transport(
                address=address,
                timeout_ms=self.speculative_config.disagg_dflash_timeout_ms,
                transport=self._transport_name,
                ipc_max_num_tokens=(
                    self.speculative_config.disagg_dflash_ipc_max_num_tokens
                ),
            )
            if not self._client.ping():
                logger.warning(
                    "Disagg-DFlash draft server did not answer ping at %s", address
                )
            self._client.handshake()
        logger.info(
            "Disagg-DFlash proxy ready (tp_rank=%d, address=%s, transport=%s, "
            "cross_step=%s, async_complete=%s, noop_reduce=%s)",
            self._tp_rank,
            self.speculative_config.disagg_dflash_address,
            self._transport_name,
            self._cross_step,
            self._async_complete,
            self._noop_reduce,
        )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        return

    def capture(self) -> None:
        return

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
    ) -> None:
        return

    def begin_reduce(
        self,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_tokens: int,
    ) -> None:
        """Launch L*H→H projector on a side stream (overlaps with sample())."""
        if self._noop_reduce or self._projector is None:
            # No-op / no projector: view only — no GEMM, no TP collective.
            self._proj_inflight = False
            self._proj_reduced = last_hidden_states[:num_tokens]
            self._proj_num_tokens = num_tokens
            self._proj_t0 = 0.0
            return
        profile = _profile_enabled() and self._tp_rank == 0
        self._proj_t0 = time.perf_counter() if profile else 0.0
        self._proj_num_tokens = num_tokens
        self._proj_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self._proj_stream):
            self._proj_reduced = reduce_aux_hidden_states(
                self._projector,
                last_hidden_states[:num_tokens],
                (
                    [h[:num_tokens] for h in aux_hidden_states]
                    if aux_hidden_states is not None
                    else None
                ),
                noop=False,
            )
        self._proj_ready.record(self._proj_stream)
        self._proj_inflight = True

    def _take_reduced(self, profile: bool) -> tuple[torch.Tensor, float, float]:
        """Wait for side-stream reduce (or no-op) and return (tensor, t0, t1)."""
        t0 = self._proj_t0 if profile else 0.0
        if self._proj_inflight:
            self._proj_ready.synchronize()
            torch.cuda.current_stream(self.device).wait_stream(self._proj_stream)
            self._proj_inflight = False
        t1 = time.perf_counter() if profile else 0.0
        assert self._proj_reduced is not None
        return self._proj_reduced, t0, t1

    def free_requests(self, req_ids: list[str]) -> None:
        if not req_ids:
            return
        # DEALER is in-order: never FREE while a speculate reply is pending.
        # Always block-drain so abort mid-flight cannot race the socket.
        self.drain_blocking(req_states=None)
        # Always notify draft for finished IDs we have seen; also drop unknown
        # IDs from local tracking so verify-side set cannot grow unbounded.
        to_free = [r for r in req_ids if r in self._known_req_ids]
        for r in req_ids:
            self._known_req_ids.discard(r)
        if not to_free or self._tp_rank != 0 or self._client is None:
            return
        try:
            with self._client_lock:
                self._client.free(DisaggDFlashFreeRequest(req_ids=to_free))
        except Exception as e:
            logger.warning("Disagg-DFlash FREE failed: %s", e)

    def _start_background_wait(self) -> None:
        """Begin speculate_wait on a worker thread (cross-step only)."""
        if (
            not self._cross_step
            or self._tp_rank != 0
            or self._client is None
            or self._wait_future is not None
        ):
            return
        client = self._client
        lock = self._client_lock

        def _wait() -> Any:
            with lock:
                return client.speculate_wait()

        self._wait_future = self._wait_executor.submit(_wait)

    def _take_background_wait(self, *, blocking: bool = True) -> Any | None:
        """Return completed bg wait result, or None if none / still running.

        When ``blocking=False`` and the future is not done, leave it in place
        and return None. When ``blocking=True``, join (legacy cross-step).
        """
        fut = self._wait_future
        if fut is None:
            return None
        if not blocking and not fut.done():
            return None
        self._wait_future = None
        return fut.result()

    def _poll_into_completed(self, *, blocking: bool = False) -> bool:
        """If the bg wait is done, stash ``(pending, response)`` in ``_completed``.

        Returns True when a completion record is available (including an
        already-stashed ``_completed``).
        """
        if self._completed is not None:
            return True
        pending = self._deferred_pending
        if pending is None or pending.early_done:
            return pending is not None and pending.early_done
        resp = self._take_background_wait(blocking=blocking)
        if resp is None and not blocking:
            # No bg wait running: allow sync wait in propose_finish.
            if self._wait_future is None:
                return True
            return False
        if resp is not None:
            pending.response = resp
            self._completed = (pending, resp)
            return True
        return False

    @property
    def cross_step_enabled(self) -> bool:
        return self._cross_step

    @property
    def async_complete_enabled(self) -> bool:
        return self._async_complete

    def inflight_req_ids(self) -> set[str]:
        pending = self._deferred_pending
        if pending is None or pending.early_done:
            return set()
        # Still in flight until applied (even if ZMQ reply is already stashed).
        return set(pending.req_ids)

    def take_ready_req_ids(self) -> list[str]:
        ready = self._ready_req_ids
        self._ready_req_ids = []
        return ready

    def defer_pending(self, pending: DisaggProposePending) -> None:
        """Hold fire-time pending until apply/drain completes it."""
        if self._deferred_pending is not None and not self._deferred_pending.early_done:
            raise RuntimeError(
                "Disagg-DFlash cross-step: new propose_begin while previous "
                "speculate is still in flight"
            )
        if self._completed is not None:
            raise RuntimeError(
                "Disagg-DFlash cross-step: new propose_begin while previous "
                "speculate completion is not yet applied"
            )
        self._deferred_pending = pending

    def _apply_pending(
        self, pending: DisaggProposePending, req_states: Any | None
    ) -> list[str]:
        """Finish ``pending`` (blocking join if needed) and write draft tokens."""
        draft = self.propose_finish(pending)
        if req_states is not None and pending.idx_mapping is not None:
            # Only write rows whose fire-time req_id is still tracked.
            alive = [rid for rid in pending.req_ids if rid in self._known_req_ids]
            if alive and len(alive) == len(pending.req_ids):
                req_states.draft_tokens[pending.idx_mapping] = draft
            elif alive:
                # Partial finish: write per surviving req by matching ids.
                for i, rid in enumerate(pending.req_ids):
                    if rid not in self._known_req_ids:
                        continue
                    # idx_mapping[i] is the request-state slot at fire time.
                    slot = int(pending.idx_mapping[i].item())
                    req_states.draft_tokens[slot] = draft[i]
        applied = list(pending.req_ids)
        self._ready_req_ids.extend(applied)
        return applied

    def try_apply_completed(self, req_states: Any | None) -> list[str] | None:
        """Non-blocking poll: apply draft if bg wait is done (TP-collective).

        All TP ranks must call this every step so the ready-bit broadcast and
        token broadcast stay matched. Returns applied req_ids, or None if the
        in-flight speculate is still outstanding / absent.
        """
        pending = self._deferred_pending
        ready = torch.zeros(1, dtype=torch.int32, device=self.device)
        if self._tp_rank == 0:
            if pending is None:
                pass
            elif pending.early_done:
                ready[0] = 1
            elif self._poll_into_completed(blocking=False):
                ready[0] = 1
        if self._tp_size > 1:
            torch.distributed.broadcast(
                ready,
                src=get_tp_group().ranks[0],
                group=get_tp_group().device_group,
            )
        if int(ready.item()) == 0:
            return None
        if pending is None:
            return None
        self._deferred_pending = None
        self._completed = None
        return self._apply_pending(pending, req_states)

    def drain_blocking(self, req_states: Any | None) -> list[str] | None:
        """Block until the deferred speculate completes and apply it."""
        pending = self._deferred_pending
        if pending is None:
            return None
        if self._tp_rank == 0 and not pending.early_done:
            self._poll_into_completed(blocking=True)
        self._deferred_pending = None
        self._completed = None
        return self._apply_pending(pending, req_states)

    def drain_deferred(self, req_states: Any | None) -> torch.Tensor | None:
        """Backward-compatible blocking drain; returns draft tensor or None."""
        pending = self._deferred_pending
        if pending is None:
            return None
        applied = self.drain_blocking(req_states)
        if applied is None:
            return None
        # Best-effort: return the proxy scratch buffer view.
        return self.draft_tokens[: len(applied)]

    def _resolve_bonus_tokens(
        self,
        input_batch: InputBatch,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        num_reqs: int,
    ) -> torch.Tensor:
        """Build per-request bonus token for the draft (batch-local)."""
        idx = input_batch.idx_mapping[:num_reqs].long()
        # Batch-local sampler output [num_reqs, K+1] (Stage 2, before postprocess)
        # vs request-state table [max_num_reqs] / [max_num_reqs, 1].
        if last_sampled.dim() == 2 and last_sampled.shape[0] == num_reqs:
            if last_sampled.shape[-1] == 1:
                ls = last_sampled[:, 0].to(dtype=torch.int64)
            else:
                ns_i64 = num_sampled[:num_reqs].to(dtype=torch.int64)
                gather_idx = (ns_i64 - 1).clamp(min=0).unsqueeze(1)
                ls = (
                    last_sampled.gather(1, gather_idx).squeeze(1).to(dtype=torch.int64)
                )
        elif last_sampled.dim() == 1:
            if last_sampled.shape[0] == num_reqs:
                ls = last_sampled[:num_reqs].to(dtype=torch.int64)
            else:
                ls = last_sampled.index_select(0, idx).to(dtype=torch.int64)
        elif last_sampled.dim() == 2 and last_sampled.shape[-1] == 1:
            ls = last_sampled[:, 0].index_select(0, idx).to(dtype=torch.int64)
        else:
            ls = last_sampled.reshape(-1).index_select(0, idx).to(dtype=torch.int64)

        if next_prefill_tokens.dim() == 1:
            if next_prefill_tokens.shape[0] == num_reqs:
                npf = next_prefill_tokens.to(dtype=torch.int64)
            else:
                npf = next_prefill_tokens.index_select(0, idx).to(dtype=torch.int64)
        else:
            npf = (
                next_prefill_tokens.reshape(-1)
                .index_select(0, idx)
                .to(dtype=torch.int64)
            )

        ns = num_sampled[:num_reqs]
        return torch.where(ns > 0, ls, npf).reshape(-1)[:num_reqs].contiguous()

    def _repair_bonus_oob(
        self,
        bonus: torch.Tensor,
        input_batch: InputBatch,
        num_rejected: torch.Tensor,
        num_reqs: int,
    ) -> torch.Tensor:
        """Clamp / repair OOB bonus tokens (may sync; call after projector join)."""
        vocab = self.vllm_config.model_config.get_vocab_size()
        oob = (bonus < 0) | (bonus >= vocab)
        if not bool(oob.any().item()):
            return bonus
        # Prefer the first logit slot (combine wrote last_sampled there)
        # over the rejected draft tail.
        qsl = input_batch.query_start_loc
        logits_indices = input_batch.logits_indices
        cu = input_batch.cu_num_logits
        before = bonus[oob][:8].detach().tolist()
        for i in torch.nonzero(oob, as_tuple=False).view(-1).tolist():
            repaired = False
            if cu is not None and logits_indices is not None:
                logit0 = int(cu[i].item())
                tok = int(input_batch.input_ids[logits_indices[logit0]].item())
                if 0 <= tok < vocab:
                    bonus[i] = tok
                    repaired = True
            if not repaired:
                s = int(qsl[i].item())
                e = int(qsl[i + 1].item())
                n_rej = int(num_rejected[i].item())
                valid_e = e - n_rej
                if valid_e > s:
                    tok = int(input_batch.input_ids[valid_e - 1].item())
                    bonus[i] = tok if 0 <= tok < vocab else 0
                else:
                    bonus[i] = 0
        logger.warning(
            "Disagg-DFlash: repaired %d/%d OOB bonus tokens "
            "(vocab=%d); sample before=%s after=%s "
            "(248320==vocab_size is the top-k sentinel, not a token)",
            int(oob.sum().item()),
            num_reqs,
            vocab,
            before,
            bonus[oob][:8].detach().tolist(),
        )
        return bonus

    @torch.inference_mode()
    def propose_begin(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        context_only: bool = False,
    ) -> DisaggProposePending:
        """Fire projector + remote speculate; does not wait for draft tokens.

        Call ``propose_finish`` after overlapping verify-side post-sample work.
        """
        del (
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            skip_attn_for_dummy_run,
            mm_inputs,
        )
        num_reqs = input_batch.num_reqs
        idx_mapping = input_batch.idx_mapping[:num_reqs].clone()
        req_ids_list = list(input_batch.req_ids[:num_reqs])
        # context_only is Design A (P/D); mutually exclusive with this proxy.
        if dummy_run or is_profile or context_only:
            self.draft_tokens[:num_reqs].zero_()
            return DisaggProposePending(
                num_reqs=num_reqs,
                early_done=True,
                idx_mapping=idx_mapping,
                req_ids=req_ids_list,
            )

        # One in-flight remote speculate: skip firing while a prior batch's
        # draft RPC is still outstanding (async-complete packs other work).
        if (
            self._async_complete
            and self._deferred_pending is not None
            and not self._deferred_pending.early_done
        ):
            return DisaggProposePending(
                num_reqs=num_reqs,
                early_done=True,
                idx_mapping=idx_mapping,
                req_ids=req_ids_list,
            )

        req_ids = req_ids_list
        # Verify kernel warmup uses "_warmup_{i}_" IDs and historically only
        # finished them via execute_model (no sample_tokens). Never allocate
        # draft slots for those synthetic requests.
        if any(rid.startswith("_warmup_") for rid in req_ids):
            self.draft_tokens[:num_reqs].zero_()
            return DisaggProposePending(
                num_reqs=num_reqs,
                early_done=True,
                idx_mapping=idx_mapping,
                req_ids=req_ids_list,
            )

        num_tokens = input_batch.num_tokens
        profile = _profile_enabled() and self._tp_rank == 0
        positions = input_batch.positions[:num_tokens]
        self._known_req_ids.update(req_ids)

        # Resolve bonus / gather meta while the side-stream projector still runs.
        # Join projector only immediately before packing hiddens for NIXL/ZMQ.
        bonus = None
        idx = None
        temp = None
        seed = None
        if self._tp_rank == 0:
            idx = input_batch.idx_mapping[:num_reqs].long()
            bonus = self._resolve_bonus_tokens(
                input_batch,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                num_reqs,
            )
            temp = (
                temperature.index_select(0, idx)
                if temperature.numel() > num_reqs
                else temperature[:num_reqs]
            )
            seed = (
                seeds.index_select(0, idx)
                if seeds.numel() > num_reqs
                else seeds[:num_reqs]
            )

        if (
            self._proj_inflight
            or self._proj_reduced is not None
        ) and self._proj_num_tokens == num_tokens:
            reduced, t_proj0, t_proj1 = self._take_reduced(profile)
        else:
            if self._proj_inflight:
                # Stale async reduce (token count changed); wait & drop.
                self._proj_ready.synchronize()
                self._proj_inflight = False
            t_proj0 = time.perf_counter() if profile else 0.0
            reduced = reduce_aux_hidden_states(
                self._projector,
                last_hidden_states[:num_tokens],
                (
                    [h[:num_tokens] for h in aux_hidden_states]
                    if aux_hidden_states is not None
                    else None
                ),
                noop=self._noop_reduce,
            )
            if profile and not self._noop_reduce:
                torch.cuda.synchronize()
            t_proj1 = time.perf_counter() if profile else 0.0
        self._proj_reduced = None

        pending = DisaggProposePending(
            num_reqs=num_reqs,
            profile=profile,
            t_proj0=t_proj0,
            t_proj1=t_proj1,
            tpv_ms=self._last_tpv_ms,
            accepted=self._last_accepted,
            idx_mapping=idx_mapping,
            req_ids=req_ids_list,
        )

        if self._tp_rank == 0:
            if self._client is None:
                raise RuntimeError(
                    "DisaggDFlashProxy client is not connected. "
                    "Ensure model_runner.load_model() calls "
                    "DisaggDFlashProxy.load_model() before propose_begin."
                )
            assert bonus is not None and temp is not None and seed is not None
            # OOB repair syncs; keep it after projector join so bonus gather
            # can overlap with the side-stream fc.
            bonus = self._repair_bonus_oob(
                bonus, input_batch, num_rejected, num_reqs
            )
            t_pack0 = time.perf_counter() if profile else 0.0
            req = pack_speculate_request(
                req_ids=req_ids,
                reduced_hiddens=reduced,
                positions=positions,
                query_start_loc=input_batch.query_start_loc,
                num_rejected=num_rejected,
                # Force draft to take the last_sampled/bonus path.
                num_sampled=torch.ones(num_reqs, dtype=torch.int32, device=bonus.device),
                last_sampled=bonus,
                next_prefill_tokens=bonus,
                temperature=temp,
                seeds=seed,
                num_speculative_tokens=self.num_speculative_steps,
                num_reqs=num_reqs,
            )
            t_pack1 = time.perf_counter() if profile else 0.0
            self._client.speculate_begin(req)
            t_fire = time.perf_counter() if profile else 0.0
            # Overlap ZMQ RTT with verify post-sample / next-step preamble.
            self._start_background_wait()
            pending.t_pack0 = t_pack0
            pending.t_pack1 = t_pack1
            pending.t_fire = t_fire
            if profile:
                pending.speculate_ms = {
                    "projector_ms": (t_proj1 - t_proj0) * 1000.0,
                    "pack_ms": (t_pack1 - t_pack0) * 1000.0,
                    "fire_ms": (t_fire - t_pack1) * 1000.0,
                }

        return pending

    @torch.inference_mode()
    def propose_finish(self, pending: DisaggProposePending) -> torch.Tensor:
        """Wait for in-flight draft tokens and broadcast to the TP group."""
        num_reqs = pending.num_reqs
        if pending.early_done:
            return self.draft_tokens[:num_reqs]

        profile = pending.profile
        speculate_ms = dict(pending.speculate_ms)
        from vllm.v1.spec_decode.disagg_dflash.timing_model import timing_model_enabled

        measure_await = profile or timing_model_enabled()
        t_wait0 = time.perf_counter() if measure_await else 0.0

        draft_tokens: torch.Tensor
        draft_forward_ms: float | None = None
        client_ms: dict[str, float] = {}
        await_ms = 0.0
        if self._tp_rank == 0:
            assert self._client is not None
            resp = pending.response
            used_bg_wait = resp is not None
            if resp is None:
                resp = self._take_background_wait(blocking=True)
                used_bg_wait = resp is not None
            if resp is None:
                with self._client_lock:
                    resp = self._client.speculate_wait()
            pending.response = None
            t_wait1 = time.perf_counter() if measure_await else 0.0
            if measure_await:
                await_ms = (t_wait1 - t_wait0) * 1000.0
            draft_forward_ms = resp.draft_forward_ms
            draft_tokens = resp.draft_tokens.to(
                device=self.device, dtype=torch.int64, non_blocking=True
            )
            raw_client_ms = getattr(self._client, "last_timings_ms", None)
            if isinstance(raw_client_ms, dict):
                client_ms = dict(raw_client_ms)
            if profile:
                speculate_ms["await_ms"] = await_ms
                if pending.t_fire > 0:
                    # Wall time from fire until we join the wait (includes any
                    # work that ran while the background wait was in flight).
                    speculate_ms["verify_overlap_ms"] = (
                        t_wait0 - pending.t_fire
                    ) * 1000.0
                speculate_ms["bg_wait"] = 1.0 if used_bg_wait else 0.0
                speculate_ms.update(client_ms)
        else:
            draft_tokens = torch.empty(
                num_reqs,
                self.num_speculative_steps,
                dtype=torch.int64,
                device=self.device,
            )

        t_bcast0 = time.perf_counter() if profile else 0.0
        if self._tp_size > 1:
            torch.distributed.broadcast(
                draft_tokens,
                src=get_tp_group().ranks[0],
                group=get_tp_group().device_group,
            )
        if profile:
            torch.cuda.synchronize()
            t_bcast1 = time.perf_counter()
            speculate_ms["broadcast_ms"] = (t_bcast1 - t_bcast0) * 1000.0
            speculate_ms["propose_total_ms"] = (t_bcast1 - pending.t_proj0) * 1000.0
            self._profile_steps += 1
            if self._profile_steps % max(1, self._profile_every) == 0:
                # Parseable profile line: compare proj/pack/fire vs await to see
                # whether draft RTT is hidden (await << Tad) under wave/async.
                logger.info(
                    "[DisaggDFlash][profile] n=%d transport=%s cross_step=%s "
                    "async=%s wave=%s ms=%s",
                    num_reqs,
                    self._transport_name,
                    self._cross_step,
                    self._async_complete,
                    bool(
                        getattr(
                            self.speculative_config,
                            "disagg_dflash_wave_schedule",
                            False,
                        )
                    ),
                    {k: round(v, 2) for k, v in speculate_ms.items()},
                )

        if self._tp_rank == 0:
            self._record_sd_timing(
                num_reqs=num_reqs,
                tpv_ms=pending.tpv_ms,
                client_ms=client_ms,
                draft_forward_ms=draft_forward_ms,
                await_ms=await_ms,
                accepted=pending.accepted,
            )

        self.draft_tokens[:num_reqs].copy_(draft_tokens[:num_reqs])
        return self.draft_tokens[:num_reqs]

    def _record_sd_timing(
        self,
        *,
        num_reqs: int,
        tpv_ms: float,
        client_ms: dict[str, float],
        draft_forward_ms: float | None,
        await_ms: float = 0.0,
        accepted: float = 0.0,
    ) -> None:
        from vllm.v1.spec_decode.disagg_dflash.timing_model import (
            record_sd_timing,
            timing_model_enabled,
        )

        if not timing_model_enabled():
            return

        if "nixl_e2e_ms" in client_ms:
            ttransfer_ms = float(client_ms["nixl_e2e_ms"])
        elif "xfer_us" in client_ms:
            ttransfer_ms = float(client_ms["xfer_us"]) / 1000.0
        else:
            ttransfer_ms = 0.0
        tzmq_ms = float(client_ms.get("zmq_recv_ms") or 0.0)
        if draft_forward_ms is not None:
            td_ms = float(draft_forward_ms)
        else:
            # Fallback: RTT − transfer − ZMQ wire piece when meta missing.
            rtt_ms = float(client_ms.get("zmq_rtt_ms") or 0.0)
            td_ms = max(0.0, rtt_ms - ttransfer_ms - tzmq_ms)
        record_sd_timing(
            mode="disagg",
            num_reqs=num_reqs,
            tpv_ms=float(tpv_ms),
            ttransfer_ms=ttransfer_ms,
            td_ms=td_ms,
            tzmq_ms=tzmq_ms,
            await_ms=float(await_ms),
            accepted=float(accepted),
        )

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        context_only: bool = False,
    ) -> torch.Tensor:
        pending = self.propose_begin(
            input_batch,
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            num_tokens_across_dp=num_tokens_across_dp,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
            mm_inputs=mm_inputs,
            is_profile=is_profile,
            context_only=context_only,
        )
        return self.propose_finish(pending)
