# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.v1.outputs import DraftTokenIds
from vllm.v1.worker.gpu.async_utils import async_copy_to_np
from vllm.v1.worker.gpu.input_batch import InputBatch


class DraftTokensHandler:
    def __init__(self, device: torch.device | None = None):
        self.device = device
        self.copy_stream = torch.cuda.Stream(device)
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.copy_event = torch.cuda.Event(blocking=True)

        self.req_ids: list[str] = []
        self.draft_tokens_np: np.ndarray | None = None
        self.num_draft_tokens: int = 0
        self._await_copy: bool = False
        # Disagg-DFlash async-complete status for the scheduler.
        # None means "do not update scheduler awaiting set this step".
        # When set: inflight ids are parked into WAITING_FOR_REMOTE_DRAFT;
        # ready ids are marked finished_recving_draft for promotion.
        self.remote_draft_inflight_req_ids: list[str] | None = None
        self.remote_draft_ready_req_ids: list[str] | None = None
        # When True, always D2H draft ids so the scheduler can fan them out
        # via SchedulerOutput (required for Disagg-DFlash TP>1).
        self.force_cpu_draft_ids: bool = False
        # Extra CPU drafts (async-ready) merged into get_draft_tokens().
        self._extra_draft_req_ids: list[str] = []
        self._extra_draft_token_ids: list[list[int]] = []

    def set_remote_draft_status(
        self,
        *,
        inflight_req_ids: list[str] | None = None,
        ready_req_ids: list[str] | None = None,
    ) -> None:
        # Always set both when called from the async-complete path so the
        # scheduler receives an authoritative snapshot.
        self.remote_draft_inflight_req_ids = (
            [] if inflight_req_ids is None else list(inflight_req_ids)
        )
        self.remote_draft_ready_req_ids = (
            [] if ready_req_ids is None else list(ready_req_ids)
        )

    def set_draft_tokens(
        self,
        input_batch: InputBatch,
        draft_tokens: torch.Tensor,
        *,
        force_cpu: bool | None = None,
    ) -> None:
        self.req_ids = list(input_batch.req_ids)
        self.num_draft_tokens = int(draft_tokens.shape[1])
        do_cpu = (
            force_cpu
            if force_cpu is not None
            else (self.force_cpu_draft_ids or input_batch.has_structured_output_reqs)
        )
        if not do_cpu:
            # No draft token validation / scheduler fan-out needed.
            self.draft_tokens_np = None
            self._await_copy = False
            return

        self._copy_draft_tokens_to_cpu(draft_tokens)

    def set_draft_tokens_for_reqs(
        self,
        req_ids: list[str],
        draft_tokens: torch.Tensor,
    ) -> None:
        """D2H draft ids for an explicit req_id list (Disagg async ready)."""
        self.req_ids = list(req_ids)
        self.num_draft_tokens = (
            int(draft_tokens.shape[1]) if draft_tokens.ndim == 2 else 0
        )
        if not req_ids:
            self.draft_tokens_np = None
            self._await_copy = False
            return
        self._copy_draft_tokens_to_cpu(draft_tokens)

    def add_cpu_draft_tokens(
        self, req_ids: list[str], draft_token_ids: list[list[int]]
    ) -> None:
        """Queue already-on-CPU draft ids (merged at get_draft_tokens)."""
        if not req_ids:
            return
        self._extra_draft_req_ids.extend(req_ids)
        self._extra_draft_token_ids.extend(draft_token_ids)
        if draft_token_ids and self.num_draft_tokens <= 0:
            self.num_draft_tokens = len(draft_token_ids[0])

    def _copy_draft_tokens_to_cpu(self, draft_tokens: torch.Tensor) -> None:
        # Spec decode + structured outputs, or Disagg TP fan-out via scheduler.
        current_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.copy_stream):
            self.draft_tokens_np = async_copy_to_np(draft_tokens)
            # draft_tokens may be a temporary allocation on the main stream and
            # is read here on copy_stream; without record_stream, the caching
            # allocator may reuse its memory before the async copy executes.
            draft_tokens.record_stream(self.copy_stream)
            self.copy_event.record()
        self._await_copy = True

    def get_draft_tokens(self) -> DraftTokenIds | None:
        if self.draft_tokens_np is not None:
            if self._await_copy:
                self.copy_event.synchronize()
                self._await_copy = False
            draft_token_ids = self.draft_tokens_np.tolist()
            req_ids = list(self.req_ids)
        elif self.req_ids:
            # This case only happens when async scheduling is disabled and
            # force_cpu_draft_ids is off.
            draft_token_ids = [[-1] * self.num_draft_tokens for _ in self.req_ids]
            req_ids = list(self.req_ids)
        else:
            draft_token_ids = []
            req_ids = []

        if self._extra_draft_req_ids:
            # Async-ready drafts for parked reqs (not necessarily in this batch).
            req_ids = self._extra_draft_req_ids + req_ids
            draft_token_ids = self._extra_draft_token_ids + draft_token_ids
            self._extra_draft_req_ids = []
            self._extra_draft_token_ids = []

        out = DraftTokenIds(
            req_ids,
            draft_token_ids,
            remote_draft_inflight_req_ids=self.remote_draft_inflight_req_ids,
            remote_draft_ready_req_ids=self.remote_draft_ready_req_ids,
        )
        # One-shot: status is consumed by EngineCore.post_step.
        self.remote_draft_inflight_req_ids = None
        self.remote_draft_ready_req_ids = None
        return out


def get_parallel_drafting_token_id(hf_config) -> int:
    """Resolve the mask token id used for parallel drafting slots.

    Checks (in order): `dflash_config.mask_token_id`, top-level `mask_token_id`,
    `dspark_noise_token_id`, `pard_token`, `ptd_token_id`. Raises ValueError if
    none are present.
    """
    dflash_config = getattr(hf_config, "dflash_config", None) or {}
    if "mask_token_id" in dflash_config:
        return int(dflash_config["mask_token_id"])
    if getattr(hf_config, "mask_token_id", None) is not None:
        return int(hf_config.mask_token_id)
    if hasattr(hf_config, "dspark_noise_token_id"):
        return int(hf_config.dspark_noise_token_id)
    if hasattr(hf_config, "pard_token"):
        return int(hf_config.pard_token)
    if hasattr(hf_config, "ptd_token_id"):
        return int(hf_config.ptd_token_id)
    raise ValueError(
        "Model config must specify `dflash_config.mask_token_id`,"
        " `mask_token_id`, `dspark_noise_token_id`, `pard_token`, or"
        " `ptd_token_id` for parallel drafting."
    )
