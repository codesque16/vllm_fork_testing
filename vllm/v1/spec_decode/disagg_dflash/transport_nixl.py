# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL data-plane transport for Disagg-DFlash (P/D-style pattern).

Control / metadata and draft token ids stay on ZMQ. Only ``context_hiddens``
move via NIXL (verify WRITE → draft-owned registered VRAM staging), matching
the same split as ``cuda_ipc`` and the register / handshake / transfer / poll
flow used by P/D ``NixlConnector`` and ``nixl_bw_sweep.py``.

Verify posts the NIXL WRITE in ``speculate_begin`` without waiting for DONE,
so HS DMA can overlap post-sample / next-step work; ``speculate_wait`` joins
the transfer before sending ZMQ meta (draft must see remote staging first).

Ping-pong: two registered staging slots on each side. Verify can post WRITE
into slot B while draft still consumes slot A (ZMQ reply in flight). Slot id
travels in the ZMQ speculate meta (``staging_slot``).

Handshake agent metadata is exchanged over the existing Disagg-DFlash ZMQ
HELLO (no separate NIXL listen-thread side channel).
"""

from __future__ import annotations

import base64
import os
import threading
import time
from collections import deque
from typing import Any

import torch

# Dual staging arenas (ping-pong). Depth matches max in-flight WRITEs.
NUM_STAGING_SLOTS = 2

from vllm.distributed.nixl_utils import NixlWrapper, is_nixl_available, nixl_agent_config
from vllm.logger import init_logger
from vllm.v1.spec_decode.disagg_dflash.protocol import (
    PAYLOAD_NIXL,
    PAYLOAD_ZMQ,
    DisaggDFlashFreeRequest,
    DisaggDFlashSpeculateRequest,
    DisaggDFlashSpeculateResponse,
    decode_hello_reply,
    encode_hello,
)
from vllm.v1.spec_decode.disagg_dflash.transport import (
    TRANSPORT_NIXL,
    DisaggDFlashClientTransport,
)
from vllm.v1.spec_decode.disagg_dflash.transport_zmq import ZmqClientTransport

logger = init_logger(__name__)

_VERIFY_AGENT_NAME = "disagg-dflash-verify"
_DRAFT_AGENT_NAME = "disagg-dflash-draft"
_MEM_TYPE = "VRAM"


def _profile_enabled() -> bool:
    from vllm.v1.spec_decode.disagg_dflash.debug_logging import disagg_profile_enabled

    return disagg_profile_enabled()


def _nixl_log_every() -> int:
    """How often to emit transfer INFO logs (verify WRITE + draft recv + ZMQ).

    - ``-1`` (default): disable these INFO logs
    - ``0``: log every transfer
    - ``N > 0``: log every Nth transfer
    """
    from vllm.v1.spec_decode.disagg_dflash.debug_logging import nixl_log_every

    return nixl_log_every()


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(data: str) -> bytes:
    return base64.b64decode(data)


def _wait_xfer_done(agent: Any, handle: Any, *, timeout_s: float = 120.0) -> None:
    """Poll NIXL transfer to DONE (same pattern as nixl_bw_sweep / P/D)."""
    t0 = time.perf_counter()
    while True:
        st = agent.check_xfer_state(handle)
        if st == "DONE":
            return
        if st == "ERR":
            raise RuntimeError("Disagg-DFlash NIXL transfer entered ERR state")
        if time.perf_counter() - t0 > timeout_s:
            raise TimeoutError(
                f"Disagg-DFlash NIXL transfer timed out after {timeout_s:.1f}s "
                f"(last state={st!r})"
            )
        time.sleep(0.0001)


def _read_xfer_telemetry(agent: Any, handle: Any) -> dict[str, float] | None:
    """Read NIXL post/xfer telemetry (same fields as P/D NixlKVConnectorStats)."""
    try:
        tel = agent.get_xfer_telemetry(handle)
    except Exception as e:
        logger.debug("Disagg-DFlash NIXL get_xfer_telemetry failed: %s", e)
        return None
    post_us = float(getattr(tel, "postDuration", 0.0))
    xfer_us = float(getattr(tel, "xferDuration", 0.0))
    total_bytes = float(getattr(tel, "totalBytes", 0.0))
    desc_count = float(getattr(tel, "descCount", 1.0))
    return {
        "post_us": post_us,
        "xfer_us": xfer_us,
        "bytes": total_bytes,
        "desc_count": desc_count,
    }


def _require_nixl() -> None:
    if not is_nixl_available() or NixlWrapper is None or nixl_agent_config is None:
        raise RuntimeError(
            "disagg_dflash_transport='nixl' requires the nixl (or rixl) package. "
            "Install NIXL or use disagg_dflash_transport='zmq' / 'cuda_ipc'."
        )


def _make_agent(name: str) -> Any:
    _require_nixl()
    # capture_telemetry=True matches P/D NixlConnector (postDuration / xferDuration).
    cfg = nixl_agent_config(
        enable_prog_thread=True,
        enable_listen_thread=False,
        backends=["UCX"],
        capture_telemetry=True,
    )
    return NixlWrapper(name, cfg)


class NixlStaging:
    """Draft-owned ping-pong NIXL-registered VRAM buffers for ``context_hiddens``.

    Two independent arenas so verify can WRITE into slot B while draft still
    reads slot A. Each slot is registered separately with NIXL.
    """

    def __init__(
        self,
        *,
        max_tokens: int,
        max_num_seqs: int,
        hidden_size: int,
        num_speculative_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
        num_slots: int = NUM_STAGING_SLOTS,
    ):
        if max_tokens < 1 or max_num_seqs < 1 or hidden_size < 1:
            raise ValueError("Invalid NIXL staging sizes")
        if num_slots < 1:
            raise ValueError("num_slots must be >= 1")
        self.max_tokens = max_tokens
        self.max_num_seqs = max_num_seqs
        self.hidden_size = hidden_size
        self.num_speculative_tokens = num_speculative_tokens
        self.dtype = dtype
        self.device = device
        self.num_slots = int(num_slots)

        self._hidden_nbytes = int(
            max_tokens * hidden_size * _dtype_nbytes(dtype)
        )
        self.buffers: list[torch.Tensor] = [
            torch.zeros(max_tokens, hidden_size, dtype=dtype, device=device)
            for _ in range(self.num_slots)
        ]
        self.device_id = int(self.buffers[0].get_device())
        self.hidden_addrs = [int(buf.data_ptr()) for buf in self.buffers]
        # Backward-compat aliases (slot 0).
        self.hidden_staging = self.buffers[0]
        self.hidden_addr = self.hidden_addrs[0]

        self._agent = _make_agent(_DRAFT_AGENT_NAME)
        self._regs: list[Any] = []
        for buf in self.buffers:
            reg = self._agent.register_memory(buf)
            if not reg:
                raise RuntimeError(
                    "Disagg-DFlash draft NIXL register_memory failed"
                )
            self._regs.append(reg)
        self._peer_name: str | None = None
        self._recv_count = 0
        self._nixl_log_every = _nixl_log_every()
        logger.info(
            "Disagg-DFlash draft NIXL ping-pong staging: slots=%d "
            "max_tokens=%d H=%d nbytes/slot=%.1fMiB",
            self.num_slots,
            self.max_tokens,
            self.hidden_size,
            self._hidden_nbytes / (1024**2),
        )

    def add_remote_verify(self, agent_metadata: bytes) -> str:
        """Register verify agent (bidirectional metadata like P/D handshake)."""
        self._peer_name = self._agent.add_remote_agent(agent_metadata)
        return self._peer_name

    def take_hiddens(
        self, num_ctx_tokens: int, staging_slot: int = 0
    ) -> torch.Tensor:
        """Return ``buffers[slot][:num_ctx]`` view after verify NIXL WRITE."""
        slot = int(staging_slot)
        if slot < 0 or slot >= self.num_slots:
            raise ValueError(
                f"staging_slot={slot} out of range [0, {self.num_slots})"
            )
        n = int(num_ctx_tokens)
        if n < 0 or n > self.max_tokens:
            raise ValueError(f"num_ctx_tokens={n} out of range")
        # NIXL DONE on verify implies remote visibility; fence the draft stream so
        # subsequent draft kernels see the written bytes (correctness, always).
        t0 = time.perf_counter()
        if n > 0 and self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        fence_us = (time.perf_counter() - t0) * 1e6
        nbytes = n * self.hidden_size * _dtype_nbytes(self.dtype)
        self._recv_count += 1
        every = self._nixl_log_every
        if every >= 0 and (every == 0 or self._recv_count % max(1, every) == 0):
            logger.info(
                "Disagg-DFlash xfer target→draft (NIXL WRITE recv/draft-side n=%d): "
                "bytes=%.1fKB n_ctx=%d slot=%d | fence_us=%.1f",
                self._recv_count,
                nbytes / 1024.0,
                n,
                slot,
                fence_us,
            )
        return self.buffers[slot][:n]

    def hello_payload(self) -> dict[str, Any]:
        return {
            "transport": TRANSPORT_NIXL,
            "max_tokens": self.max_tokens,
            "max_num_seqs": self.max_num_seqs,
            "hidden_size": self.hidden_size,
            "num_speculative_tokens": self.num_speculative_tokens,
            "dtype": str(self.dtype).removeprefix("torch."),
            "agent_metadata": _b64(self._agent.get_agent_metadata()),
            # Slot 0 aliases for older clients.
            "hidden_addr": self.hidden_addrs[0],
            "hidden_nbytes": self._hidden_nbytes,
            # Ping-pong: list of per-slot base addresses (same nbytes each).
            "num_slots": self.num_slots,
            "hidden_addrs": list(self.hidden_addrs),
            "device_id": self.device_id,
            "mem_type": _MEM_TYPE,
            "draft_tokens_payload": "zmq",
        }

    def close(self) -> None:
        try:
            if self._agent is not None:
                for reg in self._regs:
                    try:
                        self._agent.deregister_memory(reg)
                    except Exception as e:
                        logger.warning(
                            "Disagg-DFlash draft NIXL deregister failed: %s", e
                        )
        except Exception as e:
            logger.warning("Disagg-DFlash draft NIXL deregister failed: %s", e)
        self._regs = []
        self._agent = None


class NixlClientTransport(DisaggDFlashClientTransport):
    """Verify TP0: NIXL WRITE for hiddens; ZMQ for control + draft token ids.

    Supports ping-pong staging: up to ``num_slots`` WRITEs may be posted before
    the matching ZMQ speculate round-trips complete (FIFO). ``speculate_begin``
    is safe to call while a prior ``speculate_wait`` is blocked on ZMQ recv.
    """

    def __init__(
        self,
        address: str,
        timeout_ms: int = 5000,
        preferred_max_tokens: int | None = None,
    ):
        self._zmq = ZmqClientTransport(address, timeout_ms=timeout_ms)
        self._preferred_max_tokens = preferred_max_tokens
        self._handshook = False
        self._max_tokens = 0
        self._max_num_seqs = 0
        self._hidden_size = 0
        self._k = 0
        self._dtype = torch.bfloat16
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self._agent: Any = None
        self._peer_name: str | None = None
        self._num_slots = NUM_STAGING_SLOTS
        self._local_bufs: list[torch.Tensor] = []
        self._local_regs: list[Any] = []
        self._staging_ready: list[torch.cuda.Event] = []
        self._remote_addrs: list[int] = []
        self._remote_device_id = 0
        self._remote_nbytes = 0
        self._mem_type = _MEM_TYPE
        self._timeout_s = max(timeout_ms / 1000.0, 1.0)
        self.last_timings_ms: dict[str, float] = {}
        # FIFO of in-flight WRITEs / pending ZMQ round-trips (max num_slots).
        self._inflight_q: deque[dict[str, Any]] = deque()
        self._free_slots: deque[int] = deque()
        # Protects slot alloc + inflight queue; not held across ZMQ recv.
        self._slot_lock = threading.Lock()
        # Serializes NIXL agent post/join (not held across ZMQ recv so the
        # other ping-pong slot can be filled while draft computes).
        self._xfer_lock = threading.Lock()
        # Rolling NIXL telemetry (P/D-style): accumulate then log averages.
        self._xfer_count = 0
        self._reply_count = 0
        self._sum_post_us = 0.0
        self._sum_xfer_us = 0.0
        self._sum_e2e_us = 0.0
        self._sum_bytes = 0.0
        self._nixl_log_every = _nixl_log_every()

    @property
    def num_slots(self) -> int:
        return self._num_slots

    def inflight_depth(self) -> int:
        with self._slot_lock:
            return len(self._inflight_q)

    def _should_log_xfer(self, count: int) -> bool:
        every = self._nixl_log_every
        if every < 0:
            return False
        return every == 0 or (count % max(1, every) == 0)

    def _record_and_maybe_log_xfer(
        self,
        *,
        nbytes: int,
        e2e_us: float,
        telemetry: dict[str, float] | None,
        n_ctx: int,
        num_reqs: int,
    ) -> dict[str, float]:
        """Update last_timings + INFO log for target→draft NIXL WRITE."""
        post_us = float(telemetry["post_us"]) if telemetry else 0.0
        xfer_us = float(telemetry["xfer_us"]) if telemetry else 0.0
        tel_bytes = (
            float(telemetry["bytes"])
            if telemetry and telemetry.get("bytes", 0) > 0
            else float(nbytes)
        )
        # Wire BW from NIXL xferDuration; e2e includes post + poll wait.
        xfer_gbps = (
            (tel_bytes / (xfer_us * 1e-6)) / (1024**3) if xfer_us > 0 else 0.0
        )
        e2e_gbps = (tel_bytes / (e2e_us * 1e-6)) / (1024**3) if e2e_us > 0 else 0.0
        post_ms = post_us / 1000.0
        xfer_ms = xfer_us / 1000.0
        e2e_ms = e2e_us / 1000.0
        mb = tel_bytes / (1024**2)

        self._xfer_count += 1
        self._sum_post_us += post_us
        self._sum_xfer_us += xfer_us
        self._sum_e2e_us += e2e_us
        self._sum_bytes += tel_bytes

        timings = {
            "nixl_bytes": tel_bytes,
            "nixl_mb": mb,
            "nixl_post_ms": post_ms,
            "nixl_xfer_ms": xfer_ms,
            "nixl_e2e_ms": e2e_ms,
            "nixl_gbps": xfer_gbps,
            "nixl_e2e_gbps": e2e_gbps,
            "nixl_n_ctx": float(n_ctx),
            "nixl_num_reqs": float(num_reqs),
        }

        every = self._nixl_log_every
        if self._should_log_xfer(self._xfer_count):
            if every == 0:
                logger.info(
                    "Disagg-DFlash xfer target→draft (NIXL WRITE n=%d): "
                    "bytes=%.1fKB n_ctx=%d reqs=%d | "
                    "post_us=%.1f xfer_us=%.1f e2e_us=%.1f | "
                    "GB/s=%.3f e2e_GB/s=%.3f",
                    self._xfer_count,
                    tel_bytes / 1024.0,
                    n_ctx,
                    num_reqs,
                    post_us,
                    xfer_us,
                    e2e_us,
                    xfer_gbps,
                    e2e_gbps,
                )
            else:
                n = max(1, every)
                avg_post = self._sum_post_us / n
                avg_xfer = self._sum_xfer_us / n
                avg_e2e = self._sum_e2e_us / n
                avg_bytes = self._sum_bytes / n
                self._sum_post_us = 0.0
                self._sum_xfer_us = 0.0
                self._sum_e2e_us = 0.0
                self._sum_bytes = 0.0
                avg_xfer_gbps = (
                    (avg_bytes / (avg_xfer * 1e-6)) / (1024**3)
                    if avg_xfer > 0
                    else 0.0
                )
                avg_e2e_gbps = (
                    (avg_bytes / (avg_e2e * 1e-6)) / (1024**3) if avg_e2e > 0 else 0.0
                )
                logger.info(
                    "Disagg-DFlash xfer target→draft (NIXL WRITE n=%d window=%d): "
                    "bytes=%.1fKB n_ctx=%d reqs=%d | "
                    "post_us=%.1f xfer_us=%.1f e2e_us=%.1f | "
                    "GB/s=%.3f e2e_GB/s=%.3f | "
                    "avg_post_us=%.1f avg_xfer_us=%.1f avg_GB/s=%.3f "
                    "avg_e2e_GB/s=%.3f",
                    self._xfer_count,
                    n,
                    tel_bytes / 1024.0,
                    n_ctx,
                    num_reqs,
                    post_us,
                    xfer_us,
                    e2e_us,
                    xfer_gbps,
                    e2e_gbps,
                    avg_post,
                    avg_xfer,
                    avg_xfer_gbps,
                    avg_e2e_gbps,
                )
        return timings

    def _maybe_log_draft_to_target(
        self,
        *,
        num_reqs: int,
        k: int,
        token_nbytes: int,
        recv_us: float,
        rtt_us: float | None,
    ) -> None:
        """INFO log for draft→target ZMQ draft-token reply."""
        self._reply_count += 1
        if not self._should_log_xfer(self._reply_count):
            return
        if rtt_us is not None:
            logger.info(
                "Disagg-DFlash xfer draft→target (ZMQ tokens n=%d): "
                "reqs=%d K=%d bytes=%.1fKB | recv_us=%.1f rtt_us=%.1f",
                self._reply_count,
                num_reqs,
                k,
                token_nbytes / 1024.0,
                recv_us,
                rtt_us,
            )
        else:
            logger.info(
                "Disagg-DFlash xfer draft→target (ZMQ tokens n=%d): "
                "reqs=%d K=%d bytes=%.1fKB | recv_us=%.1f",
                self._reply_count,
                num_reqs,
                k,
                token_nbytes / 1024.0,
                recv_us,
            )

    def handshake(self) -> None:
        if self._handshook:
            return
        _require_nixl()
        self._agent = _make_agent(_VERIFY_AGENT_NAME)

        hello_kwargs: dict[str, Any] = {
            "agent_metadata": _b64(self._agent.get_agent_metadata()),
        }
        if self._preferred_max_tokens is not None:
            hello_kwargs["preferred_max_tokens"] = int(self._preferred_max_tokens)

        frames = self._zmq.send_recv(encode_hello(TRANSPORT_NIXL, **hello_kwargs))
        meta = decode_hello_reply(frames)
        if meta.get("error"):
            raise RuntimeError(f"Disagg-DFlash NIXL HELLO failed: {meta}")
        server_t = meta.get("transport")
        if server_t != TRANSPORT_NIXL:
            raise RuntimeError(
                f"Verify requested nixl but draft replied transport={server_t!r}. "
                "Start the draft server with --transport nixl."
            )

        self._peer_name = self._agent.add_remote_agent(_unb64(meta["agent_metadata"]))
        self._max_tokens = int(meta["max_tokens"])
        self._max_num_seqs = int(meta["max_num_seqs"])
        self._hidden_size = int(meta["hidden_size"])
        self._k = int(meta["num_speculative_tokens"])
        dtype_name = str(meta.get("dtype", "bfloat16"))
        self._dtype = getattr(torch, dtype_name, torch.bfloat16)
        self._remote_nbytes = int(meta["hidden_nbytes"])
        self._remote_device_id = int(meta["device_id"])
        self._mem_type = str(meta.get("mem_type", _MEM_TYPE))

        # Prefer ping-pong addr list; fall back to single hidden_addr.
        addrs = meta.get("hidden_addrs")
        if isinstance(addrs, list) and len(addrs) >= 1:
            self._remote_addrs = [int(a) for a in addrs]
            self._num_slots = len(self._remote_addrs)
        else:
            self._remote_addrs = [int(meta["hidden_addr"])]
            self._num_slots = 1
        # Cap at local ping-pong depth.
        self._num_slots = min(self._num_slots, NUM_STAGING_SLOTS)
        self._remote_addrs = self._remote_addrs[: self._num_slots]

        self._local_bufs = []
        self._local_regs = []
        self._staging_ready = []
        for _ in range(self._num_slots):
            buf = torch.zeros(
                self._max_tokens,
                self._hidden_size,
                dtype=self._dtype,
                device=self.device,
            )
            reg = self._agent.register_memory(buf)
            if not reg:
                raise RuntimeError(
                    "Disagg-DFlash verify NIXL register_memory failed"
                )
            self._local_bufs.append(buf)
            self._local_regs.append(reg)
            self._staging_ready.append(
                torch.cuda.Event(enable_timing=False)
            )
        self._free_slots = deque(range(self._num_slots))
        self._inflight_q.clear()

        self._handshook = True
        logger.info(
            "Disagg-DFlash NIXL handshake ok: max_tokens=%d max_seqs=%d H=%d K=%d "
            "slots=%d peer=%s (draft_tokens via ZMQ, ping-pong)",
            self._max_tokens,
            self._max_num_seqs,
            self._hidden_size,
            self._k,
            self._num_slots,
            self._peer_name,
        )

    def speculate(
        self, request: DisaggDFlashSpeculateRequest
    ) -> DisaggDFlashSpeculateResponse:
        self.speculate_begin(request)
        return self.speculate_wait()

    def speculate_begin(self, request: DisaggDFlashSpeculateRequest) -> None:
        """Kick NIXL WRITE into a free ping-pong slot (non-blocking).

        Safe to call while a prior ``speculate_wait`` is blocked on ZMQ recv —
        that is the point of ping-pong: fill slot B while draft holds slot A.
        """
        if not self._handshook:
            self.handshake()
        assert self._agent is not None
        assert self._peer_name is not None
        if not self._local_bufs:
            raise RuntimeError("Disagg-DFlash NIXL local staging not initialized")

        profile = _profile_enabled()
        t0 = time.perf_counter() if profile else 0.0

        hiddens = request.context_hiddens
        if not hiddens.is_cuda:
            hiddens = hiddens.to(device=self.device, non_blocking=True)
        hiddens = hiddens.to(dtype=self._dtype).contiguous()
        n_ctx = int(hiddens.shape[0])
        h = int(hiddens.shape[1]) if hiddens.ndim == 2 else 0
        num_reqs = len(request.req_ids)
        if n_ctx > self._max_tokens:
            raise RuntimeError(
                f"Disagg-DFlash NIXL staging too small for {n_ctx} context tokens "
                f"(max_tokens={self._max_tokens}). Raise disagg_dflash_ipc_max_num_tokens "
                "/ --ipc-max-num-tokens on draft."
            )
        if num_reqs > self._max_num_seqs:
            raise RuntimeError(
                f"Disagg-DFlash NIXL staging too small for {num_reqs} reqs "
                f"(max_num_seqs={self._max_num_seqs})."
            )
        if h and h != self._hidden_size:
            raise RuntimeError(
                f"Hidden size mismatch: verify H={h}, draft staging H={self._hidden_size}"
            )

        with self._slot_lock:
            if not self._free_slots:
                raise RuntimeError(
                    "Disagg-DFlash NIXL: no free ping-pong staging slot "
                    f"(depth={self._num_slots}, inflight={len(self._inflight_q)})"
                )
            slot = int(self._free_slots.popleft())

        local_buf = self._local_bufs[slot]
        ready_evt = self._staging_ready[slot]
        remote_addr = self._remote_addrs[slot]

        nbytes = n_ctx * self._hidden_size * hiddens.element_size()
        handle = None
        t_xfer0 = 0.0
        try:
            with self._xfer_lock:
                if n_ctx > 0:
                    local_buf[:n_ctx].copy_(hiddens, non_blocking=True)
                    ready_evt.record(torch.cuda.current_stream(self.device))
                    ready_evt.synchronize()

                t1 = time.perf_counter()

                if nbytes > 0:
                    local_dev = int(local_buf.get_device())
                    local_descs = self._agent.get_xfer_descs(
                        [(int(local_buf.data_ptr()), nbytes, local_dev)],
                        mem_type=self._mem_type,
                    )
                    remote_descs = self._agent.get_xfer_descs(
                        [(remote_addr, nbytes, self._remote_device_id)],
                        mem_type=self._mem_type,
                    )
                    t_xfer0 = time.perf_counter()
                    handle = self._agent.initialize_xfer(
                        "WRITE",
                        local_descs,
                        remote_descs,
                        self._peer_name,
                        b"",
                    )
                    st = self._agent.transfer(handle)
                    if st == "ERR":
                        self._agent.release_xfer_handle(handle)
                        handle = None
                        raise RuntimeError(
                            "Disagg-DFlash NIXL WRITE transfer failed"
                        )
        except Exception:
            with self._slot_lock:
                self._free_slots.append(slot)
            raise

        nixl_req = DisaggDFlashSpeculateRequest(
            req_ids=request.req_ids,
            context_hiddens=hiddens,
            context_positions=request.context_positions,
            query_start_loc=request.query_start_loc,
            num_rejected=request.num_rejected,
            num_sampled=request.num_sampled,
            last_sampled=request.last_sampled,
            next_prefill_tokens=request.next_prefill_tokens,
            temperature=request.temperature,
            seeds=request.seeds,
            num_speculative_tokens=request.num_speculative_tokens,
            payload_mode=PAYLOAD_NIXL,
            staging_slot=slot,
        )
        t_kick = time.perf_counter()

        k = int(request.num_speculative_tokens)
        with self._slot_lock:
            self._inflight_q.append(
                {
                    "slot": slot,
                    "num_reqs": num_reqs,
                    "k": k,
                    "n_ctx": n_ctx,
                    "nbytes": nbytes,
                    "profile": profile,
                    "t0": t0,
                    "t1": t1,
                    "t_kick": t_kick,
                    "t_xfer0": t_xfer0,
                    "handle": handle,
                    "nixl_req": nixl_req,
                }
            )

    def speculate_wait(self) -> DisaggDFlashSpeculateResponse:
        """Join oldest NIXL WRITE, send ZMQ meta (with slot), recv draft tokens.

        Slot is returned to the free pool only after the ZMQ reply (draft has
        finished consuming that staging arena).
        """
        with self._slot_lock:
            if not self._inflight_q:
                raise RuntimeError(
                    "Disagg-DFlash NIXL speculate_wait without speculate_begin"
                )
            inflight = self._inflight_q.popleft()

        slot = int(inflight["slot"])
        num_reqs = int(inflight["num_reqs"])
        k = int(inflight["k"])
        n_ctx = int(inflight["n_ctx"])
        nbytes = int(inflight["nbytes"])
        profile = bool(inflight["profile"])
        handle = inflight.get("handle")
        nixl_req: DisaggDFlashSpeculateRequest = inflight["nixl_req"]
        assert self._agent is not None

        xfer_timings: dict[str, float] = {}
        t2 = time.perf_counter()
        try:
            with self._xfer_lock:
                if handle is not None:
                    try:
                        st = self._agent.check_xfer_state(handle)
                        if st == "ERR":
                            raise RuntimeError(
                                "Disagg-DFlash NIXL WRITE entered ERR state"
                            )
                        if st != "DONE":
                            _wait_xfer_done(
                                self._agent, handle, timeout_s=self._timeout_s
                            )
                        t2 = time.perf_counter()
                        e2e_us = (t2 - float(inflight["t_xfer0"])) * 1e6
                        telemetry = _read_xfer_telemetry(self._agent, handle)
                        xfer_timings = self._record_and_maybe_log_xfer(
                            nbytes=nbytes,
                            e2e_us=e2e_us,
                            telemetry=telemetry,
                            n_ctx=n_ctx,
                            num_reqs=num_reqs,
                        )
                    finally:
                        try:
                            self._agent.release_xfer_handle(handle)
                        except Exception as e:
                            logger.warning(
                                "Disagg-DFlash NIXL release_xfer_handle failed: %s",
                                e,
                            )

            # Only notify draft after remote staging is visible (NIXL DONE).
            # xfer_lock is released so begin can fill the other slot during recv.
            frames = nixl_req.encode()
            self._zmq.send(frames)
            t_send = time.perf_counter()

            t_recv0 = time.perf_counter()
            resp_frames = self._zmq.recv()
            t3 = time.perf_counter()
            recv_us = (t3 - t_recv0) * 1e6
            rtt_us = (t3 - t_send) * 1e6

            resp = DisaggDFlashSpeculateResponse.decode(resp_frames)
            out = resp.draft_tokens
            if out.shape[0] < num_reqs or out.shape[1] < k:
                raise RuntimeError(
                    f"Disagg-DFlash draft token shape {tuple(out.shape)} "
                    f"incompatible with num_reqs={num_reqs} K={k}"
                )
            out = out[:num_reqs, :k].to(
                device=self.device, dtype=torch.int64, non_blocking=True
            )
            t4 = time.perf_counter() if profile else 0.0

            token_nbytes = num_reqs * k * 8
            self._maybe_log_draft_to_target(
                num_reqs=num_reqs,
                k=k,
                token_nbytes=token_nbytes,
                recv_us=recv_us,
                rtt_us=rtt_us,
            )

            self.last_timings_ms = (
                dict(xfer_timings) if isinstance(xfer_timings, dict) else {}
            )
            self.last_timings_ms["zmq_recv_ms"] = (t3 - t_send) * 1000.0
            self.last_timings_ms["zmq_rtt_ms"] = (t3 - t2) * 1000.0
            self.last_timings_ms["staging_slot"] = float(slot)
            if profile:
                t0 = float(inflight["t0"])
                t1 = float(inflight["t1"])
                t_kick = float(inflight["t_kick"])
                self.last_timings_ms.update(
                    {
                        "nixl_prep_ms": (t1 - t0) * 1000.0,
                        "nixl_kick_ms": (t_kick - t1) * 1000.0,
                        "nixl_write_ms": (
                            (t2 - float(inflight["t_xfer0"] or t_kick)) * 1000.0
                            if nbytes > 0
                            else 0.0
                        ),
                        "nixl_await_ms": (t2 - t_kick) * 1000.0,
                        "zmq_send_ms": (t_send - t2) * 1000.0,
                        "token_h2d_ms": (t4 - t3) * 1000.0,
                        "nixl_total_ms": (t4 - t0) * 1000.0,
                    }
                )
            return DisaggDFlashSpeculateResponse(
                draft_tokens=out,
                payload_mode=PAYLOAD_ZMQ,
                draft_forward_ms=resp.draft_forward_ms,
            )
        finally:
            # Free slot only after draft finished with that arena (ZMQ replied
            # or this wait failed). Enables the next WRITE into this slot.
            with self._slot_lock:
                self._free_slots.append(slot)

    def free(self, request: DisaggDFlashFreeRequest) -> None:
        self._zmq.free(request)

    def ping(self) -> bool:
        return self._zmq.ping()

    def close(self) -> None:
        try:
            if self._agent is not None:
                for reg in self._local_regs:
                    try:
                        self._agent.deregister_memory(reg)
                    except Exception as e:
                        logger.warning(
                            "Disagg-DFlash verify NIXL deregister failed: %s", e
                        )
        except Exception as e:
            logger.warning("Disagg-DFlash verify NIXL deregister failed: %s", e)
        self._local_regs = []
        self._local_bufs = []
        self._agent = None
        self._zmq.close()
