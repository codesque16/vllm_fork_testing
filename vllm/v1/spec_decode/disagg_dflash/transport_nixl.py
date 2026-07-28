# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL data-plane transport for Disagg-DFlash (P/D-style pattern).

Control / metadata and draft token ids stay on ZMQ. Only ``context_hiddens``
move via NIXL (verify WRITE → draft-owned registered VRAM staging), matching
the same split as ``cuda_ipc`` and the register / handshake / transfer / poll
flow used by P/D ``NixlConnector`` and ``nixl_bw_sweep.py``.

Handshake agent metadata is exchanged over the existing Disagg-DFlash ZMQ
HELLO (no separate NIXL listen-thread side channel).
"""

from __future__ import annotations

import base64
import os
import time
from typing import Any

import torch

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
    """Draft-owned NIXL-registered VRAM buffer for ``context_hiddens``."""

    def __init__(
        self,
        *,
        max_tokens: int,
        max_num_seqs: int,
        hidden_size: int,
        num_speculative_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        if max_tokens < 1 or max_num_seqs < 1 or hidden_size < 1:
            raise ValueError("Invalid NIXL staging sizes")
        self.max_tokens = max_tokens
        self.max_num_seqs = max_num_seqs
        self.hidden_size = hidden_size
        self.num_speculative_tokens = num_speculative_tokens
        self.dtype = dtype
        self.device = device

        self.hidden_staging = torch.zeros(
            max_tokens, hidden_size, dtype=dtype, device=device
        )
        self._hidden_nbytes = int(
            max_tokens * hidden_size * _dtype_nbytes(dtype)
        )
        self.device_id = int(self.hidden_staging.get_device())
        self.hidden_addr = int(self.hidden_staging.data_ptr())

        self._agent = _make_agent(_DRAFT_AGENT_NAME)
        # P/D-style registration: tensor → NIXL reg list.
        self._reg = self._agent.register_memory(self.hidden_staging)
        if not self._reg:
            raise RuntimeError("Disagg-DFlash draft NIXL register_memory failed")
        self._peer_name: str | None = None
        self._recv_count = 0
        self._nixl_log_every = _nixl_log_every()

    def add_remote_verify(self, agent_metadata: bytes) -> str:
        """Register verify agent (bidirectional metadata like P/D handshake)."""
        self._peer_name = self._agent.add_remote_agent(agent_metadata)
        return self._peer_name

    def take_hiddens(self, num_ctx_tokens: int) -> torch.Tensor:
        """Return ``[:num_ctx]`` view after verify NIXL WRITE completed."""
        n = int(num_ctx_tokens)
        if n < 0 or n > self.max_tokens:
            raise ValueError(f"num_ctx_tokens={n} out of range")
        # NIXL DONE on verify implies remote visibility; sync draft stream so
        # subsequent draft kernels see the written bytes.
        # There is no NIXL READ op on this path — verify WRITEs into this
        # buffer; draft-side log is the receiver/"read" view of that WRITE.
        t0 = time.perf_counter()
        if n > 0 and self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        sync_us = (time.perf_counter() - t0) * 1e6
        nbytes = n * self.hidden_size * _dtype_nbytes(self.dtype)
        self._recv_count += 1
        every = self._nixl_log_every
        if every >= 0 and (every == 0 or self._recv_count % max(1, every) == 0):
            logger.info(
                "Disagg-DFlash xfer target→draft (NIXL WRITE recv/draft-side n=%d): "
                "bytes=%.1fKB n_ctx=%d | sync_us=%.1f",
                self._recv_count,
                nbytes / 1024.0,
                n,
                sync_us,
            )
        return self.hidden_staging[:n]

    def hello_payload(self) -> dict[str, Any]:
        return {
            "transport": TRANSPORT_NIXL,
            "max_tokens": self.max_tokens,
            "max_num_seqs": self.max_num_seqs,
            "hidden_size": self.hidden_size,
            "num_speculative_tokens": self.num_speculative_tokens,
            "dtype": str(self.dtype).removeprefix("torch."),
            "agent_metadata": _b64(self._agent.get_agent_metadata()),
            "hidden_addr": self.hidden_addr,
            "hidden_nbytes": self._hidden_nbytes,
            "device_id": self.device_id,
            "mem_type": _MEM_TYPE,
            "draft_tokens_payload": "zmq",
        }

    def close(self) -> None:
        try:
            if self._agent is not None and self._reg is not None:
                self._agent.deregister_memory(self._reg)
        except Exception as e:
            logger.warning("Disagg-DFlash draft NIXL deregister failed: %s", e)
        self._reg = None
        self._agent = None


class NixlClientTransport(DisaggDFlashClientTransport):
    """Verify TP0: NIXL WRITE for hiddens; ZMQ for control + draft token ids."""

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
        self._local_buf: torch.Tensor | None = None
        self._local_reg: Any = None
        self._staging_ready = torch.cuda.Event(enable_timing=False)
        self._remote_addr = 0
        self._remote_device_id = 0
        self._remote_nbytes = 0
        self._mem_type = _MEM_TYPE
        self._timeout_s = max(timeout_ms / 1000.0, 1.0)
        self.last_timings_ms: dict[str, float] = {}
        self._inflight: dict[str, Any] | None = None
        # Rolling NIXL telemetry (P/D-style): accumulate then log averages.
        self._xfer_count = 0
        self._reply_count = 0
        self._sum_post_us = 0.0
        self._sum_xfer_us = 0.0
        self._sum_e2e_us = 0.0
        self._sum_bytes = 0.0
        self._nixl_log_every = _nixl_log_every()

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
        self._remote_addr = int(meta["hidden_addr"])
        self._remote_nbytes = int(meta["hidden_nbytes"])
        self._remote_device_id = int(meta["device_id"])
        self._mem_type = str(meta.get("mem_type", _MEM_TYPE))

        self._local_buf = torch.zeros(
            self._max_tokens,
            self._hidden_size,
            dtype=self._dtype,
            device=self.device,
        )
        self._local_reg = self._agent.register_memory(self._local_buf)
        if not self._local_reg:
            raise RuntimeError("Disagg-DFlash verify NIXL register_memory failed")

        self._handshook = True
        logger.info(
            "Disagg-DFlash NIXL handshake ok: max_tokens=%d max_seqs=%d H=%d K=%d "
            "peer=%s (draft_tokens via ZMQ)",
            self._max_tokens,
            self._max_num_seqs,
            self._hidden_size,
            self._k,
            self._peer_name,
        )

    def speculate(
        self, request: DisaggDFlashSpeculateRequest
    ) -> DisaggDFlashSpeculateResponse:
        self.speculate_begin(request)
        return self.speculate_wait()

    def speculate_begin(self, request: DisaggDFlashSpeculateRequest) -> None:
        """WRITE hiddens over NIXL + send ZMQ meta; draft runs while caller overlaps."""
        if not self._handshook:
            self.handshake()
        assert self._agent is not None
        assert self._local_buf is not None
        assert self._peer_name is not None

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

        nbytes = n_ctx * self._hidden_size * hiddens.element_size()
        if n_ctx > 0:
            # Copy into NIXL-registered staging; sync only the copy event
            # (not the whole device) before the host-driven WRITE.
            self._local_buf[:n_ctx].copy_(hiddens, non_blocking=True)
            self._staging_ready.record(torch.cuda.current_stream(self.device))
            self._staging_ready.synchronize()

        t1 = time.perf_counter()

        xfer_timings: dict[str, float] = {}
        if nbytes > 0:
            local_dev = int(self._local_buf.get_device())
            local_descs = self._agent.get_xfer_descs(
                [(int(self._local_buf.data_ptr()), nbytes, local_dev)],
                mem_type=self._mem_type,
            )
            remote_descs = self._agent.get_xfer_descs(
                [(self._remote_addr, nbytes, self._remote_device_id)],
                mem_type=self._mem_type,
            )
            # Same one-shot pattern as nixl_bw_sweep (variable nbytes per step).
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
                raise RuntimeError("Disagg-DFlash NIXL WRITE transfer failed")
            if st != "DONE":
                _wait_xfer_done(self._agent, handle, timeout_s=self._timeout_s)
            e2e_us = (time.perf_counter() - t_xfer0) * 1e6
            # Must read telemetry before release (P/D NixlConnector pattern).
            telemetry = _read_xfer_telemetry(self._agent, handle)
            self._agent.release_xfer_handle(handle)
            xfer_timings = self._record_and_maybe_log_xfer(
                nbytes=nbytes,
                e2e_us=e2e_us,
                telemetry=telemetry,
                n_ctx=n_ctx,
                num_reqs=num_reqs,
            )

        t2 = time.perf_counter()

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
        )
        frames = nixl_req.encode()
        self._zmq.send(frames)
        t_send = time.perf_counter()

        k = int(request.num_speculative_tokens)
        self._inflight = {
            "num_reqs": num_reqs,
            "k": k,
            "profile": profile,
            "t0": t0,
            "t1": t1,
            "t2": t2,
            "t_send": t_send,
            "xfer_timings": xfer_timings,
        }

    def speculate_wait(self) -> DisaggDFlashSpeculateResponse:
        """Recv draft token ids on ZMQ (always — payload is tiny)."""
        inflight = getattr(self, "_inflight", None)
        if inflight is None:
            raise RuntimeError(
                "Disagg-DFlash NIXL speculate_wait without speculate_begin"
            )
        self._inflight = None

        num_reqs = int(inflight["num_reqs"])
        k = int(inflight["k"])
        profile = bool(inflight["profile"])

        t_recv0 = time.perf_counter()
        resp_frames = self._zmq.recv()
        t3 = time.perf_counter()
        recv_us = (t3 - t_recv0) * 1e6
        t_send = float(inflight.get("t_send") or 0.0)
        rtt_us = (t3 - t_send) * 1e6 if t_send > 0.0 else None

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

        # Draft tokens are int64 [num_reqs, K] on the wire (+ msgpack meta).
        token_nbytes = num_reqs * k * 8
        self._maybe_log_draft_to_target(
            num_reqs=num_reqs,
            k=k,
            token_nbytes=token_nbytes,
            recv_us=recv_us,
            rtt_us=rtt_us,
        )

        # Always surface transfer / ZMQ pieces for Tad (cheap wall clocks).
        xfer_timings = inflight.get("xfer_timings") or {}
        self.last_timings_ms = dict(xfer_timings) if isinstance(xfer_timings, dict) else {}
        self.last_timings_ms["zmq_recv_ms"] = (t3 - t_send) * 1000.0 if t_send > 0 else 0.0
        self.last_timings_ms["zmq_rtt_ms"] = (
            (t3 - float(inflight["t2"])) * 1000.0 if inflight.get("t2") else 0.0
        )
        if profile:
            t0 = float(inflight["t0"])
            t1 = float(inflight["t1"])
            t2 = float(inflight["t2"])
            self.last_timings_ms.update(
                {
                    "nixl_prep_ms": (t1 - t0) * 1000.0,
                    "nixl_write_ms": (t2 - t1) * 1000.0,
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

    def free(self, request: DisaggDFlashFreeRequest) -> None:
        self._zmq.free(request)

    def ping(self) -> bool:
        return self._zmq.ping()

    def close(self) -> None:
        try:
            if self._agent is not None and self._local_reg is not None:
                self._agent.deregister_memory(self._local_reg)
        except Exception as e:
            logger.warning("Disagg-DFlash verify NIXL deregister failed: %s", e)
        self._local_reg = None
        self._agent = None
        self._local_buf = None
        self._zmq.close()
