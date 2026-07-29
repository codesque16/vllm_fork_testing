# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA IPC data-plane transport for same-node Disagg-DFlash.

Control / metadata and the small draft-token response stay on ZMQ. Only the
large ``context_hiddens`` tensor moves via draft-owned ``cudaMalloc`` staging
exported with ``cudaIpcGetMemHandle`` (NVLink / PCIe P2P).

Draft token ids are tiny (``num_reqs × K`` int64) so they always return as
ZMQ frames — IPC staging for tokens is intentionally unused.

Staging is allocated with the CUDA runtime allocator (not the PyTorch caching
allocator) so IPC handles refer to the exact base pointer. Draft copies
into ordinary torch tensors around each speculate.

Copies use ``cudaMemcpyAsync`` on a dedicated stream with CUDA events instead of
full ``cudaDeviceSynchronize`` after every transfer.
"""

from __future__ import annotations

import base64
import ctypes
import os
import time
from typing import Any

import torch

from vllm.distributed.device_communicators.cuda_wrapper import (
    CudaRTLibrary,
    cudaIpcMemHandle_t,
)
from vllm.logger import init_logger
from vllm.v1.spec_decode.disagg_dflash.protocol import (
    PAYLOAD_IPC,
    PAYLOAD_ZMQ,
    DisaggDFlashFreeRequest,
    DisaggDFlashSpeculateRequest,
    DisaggDFlashSpeculateResponse,
    decode_hello_reply,
    encode_hello,
)
from vllm.v1.spec_decode.disagg_dflash.transport import (
    TRANSPORT_CUDA_IPC,
    DisaggDFlashClientTransport,
)
from vllm.v1.spec_decode.disagg_dflash.transport_zmq import ZmqClientTransport

logger = init_logger(__name__)

_ALIGN = 1 << 21  # 2 MiB — CUDA IPC allocation alignment
_CUDA_MEMCPY_DEFAULT = 4


def _aligned_nbytes(nbytes: int) -> int:
    return ((nbytes + _ALIGN - 1) // _ALIGN) * _ALIGN


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _profile_enabled() -> bool:
    from vllm.v1.spec_decode.disagg_dflash.debug_logging import disagg_profile_enabled

    return disagg_profile_enabled()


def export_ipc_handle_ptr(ptr: int | ctypes.c_void_p) -> str:
    """Export ``cudaIpcGetMemHandle`` for a cudaMalloc pointer (base64)."""
    lib = CudaRTLibrary()
    c_ptr = ptr if isinstance(ptr, ctypes.c_void_p) else ctypes.c_void_p(int(ptr))
    handle = lib.cudaIpcGetMemHandle(c_ptr)
    return base64.b64encode(bytes(handle.internal)).decode("ascii")


def open_ipc_ptr(handle_b64: str) -> int:
    """Open a peer CUDA IPC allocation; return device pointer."""
    lib = CudaRTLibrary()
    try:
        handle = cudaIpcMemHandle_t()
        raw = base64.b64decode(handle_b64)
        # cudaIpcMemHandle_t is opaque; libcudart historically uses 64 bytes,
        # while vLLM's ctypes wrapper allocates a 128-byte ``internal`` buffer.
        # Accept either and zero-pad into the structure.
        if len(raw) not in (64, 128):
            raise ValueError(
                f"IPC handle must be 64 or 128 bytes, got {len(raw)}"
            )
        capacity = len(handle.internal)
        for i in range(capacity):
            handle.internal[i] = raw[i] if i < len(raw) else 0
        ptr = lib.cudaIpcOpenMemHandle(handle)
        if not ptr or not ptr.value:
            raise RuntimeError("cudaIpcOpenMemHandle returned NULL")
        return int(ptr.value)
    except Exception as e:
        raise RuntimeError(
            "Failed to open Disagg-DFlash CUDA IPC handle. "
            "cuda_ipc requires draft and verify on the same host with "
            "P2P/NVLink (or PCIe P2P) between TP0's GPU and the draft GPU. "
            f"Underlying error: {e}"
        ) from e


def close_ipc_ptr(ptr: int) -> None:
    if not ptr:
        return
    try:
        lib = CudaRTLibrary()
        if "cudaIpcCloseMemHandle" in getattr(lib, "funcs", {}):
            lib.CUDART_CHECK(
                lib.funcs["cudaIpcCloseMemHandle"](ctypes.c_void_p(ptr))
            )
        else:
            f = getattr(lib.lib, "cudaIpcCloseMemHandle", None)
            if f is not None:
                f.restype = ctypes.c_int
                f.argtypes = [ctypes.c_void_p]
                err = f(ctypes.c_void_p(ptr))
                if err != 0:
                    raise RuntimeError(f"cudaIpcCloseMemHandle err={err}")
    except Exception as e:
        logger.warning("cudaIpcCloseMemHandle failed: %s", e)


def memcpy_d2d_async(
    dst_ptr: int, src_ptr: int, nbytes: int, stream: torch.cuda.Stream | None = None
) -> None:
    """Device-to-device copy on ``stream`` (default stream if None). No device sync."""
    if nbytes <= 0:
        return
    lib = CudaRTLibrary()
    stream_handle = 0 if stream is None else int(stream.cuda_stream)
    f = getattr(lib.lib, "cudaMemcpyAsync", None)
    if f is None:
        # Fallback: synchronous memcpy (still no cudaDeviceSynchronize).
        lib.cudaMemcpy(ctypes.c_void_p(dst_ptr), ctypes.c_void_p(src_ptr), nbytes)
        return
    f.restype = ctypes.c_int
    f.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    err = f(
        ctypes.c_void_p(dst_ptr),
        ctypes.c_void_p(src_ptr),
        ctypes.c_size_t(nbytes),
        ctypes.c_int(_CUDA_MEMCPY_DEFAULT),
        ctypes.c_void_p(stream_handle),
    )
    if err != 0:
        raise RuntimeError(f"cudaMemcpyAsync failed: err={err}")


def memcpy_d2d(dst_ptr: int, src_ptr: int, nbytes: int) -> None:
    """Compatibility helper: async copy on current stream + stream sync."""
    stream = torch.cuda.current_stream()
    memcpy_d2d_async(dst_ptr, src_ptr, nbytes, stream=stream)
    stream.synchronize()


class CudaIpcStaging:
    """Draft-owned cudaMalloc IPC buffer for ``context_hiddens`` only."""

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
            raise ValueError("Invalid CUDA IPC staging sizes")
        self.max_tokens = max_tokens
        self.max_num_seqs = max_num_seqs
        self.hidden_size = hidden_size
        self.num_speculative_tokens = num_speculative_tokens
        self.dtype = dtype
        self.device = device
        self._lib = CudaRTLibrary()
        self._stream = torch.cuda.Stream(device=device)

        # Torch mirror (caching allocator) — draft model reads this after IPC pull.
        self.hidden_staging = torch.zeros(
            max_tokens, hidden_size, dtype=dtype, device=device
        )

        # IPC-exported buffer (cudaMalloc base == handle base) for context hiddens.
        self._hidden_nbytes = int(
            max_tokens * hidden_size * _dtype_nbytes(dtype)
        )
        self._hidden_ipc_ptr = self._lib.cudaMalloc(
            _aligned_nbytes(self._hidden_nbytes)
        )
        self._lib.cudaMemset(self._hidden_ipc_ptr, 0, self._hidden_nbytes)
        self._hidden_handle = export_ipc_handle_ptr(self._hidden_ipc_ptr)

    @property
    def hidden_ipc_ptr(self) -> int:
        return int(self._hidden_ipc_ptr.value)  # type: ignore[arg-type]

    def pull_hiddens_from_ipc(self, num_ctx_tokens: int) -> torch.Tensor:
        """IPC buffer → torch hidden_staging; return ``[:num_ctx]`` view."""
        n = int(num_ctx_tokens)
        if n < 0 or n > self.max_tokens:
            raise ValueError(f"num_ctx_tokens={n} out of range")
        nbytes = n * self.hidden_size * _dtype_nbytes(self.dtype)
        if n > 0:
            # Wait for any prior default-stream work, then async pull + stream sync.
            self._stream.wait_stream(torch.cuda.current_stream(self.device))
            memcpy_d2d_async(
                int(self.hidden_staging.data_ptr()),
                self.hidden_ipc_ptr,
                nbytes,
                stream=self._stream,
            )
            self._stream.synchronize()
        return self.hidden_staging[:n]

    def hello_payload(self) -> dict[str, Any]:
        return {
            "transport": TRANSPORT_CUDA_IPC,
            "max_tokens": self.max_tokens,
            "max_num_seqs": self.max_num_seqs,
            "hidden_size": self.hidden_size,
            "num_speculative_tokens": self.num_speculative_tokens,
            "dtype": str(self.dtype).removeprefix("torch."),
            "hidden_ipc": self._hidden_handle,
            "hidden_nbytes": self._hidden_nbytes,
            # Draft tokens always return on ZMQ (small payload).
            "draft_tokens_payload": "zmq",
        }

    def close(self) -> None:
        try:
            if self._hidden_ipc_ptr:
                self._lib.cudaFree(self._hidden_ipc_ptr)
        except Exception as e:
            logger.warning("CUDA IPC staging free failed: %s", e)
        self._hidden_ipc_ptr = ctypes.c_void_p()


class CudaIpcClientTransport(DisaggDFlashClientTransport):
    """Verify TP0: CUDA IPC for hiddens; ZMQ for control + draft token ids."""

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
        self._hidden_ptr = 0
        self._dtype = torch.bfloat16
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self._stream = torch.cuda.Stream(device=self.device)
        self._hidden_ready = torch.cuda.Event(enable_timing=False)
        # Filled by speculate() for optional profiling.
        self.last_timings_ms: dict[str, float] = {}
        self._inflight: dict[str, Any] | None = None

    def handshake(self) -> None:
        if self._handshook:
            return
        hello_kwargs: dict[str, Any] = {}
        if self._preferred_max_tokens is not None:
            hello_kwargs["preferred_max_tokens"] = int(self._preferred_max_tokens)
        frames = self._zmq.send_recv(
            encode_hello(TRANSPORT_CUDA_IPC, **hello_kwargs)
        )
        meta = decode_hello_reply(frames)
        if meta.get("error"):
            raise RuntimeError(f"Disagg-DFlash CUDA IPC HELLO failed: {meta}")
        server_t = meta.get("transport")
        if server_t != TRANSPORT_CUDA_IPC:
            raise RuntimeError(
                f"Verify requested cuda_ipc but draft replied transport={server_t!r}. "
                "Start the draft server with --transport cuda_ipc."
            )
        self._hidden_ptr = open_ipc_ptr(meta["hidden_ipc"])
        self._max_tokens = int(meta["max_tokens"])
        self._max_num_seqs = int(meta["max_num_seqs"])
        self._hidden_size = int(meta["hidden_size"])
        self._k = int(meta["num_speculative_tokens"])
        dtype_name = str(meta.get("dtype", "bfloat16"))
        self._dtype = getattr(torch, dtype_name, torch.bfloat16)
        self._handshook = True
        logger.info(
            "Disagg-DFlash CUDA IPC handshake ok: max_tokens=%d max_seqs=%d H=%d K=%d "
            "(draft_tokens via ZMQ)",
            self._max_tokens,
            self._max_num_seqs,
            self._hidden_size,
            self._k,
        )

    def speculate(
        self, request: DisaggDFlashSpeculateRequest
    ) -> DisaggDFlashSpeculateResponse:
        self.speculate_begin(request)
        return self.speculate_wait()

    def speculate_begin(self, request: DisaggDFlashSpeculateRequest) -> None:
        """Publish hiddens + send ZMQ meta; draft runs while caller overlaps."""
        if not self._handshook:
            self.handshake()

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
                f"Disagg-DFlash IPC staging too small for {n_ctx} context tokens "
                f"(max_tokens={self._max_tokens}). Raise disagg_dflash_ipc_max_num_tokens "
                "/ --ipc-max-num-tokens on draft."
            )
        if num_reqs > self._max_num_seqs:
            raise RuntimeError(
                f"Disagg-DFlash IPC token staging too small for {num_reqs} reqs "
                f"(max_num_seqs={self._max_num_seqs})."
            )
        if h and h != self._hidden_size:
            raise RuntimeError(
                f"Hidden size mismatch: verify H={h}, draft staging H={self._hidden_size}"
            )

        # Publish hiddens on IPC stream; overlap CPU encode with the copy.
        self._stream.wait_stream(torch.cuda.current_stream(self.device))
        nbytes = n_ctx * self._hidden_size * hiddens.element_size()
        memcpy_d2d_async(
            self._hidden_ptr, int(hiddens.data_ptr()), nbytes, stream=self._stream
        )
        self._hidden_ready.record(self._stream)

        ipc_req = DisaggDFlashSpeculateRequest(
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
            payload_mode=PAYLOAD_IPC,
        )
        frames = ipc_req.encode()

        t1 = time.perf_counter() if profile else 0.0
        # Must complete publish before draft pulls over ZMQ.
        self._hidden_ready.synchronize()
        t2 = time.perf_counter() if profile else 0.0

        self._zmq.send(frames)
        # Always stamp send time so Tad can use zmq_recv_ms without profile.
        t_send = time.perf_counter()
        if not profile:
            t2 = t_send

        k = int(request.num_speculative_tokens)
        self._inflight = {
            "num_reqs": num_reqs,
            "k": k,
            "profile": profile,
            "t0": t0,
            "t1": t1,
            "t2": t2,
            "t_send": t_send,
        }

    def speculate_wait(self) -> DisaggDFlashSpeculateResponse:
        """Recv draft token ids on ZMQ (always — payload is tiny)."""
        inflight = getattr(self, "_inflight", None)
        if inflight is None:
            raise RuntimeError(
                "Disagg-DFlash CUDA IPC speculate_wait without speculate_begin"
            )
        self._inflight = None

        num_reqs = int(inflight["num_reqs"])
        k = int(inflight["k"])
        profile = bool(inflight["profile"])

        resp_frames = self._zmq.recv()
        t3 = time.perf_counter()

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
        # Ensure tokens are visible on the default stream before returning to the
        # caller (DraftTokenIds / scheduler fan-out; no TP draft-id NCCL).
        torch.cuda.current_stream(self.device).wait_stream(self._stream)
        t4 = time.perf_counter() if profile else 0.0

        t_send = float(inflight.get("t_send") or 0.0)
        t2 = float(inflight.get("t2") or 0.0)
        # Always surface ZMQ pieces for Tad (cheap wall clocks).
        self.last_timings_ms = {
            "zmq_recv_ms": (t3 - t_send) * 1000.0 if t_send > 0 else 0.0,
            "zmq_rtt_ms": (t3 - t2) * 1000.0 if t2 > 0 else 0.0,
        }
        if profile:
            t0 = float(inflight["t0"])
            t1 = float(inflight["t1"])
            self.last_timings_ms.update(
                {
                    "ipc_prep_ms": (t1 - t0) * 1000.0,
                    "ipc_hidden_wait_ms": (t2 - t1) * 1000.0,
                    "zmq_send_ms": (t_send - t2) * 1000.0,
                    "token_h2d_ms": (t4 - t3) * 1000.0,
                    "ipc_total_ms": (t4 - t0) * 1000.0,
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
        close_ipc_ptr(self._hidden_ptr)
        self._hidden_ptr = 0
        self._zmq.close()
