# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire protocol for Disagg-DFlash (msgpack metadata + raw tensor frames)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from msgspec import msgpack

CMD_SPECULATE = "speculate"
CMD_FREE = "free"
CMD_PING = "ping"
CMD_HELLO = "hello"

PAYLOAD_ZMQ = "zmq"
PAYLOAD_IPC = "ipc"
PAYLOAD_NIXL = "nixl"
# Hiddens live in peer staging (CUDA IPC / NIXL); ZMQ carries meta only.
_OFFWIRE_PAYLOAD_MODES = frozenset({PAYLOAD_IPC, PAYLOAD_NIXL})

_DTYPE_TO_STR = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.int32: "int32",
    torch.int64: "int64",
}
_STR_TO_DTYPE = {v: k for k, v in _DTYPE_TO_STR.items()}


def _dtype_str(dtype: torch.dtype) -> str:
    if dtype not in _DTYPE_TO_STR:
        raise ValueError(f"Unsupported dtype for Disagg-DFlash wire protocol: {dtype}")
    return _DTYPE_TO_STR[dtype]


def _tensor_meta(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    t = tensor.detach().contiguous().cpu()
    return {
        "name": name,
        "dtype": _dtype_str(t.dtype),
        "shape": list(t.shape),
        "nbytes": t.numel() * t.element_size(),
    }


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    t = tensor.detach().contiguous().cpu()
    # torch.bfloat16 has no numpy dtype; ship raw 16-bit payload.
    if t.dtype == torch.bfloat16:
        return t.view(torch.uint16).numpy().tobytes()
    return t.numpy().tobytes()


def _bytes_to_tensor(buf: bytes, dtype: str, shape: list[int]) -> torch.Tensor:
    np_dtype = {
        "float16": np.float16,
        "bfloat16": np.dtype(np.uint16),  # handled below
        "float32": np.float32,
        "int32": np.int32,
        "int64": np.int64,
    }[dtype]
    if dtype == "bfloat16":
        # Interpret raw bytes as uint16 then view as bfloat16 via torch.
        arr = np.frombuffer(buf, dtype=np.uint16).reshape(shape)
        return torch.from_numpy(arr.copy()).view(torch.bfloat16)
    arr = np.frombuffer(buf, dtype=np_dtype).reshape(shape).copy()
    return torch.from_numpy(arr)


_META_TENSOR_NAMES = (
    "context_positions",
    "query_start_loc",
    "num_rejected",
    "num_sampled",
    "last_sampled",
    "next_prefill_tokens",
    "temperature",
    "seeds",
)


@dataclass
class DisaggDFlashSpeculateRequest:
    """One verify→draft speculate round (incremental reduced hiddens)."""

    req_ids: list[str]
    # Flat [num_context_tokens, hidden] — already L*H→H reduced on verify.
    # May be a GPU view into IPC staging when payload_mode=ipc.
    context_hiddens: torch.Tensor
    context_positions: torch.Tensor  # [num_context_tokens]
    query_start_loc: torch.Tensor  # [num_reqs + 1] into context tokens
    num_rejected: torch.Tensor  # [num_reqs]
    num_sampled: torch.Tensor  # [num_reqs]
    last_sampled: torch.Tensor  # [num_reqs]
    next_prefill_tokens: torch.Tensor  # [num_reqs]
    temperature: torch.Tensor  # [num_reqs]
    seeds: torch.Tensor  # [num_reqs]
    num_speculative_tokens: int
    payload_mode: str = PAYLOAD_ZMQ

    def encode(self) -> list[bytes]:
        if self.payload_mode in _OFFWIRE_PAYLOAD_MODES:
            return self._encode_offwire()
        tensors = {
            "context_hiddens": self.context_hiddens,
            "context_positions": self.context_positions,
            "query_start_loc": self.query_start_loc,
            "num_rejected": self.num_rejected,
            "num_sampled": self.num_sampled,
            "last_sampled": self.last_sampled,
            "next_prefill_tokens": self.next_prefill_tokens,
            "temperature": self.temperature,
            "seeds": self.seeds,
        }
        meta = {
            "cmd": CMD_SPECULATE,
            "payload_mode": PAYLOAD_ZMQ,
            "req_ids": self.req_ids,
            "num_speculative_tokens": self.num_speculative_tokens,
            "tensors": [_tensor_meta(k, v) for k, v in tensors.items()],
        }
        frames = [msgpack.encode(meta)]
        for t in tensors.values():
            frames.append(_tensor_bytes(t))
        return frames

    def _encode_offwire(self) -> list[bytes]:
        """Metadata + small tensors only; hiddens live in IPC/NIXL staging."""
        tensors = {
            name: getattr(self, name) for name in _META_TENSOR_NAMES
        }
        meta = {
            "cmd": CMD_SPECULATE,
            "payload_mode": self.payload_mode,
            "req_ids": self.req_ids,
            "num_speculative_tokens": self.num_speculative_tokens,
            "num_ctx_tokens": int(self.context_hiddens.shape[0]),
            "hidden_size": int(self.context_hiddens.shape[1])
            if self.context_hiddens.ndim == 2
            else 0,
            "num_reqs": len(self.req_ids),
            "tensors": [_tensor_meta(k, v) for k, v in tensors.items()],
        }
        frames = [msgpack.encode(meta)]
        for t in tensors.values():
            frames.append(_tensor_bytes(t))
        return frames

    @staticmethod
    def decode(
        frames: list[bytes],
        *,
        context_hiddens: torch.Tensor | None = None,
    ) -> DisaggDFlashSpeculateRequest:
        meta = msgpack.decode(frames[0])
        assert meta["cmd"] == CMD_SPECULATE
        payload_mode = meta.get("payload_mode", PAYLOAD_ZMQ)
        tensors: dict[str, torch.Tensor] = {}
        for i, tmeta in enumerate(meta["tensors"]):
            tensors[tmeta["name"]] = _bytes_to_tensor(
                frames[1 + i], tmeta["dtype"], tmeta["shape"]
            )
        if payload_mode in _OFFWIRE_PAYLOAD_MODES:
            if context_hiddens is None:
                raise ValueError(
                    f"{payload_mode} speculate decode requires "
                    "context_hiddens staging view"
                )
            n_ctx = int(meta["num_ctx_tokens"])
            h_size = int(meta["hidden_size"])
            if context_hiddens.shape[0] < n_ctx or (
                h_size and context_hiddens.shape[1] != h_size
            ):
                raise ValueError(
                    f"Staging shape {tuple(context_hiddens.shape)} "
                    f"incompatible with num_ctx={n_ctx} hidden={h_size}"
                )
            hiddens = context_hiddens[:n_ctx]
        else:
            hiddens = tensors["context_hiddens"]
        return DisaggDFlashSpeculateRequest(
            req_ids=list(meta["req_ids"]),
            context_hiddens=hiddens,
            context_positions=tensors["context_positions"],
            query_start_loc=tensors["query_start_loc"],
            num_rejected=tensors["num_rejected"],
            num_sampled=tensors["num_sampled"],
            last_sampled=tensors["last_sampled"],
            next_prefill_tokens=tensors["next_prefill_tokens"],
            temperature=tensors["temperature"],
            seeds=tensors["seeds"],
            num_speculative_tokens=int(meta["num_speculative_tokens"]),
            payload_mode=payload_mode,
        )


@dataclass
class DisaggDFlashSpeculateResponse:
    """Draft → verify reply.

    Production always uses ``PAYLOAD_ZMQ`` for ``draft_tokens`` (tiny payload).
    ``PAYLOAD_IPC`` meta-only replies remain decodable for legacy compatibility.
    """

    draft_tokens: torch.Tensor  # [num_reqs, K] int64
    payload_mode: str = PAYLOAD_ZMQ
    # Optional draft forward latency (ms) for Tpv/Tad timing model.
    draft_forward_ms: float | None = None

    def encode(self) -> list[bytes]:
        if self.payload_mode == PAYLOAD_IPC:
            meta = {
                "cmd": "speculate_response",
                "payload_mode": PAYLOAD_IPC,
                "num_reqs": int(self.draft_tokens.shape[0]),
                "num_speculative_tokens": int(self.draft_tokens.shape[1]),
            }
            if self.draft_forward_ms is not None:
                meta["draft_forward_ms"] = float(self.draft_forward_ms)
            return [msgpack.encode(meta)]
        meta: dict[str, Any] = {
            "cmd": "speculate_response",
            "payload_mode": PAYLOAD_ZMQ,
            "tensors": [_tensor_meta("draft_tokens", self.draft_tokens)],
        }
        if self.draft_forward_ms is not None:
            meta["draft_forward_ms"] = float(self.draft_forward_ms)
        return [msgpack.encode(meta), _tensor_bytes(self.draft_tokens)]

    @staticmethod
    def decode(
        frames: list[bytes],
        *,
        draft_tokens: torch.Tensor | None = None,
    ) -> DisaggDFlashSpeculateResponse:
        meta = msgpack.decode(frames[0])
        payload_mode = meta.get("payload_mode", PAYLOAD_ZMQ)
        draft_forward_ms = meta.get("draft_forward_ms")
        if draft_forward_ms is not None:
            draft_forward_ms = float(draft_forward_ms)
        if payload_mode == PAYLOAD_IPC:
            if draft_tokens is None:
                raise ValueError(
                    "IPC speculate response decode requires draft_tokens staging view"
                )
            n = int(meta["num_reqs"])
            k = int(meta["num_speculative_tokens"])
            return DisaggDFlashSpeculateResponse(
                draft_tokens=draft_tokens[:n, :k],
                payload_mode=PAYLOAD_IPC,
                draft_forward_ms=draft_forward_ms,
            )
        tmeta = meta["tensors"][0]
        tokens = _bytes_to_tensor(frames[1], tmeta["dtype"], tmeta["shape"])
        return DisaggDFlashSpeculateResponse(
            draft_tokens=tokens,
            payload_mode=PAYLOAD_ZMQ,
            draft_forward_ms=draft_forward_ms,
        )


@dataclass
class DisaggDFlashFreeRequest:
    req_ids: list[str]

    def encode(self) -> list[bytes]:
        meta = {"cmd": CMD_FREE, "req_ids": self.req_ids}
        return [msgpack.encode(meta)]

    @staticmethod
    def decode(frames: list[bytes]) -> DisaggDFlashFreeRequest:
        meta = msgpack.decode(frames[0])
        assert meta["cmd"] == CMD_FREE
        return DisaggDFlashFreeRequest(req_ids=list(meta["req_ids"]))


def encode_ping() -> list[bytes]:
    return [msgpack.encode({"cmd": CMD_PING})]


def encode_pong() -> list[bytes]:
    return [msgpack.encode({"cmd": "pong"})]


def encode_hello(transport: str, **extra: Any) -> list[bytes]:
    meta = {"cmd": CMD_HELLO, "transport": transport, **extra}
    return [msgpack.encode(meta)]


def encode_hello_reply(payload: dict[str, Any]) -> list[bytes]:
    meta = {"cmd": "hello_reply", **payload}
    return [msgpack.encode(meta)]


def decode_hello_reply(frames: list[bytes]) -> dict[str, Any]:
    meta = msgpack.decode(frames[0])
    if meta.get("cmd") != "hello_reply":
        raise ValueError(f"Expected hello_reply, got {meta.get('cmd')}")
    return meta


def peek_cmd(frames: list[bytes]) -> str:
    meta = msgpack.decode(frames[0])
    return str(meta["cmd"])
