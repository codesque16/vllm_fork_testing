# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pluggable Disagg-DFlash transport interface + factory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.spec_decode.disagg_dflash.protocol import (
        DisaggDFlashFreeRequest,
        DisaggDFlashSpeculateRequest,
        DisaggDFlashSpeculateResponse,
    )

logger = init_logger(__name__)

DisaggDFlashTransportName = Literal["zmq", "cuda_ipc", "nccl", "nixl"]

TRANSPORT_ZMQ = "zmq"
TRANSPORT_CUDA_IPC = "cuda_ipc"
TRANSPORT_NCCL = "nccl"
TRANSPORT_NIXL = "nixl"


class DisaggDFlashClientTransport(ABC):
    """Verify-side transport (TP0). Control plane is always ZMQ under the hood."""

    @abstractmethod
    def handshake(self) -> None:
        """Negotiate transport / open IPC handles. Idempotent."""

    @abstractmethod
    def speculate(
        self, request: DisaggDFlashSpeculateRequest
    ) -> DisaggDFlashSpeculateResponse:
        ...

    def speculate_begin(self, request: DisaggDFlashSpeculateRequest) -> None:
        """Send/start speculate so the caller can overlap other work.

        Default falls back to buffering the request for ``speculate_wait``.
        ZMQ / CUDA IPC override this to avoid a host-side stall on send+recv.
        """
        self._buffered_speculate = request  # type: ignore[attr-defined]

    def speculate_wait(self) -> "DisaggDFlashSpeculateResponse":
        """Complete an in-flight speculate started by ``speculate_begin``."""
        request = getattr(self, "_buffered_speculate", None)
        if request is None:
            raise RuntimeError(
                "Disagg-DFlash speculate_wait without matching speculate_begin"
            )
        self._buffered_speculate = None  # type: ignore[attr-defined]
        # Call the transport's blocking path; subclasses that override
        # speculate_begin must also override speculate_wait.
        return self._speculate_blocking(request)

    def _speculate_blocking(
        self, request: DisaggDFlashSpeculateRequest
    ) -> "DisaggDFlashSpeculateResponse":
        """Blocking speculate used by the default begin/wait pair."""
        return self.speculate(request)

    @abstractmethod
    def free(self, request: DisaggDFlashFreeRequest) -> None:
        ...

    @abstractmethod
    def ping(self) -> bool:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


def create_client_transport(
    *,
    address: str,
    timeout_ms: int,
    transport: str = TRANSPORT_ZMQ,
    ipc_max_num_tokens: int | None = None,
) -> DisaggDFlashClientTransport:
    """Factory for verify-side Disagg-DFlash transports."""
    name = (transport or TRANSPORT_ZMQ).lower()
    if name == TRANSPORT_ZMQ:
        from vllm.v1.spec_decode.disagg_dflash.transport_zmq import (
            ZmqClientTransport,
        )

        return ZmqClientTransport(address, timeout_ms=timeout_ms)
    if name == TRANSPORT_CUDA_IPC:
        from vllm.v1.spec_decode.disagg_dflash.transport_cuda_ipc import (
            CudaIpcClientTransport,
        )

        return CudaIpcClientTransport(
            address,
            timeout_ms=timeout_ms,
            preferred_max_tokens=ipc_max_num_tokens,
        )
    if name == TRANSPORT_NIXL:
        from vllm.v1.spec_decode.disagg_dflash.transport_nixl import (
            NixlClientTransport,
        )

        return NixlClientTransport(
            address,
            timeout_ms=timeout_ms,
            preferred_max_tokens=ipc_max_num_tokens,
        )
    if name == TRANSPORT_NCCL:
        raise NotImplementedError(
            "disagg_dflash_transport='nccl' is reserved but not implemented yet. "
            "Use 'zmq', 'cuda_ipc', or 'nixl'."
        )
    raise ValueError(
        f"Unknown disagg_dflash_transport={transport!r}. "
        f"Supported: zmq, cuda_ipc, nixl (stub: nccl)."
    )
