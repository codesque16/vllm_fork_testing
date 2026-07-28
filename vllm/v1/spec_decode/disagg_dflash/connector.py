# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ZMQ DEALER/ROUTER connector for Disagg-DFlash.

Thin facade that re-exports the ZMQ transport so existing imports keep working.
Prefer ``transport.create_client_transport`` for new call sites.
"""

from __future__ import annotations

from vllm.v1.spec_decode.disagg_dflash.transport_zmq import (
    DisaggDFlashClient,
    DisaggDFlashServerSocket,
    ZmqClientTransport,
)

__all__ = [
    "DisaggDFlashClient",
    "DisaggDFlashServerSocket",
    "ZmqClientTransport",
]
