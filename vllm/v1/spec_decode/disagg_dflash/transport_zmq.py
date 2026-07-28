# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ZMQ DEALER/ROUTER transport for Disagg-DFlash (control + full tensor frames)."""

from __future__ import annotations

import threading
from typing import Any, Callable

import zmq
from msgspec import msgpack

from vllm.logger import init_logger
from vllm.v1.spec_decode.disagg_dflash.protocol import (
    CMD_FREE,
    CMD_HELLO,
    CMD_PING,
    CMD_SPECULATE,
    DisaggDFlashFreeRequest,
    DisaggDFlashSpeculateRequest,
    DisaggDFlashSpeculateResponse,
    encode_hello_reply,
    encode_ping,
    encode_pong,
    peek_cmd,
)
from vllm.v1.spec_decode.disagg_dflash.transport import (
    TRANSPORT_ZMQ,
    DisaggDFlashClientTransport,
)

logger = init_logger(__name__)

# Handler: (cmd, payload) -> response frames.
# For SPECulate/FREE, payload is the decoded request dataclass.
# For HELLO, payload is the decoded hello meta dict.
ServerHandler = Callable[[str, Any], list[bytes]]
# Optional: custom speculate frame decoder (e.g. IPC staging views).
SpeculateDecoder = Callable[[list[bytes]], DisaggDFlashSpeculateRequest]


class ZmqClientTransport(DisaggDFlashClientTransport):
    """Verify-side ZMQ DEALER client (pure tensor-in-frames data plane)."""

    def __init__(self, address: str, timeout_ms: int = 5000):
        self.address = address
        self.timeout_ms = timeout_ms
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.DEALER)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._sock.connect(address)
        self._handshook = False
        logger.info("Disagg-DFlash ZMQ client connected to %s", address)

    def handshake(self) -> None:
        """Announce zmq transport; accept zmq or noop-compatible hello_reply."""
        if self._handshook:
            return
        from vllm.v1.spec_decode.disagg_dflash.protocol import (
            decode_hello_reply,
            encode_hello,
        )

        try:
            self._sock.send_multipart(encode_hello(TRANSPORT_ZMQ))
            meta = decode_hello_reply(self._sock.recv_multipart())
            server_t = meta.get("transport", TRANSPORT_ZMQ)
            if server_t not in (TRANSPORT_ZMQ, "zmq"):
                raise RuntimeError(
                    f"Disagg-DFlash ZMQ client got hello_reply transport={server_t!r}"
                )
        except zmq.ZMQError as e:
            # Older draft servers may not speak HELLO; ZMQ path still works.
            logger.warning(
                "Disagg-DFlash HELLO failed (%s); continuing with legacy ZMQ frames",
                e,
            )
        self._handshook = True

    def speculate(
        self, request: DisaggDFlashSpeculateRequest
    ) -> DisaggDFlashSpeculateResponse:
        self.speculate_begin(request)
        return self.speculate_wait()

    def speculate_begin(self, request: DisaggDFlashSpeculateRequest) -> None:
        """Send speculate frames; draft starts while caller does other work."""
        self._sock.send_multipart(request.encode())

    def speculate_wait(self) -> DisaggDFlashSpeculateResponse:
        """Block until the draft replies to the in-flight speculate."""
        return DisaggDFlashSpeculateResponse.decode(self._sock.recv_multipart())

    def free(self, request: DisaggDFlashFreeRequest) -> None:
        # Fire-and-forget: draft may be mid-speculate; don't fail verify on timeout.
        try:
            self._sock.send_multipart(request.encode())
            self._sock.recv_multipart()
        except zmq.ZMQError as e:
            logger.warning("Disagg-DFlash FREE ack failed: %s", e)

    def ping(self) -> bool:
        try:
            self._sock.send_multipart(encode_ping())
            frames = self._sock.recv_multipart()
            return peek_cmd(frames) == "pong"
        except zmq.ZMQError as e:
            logger.warning("Disagg-DFlash ping failed: %s", e)
            return False

    def close(self) -> None:
        self._sock.close(linger=0)

    # Low-level access for CUDA IPC transport wrappers.
    def send(self, frames: list[bytes]) -> None:
        self._sock.send_multipart(frames)

    def recv(self) -> list[bytes]:
        return self._sock.recv_multipart()

    def send_recv(self, frames: list[bytes]) -> list[bytes]:
        self.send(frames)
        return self.recv()


class DisaggDFlashServerSocket:
    """Draft-side ZMQ ROUTER socket with a request handler callback."""

    def __init__(self, bind_address: str):
        self.bind_address = bind_address
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.ROUTER)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(bind_address)
        self._stop = threading.Event()
        logger.info("Disagg-DFlash draft server listening on %s", bind_address)

    def serve(
        self,
        handler: ServerHandler,
        *,
        speculate_decoder: SpeculateDecoder | None = None,
    ) -> None:
        """Block serving until stop() is called.

        ``handler(cmd, payload) -> list[bytes]`` returns response frames.
        """
        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)
        while not self._stop.is_set():
            events = dict(poller.poll(timeout=200))
            if self._sock not in events:
                continue
            parts = self._sock.recv_multipart()
            # ROUTER: [identity, ...]
            identity, frames = parts[0], parts[1:]
            cmd = peek_cmd(frames)
            decoded_speculate = None
            try:
                if cmd == CMD_PING:
                    resp = encode_pong()
                elif cmd == CMD_HELLO:
                    hello_meta = msgpack.decode(frames[0])
                    resp = handler(CMD_HELLO, hello_meta)
                elif cmd == CMD_SPECULATE:
                    if speculate_decoder is not None:
                        decoded_speculate = speculate_decoder(frames)
                    else:
                        decoded_speculate = DisaggDFlashSpeculateRequest.decode(
                            frames
                        )
                    resp = handler(CMD_SPECULATE, decoded_speculate)
                elif cmd == CMD_FREE:
                    req = DisaggDFlashFreeRequest.decode(frames)
                    resp = handler(CMD_FREE, req)
                else:
                    logger.warning("Unknown Disagg-DFlash cmd=%s", cmd)
                    resp = encode_pong()
            except Exception:
                logger.exception("Disagg-DFlash handler failed for cmd=%s", cmd)
                # Never reply pong to speculate — verify would fail to decode
                # and can leave the ZMQ DEALER desynchronized.
                if cmd == CMD_SPECULATE:
                    import torch

                    # Prefer the already-decoded request (NIXL take_hiddens
                    # consumes the staging buffer; re-decode often fails).
                    if decoded_speculate is not None:
                        n = len(decoded_speculate.req_ids)
                        k = int(decoded_speculate.num_speculative_tokens)
                    else:
                        try:
                            if speculate_decoder is not None:
                                failed = speculate_decoder(frames)
                            else:
                                failed = DisaggDFlashSpeculateRequest.decode(frames)
                            n = len(failed.req_ids)
                            k = int(failed.num_speculative_tokens)
                        except Exception:
                            n, k = 1, 7
                    # Always ZMQ for draft tokens (IPC is hiddens-only).
                    resp = DisaggDFlashSpeculateResponse(
                        draft_tokens=torch.zeros(n, k, dtype=torch.int64)
                    ).encode()

                elif cmd == CMD_HELLO:
                    resp = encode_hello_reply(
                        {"transport": TRANSPORT_ZMQ, "error": "hello_failed"}
                    )
                else:
                    resp = encode_pong()
            self._sock.send_multipart([identity, *resp])

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self.stop()
        self._sock.close(linger=0)


# Backward-compatible aliases (connector.py re-exports these).
DisaggDFlashClient = ZmqClientTransport
