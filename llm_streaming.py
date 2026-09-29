"""Incremental SSE parsing and durable chat-completion stream capture."""

from __future__ import annotations

import codecs
import http.client
import json
from pathlib import Path
import queue
import select
import socket
import sys
import threading
import time
from typing import Any
import urllib.parse
import urllib.request


class StreamTransportError(RuntimeError):
    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = str(category)


class HTTPStreamResponse:
    """Small cancellable HTTP/1.1 body reader, including chunk decoding."""

    def __init__(
        self,
        connection,
        transport_socket,
        *,
        status,
        reason,
        headers,
        initial_body,
    ):
        self.connection = connection
        self.transport_socket = transport_socket
        self.status = int(status)
        self.reason = str(reason or "")
        self.headers = dict(headers)
        self._raw = bytearray(initial_body)
        self._decoded = bytearray()
        self._aborted = threading.Event()
        self._eof = False
        self._chunked = "chunked" in self.headers.get("transfer-encoding", "").lower()
        length = self.headers.get("content-length")
        self._remaining_length = None if length is None else int(length)
        self._chunk_remaining: int | None = None
        self._chunk_needs_crlf = False

    def _recv_more(self) -> None:
        while not self._aborted.is_set():
            readable, _, _ = select.select([self.transport_socket], [], [], 0.1)
            if not readable:
                continue
            data = self.transport_socket.recv(65_536)
            if data:
                self._raw.extend(data)
            else:
                self._eof = True
            return
        self._eof = True

    def _decode_chunked(self, wanted: int) -> None:
        while len(self._decoded) < wanted and not self._eof:
            if self._chunk_needs_crlf:
                if len(self._raw) < 2:
                    self._recv_more()
                    continue
                if self._raw[:2] != b"\r\n":
                    raise OSError("invalid HTTP chunk delimiter")
                del self._raw[:2]
                self._chunk_needs_crlf = False
                self._chunk_remaining = None
            if self._chunk_remaining is None:
                end = self._raw.find(b"\r\n")
                if end < 0:
                    self._recv_more()
                    continue
                line = bytes(self._raw[:end])
                del self._raw[: end + 2]
                try:
                    self._chunk_remaining = int(line.split(b";", 1)[0], 16)
                except ValueError as exc:
                    raise OSError(f"invalid HTTP chunk size {line!r}") from exc
                if self._chunk_remaining == 0:
                    self._eof = True
                    return
            if self._chunk_remaining:
                if not self._raw:
                    self._recv_more()
                    continue
                count = min(self._chunk_remaining, len(self._raw), wanted - len(self._decoded))
                self._decoded.extend(self._raw[:count])
                del self._raw[:count]
                self._chunk_remaining -= count
                if self._chunk_remaining == 0:
                    self._chunk_needs_crlf = True
                return

    def _decode_plain(self, wanted: int) -> None:
        while len(self._decoded) < wanted and not self._eof:
            if self._raw:
                count = min(len(self._raw), wanted - len(self._decoded))
                if self._remaining_length is not None:
                    count = min(count, self._remaining_length)
                self._decoded.extend(self._raw[:count])
                del self._raw[:count]
                if self._remaining_length is not None:
                    self._remaining_length -= count
                    if self._remaining_length == 0:
                        self._eof = True
                return
            self._recv_more()

    def read1(self, size):
        wanted = max(1, int(size))
        if self._chunked:
            self._decode_chunked(wanted)
        else:
            self._decode_plain(wanted)
        if not self._decoded:
            return b""
        result = bytes(self._decoded[:wanted])
        del self._decoded[:wanted]
        return result

    def abort(self):
        self._aborted.set()

    def close(self):
        self._aborted.set()
        try:
            self.transport_socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.connection.close()


def _remaining(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def open_http_stream(
    request: urllib.request.Request,
    *,
    connect_timeout: float,
    header_timeout: float,
    total_timeout: float,
) -> HTTPStreamResponse:
    """Open an HTTP response with separate TCP and post-connect deadlines.

    ``connect_timeout`` covers TCP/TLS establishment only.  Sending the request
    and waiting for response headers use ``header_timeout`` while the absolute
    ``total_timeout`` covers every phase.  Body timeouts are enforced by
    :func:`capture_chat_stream` after this function returns.
    """

    started = time.monotonic()
    deadline = started + float(total_timeout)
    parsed = urllib.parse.urlsplit(request.full_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise StreamTransportError(
            "connection_failure", f"unsupported streaming URL: {request.full_url!r}"
        )
    connection_class = (
        http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    )
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    connection = connection_class(
        parsed.hostname, port, timeout=float(connect_timeout)
    )
    phase = "connection"
    try:
        connect_limit = min(float(connect_timeout), _remaining(deadline))
        if connect_limit <= 0.0:
            raise StreamTransportError(
                "stream_total_timeout",
                f"request exceeded total timeout of {float(total_timeout):g} seconds before connection",
            )
        connection.timeout = connect_limit
        connection.connect()
        phase = "request_and_headers"
        transport_socket = connection.sock
        remaining = _remaining(deadline)
        if remaining <= 0.0:
            raise StreamTransportError(
                "stream_total_timeout",
                "request exceeded total timeout of "
                f"{float(total_timeout):g} seconds after connection",
            )
        header_limit = min(float(header_timeout), remaining)
        if transport_socket is not None:
            transport_socket.settimeout(header_limit)
        path = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        headers = {key: value for key, value in request.header_items()}
        connection.putrequest(request.get_method(), path)
        for key, value in headers.items():
            connection.putheader(key, value)
        if request.data is not None and not any(
            key.lower() == "content-length" for key in headers
        ):
            connection.putheader("Content-Length", str(len(request.data)))
        connection.endheaders(request.data)
        header_bytes = bytearray()
        header_deadline = min(deadline, time.monotonic() + float(header_timeout))
        while b"\r\n\r\n" not in header_bytes:
            remaining = _remaining(header_deadline)
            if remaining <= 0.0:
                category = (
                    "stream_total_timeout"
                    if _remaining(deadline) <= 1e-3
                    else "response_header_timeout"
                )
                raise StreamTransportError(
                    category,
                    "timed out waiting for response headers after "
                    f"{time.monotonic() - started:.3f} seconds",
                )
            readable, _, _ = select.select(
                [transport_socket], [], [], min(0.1, remaining)
            )
            if not readable:
                continue
            data = transport_socket.recv(65_536)
            if not data:
                raise StreamTransportError(
                    "connection_failure", "connection closed before response headers"
                )
            header_bytes.extend(data)
            if len(header_bytes) > 1_048_576:
                raise StreamTransportError(
                    "stream_protocol_error", "response headers exceed 1 MiB"
                )
        raw_headers, initial_body = bytes(header_bytes).split(b"\r\n\r\n", 1)
        lines = raw_headers.decode("iso-8859-1").split("\r\n")
        status_parts = lines[0].split(" ", 2)
        if len(status_parts) < 2 or not status_parts[1].isdigit():
            raise StreamTransportError(
                "stream_protocol_error", f"invalid HTTP status line: {lines[0]!r}"
            )
        status = int(status_parts[1])
        reason = status_parts[2] if len(status_parts) > 2 else ""
        response_headers: dict[str, str] = {}
        current_name = None
        for line in lines[1:]:
            if line[:1] in {" ", "\t"} and current_name is not None:
                response_headers[current_name] += " " + line.strip()
                continue
            name, separator, value = line.partition(":")
            if not separator:
                raise StreamTransportError(
                    "stream_protocol_error", f"invalid HTTP header line: {line!r}"
                )
            current_name = name.strip().lower()
            response_headers[current_name] = value.strip()
        transport_socket.settimeout(None)
        return HTTPStreamResponse(
            connection,
            transport_socket,
            status=status,
            reason=reason,
            headers=response_headers,
            initial_body=initial_body,
        )
    except (StreamTransportError, KeyboardInterrupt):
        if connection.sock is not None:
            try:
                connection.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        connection.close()
        raise
    except (TimeoutError, socket.timeout) as exc:
        elapsed = time.monotonic() - started
        category = (
            "stream_total_timeout"
            if elapsed >= float(total_timeout) - 1e-3
            else "response_header_timeout"
            if phase == "request_and_headers"
            else "connection_timeout"
        )
        if connection.sock is not None:
            try:
                connection.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        connection.close()
        description = (
            "sending the request or waiting for response headers"
            if category == "response_header_timeout"
            else "opening the connection"
        )
        raise StreamTransportError(
            category, f"timed out {description} after {elapsed:.3f} seconds"
        ) from exc
    except (OSError, http.client.HTTPException) as exc:
        if connection.sock is not None:
            try:
                connection.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        connection.close()
        raise StreamTransportError(
            "connection_failure",
            f"stream connection failed: {type(exc).__name__}: {exc}",
        ) from exc


class SSEDecoder:
    """Decode UTF-8 and SSE event boundaries independently of TCP chunks."""

    def __init__(self):
        self._decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self._text = ""
        self._data_lines: list[str] = []

    def _line(self, line: str, events: list[str]) -> None:
        if line == "":
            if self._data_lines:
                events.append("\n".join(self._data_lines))
                self._data_lines.clear()
            return
        if line.startswith(":"):
            return
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "data":
            self._data_lines.append(value)

    def feed(self, chunk: bytes, *, final: bool = False) -> list[str]:
        try:
            self._text += self._decoder.decode(chunk, final=final)
        except UnicodeDecodeError as exc:
            raise StreamTransportError(
                "stream_protocol_error", f"stream contains invalid UTF-8: {exc}"
            ) from exc
        events: list[str] = []
        while True:
            newline = self._text.find("\n")
            carriage = self._text.find("\r")
            candidates = [value for value in (newline, carriage) if value >= 0]
            if not candidates:
                break
            index = min(candidates)
            # A trailing CR may be the first half of CRLF in the next TCP chunk.
            if self._text[index] == "\r" and index + 1 == len(self._text) and not final:
                break
            line = self._text[:index]
            consumed = 1
            if self._text[index:index + 2] == "\r\n":
                consumed = 2
            self._text = self._text[index + consumed:]
            self._line(line, events)
        if final:
            if self._text:
                self._line(self._text, events)
                self._text = ""
            if self._data_lines:
                events.append("\n".join(self._data_lines))
                self._data_lines.clear()
        return events


def _write_status(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _call_bounded(function, timeout: float) -> bool:
    """Run best-effort transport cleanup without letting it own the deadline."""

    finished = threading.Event()

    def invoke() -> None:
        try:
            function()
        except BaseException:
            pass
        finally:
            finished.set()

    cleanup = threading.Thread(target=invoke, name="llm-stream-cleanup", daemon=True)
    cleanup.start()
    return finished.wait(max(0.0, float(timeout)))


def _abort_response(response, timeout: float) -> bool:
    abort = getattr(response, "abort", None)
    if callable(abort):
        return _call_bounded(abort, timeout)
    # Compatibility fallback for injected/test response objects.  Production
    # network responses always expose abort(), which performs socket shutdown.
    close = getattr(response, "close", None)
    return True if not callable(close) else _call_bounded(close, timeout)


def read_response_body_bounded(
    response,
    *,
    idle_timeout: float,
    total_timeout: float,
    maximum_bytes: int = 65_536,
    redact=None,
) -> tuple[bytes, dict[str, Any]]:
    """Read an HTTP error body with the same bounded cancellation semantics."""

    chunks: queue.Queue[tuple[str, Any]] = queue.Queue()
    started = time.monotonic()
    last_bytes = started
    body = bytearray()

    def reader() -> None:
        total_read = 0
        try:
            read = getattr(response, "read1", None) or response.read
            while total_read < int(maximum_bytes):
                chunk = read(min(4096, int(maximum_bytes) - total_read))
                if not chunk:
                    chunks.put(("eof", None))
                    return
                total_read += len(chunk)
                chunks.put(("bytes", chunk))
            chunks.put(("limit", None))
        except BaseException as exc:
            chunks.put(("error", exc))

    thread = threading.Thread(target=reader, name="llm-http-error-reader", daemon=True)
    thread.start()
    status = "complete"
    failure = None
    try:
        while True:
            now = time.monotonic()
            if now - started >= float(total_timeout):
                status = "stream_total_timeout"
                failure = (
                    "HTTP error body exceeded total timeout of "
                    f"{float(total_timeout):g} seconds"
                )
                break
            if now - last_bytes >= float(idle_timeout):
                status = "stream_idle_timeout"
                failure = f"HTTP error body was idle for {float(idle_timeout):g} seconds"
                break
            wait = min(
                0.1,
                max(0.01, float(total_timeout) - (now - started)),
                max(0.01, float(idle_timeout) - (now - last_bytes)),
            )
            try:
                kind, value = chunks.get(timeout=wait)
            except queue.Empty:
                continue
            if kind == "bytes":
                last_bytes = time.monotonic()
                body.extend(value)
            elif kind in {"eof", "limit"}:
                break
            else:
                status = "stream_connection_error"
                failure = f"HTTP error body read failed: {type(value).__name__}: {value}"
                if callable(redact):
                    failure = redact(failure)
                break
    except KeyboardInterrupt:
        status = "stream_cancelled"
        failure = "HTTP error body read cancelled by user"
    finally:
        _abort_response(response, 0.1)
        thread.join(0.5)
        if not thread.is_alive():
            _call_bounded(response.close, 0.1)
    return bytes(body), {
        "status": status,
        "failure": failure,
        "reader_terminated": not thread.is_alive(),
        "elapsed_seconds": time.monotonic() - started,
        "truncated": len(body) >= int(maximum_bytes),
    }


def capture_chat_stream(
    response,
    *,
    directory: str | Path,
    idle_timeout: float,
    total_timeout: float,
    progress_interval: float = 10.0,
    cancel_event: threading.Event | None = None,
    cleanup_timeout: float = 0.5,
    redact=None,
) -> dict[str, Any]:
    """Consume one HTTP SSE body while preserving every completed event."""

    def safe_text(value: Any) -> str:
        return redact(value) if callable(redact) else str(value)

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    event_path = directory / "stream_events.jsonl"
    content_path = directory / "response_content.partial.txt"
    reasoning_path = directory / "reasoning.partial.txt"
    status_path = directory / "stream_status.json"
    started = time.monotonic()
    last_bytes = started
    decoder = SSEDecoder()
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    refusal_parts: list[str] = []
    finish_reason = None
    actual_model = None
    usage = None
    done_received = False
    terminal_chunk_received = False
    tool_calls_seen = False
    tool_call_parts: dict[int, dict[str, Any]] = {}
    event_count = 0
    byte_count = 0
    first_output_elapsed = None
    final_status = "receiving"
    failure = None
    reader_terminated = False
    abort_completed = False
    cleanup_elapsed = 0.0

    chunks: queue.Queue[tuple[str, Any]] = queue.Queue()

    def reader() -> None:
        try:
            read = getattr(response, "read1", None) or response.read
            while True:
                chunk = read(4096)
                if not chunk:
                    chunks.put(("eof", None))
                    return
                chunks.put(("bytes", chunk))
        except BaseException as exc:  # delivered to and classified by the owner thread
            chunks.put(("error", exc))

    thread = threading.Thread(target=reader, name="llm-sse-reader", daemon=True)
    thread.start()
    next_progress = started + max(float(progress_interval), 0.1)

    def snapshot() -> dict[str, Any]:
        return {
            "status": final_status,
            "done_received": bool(done_received),
            "terminal_chunk_received": bool(terminal_chunk_received),
            "finish_reason": finish_reason,
            "event_count": int(event_count),
            "received_bytes": int(byte_count),
            "content_characters": sum(map(len, content_parts)),
            "reasoning_characters": sum(map(len, reasoning_parts)),
            "refusal_characters": sum(map(len, refusal_parts)),
            "first_output_elapsed_seconds": first_output_elapsed,
            "elapsed_seconds": time.monotonic() - started,
            "failure": failure,
            "reader_terminated": bool(reader_terminated),
            "abort_completed": bool(abort_completed),
            "cleanup_elapsed_seconds": float(cleanup_elapsed),
        }

    _write_status(status_path, snapshot())
    try:
        with event_path.open("a", encoding="utf-8", newline="\n") as event_file, content_path.open(
            "a", encoding="utf-8", newline=""
        ) as content_file, reasoning_path.open("a", encoding="utf-8", newline="") as reasoning_file:

            def handle_event(data: str) -> None:
                nonlocal event_count, done_received, finish_reason
                nonlocal terminal_chunk_received, actual_model, usage
                nonlocal tool_calls_seen, first_output_elapsed
                event_count += 1
                safe_data = redact(data) if callable(redact) else data
                record: dict[str, Any] = {
                    "sequence": event_count,
                    "elapsed_seconds": time.monotonic() - started,
                    "data": safe_data,
                }
                # Persist the complete SSE data event before interpreting it so
                # a malformed JSON event is still available after failure.
                event_file.write(
                    json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                )
                event_file.flush()
                if safe_data.strip() == "[DONE]":
                    done_received = True
                else:
                    try:
                        value = json.loads(safe_data)
                    except json.JSONDecodeError as exc:
                        raise StreamTransportError(
                            "stream_protocol_error",
                            f"SSE data event {event_count} is invalid JSON: {exc}",
                        ) from exc
                    if not isinstance(value, dict):
                        raise StreamTransportError(
                            "stream_protocol_error",
                            f"SSE data event {event_count} is not a JSON object",
                        )
                    if value.get("error") is not None:
                        raise StreamTransportError(
                            "stream_server_error",
                            f"chat completion stream error: {value['error']}",
                        )
                    if value.get("model") is not None:
                        actual_model = value["model"]
                    if value.get("usage") is not None:
                        usage = value["usage"]
                    choices = value.get("choices") or []
                    if not isinstance(choices, list):
                        raise StreamTransportError(
                            "stream_protocol_error", "stream choices is not a list"
                        )
                    for choice in choices:
                        if not isinstance(choice, dict):
                            raise StreamTransportError(
                                "stream_protocol_error", "stream choice is not an object"
                            )
                        delta = choice.get("delta") or {}
                        if not isinstance(delta, dict):
                            raise StreamTransportError(
                                "stream_protocol_error", "stream delta is not an object"
                            )
                        if delta.get("tool_calls") or delta.get("function_call"):
                            tool_calls_seen = True
                        for fragment in delta.get("tool_calls") or []:
                            if not isinstance(fragment, dict):
                                raise StreamTransportError(
                                    "stream_protocol_error",
                                    "stream tool-call delta is not an object",
                                )
                            index = int(fragment.get("index", 0))
                            accumulated = tool_call_parts.setdefault(
                                index,
                                {
                                    "index": index,
                                    "id": None,
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                },
                            )
                            if fragment.get("id") is not None:
                                accumulated["id"] = str(fragment["id"])
                            if fragment.get("type") is not None:
                                accumulated["type"] = str(fragment["type"])
                            function = fragment.get("function") or {}
                            if not isinstance(function, dict):
                                raise StreamTransportError(
                                    "stream_protocol_error",
                                    "stream tool-call function delta is not an object",
                                )
                            if function.get("name") is not None:
                                accumulated["function"]["name"] += str(
                                    function["name"]
                                )
                            if function.get("arguments") is not None:
                                accumulated["function"]["arguments"] += str(
                                    function["arguments"]
                                )
                        content = delta.get("content")
                        reasoning = delta.get(
                            "reasoning_content", delta.get("reasoning")
                        )
                        refusal = delta.get("refusal")
                        if content is not None:
                            if not isinstance(content, str):
                                raise StreamTransportError(
                                    "stream_protocol_error",
                                    "stream content delta is not text",
                                )
                            if content:
                                content_parts.append(content)
                                content_file.write(content)
                                content_file.flush()
                                first_output_elapsed = first_output_elapsed or (
                                    time.monotonic() - started
                                )
                        if refusal is not None:
                            if not isinstance(refusal, str):
                                raise StreamTransportError(
                                    "stream_protocol_error",
                                    "stream refusal delta is not text",
                                )
                            if refusal:
                                refusal_parts.append(refusal)
                                first_output_elapsed = first_output_elapsed or (
                                    time.monotonic() - started
                                )
                        if reasoning is not None:
                            if not isinstance(reasoning, str):
                                raise StreamTransportError(
                                    "stream_protocol_error",
                                    "stream reasoning delta is not text",
                                )
                            if reasoning:
                                reasoning_parts.append(reasoning)
                                reasoning_file.write(reasoning)
                                reasoning_file.flush()
                                first_output_elapsed = first_output_elapsed or (
                                    time.monotonic() - started
                                )
                        current_finish = choice.get("finish_reason")
                        if current_finish is not None:
                            finish_reason = current_finish
                            terminal_chunk_received = True
                _write_status(status_path, snapshot())

            while True:
                now = time.monotonic()
                if cancel_event is not None and cancel_event.is_set():
                    raise StreamTransportError(
                        "stream_cancelled", "stream cancelled by caller"
                    )
                if now - started >= float(total_timeout):
                    raise StreamTransportError(
                        "stream_total_timeout",
                        f"stream exceeded total timeout of {float(total_timeout):g} seconds",
                    )
                if now - last_bytes >= float(idle_timeout):
                    raise StreamTransportError(
                        "stream_idle_timeout",
                        f"stream was idle for {float(idle_timeout):g} seconds",
                    )
                if now >= next_progress:
                    print(
                        "LLM stream: "
                        f"waited={now - started:.0f}s, "
                        f"first_output={'yes' if first_output_elapsed is not None else 'no'}, "
                        f"content_chars={sum(map(len, content_parts))}, "
                        f"reasoning_chars={sum(map(len, reasoning_parts))}",
                        file=sys.stderr,
                        flush=True,
                    )
                    _write_status(status_path, snapshot())
                    next_progress = now + max(float(progress_interval), 0.1)
                wait = min(
                    0.05 if cancel_event is not None else 1.0,
                    max(0.01, float(total_timeout) - (now - started)),
                    max(0.01, float(idle_timeout) - (now - last_bytes)),
                )
                try:
                    kind, value = chunks.get(timeout=wait)
                except queue.Empty:
                    continue
                if kind == "bytes":
                    last_bytes = time.monotonic()
                    byte_count += len(value)
                    for event in decoder.feed(value):
                        handle_event(event)
                    if done_received:
                        break
                    continue
                if kind == "error":
                    raise StreamTransportError(
                        "stream_connection_error",
                        safe_text(
                            f"stream read failed: {type(value).__name__}: {value}"
                        ),
                    ) from value
                if kind == "eof":
                    for event in decoder.feed(b"", final=True):
                        handle_event(event)
                    break
            if not done_received:
                raise StreamTransportError(
                    "stream_incomplete", "stream ended without the [DONE] completion signal"
                )
            if not terminal_chunk_received:
                raise StreamTransportError(
                    "stream_incomplete", "stream ended without a terminal finish_reason"
                )
            final_status = "complete"
    except KeyboardInterrupt as exc:
        final_status = "cancelled"
        failure = "stream cancelled by user"
        raise StreamTransportError("stream_cancelled", failure) from exc
    except StreamTransportError as exc:
        final_status = exc.category
        failure = safe_text(exc)
        raise
    finally:
        cleanup_started = time.monotonic()
        abort_completed = _abort_response(response, min(float(cleanup_timeout), 0.1))
        thread.join(max(0.0, float(cleanup_timeout)))
        reader_terminated = not thread.is_alive()
        if reader_terminated:
            _call_bounded(response.close, min(float(cleanup_timeout), 0.1))
        cleanup_elapsed = time.monotonic() - cleanup_started
        _write_status(status_path, snapshot())

    return {
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
        "refusal": "".join(refusal_parts) or None,
        "finish_reason": finish_reason,
        "actual_model": actual_model,
        "usage": usage,
        "done_received": done_received,
        "terminal_chunk_received": terminal_chunk_received,
        "transport_completed": True,
        "tool_calls_seen": tool_calls_seen,
        "tool_calls": [tool_call_parts[index] for index in sorted(tool_call_parts)],
        "elapsed_seconds": time.monotonic() - started,
        "event_count": event_count,
        "received_bytes": byte_count,
        "reader_terminated": reader_terminated,
        "cleanup_elapsed_seconds": cleanup_elapsed,
    }
