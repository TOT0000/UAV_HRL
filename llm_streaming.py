"""Incremental SSE parsing and durable LM Studio stream capture."""

from __future__ import annotations

import codecs
import json
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any


class StreamTransportError(RuntimeError):
    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = str(category)


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


def capture_chat_stream(
    response,
    *,
    directory: str | Path,
    idle_timeout: float,
    total_timeout: float,
    progress_interval: float = 10.0,
) -> dict[str, Any]:
    """Consume one HTTP SSE body while preserving every completed event."""

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
    finish_reason = None
    actual_model = None
    usage = None
    done_received = False
    terminal_chunk_received = False
    tool_calls_seen = False
    event_count = 0
    byte_count = 0
    first_output_elapsed = None
    final_status = "receiving"
    failure = None

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
            "first_output_elapsed_seconds": first_output_elapsed,
            "elapsed_seconds": time.monotonic() - started,
            "failure": failure,
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
                record: dict[str, Any] = {
                    "sequence": event_count,
                    "elapsed_seconds": time.monotonic() - started,
                    "data": data,
                }
                # Persist the complete SSE data event before interpreting it so
                # a malformed JSON event is still available after failure.
                event_file.write(
                    json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                )
                event_file.flush()
                if data.strip() == "[DONE]":
                    done_received = True
                else:
                    try:
                        value = json.loads(data)
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
                            f"LM Studio stream error: {value['error']}",
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
                        content = delta.get("content")
                        reasoning = delta.get(
                            "reasoning_content", delta.get("reasoning")
                        )
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
                    1.0,
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
                        f"stream read failed: {type(value).__name__}: {value}",
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
        failure = str(exc)
        raise
    finally:
        try:
            response.close()
        except BaseException:
            pass
        _write_status(status_path, snapshot())

    return {
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
        "finish_reason": finish_reason,
        "actual_model": actual_model,
        "usage": usage,
        "done_received": done_received,
        "terminal_chunk_received": terminal_chunk_received,
        "transport_completed": True,
        "tool_calls_seen": tool_calls_seen,
        "elapsed_seconds": time.monotonic() - started,
        "event_count": event_count,
        "received_bytes": byte_count,
    }
