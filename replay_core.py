# replay_core.py
"""
ATIF-v1.5 loader, validator, and normalizer.

This module is the sole layer that names ATIF schema fields.
All downstream code operates exclusively on Event objects.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


_ATTACHMENTS_BLOCK_RE = re.compile(r"<attachments>.*?</attachments>", re.DOTALL | re.IGNORECASE)
_ATTACHMENT_RE = re.compile(r"<attachment\b[^>]*>.*?</attachment>", re.DOTALL | re.IGNORECASE)
_USER_REQUEST_RE = re.compile(r"<userRequest>\s*(.*?)\s*</userRequest>", re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# Public data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    """
    A single, self-contained unit of replay data.

    kind    — one of: "message" | "reasoning" | "tool_call" | "tool_result" | "metrics"
    step_id — the originating step index (1-based by convention)
    ts      — ISO-8601 timestamp from the source document, or None
    role    — canonical: "system" | "user" | "agent" | "tool"
    title   — tool name (tool_call), or sub-kind label (reasoning / metrics)
    body    — str for message/reasoning/tool_result; dict for tool_call args / metrics
    ref     — tool_call_id (on tool_call) or source_call_id (on tool_result)
    """
    kind:    str
    step_id: int
    ts:      Optional[str]
    role:    str
    title:   Optional[str]
    body:    Any            # str | dict | None
    ref:     Optional[str]


@dataclass(frozen=True)
class ToolCall:
    """
    Parsed tool call with normalized field access.
    Supports dual naming conventions: id/tool_call_id, name/function_name.
    """
    call_id:   str
    name:      str
    arguments: dict[str, Any]
    timestamp: Optional[str]


@dataclass(frozen=True)
class ToolResult:
    """
    Parsed tool result with normalized field access.
    Supports dual field names: source_call_id/ref.
    """
    source_call_id: Optional[str]
    content:        Any
    timestamp:      Optional[str]


# ---------------------------------------------------------------------------
# Data model utilities
# ---------------------------------------------------------------------------

def parse_tool_call(raw: dict) -> ToolCall:
    """
    Extract a ToolCall from a raw dict with flexible field names.
    Supports: id/tool_call_id, name/function_name.
    """
    call_id = raw.get("id") or raw.get("tool_call_id") or "unknown"
    name = raw.get("name") or raw.get("function_name") or "unknown"
    arguments = _decode_args(raw.get("arguments", {}))
    timestamp = raw.get("timestamp")
    return ToolCall(call_id=call_id, name=name, arguments=arguments, timestamp=timestamp)


def parse_tool_result(raw: dict) -> ToolResult:
    """
    Extract a ToolResult from a raw dict with flexible field names.
    Supports: source_call_id/ref.
    """
    source_call_id = raw.get("source_call_id") or raw.get("ref")
    content = raw.get("content")
    timestamp = raw.get("timestamp")
    return ToolResult(source_call_id=source_call_id, content=content, timestamp=timestamp)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ATIFValidationError(ValueError):
    """Raised when the input document does not conform to ATIF-v1.5."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ATIFValidationError(message)


def _validate(doc: dict) -> None:
    """
    Validate the structure of a (possibly already-adapted) ATIF document.
    Raises ATIFValidationError on the first structural violation found.
    """
    _require(isinstance(doc, dict), "Root must be a JSON object.")

    version = doc.get("schema_version")
    _require(
        isinstance(version, str)
        and bool(re.match(r"^(?:ATIF-v)?1\.5(?:\..*)?$", version)),
        f"Expected schema_version matching '(ATIF-v)1.5.*', got {version!r}.",
    )
    _require("session_id" in doc, "Missing required field: session_id.")
    _require(
        "agent" in doc and isinstance(doc["agent"], dict),
        "Missing or invalid 'agent' object.",
    )
    _require(
        "steps" in doc and isinstance(doc["steps"], list),
        "Missing or invalid 'steps' list.",
    )

    for i, step in enumerate(doc["steps"]):
        _require(isinstance(step, dict), f"Step {i}: must be a JSON object.")

        msg = step.get("message")
        if msg is not None:
            _require(
                isinstance(msg, (dict, str)),
                f"Step {i}: 'message' must be an object or string.",
            )
            if isinstance(msg, dict):
                _require(
                    "content" in msg,
                    f"Step {i}: message object missing 'content'.",
                )
                _require(
                    "role" in msg or "source" in step,
                    f"Step {i}: message object missing 'role' and step missing 'source'.",
                )

        source = step.get("source")
        if source is not None:
            _require(isinstance(source, str), f"Step {i}: 'source' must be a string.")

        tool_calls = step.get("tool_calls")
        if tool_calls is not None:
            _require(
                isinstance(tool_calls, list),
                f"Step {i}: 'tool_calls' must be a list.",
            )
            for j, tc in enumerate(tool_calls):
                _require(isinstance(tc, dict), f"Step {i}, tool_call {j}: must be an object.")
                _require(
                    "id" in tc or "tool_call_id" in tc,
                    f"Step {i}, tool_call {j}: missing 'id'/'tool_call_id'.",
                )
                _require(
                    "name" in tc or "function_name" in tc,
                    f"Step {i}, tool_call {j}: missing 'name'/'function_name'.",
                )

        obs = step.get("observation")
        if obs is not None:
            _require(isinstance(obs, dict), f"Step {i}: 'observation' must be an object.")
            results = obs.get("results")
            if results is not None:
                _require(
                    isinstance(results, list),
                    f"Step {i}: observation.results must be a list.",
                )
                for k, r in enumerate(results):
                    _require(
                        isinstance(r, dict),
                        f"Step {i}, result {k}: must be an object.",
                    )
                    _require(
                        "source_call_id" in r or "ref" in r,
                        f"Step {i}, result {k}: missing 'source_call_id'/'ref'.",
                    )


# ---------------------------------------------------------------------------
# Adapter: non-canonical formats → ATIF-v1.5
# ---------------------------------------------------------------------------

def copilot_to_atif(raw: dict) -> dict:
    """
    Convert a non-canonical agent log to ATIF-v1.5, if necessary.

    Detection heuristic: a document that lacks 'steps' but contains a
    top-level 'messages' list is treated as an OpenAI / GitHub Copilot
    chat-completion log.  Any document that already has 'steps' is returned
    unchanged — this function must be idempotent on canonical input.
    """
    if "steps" in raw:
        return raw

    messages = raw.get("messages") or []
    steps: list[dict] = []

    _role_map = {"assistant": "agent", "user": "user", "system": "system"}

    for i, m in enumerate(messages):
        raw_role = m.get("role", "user")
        role = _role_map.get(raw_role, "user")

        step: dict = {
            "message": {
                "role": role,
                "content": m.get("content") or "",
            }
        }

        raw_tcs = m.get("tool_calls") or []
        if raw_tcs:
            step["tool_calls"] = [
                {
                    "id":        tc.get("id", f"tc_{i}_{j}"),
                    "name":      (tc.get("function") or {}).get("name", "unknown"),
                    "arguments": _decode_args(
                        (tc.get("function") or {}).get("arguments", "{}")
                    ),
                }
                for j, tc in enumerate(raw_tcs)
            ]

        steps.append(step)

    return {
        "schema_version": "1.5",
        "session_id":     raw.get("id", "converted"),
        "agent":          {"name": raw.get("model", "unknown")},
        "steps":          steps,
    }


def _decode_args(value: Any) -> Any:
    """Parse a JSON string into a dict; return the value unchanged otherwise."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _strip_attachment_content(value: Any) -> Any:
    """
    Remove embedded attachment payload markup from message text.

    This keeps the actual conversation prompt visible while dropping verbose
    `<attachments>...</attachments>` and standalone `<attachment>...</attachment>`
    blocks from rendered message bodies.
    """
    if not isinstance(value, str):
        return value

    text = value
    had_attachments = False

    text, attachments_block_count = _ATTACHMENTS_BLOCK_RE.subn("", text)
    had_attachments = had_attachments or attachments_block_count > 0

    text, attachment_count = _ATTACHMENT_RE.subn("", text)
    had_attachments = had_attachments or attachment_count > 0

    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if had_attachments and not text:
        return "[attachments omitted]"

    return text


def _extract_user_request_content(value: Any) -> Any:
    """
    If a message contains one or more <userRequest> blocks, keep only their
    inner text. Otherwise return the original value unchanged.
    """
    if not isinstance(value, str):
        return value

    matches = [match.strip() for match in _USER_REQUEST_RE.findall(value) if match.strip()]
    if not matches:
        return value

    return "\n\n".join(matches)


def _normalize_message_body(value: Any, role: str) -> Any:
    """
    Clean message content for rendering.

    - Remove bulky attachment payload sections for all roles.
    - For user messages that include wrapper metadata, keep only the content
      inside <userRequest>...</userRequest>.
    """
    body = _strip_attachment_content(value)
    if role == "user":
        body = _extract_user_request_content(body)
    return body


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load(path: str | Path) -> dict:
    """
    Read an ATIF file from disk, apply the non-canonical adapter if needed,
    and validate the result.

    Raises:
        json.JSONDecodeError      — if the file is not valid JSON.
        ATIFValidationError       — if the document fails schema validation.
        OSError / FileNotFoundError — if the file cannot be read.
    """
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    doc = copilot_to_atif(raw)
    _validate(doc)
    return doc


# ---------------------------------------------------------------------------
# Normalizer — sole consumer of ATIF schema field names
# ---------------------------------------------------------------------------

def normalize(doc: dict) -> list[Event]:
    """
    Convert a validated ATIF-v1.5 document into a flat, ordered Event list.

        Per-step emission order:
            1. message      — one event for step.message (object or string)
            2. reasoning    — if step.reasoning_content is present
            3. tool_calls   — one event per item in step.tool_calls[]
            4. tool_results — one event per item in step.observation.results[]
            5. metrics      — if step.metrics is present, as the final event

        Supports both of these step shapes:
            - canonical: message={role, content, timestamp}, tool_calls={id, name, arguments}
            - spec-adherent trajectory: source + message="...", tool_calls={tool_call_id, function_name, arguments}
    """
    events: list[Event] = []

    for step_index, step in enumerate(doc["steps"]):
        step_id = int(step.get("step_id", step_index + 1))

        # ── 1. message ──────────────────────────────────────────────────────
        raw_message = step.get("message")
        step_ts: Optional[str] = step.get("timestamp")

        if raw_message is not None:
            if isinstance(raw_message, dict):
                step_ts = raw_message.get("timestamp") or step_ts
                role = _canonical_role(raw_message.get("role") or step.get("source", "user"))
                body = _normalize_message_body(raw_message.get("content", ""), role)
            else:
                role = _canonical_role(str(step.get("source", "user")))
                body = _normalize_message_body(str(raw_message), role)

            events.append(Event(
                kind="message",
                step_id=step_id,
                ts=step_ts,
                role=role,
                title=None,
                body=body,
                ref=None,
            ))

        # ── 2. reasoning ─────────────────────────────────────────────────────
        reasoning = step.get("reasoning_content")
        if reasoning is not None:
            events.append(Event(
                kind="reasoning",
                step_id=step_id,
                ts=step_ts,
                role="agent",
                title="reasoning",
                body=reasoning,
                ref=None,
            ))

        # ── 3. tool_calls ────────────────────────────────────────────────────
        for raw_tc in step.get("tool_calls") or []:
            tc = parse_tool_call(raw_tc)
            events.append(Event(
                kind="tool_call",
                step_id=step_id,
                ts=tc.timestamp or step_ts,
                role="agent",
                title=tc.name,
                body=tc.arguments,
                ref=tc.call_id,
            ))

        # ── 4. tool_results ──────────────────────────────────────────────────
        obs = step.get("observation") or {}
        for raw_result in obs.get("results") or []:
            result = parse_tool_result(raw_result)
            events.append(Event(
                kind="tool_result",
                step_id=step_id,
                ts=result.timestamp or step_ts,
                role="tool",
                title=None,
                body=result.content,
                ref=result.source_call_id,
            ))

        # ── 5. metrics ───────────────────────────────────────────────────────
        metrics = step.get("metrics")
        if metrics is not None:
            events.append(Event(
                kind="metrics",
                step_id=step_id,
                ts=None,
                role="agent",
                title="metrics",
                body=metrics,
                ref=None,
            ))

    final_metrics = doc.get("final_metrics")
    if final_metrics is not None:
        final_step_id = (events[-1].step_id + 1) if events else 1
        events.append(Event(
            kind="metrics",
            step_id=final_step_id,
            ts=None,
            role="agent",
            title="final_metrics",
            body=final_metrics,
            ref=None,
        ))

    return events


def _canonical_role(raw: str) -> str:
    """Map any observed role string to one of the four canonical values."""
    table = {
        "user":      "user",
        "human":     "user",
        "assistant": "agent",
        "agent":     "agent",
        "system":    "system",
        "tool":      "tool",
        "function":  "tool",
    }
    return table.get(raw.lower(), "user")
