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
        isinstance(version, str) and re.match(r"^1\.5", version),
        f"Expected schema_version matching '1.5.*', got {version!r}.",
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
            _require(isinstance(msg, dict), f"Step {i}: 'message' must be an object.")
            _require("role" in msg,    f"Step {i}: message missing 'role'.")
            _require("content" in msg, f"Step {i}: message missing 'content'.")

        tool_calls = step.get("tool_calls")
        if tool_calls is not None:
            _require(
                isinstance(tool_calls, list),
                f"Step {i}: 'tool_calls' must be a list.",
            )
            for j, tc in enumerate(tool_calls):
                _require(isinstance(tc, dict), f"Step {i}, tool_call {j}: must be an object.")
                _require("id"   in tc, f"Step {i}, tool_call {j}: missing 'id'.")
                _require("name" in tc, f"Step {i}, tool_call {j}: missing 'name'.")

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
                        "source_call_id" in r,
                        f"Step {i}, result {k}: missing 'source_call_id'.",
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
      1. message      — one event for step.message
      2. reasoning    — if step.reasoning_content is present
      3. tool_calls   — one event per item in step.tool_calls[]
      4. tool_results — one event per item in step.observation.results[]
      5. metrics      — if step.metrics is present, as the final event
    """
    events: list[Event] = []

    for step_index, step in enumerate(doc["steps"]):
        step_id = int(step.get("step_id", step_index + 1))

        # ── 1. message ──────────────────────────────────────────────────────
        raw_message = step.get("message")
        step_ts: Optional[str] = None

        if raw_message is not None:
            step_ts = raw_message.get("timestamp")
            role = _canonical_role(raw_message.get("role", "user"))
            events.append(Event(
                kind="message",
                step_id=step_id,
                ts=step_ts,
                role=role,
                title=None,
                body=raw_message.get("content", ""),
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
        for tc in step.get("tool_calls") or []:
            events.append(Event(
                kind="tool_call",
                step_id=step_id,
                ts=tc.get("timestamp"),
                role="agent",
                title=tc["name"],
                body=tc.get("arguments"),
                ref=tc["id"],
            ))

        # ── 4. tool_results ──────────────────────────────────────────────────
        obs = step.get("observation") or {}
        for result in obs.get("results") or []:
            events.append(Event(
                kind="tool_result",
                step_id=step_id,
                ts=result.get("timestamp"),
                role="tool",
                title=None,
                body=result.get("content"),
                ref=result["source_call_id"],
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
