# atif_replay.py
"""
ATIF-v1.5 local session replayer — CLI entry point and terminal renderer.

Usage:
    python atif_replay.py play trajectory.json
    python atif_replay.py play trajectory.json --show-reasoning --speed 2.0
    python atif_replay.py serve trajectory.json --port 8000
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from typing import Any

from replay_core import ATIFValidationError, Event, load, normalize


# ---------------------------------------------------------------------------
# Terminal geometry
# ---------------------------------------------------------------------------

_MIN_WIDTH = 60
_MAX_WIDTH = 100


def _terminal_width() -> int:
    try:
        cols = shutil.get_terminal_size().columns
    except Exception:
        cols = 80
    return min(_MAX_WIDTH, max(_MIN_WIDTH, cols - 2))


# ---------------------------------------------------------------------------
# ANSI color support
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


class _C:
    """ANSI escape sequences, or empty strings when color is not supported."""
    RESET   = "\033[0m"     if _USE_COLOR else ""
    BOLD    = "\033[1m"     if _USE_COLOR else ""
    DIM     = "\033[2m"     if _USE_COLOR else ""
    CYAN    = "\033[1;36m"  if _USE_COLOR else ""
    YELLOW  = "\033[1;33m"  if _USE_COLOR else ""
    GREEN   = "\033[1;32m"  if _USE_COLOR else ""
    RED     = "\033[1;31m"  if _USE_COLOR else ""
    MAGENTA = "\033[1;35m"  if _USE_COLOR else ""
    BLUE    = "\033[34m"    if _USE_COLOR else ""


# role → (display label, colour prefix)
_ROLE_STYLE: dict[str, tuple[str, str]] = {
    "system": ("SYSTEM", _C.YELLOW),
    "user":   ("USER",   _C.CYAN),
    "agent":  ("AGENT",  _C.GREEN),
    "tool":   ("TOOL",   _C.MAGENTA),
}

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


TOOL_VISIBILITY_FULL = "full"
TOOL_VISIBILITY_MINIMAL = "minimal"
TOOL_VISIBILITY_HIDDEN = "hidden"

_NOISY_TOOLS = {
    "read_file",
    "readfile",
    "grep_code",
    "grep_search",
    "file_search",
    "list_dir",
    "get_errors",
    "manage_todo_list",
    "get_changed_files",
}


def _is_noisy_tool(name: str | None) -> bool:
    tool = (name or "").strip().lower()
    return (
        tool in _NOISY_TOOLS
        or tool.startswith("read_")
        or tool.startswith("grep_")
    )


def _visible_len(s: str) -> int:
    """Return the printable character length of a string, ignoring ANSI codes."""
    return len(_ANSI_RE.sub("", s))


def _ellipsize(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


# ---------------------------------------------------------------------------
# Text wrapping (ANSI-free content only)
# ---------------------------------------------------------------------------

def _wrap_text(text: str, max_width: int) -> list[str]:
    """
    Wrap plain text to max_width characters per line, preserving existing
    newlines.  Always returns at least one element.
    """
    result: list[str] = []
    for paragraph in str(text).splitlines():
        if not paragraph:
            result.append("")
            continue
        while len(paragraph) > max_width:
            result.append(paragraph[:max_width])
            paragraph = paragraph[max_width:]
        result.append(paragraph)
    return result or [""]


# ---------------------------------------------------------------------------
# Box drawing primitives
# ---------------------------------------------------------------------------

def _box_top(step_id: int, role: str, width: int) -> str:
    label, colour = _ROLE_STYLE.get(role, (role.upper(), ""))
    coloured = f"{colour}{label}{_C.RESET}"
    header = f" [{step_id}] {coloured} "
    # total visible length should equal `width`:
    # 2 chars for the left corner and dash (╭─) + visible_len(header) + dashes + 1 char for the right corner (╮)
    # => dashes = width - 3 - visible_len(header)
    dashes = max(0, width - 3 - _visible_len(header))
    return f"╭─{header}" + "─" * dashes + "╮"


def _box_bottom(width: int) -> str:
    return "╰" + ("─" * (width - 2)) + "╯"


def _box_line(content: str) -> str:
    """Return raw line content; box framing is applied in a later pass."""
    return str(content)


def _format_box_line(content: str, inner_width: int) -> str:
    """Render a single boxed line padded to the computed step width."""
    visible = _visible_len(str(content))
    pad = max(0, inner_width - visible)
    return f"│ {content}" + (" " * pad) + " │"


def _header_inner_width(step_id: int, role: str) -> int:
    """Minimum inner width required so the header fits inside the frame."""
    label, _colour = _ROLE_STYLE.get(role, (role.upper(), ""))
    header = f" [{step_id}] {label} "
    return max(0, len(header) - 1)


# ---------------------------------------------------------------------------
# Per-kind event renderers
# ---------------------------------------------------------------------------

def _render_message(body: str, inner_width: int) -> list[str]:
    return [_box_line(line) for line in _wrap_text(body, inner_width)]


def _render_reasoning(body: str, inner_width: int) -> list[str]:
    label = f" {_C.DIM}◆ reasoning{_C.RESET}"
    lines = [_box_line(label)]
    for line in _wrap_text(body, inner_width - 4):
        lines.append(_box_line(f" {_C.DIM}│ {line}{_C.RESET}"))
    return lines


def _render_tool_call(title: str, body: Any, inner_width: int) -> list[str]:
    if isinstance(body, dict) and title == "apply_patch":
        patch_input = body.get("input")
        if isinstance(patch_input, str):
            return _render_apply_patch_diff(patch_input, inner_width)

    if isinstance(body, dict):
        compact_args = json.dumps(body, separators=(",", ":"))
    elif body is None:
        compact_args = ""
    else:
        compact_args = str(body)

    label_ansi  = f" {_C.BLUE}▶ tool:{_C.RESET} {_C.BOLD}{title}{_C.RESET}"
    label_plain = f" ▶ tool: {title}"
    single_visible = len(label_plain) + len(f"({compact_args})")

    if single_visible <= inner_width:
        return [_box_line(f"{label_ansi}({compact_args})")]

    # Fall back to multi-line for long argument objects.
    formatted = json.dumps(body, indent=2) if isinstance(body, dict) else compact_args
    result = [_box_line(f"{label_ansi}(")]
    for line in formatted.splitlines():
        result.append(_box_line(f"   {line}"))
    result.append(_box_line(" )"))
    return result


def _render_apply_patch_diff(patch_text: str, inner_width: int) -> list[str]:
    lines = [_box_line(f" {_C.BLUE}▶ tool:{_C.RESET} {_C.BOLD}apply_patch{_C.RESET}")]
    lines.append(_box_line(f" {_C.DIM}↳ diff preview{_C.RESET}"))

    raw_lines = patch_text.splitlines()
    max_lines = 140
    shown = raw_lines[:max_lines]

    for raw in shown:
        color = ""
        if raw.startswith("+++"):
            color = _C.GREEN
        elif raw.startswith("---"):
            color = _C.RED
        elif raw.startswith("*** Begin Patch") or raw.startswith("*** End Patch"):
            color = _C.BLUE
        elif raw.startswith("*** Update File:") or raw.startswith("*** Add File:") or raw.startswith("*** Delete File:"):
            color = _C.BLUE
        elif raw.startswith("@@"):
            color = _C.YELLOW
        elif raw.startswith("+"):
            color = _C.GREEN
        elif raw.startswith("-"):
            color = _C.RED
        elif raw.startswith(" "):
            color = _C.DIM

        for segment in _wrap_text(raw, inner_width - 3):
            lines.append(_box_line(f"   {color}{segment}{_C.RESET}"))

    if len(raw_lines) > max_lines:
        hidden = len(raw_lines) - max_lines
        lines.append(_box_line(f"   {_C.DIM}… {hidden} more patch lines hidden{_C.RESET}"))

    return lines


def _render_tool_call_minimal(title: str, body: Any, inner_width: int) -> list[str]:
    summary = ""
    if title in {"read_file", "readfile"} and isinstance(body, dict):
        path = body.get("filePath") or body.get("path") or "?"
        start = body.get("startLine")
        end = body.get("endLine")
        summary = f"{path}:{start}-{end}" if start is not None and end is not None else str(path)
    elif title in {"grep_search", "grep_code"} and isinstance(body, dict):
        query = str(body.get("query", ""))
        summary = f"query={query!r}"
    elif title == "file_search" and isinstance(body, dict):
        summary = f"query={body.get('query', '')!r}"
    elif title == "list_dir" and isinstance(body, dict):
        summary = f"path={body.get('path', '')}"
    elif title == "get_errors" and isinstance(body, dict):
        fps = body.get("filePaths")
        summary = f"files={len(fps)}" if isinstance(fps, list) else "workspace"
    elif title == "manage_todo_list" and isinstance(body, dict):
        todos = body.get("todoList")
        summary = f"items={len(todos)}" if isinstance(todos, list) else "updated"
    else:
        if isinstance(body, dict):
            summary = _ellipsize(json.dumps(body, separators=(",", ":")), inner_width // 2)
        else:
            summary = _ellipsize(str(body), inner_width // 2)

    label = f" {_C.BLUE}▶ tool:{_C.RESET} {_C.BOLD}{title}{_C.RESET} {_C.DIM}{_ellipsize(summary, inner_width - 16)}{_C.RESET}"
    return [_box_line(label)]


def _render_tool_result(body: Any, inner_width: int) -> list[str]:
    header = f" {_C.MAGENTA}◀ result:{_C.RESET}"
    lines = [_box_line(header)]
    content = body if isinstance(body, str) else json.dumps(body, indent=2)
    for line in _wrap_text(str(content), inner_width - 4):
        lines.append(_box_line(f"   {line}"))
    return lines


def _collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _remove_all_whitespace(text: str) -> str:
    return "".join(text.split())


def _extract_run_task_field(
    block: str,
    key: str,
    *,
    next_key: str | None = None,
    kind: str = "string",
) -> Any:
    if kind == "bool":
        match = re.search(rf'"{re.escape(key)}":\s*(true|false)', block)
        return None if match is None else (match.group(1) == "true")

    if kind == "int":
        match = re.search(rf'"{re.escape(key)}":\s*(-?\d+)', block)
        return None if match is None else int(match.group(1))

    if next_key is None:
        pattern = rf'"{re.escape(key)}":\s*"(.*)"\s*}}\s*$'
    else:
        pattern = rf'"{re.escape(key)}":\s*"(.*?)",\s*"{re.escape(next_key)}":'

    match = re.search(pattern, block, flags=re.DOTALL)
    if match is None:
        return None

    value = match.group(1)
    if key == "log_file":
        return _remove_all_whitespace(value)
    return _collapse_spaces(value)


def _extract_run_task_payload(text: str) -> dict[str, Any] | None:
    start = text.rfind("{\n")
    if start == -1:
        return None

    block = text[start:]
    payload: dict[str, Any] = {}

    for key, kind in [
        ("ok", "bool"),
        ("exit_code", "int"),
        ("duration_ms", "int"),
        ("tasks_executed", "int"),
        ("tasks_up_to_date", "int"),
        ("tasks_from_cache", "int"),
        ("build_successful", "bool"),
        ("test_task_detected", "bool"),
    ]:
        value = _extract_run_task_field(block, key, kind=kind)
        if value is not None:
            payload[key] = value

    string_fields = [
        ("task", "exit_code"),
        ("log_file", "tasks_executed"),
        ("failure_summary", None),
    ]
    for key, next_key in string_fields:
        value = _extract_run_task_field(block, key, next_key=next_key)
        if value is not None:
            payload[key] = value

    return payload or None


def _is_gradle_run_task_invocation(tool_name: str | None, tool_body: Any) -> bool:
    if tool_name != "run_in_terminal" or not isinstance(tool_body, dict):
        return False

    command = str(tool_body.get("command", ""))
    return "scripts/gradle/run_task.sh" in command


def _trim_gradle_run_task_payload(payload: dict[str, Any]) -> dict[str, Any]:
    preferred_order = [
        "ok",
        "task",
        "exit_code",
        "duration_ms",
        "log_file",
        "tasks_executed",
        "tasks_up_to_date",
        "tasks_from_cache",
        "build_successful",
        "test_task_detected",
    ]

    trimmed: dict[str, Any] = {}
    for key in preferred_order:
        if key in payload:
            trimmed[key] = payload[key]

    for key, value in payload.items():
        if key not in trimmed and key != "failure_summary":
            trimmed[key] = value

    return trimmed


def _render_gradle_run_task_result(body: Any, inner_width: int) -> list[str]:
    if not isinstance(body, str):
        return _render_tool_result(body, inner_width)

    payload = _extract_run_task_payload(body)
    if payload is None:
        return _render_tool_result(body, inner_width)

    header = f" {_C.MAGENTA}◀ result:{_C.RESET}"
    lines = [_box_line(header)]

    failure_summary = payload.get("failure_summary")
    if isinstance(failure_summary, str) and failure_summary.strip():
        for line in _wrap_text(failure_summary.strip(), inner_width - 4):
            lines.append(_box_line(f"   {_C.RED}{line}{_C.RESET}"))

    trimmed_payload = _trim_gradle_run_task_payload(payload)
    for line in json.dumps(trimmed_payload, indent=2).splitlines():
        for segment in _wrap_text(line, inner_width - 4):
            lines.append(_box_line(f"   {segment}"))

    return lines


def _render_tool_result_minimal(body: Any, inner_width: int, tool_name: str | None) -> list[str]:
    text = str(body) if isinstance(body, str) else json.dumps(body, separators=(",", ":"))

    if tool_name == "read_file":
        summary = f"{len(text.splitlines())} lines suppressed"
    else:
        first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        summary = _ellipsize(first or "result suppressed", inner_width - 8)

    return [_box_line(f" {_C.MAGENTA}◀ result:{_C.RESET} {_C.DIM}{summary}{_C.RESET}")]


def _render_metrics(body: dict, inner_width: int) -> list[str]:
    parts: list[str] = []
    if "prompt_tokens"     in body: parts.append(f"prompt={body['prompt_tokens']}")
    if "completion_tokens" in body: parts.append(f"completion={body['completion_tokens']}")
    if "cost_usd"          in body: parts.append(f"cost=${body['cost_usd']:.4f}")
    for k, v in body.items():
        if k not in ("prompt_tokens", "completion_tokens", "cost_usd"):
            parts.append(f"{k}={v}")
    summary = "  ".join(parts)
    return [_box_line(f" {_C.DIM}⬡ metrics  {summary}{_C.RESET}")]


# ---------------------------------------------------------------------------
# Step renderer — groups all events for one step into a single box
# ---------------------------------------------------------------------------

def _render_step(
    step_id: int,
    events: list[Event],
    show_reasoning: bool,
    tool_visibility: str,
    width: int,
) -> list[str]:
    """
    Render a list of same-step events into one annotated box.

    Gap rules:
      - No blank line before the first rendered item.
      - No blank line before tool_result (it reads as a direct continuation
        of the preceding tool_call).
      - A blank separator line before every other non-first item.
    """
    if not events:
        return []

    role = next((ev.role for ev in events if ev.kind == "message"), "agent")
    call_name_by_ref = {
        ev.ref: (ev.title or "")
        for ev in events
        if ev.kind == "tool_call" and ev.ref
    }
    call_body_by_ref = {
        ev.ref: ev.body
        for ev in events
        if ev.kind == "tool_call" and ev.ref
    }
    inner_hint = max(1, width - 4)
    content_lines: list[str] = []
    first_rendered = True

    for ev in events:
        rendered: list[str] = []

        if ev.kind == "message":
            rendered = _render_message(str(ev.body or ""), inner_hint)
        elif ev.kind == "reasoning" and show_reasoning:
            rendered = _render_reasoning(str(ev.body or ""), inner_hint)
        elif ev.kind == "tool_call":
            tool_name = ev.title or "?"
            is_noisy = _is_noisy_tool(tool_name)

            if tool_visibility == TOOL_VISIBILITY_HIDDEN and is_noisy:
                rendered = []
            elif tool_visibility == TOOL_VISIBILITY_MINIMAL and is_noisy:
                rendered = _render_tool_call_minimal(tool_name, ev.body, inner_hint)
            else:
                rendered = _render_tool_call(tool_name, ev.body, inner_hint)
        elif ev.kind == "tool_result":
            origin_tool = call_name_by_ref.get(ev.ref) if ev.ref else None
            origin_tool_body = call_body_by_ref.get(ev.ref) if ev.ref else None
            is_noisy = _is_noisy_tool(origin_tool)

            if _is_gradle_run_task_invocation(origin_tool, origin_tool_body):
                rendered = _render_gradle_run_task_result(ev.body, inner_hint)
            elif tool_visibility == TOOL_VISIBILITY_HIDDEN and is_noisy:
                rendered = []
            elif tool_visibility == TOOL_VISIBILITY_MINIMAL and is_noisy:
                rendered = _render_tool_result_minimal(ev.body, inner_hint, origin_tool)
            else:
                rendered = _render_tool_result(ev.body, inner_hint)
        elif ev.kind == "metrics":
            rendered = _render_metrics(
                ev.body if isinstance(ev.body, dict) else {}, inner_hint
            )

        if not rendered:
            continue

        needs_gap = not first_rendered and ev.kind != "tool_result"
        if needs_gap:
            content_lines.append(_box_line(""))
        content_lines.extend(rendered)
        first_rendered = False

    inner = max(
        _header_inner_width(step_id, role),
        max((_visible_len(line) for line in content_lines), default=0),
    )
    box_width = inner + 4

    output = [_box_top(step_id, role, box_width)]
    output.extend(_format_box_line(line, inner) for line in content_lines)
    output.append(_box_bottom(box_width))
    return output


# ---------------------------------------------------------------------------
# play command
# ---------------------------------------------------------------------------

def run_play(path: str, show_reasoning: bool, speed: float, tool_visibility: str) -> None:
    try:
        doc = load(path)
    except (ATIFValidationError, json.JSONDecodeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    events = normalize(doc)
    width  = _terminal_width()

    # Group events by step_id, preserving insertion order.
    steps: dict[int, list[Event]] = {}
    for ev in events:
        steps.setdefault(ev.step_id, []).append(ev)

    # Delay between steps is only applied when writing to a real terminal.
    is_tty = sys.stdout.isatty()
    delay  = (0.35 / max(speed, 0.01)) if is_tty else 0.0

    for step_id, step_events in steps.items():
        for line in _render_step(step_id, step_events, show_reasoning, tool_visibility, width):
            print(line)
        print()
        if delay:
            time.sleep(delay)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atif_replay",
        description="ATIF-v1.5 local session replayer.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    play = sub.add_parser("play", help="Render a session to the terminal.")
    play.add_argument("file", help="Path to an ATIF-v1.5 JSON file.")
    play.add_argument(
        "--show-reasoning",
        action="store_true",
        help="Render reasoning_content blocks (hidden by default).",
    )
    play.add_argument(
        "--speed",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help="Playback speed multiplier (default: 1.0).  2.0 = twice as fast.",
    )
    play.add_argument(
        "--tool-visibility",
        choices=[TOOL_VISIBILITY_FULL, TOOL_VISIBILITY_MINIMAL, TOOL_VISIBILITY_HIDDEN],
        default=TOOL_VISIBILITY_MINIMAL,
        help="Tool event verbosity: full, minimal (default), or hidden for noisy tools.",
    )

    serve = sub.add_parser("serve", help="Launch a local browser viewer.")
    serve.add_argument("file", help="Path to an ATIF-v1.5 JSON file.")
    serve.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTTP listen port (default: 8000).",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _build_parser().parse_args()

    if args.command == "play":
        run_play(args.file, args.show_reasoning, args.speed, args.tool_visibility)
    elif args.command == "serve":
        from browser_view import run_serve
        run_serve(args.file, args.port)


if __name__ == "__main__":
    main()
