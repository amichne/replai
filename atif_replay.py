"""Terminal replay renderer and playback command."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from replay_core import Event, choose_trajectory_file, load, normalize


_MIN_WIDTH = 60
_MAX_WIDTH = 100

_USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")

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


class _C:
    """ANSI escape sequences, or empty strings when color is not supported."""

    RESET = "\033[0m" if _USE_COLOR else ""
    BOLD = "\033[1m" if _USE_COLOR else ""
    DIM = "\033[2m" if _USE_COLOR else ""
    CYAN = "\033[1;36m" if _USE_COLOR else ""
    YELLOW = "\033[1;33m" if _USE_COLOR else ""
    GREEN = "\033[1;32m" if _USE_COLOR else ""
    RED = "\033[1;31m" if _USE_COLOR else ""
    MAGENTA = "\033[1;35m" if _USE_COLOR else ""
    BLUE = "\033[34m" if _USE_COLOR else ""


_ROLE_STYLE = {
    "system": ("SYSTEM", _C.YELLOW),
    "user": ("USER", _C.CYAN),
    "agent": ("AGENT", _C.GREEN),
    "tool": ("TOOL", _C.MAGENTA),
}

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _terminal_width() -> int:
    try:
        cols = shutil.get_terminal_size().columns
    except Exception:
        cols = 80
    return min(_MAX_WIDTH, max(_MIN_WIDTH, cols - 2))


def _is_noisy_tool(name: Optional[str]) -> bool:
    tool = (name or "").strip().lower()
    return tool in _NOISY_TOOLS or tool.startswith("read_") or tool.startswith("grep_")


def _visible_len(s: str) -> int:
    return len(_ANSI_RE.sub("", s))


def _ellipsize(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _wrap_text(text: str, max_width: int) -> List[str]:
    result: List[str] = []
    for paragraph in str(text).splitlines():
        if not paragraph:
            result.append("")
            continue
        while len(paragraph) > max_width:
            result.append(paragraph[:max_width])
            paragraph = paragraph[max_width:]
        result.append(paragraph)
    return result or [""]


def _box_top(step_id: int, role: str, width: int) -> str:
    label, colour = _ROLE_STYLE.get(role, (role.upper(), ""))
    coloured = f"{colour}{label}{_C.RESET}"
    header = f" [{step_id}] {coloured} "
    dashes = max(0, width - 3 - _visible_len(header))
    return f"╭─{header}" + "─" * dashes + "╮"


def _box_bottom(width: int) -> str:
    return "╰" + ("─" * (width - 2)) + "╯"


def _box_line(content: str) -> str:
    return str(content)


def _format_box_line(content: str, inner_width: int) -> str:
    visible = _visible_len(str(content))
    pad = max(0, inner_width - visible)
    return f"│ {content}" + (" " * pad) + " │"


def _header_inner_width(step_id: int, role: str) -> int:
    label, _colour = _ROLE_STYLE.get(role, (role.upper(), ""))
    header = f" [{step_id}] {label} "
    return max(0, len(header) - 1)


def _render_message(body: str, inner_width: int) -> List[str]:
    return [_box_line(line) for line in _wrap_text(body, inner_width)]


def _render_reasoning(body: str, inner_width: int) -> List[str]:
    label = f" {_C.DIM}◆ reasoning{_C.RESET}"
    lines = [_box_line(label)]
    for line in _wrap_text(body, inner_width - 4):
        lines.append(_box_line(f" {_C.DIM}│ {line}{_C.RESET}"))
    return lines


def _render_tool_call(title: str, body: Any, inner_width: int) -> List[str]:
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

    label_ansi = f" {_C.BLUE}▶ tool:{_C.RESET} {_C.BOLD}{title}{_C.RESET}"
    label_plain = f" ▶ tool: {title}"
    single_visible = len(label_plain) + len(f"({compact_args})")

    if single_visible <= inner_width:
        return [_box_line(f"{label_ansi}({compact_args})")]

    formatted = json.dumps(body, indent=2) if isinstance(body, dict) else compact_args
    result = [_box_line(f"{label_ansi}(")]
    for line in formatted.splitlines():
        result.append(_box_line(f"   {line}"))
    result.append(_box_line(" )"))
    return result


def _render_apply_patch_diff(patch_text: str, inner_width: int) -> List[str]:
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


def _render_tool_call_minimal(title: str, body: Any, inner_width: int) -> List[str]:
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


def _render_tool_result(body: Any, inner_width: int) -> List[str]:
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


def _extract_run_task_field(block: str, key: str, *, next_key: Optional[str] = None, kind: str = "string") -> Any:
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


def _extract_run_task_payload(text: str) -> Optional[Dict[str, Any]]:
    start = text.rfind("{\n")
    if start == -1:
        return None

    block = text[start:]
    payload: Dict[str, Any] = {}

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

    for key, next_key in [("task", "exit_code"), ("log_file", "tasks_executed"), ("failure_summary", None)]:
        value = _extract_run_task_field(block, key, next_key=next_key)
        if value is not None:
            payload[key] = value

    return payload or None


def _is_gradle_run_task_invocation(tool_name: Optional[str], tool_body: Any) -> bool:
    if tool_name != "run_in_terminal" or not isinstance(tool_body, dict):
        return False

    command = str(tool_body.get("command", ""))
    return "scripts/gradle/run_task.sh" in command


def _trim_gradle_run_task_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
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

    trimmed: Dict[str, Any] = {}
    for key in preferred_order:
        if key in payload:
            trimmed[key] = payload[key]

    for key, value in payload.items():
        if key not in trimmed and key != "failure_summary":
            trimmed[key] = value

    return trimmed


def _render_gradle_run_task_result(body: Any, inner_width: int) -> List[str]:
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


def _render_tool_result_minimal(body: Any, inner_width: int, tool_name: Optional[str]) -> List[str]:
    text = str(body) if isinstance(body, str) else json.dumps(body, separators=(",", ":"))

    if tool_name == "read_file":
        summary = f"{len(text.splitlines())} lines suppressed"
    else:
        first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        summary = _ellipsize(first or "result suppressed", inner_width - 8)

    return [_box_line(f" {_C.MAGENTA}◀ result:{_C.RESET} {_C.DIM}{summary}{_C.RESET}")]


def _render_metrics(body: dict, inner_width: int) -> List[str]:
    parts: List[str] = []
    if "prompt_tokens" in body:
        parts.append(f"prompt={body['prompt_tokens']}")
    if "completion_tokens" in body:
        parts.append(f"completion={body['completion_tokens']}")
    if "cost_usd" in body:
        parts.append(f"cost=${body['cost_usd']:.4f}")
    for key, value in body.items():
        if key not in ("prompt_tokens", "completion_tokens", "cost_usd"):
            parts.append(f"{key}={value}")
    summary = "  ".join(parts)
    return [_box_line(f" {_C.DIM}⬡ metrics  {summary}{_C.RESET}")]


def _render_step(step_id: int, events: List[Event], show_reasoning: bool, tool_visibility: str, width: int) -> List[str]:
    if not events:
        return []

    role = next((ev.role for ev in events if ev.kind == "message"), "agent")
    call_name_by_ref = {ev.ref: (ev.title or "") for ev in events if ev.kind == "tool_call" and ev.ref}
    call_body_by_ref = {ev.ref: ev.body for ev in events if ev.kind == "tool_call" and ev.ref}
    inner_hint = max(1, width - 4)
    content_lines: List[str] = []
    first_rendered = True

    for ev in events:
        rendered: List[str] = []

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
            rendered = _render_metrics(ev.body if isinstance(ev.body, dict) else {}, inner_hint)

        if not rendered:
            continue

        needs_gap = not first_rendered and ev.kind != "tool_result"
        if needs_gap:
            content_lines.append(_box_line(""))
        content_lines.extend(rendered)
        first_rendered = False

    inner = max(_header_inner_width(step_id, role), max((_visible_len(line) for line in content_lines), default=0))
    box_width = inner + 4

    output = [_box_top(step_id, role, box_width)]
    output.extend(_format_box_line(line, inner) for line in content_lines)
    output.append(_box_bottom(box_width))
    return output


def run_play(path: Optional[str], show_reasoning: bool, speed: float, tool_visibility: str) -> None:
    selected_path = Path(path).expanduser() if path else choose_trajectory_file(Path.cwd())
    doc = load(selected_path)
    events = normalize(doc)
    width = _terminal_width()

    steps: Dict[int, List[Event]] = {}
    for ev in events:
        steps.setdefault(ev.step_id, []).append(ev)

    is_tty = sys.stdout.isatty()
    delay = (0.35 / max(speed, 0.01)) if is_tty else 0.0

    for step_id, step_events in steps.items():
        for line in _render_step(step_id, step_events, show_reasoning, tool_visibility, width):
            print(line)
        print()
        if delay:
            time.sleep(delay)


def _build_play_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render an ATIF replay to the terminal.")
    parser.add_argument("file", nargs="?", help="Path to an ATIF-v1.5 JSON file.")
    parser.add_argument("--show-reasoning", action="store_true", help="Render reasoning_content blocks.")
    parser.add_argument("--speed", type=float, default=1.0, metavar="FACTOR", help="Playback speed multiplier.")
    parser.add_argument(
        "--tool-visibility",
        choices=[TOOL_VISIBILITY_FULL, TOOL_VISIBILITY_MINIMAL, TOOL_VISIBILITY_HIDDEN],
        default=TOOL_VISIBILITY_MINIMAL,
        help="Tool event verbosity: full, minimal, or hidden for noisy tools.",
    )
    return parser


def main() -> None:
    args = _build_play_parser().parse_args()
    run_play(args.file, args.show_reasoning, args.speed, args.tool_visibility)


__all__ = [
    "TOOL_VISIBILITY_FULL",
    "TOOL_VISIBILITY_HIDDEN",
    "TOOL_VISIBILITY_MINIMAL",
    "run_play",
]


if __name__ == "__main__":
    main()
