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


def _visible_len(s: str) -> int:
    """Return the printable character length of a string, ignoring ANSI codes."""
    return len(_ANSI_RE.sub("", s))


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
    dashes = max(0, width - 3 - _visible_len(header))
    return f"╭─{header}" + "─" * dashes + "╮"


def _box_bottom(width: int) -> str:
    return "╰" + "─" * (width - 2) + "╯"


def _box_line(content: str) -> str:
    return f"│ {content}"


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


def _render_tool_result(body: Any, inner_width: int) -> list[str]:
    header = f" {_C.MAGENTA}◀ result:{_C.RESET}"
    lines = [_box_line(header)]
    content = body if isinstance(body, str) else json.dumps(body, indent=2)
    for line in _wrap_text(str(content), inner_width - 4):
        lines.append(_box_line(f"   {line}"))
    return lines


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
    inner = width - 4
    output = [_box_top(step_id, role, width)]
    first_rendered = True

    for ev in events:
        rendered: list[str] = []

        if ev.kind == "message":
            rendered = _render_message(str(ev.body or ""), inner)
        elif ev.kind == "reasoning" and show_reasoning:
            rendered = _render_reasoning(str(ev.body or ""), inner)
        elif ev.kind == "tool_call":
            rendered = _render_tool_call(ev.title or "?", ev.body, inner)
        elif ev.kind == "tool_result":
            rendered = _render_tool_result(ev.body, inner)
        elif ev.kind == "metrics":
            rendered = _render_metrics(
                ev.body if isinstance(ev.body, dict) else {}, inner
            )

        if not rendered:
            continue

        needs_gap = not first_rendered and ev.kind != "tool_result"
        if needs_gap:
            output.append(_box_line(""))
        output.extend(rendered)
        first_rendered = False

    output.append(_box_bottom(width))
    return output


# ---------------------------------------------------------------------------
# play command
# ---------------------------------------------------------------------------

def run_play(path: str, show_reasoning: bool, speed: float) -> None:
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
        for line in _render_step(step_id, step_events, show_reasoning, width):
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
        run_play(args.file, args.show_reasoning, args.speed)
    elif args.command == "serve":
        from browser_view import run_serve
        run_serve(args.file, args.port)


if __name__ == "__main__":
    main()
