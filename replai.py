"""CLI entrypoint and compatibility exports for the replay app."""

from __future__ import annotations

import argparse

from atif_replay import (
    TOOL_VISIBILITY_FULL,
    TOOL_VISIBILITY_HIDDEN,
    TOOL_VISIBILITY_MINIMAL,
    run_play,
)
from browser_view import build_html, run_html, run_serve
from replay_core import (
    ATIFValidationError,
    Event,
    ToolCall,
    ToolResult,
    choose_trajectory_file,
    copilot_to_atif,
    discover_trajectory_files,
    load,
    normalize,
    parse_tool_call,
    parse_tool_result,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="replai",
        description="Portable ATIF-v1.5 local session replayer.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    _no_latest_help = (
        "Skip auto-discovery of the latest Copilot session and require an explicit path "
        "or interactive selection."
    )

    play = sub.add_parser("play", help="Render a session to the terminal.")
    play.add_argument("file", nargs="?", help="Path to an ATIF-v1.5 JSON file, a Copilot `main.jsonl`, or a session directory.")
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
        help="Playback speed multiplier (default: 1.0). 2.0 = twice as fast.",
    )
    play.add_argument(
        "--tool-visibility",
        choices=[TOOL_VISIBILITY_FULL, TOOL_VISIBILITY_MINIMAL, TOOL_VISIBILITY_HIDDEN],
        default=TOOL_VISIBILITY_MINIMAL,
        help="Tool event verbosity: full, minimal (default), or hidden for noisy tools.",
    )
    play.add_argument("--no-latest", action="store_true", help=_no_latest_help)

    serve = sub.add_parser("serve", help="Launch a browser viewer (direct HTML mode by default).")
    serve.add_argument("file", nargs="?", help="Path to an ATIF-v1.5 JSON file, a Copilot `main.jsonl`, or a session directory.")
    serve.add_argument("--output", help="Optional HTML output path. If omitted, a temporary file is created.")
    serve.add_argument(
        "--http",
        action="store_true",
        help="Use the legacy localhost HTTP server instead of opening the static HTML file directly.",
    )
    serve.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTTP listen port when --http is used (default: 8000).",
    )
    serve.add_argument("--no-latest", action="store_true", help=_no_latest_help)

    html = sub.add_parser("html", help="Write the static HTML viewer or print it to stdout.")
    html.add_argument("file", nargs="?", help="Path to an ATIF-v1.5 JSON file, a Copilot `main.jsonl`, or a session directory.")
    html.add_argument("--output", help="Write the HTML viewer to this path instead of stdout.")
    html.add_argument("--open", action="store_true", help="Open the generated HTML file in the default browser.")
    html.add_argument("--no-latest", action="store_true", help=_no_latest_help)

    return parser


def _resolve_file(raw: str | None, no_latest: bool) -> str:
    """Normalize the positional file arg to an absolute path string.

    If *raw* is provided it is resolved via :func:`choose_trajectory_file` so
    session directories are automatically expanded to ``main.jsonl``.  When
    *raw* is omitted the function auto-discovers the latest Copilot session
    unless *no_latest* is True, in which case it falls back to an interactive
    ATIF-file selection.
    """
    root = raw if raw is not None else "."
    return str(choose_trajectory_file(root, use_latest=not no_latest))


def main() -> None:
    args = _build_parser().parse_args()
    no_latest: bool = getattr(args, "no_latest", False)
    resolved = _resolve_file(args.file, no_latest)

    if args.command == "play":
        run_play(resolved, args.show_reasoning, args.speed, args.tool_visibility)
    elif args.command == "serve":
        run_serve(resolved, args.port, args.output, args.http)
    elif args.command == "html":
        run_html(resolved, args.output, args.open)


__all__ = [
    "ATIFValidationError",
    "Event",
    "ToolCall",
    "ToolResult",
    "TOOL_VISIBILITY_FULL",
    "TOOL_VISIBILITY_HIDDEN",
    "TOOL_VISIBILITY_MINIMAL",
    "build_html",
    "choose_trajectory_file",
    "copilot_to_atif",
    "discover_trajectory_files",
    "load",
    "main",
    "normalize",
    "parse_tool_call",
    "parse_tool_result",
    "run_html",
    "run_play",
    "run_serve",
]


if __name__ == "__main__":
    main()
