"""Static HTML browser viewer and optional localhost serving."""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import tempfile
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from browser_assets import BROWSER_CSS, BROWSER_JS
from replay_core import ATIFValidationError, Event, load, normalize

try:
	from jinja2 import Environment, FileSystemLoader, select_autoescape
except ImportError:  # pragma: no cover - fallback keeps the viewer portable without Jinja2 installed.
	Environment = None
	FileSystemLoader = None
	select_autoescape = None


_TEMPLATE_NAME = "browser_template.html.j2"


def _js_safe_json(obj: Any) -> str:
	return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


@lru_cache(maxsize=1)
def _template_source() -> str:
	template_path = Path(__file__).resolve().with_name(_TEMPLATE_NAME)
	return template_path.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _jinja_environment() -> Optional["Environment"]:
	if Environment is None or FileSystemLoader is None or select_autoescape is None:
		return None
	return Environment(
		loader=FileSystemLoader(str(Path(__file__).resolve().parent)),
		autoescape=select_autoescape(("html", "xml")),
		keep_trailing_newline=True,
	)


def _render_template(context: Dict[str, Any]) -> str:
	environment = _jinja_environment()
	if environment is not None:
		return environment.get_template(_TEMPLATE_NAME).render(**context)

	template = _template_source()
	replacements = {
		"{{ page_title }}": str(context["page_title"]),
		"{{ browser_css | safe }}": str(context["browser_css"]),
		"{{ initial_events_json | safe }}": str(context["initial_events_json"]),
		"{{ initial_raw_doc_json | safe }}": str(context["initial_raw_doc_json"]),
		"{{ browser_js | safe }}": str(context["browser_js"]),
	}
	for needle, value in replacements.items():
		template = template.replace(needle, value)
	return template


def build_html(doc: Optional[dict], events: List[Event]) -> str:
	context = {
		"page_title": str((doc or {}).get("session_id", "open local trajectory")),
		"browser_css": BROWSER_CSS,
		"browser_js": BROWSER_JS,
		"initial_events_json": _js_safe_json([dataclasses.asdict(ev) for ev in events]),
		"initial_raw_doc_json": _js_safe_json(doc if doc is not None else None),
	}
	return _render_template(context)


def _load_doc_and_events(path: Optional[str]) -> Tuple[Optional[dict], List[Event]]:
	doc: Optional[dict] = None
	events: List[Event] = []

	if path:
		try:
			doc = load(path)
		except (ATIFValidationError, json.JSONDecodeError, OSError) as exc:
			print(f"Error: {exc}", file=sys.stderr)
			sys.exit(1)
		events = normalize(doc)

	return doc, events


def _safe_session_slug(doc: Optional[dict]) -> str:
	raw_value = str((doc or {}).get("session_id", "replai-session"))
	slug = re.sub(r"[^A-Za-z0-9._-]+", "-", raw_value).strip("-.")
	return slug or "replai-session"


def _write_html_output(html: str, doc: Optional[dict], output_path: Optional[str]) -> Path:
	if output_path:
		target = Path(output_path).expanduser().resolve()
		target.parent.mkdir(parents=True, exist_ok=True)
		target.write_text(html, encoding="utf-8")
		return target

	with tempfile.NamedTemporaryFile(
		mode="w",
		encoding="utf-8",
		suffix=".html",
		prefix=f"replai-{_safe_session_slug(doc)}-",
		delete=False,
	) as fh:
		fh.write(html)
		return Path(fh.name)


def _open_browser_url(url: str) -> None:
	try:
		import webbrowser

		webbrowser.open(url)
	except Exception:
		pass


def run_html(path: Optional[str], output: Optional[str], open_browser: bool) -> None:
	doc, events = _load_doc_and_events(path)
	html = build_html(doc, events)

	if output:
		target = _write_html_output(html, doc, output)
		print(f"Wrote HTML viewer to {target}")
		if open_browser:
			_open_browser_url(target.as_uri())
		return

	if open_browser:
		target = _write_html_output(html, doc, None)
		print(f"Opened static HTML viewer at {target}")
		print("This is a local file viewer — no server is required.")
		_open_browser_url(target.as_uri())
		return

	sys.stdout.write(html)


def run_serve(path: Optional[str], port: int, output: Optional[str], http_mode: bool) -> None:
	doc, events = _load_doc_and_events(path)
	html = build_html(doc, events)

	if not http_mode:
		target = _write_html_output(html, doc, output)
		session_id = (doc or {}).get("session_id", "(open local file in browser)")
		agent_name = ((doc or {}).get("agent") or {}).get("name", "local-file")
		step_count = len(set(ev.step_id for ev in events)) if events else 0

		print(f"ATIF Replay  ·  session: {session_id}  ·  agent: {agent_name}  ·  {step_count} steps")
		print(f"Opened static viewer at {target}")
		print("Running in direct HTML mode — no server required.")
		if not path:
			print("No file was passed — open a local trajectory from the browser UI.")
		_open_browser_url(target.as_uri())
		return

	body = html.encode("utf-8")
	session_id = (doc or {}).get("session_id", "(open local file in browser)")
	agent_name = ((doc or {}).get("agent") or {}).get("name", "local-file")
	step_count = len(set(ev.step_id for ev in events)) if events else 0

	class _Handler(BaseHTTPRequestHandler):
		def do_GET(self) -> None:
			if self.path in ("/", "/index.html"):
				self.send_response(200)
				self.send_header("Content-Type", "text/html; charset=utf-8")
				self.send_header("Content-Length", str(len(body)))
				self.send_header("Cache-Control", "no-store")
				self.end_headers()
				self.wfile.write(body)
			else:
				self.send_error(404)

		def log_message(self, fmt: str, *args: object) -> None:  # type: ignore[override]
			pass

	url = f"http://127.0.0.1:{port}/"

	print(f"ATIF Replay  ·  session: {session_id}  ·  agent: {agent_name}  ·  {step_count} steps")
	print(f"Serving at {url}")
	if not path:
		print("No file was passed — open a local trajectory from the browser UI.")
	print("Press Ctrl-C to stop.")

	_open_browser_url(url)

	server = HTTPServer(("127.0.0.1", port), _Handler)
	try:
		server.serve_forever()
	except KeyboardInterrupt:
		print("\nStopped.")
	finally:
		server.server_close()


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Open or export the ATIF browser viewer.")
	sub = parser.add_subparsers(dest="command", required=True)

	serve = sub.add_parser("serve", help="Open the browser viewer.")
	serve.add_argument("file", nargs="?", help="Path to an ATIF-v1.5 JSON file.")
	serve.add_argument("--output", help="Optional HTML output path.")
	serve.add_argument("--http", action="store_true", help="Use a localhost HTTP server instead of direct HTML mode.")
	serve.add_argument("--port", type=int, default=8000, help="HTTP listen port when --http is used.")

	html = sub.add_parser("html", help="Write the static HTML viewer or print it to stdout.")
	html.add_argument("file", nargs="?", help="Path to an ATIF-v1.5 JSON file.")
	html.add_argument("--output", help="Write the HTML viewer to this path instead of stdout.")
	html.add_argument("--open", action="store_true", help="Open the generated HTML file in the default browser.")
	return parser


def main() -> None:
	args = _build_parser().parse_args()
	if args.command == "serve":
		run_serve(args.file, args.port, args.output, args.http)
	elif args.command == "html":
		run_html(args.file, args.output, args.open)


__all__ = ["build_html", "run_html", "run_serve"]


if __name__ == "__main__":
	main()
