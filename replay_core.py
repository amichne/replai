"""Core ATIF replay models, normalization, and file discovery."""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


_ATTACHMENTS_BLOCK_RE = re.compile(r"<attachments>.*?</attachments>", re.DOTALL | re.IGNORECASE)
_ATTACHMENT_RE = re.compile(r"<attachment\b[^>]*>.*?</attachment>", re.DOTALL | re.IGNORECASE)
_USER_REQUEST_RE = re.compile(r"<userRequest>\s*(.*?)\s*</userRequest>", re.DOTALL | re.IGNORECASE)
# Matches text-typed parts in a (possibly truncated) Copilot response JSON string.
# Handles both key orderings: {"type":"text","content":"..."} and the reverse.
_RESPONSE_TEXT_PART_RE = re.compile(
	r'"type"\s*:\s*"text"[^{}\[\]]*?"content"\s*:\s*"((?:[^"\\]|\\.)*)'
	r'|"content"\s*:\s*"((?:[^"\\]|\\.)*)[^{}\[\]]*?"type"\s*:\s*"text"',
	re.DOTALL,
)

_TRAJECTORY_SUFFIXES = (".json", ".jsonl", ".atif", ".atif.json")
_IGNORED_SCAN_DIRS = {
	".git",
	".venv",
	"venv",
	"node_modules",
	"__pycache__",
	".pytest_cache",
	".mypy_cache",
	".ruff_cache",
	".idea",
	".vscode",
	"dist",
	"build",
}


@dataclass(frozen=True)
class Event:
	"""A normalized replay event."""

	kind: str
	step_id: int
	ts: Optional[str]
	role: str
	title: Optional[str]
	body: Any
	ref: Optional[str]


@dataclass(frozen=True)
class ToolCall:
	"""Parsed tool call with normalized field access."""

	call_id: str
	name: str
	arguments: Any
	timestamp: Optional[str]


@dataclass(frozen=True)
class ToolResult:
	"""Parsed tool result with normalized field access."""

	source_call_id: Optional[str]
	content: Any
	timestamp: Optional[str]


class ATIFValidationError(ValueError):
	"""Raised when the input document does not conform to ATIF-v1.5."""


def parse_tool_call(raw: dict) -> ToolCall:
	call_id = raw.get("id") or raw.get("tool_call_id") or "unknown"
	name = raw.get("name") or raw.get("function_name") or "unknown"
	arguments = _decode_args(raw.get("arguments", {}))
	timestamp = raw.get("timestamp")
	return ToolCall(call_id=call_id, name=name, arguments=arguments, timestamp=timestamp)


def parse_tool_result(raw: dict) -> ToolResult:
	source_call_id = raw.get("source_call_id") or raw.get("ref")
	content = raw.get("content")
	timestamp = raw.get("timestamp")
	return ToolResult(source_call_id=source_call_id, content=content, timestamp=timestamp)


def _require(condition: bool, message: str) -> None:
	if not condition:
		raise ATIFValidationError(message)


def _validate(doc: dict) -> None:
	_require(isinstance(doc, dict), "Root must be a JSON object.")

	version = doc.get("schema_version")
	_require(
		isinstance(version, str) and bool(re.match(r"^(?:ATIF-v)?1\.5(?:\..*)?$", version)),
		f"Expected schema_version matching '(ATIF-v)1.5.*', got {version!r}.",
	)
	_require("session_id" in doc, "Missing required field: session_id.")
	_require("agent" in doc and isinstance(doc["agent"], dict), "Missing or invalid 'agent' object.")
	_require("steps" in doc and isinstance(doc["steps"], list), "Missing or invalid 'steps' list.")

	for i, step in enumerate(doc["steps"]):
		_require(isinstance(step, dict), f"Step {i}: must be a JSON object.")

		msg = step.get("message")
		if msg is not None:
			_require(isinstance(msg, (dict, str)), f"Step {i}: 'message' must be an object or string.")
			if isinstance(msg, dict):
				_require("content" in msg, f"Step {i}: message object missing 'content'.")
				_require(
					"role" in msg or "source" in step,
					f"Step {i}: message object missing 'role' and step missing 'source'.",
				)

		source = step.get("source")
		if source is not None:
			_require(isinstance(source, str), f"Step {i}: 'source' must be a string.")

		tool_calls = step.get("tool_calls")
		if tool_calls is not None:
			_require(isinstance(tool_calls, list), f"Step {i}: 'tool_calls' must be a list.")
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
				_require(isinstance(results, list), f"Step {i}: observation.results must be a list.")
				for k, result in enumerate(results):
					_require(isinstance(result, dict), f"Step {i}, result {k}: must be an object.")
					_require(
						"source_call_id" in result or "ref" in result,
						f"Step {i}, result {k}: missing 'source_call_id'/'ref'.",
					)


def copilot_to_atif(raw: dict) -> dict:
	if "steps" in raw:
		return raw

	messages = raw.get("messages") or []
	steps: List[dict] = []
	role_map = {"assistant": "agent", "user": "user", "system": "system"}

	for i, message in enumerate(messages):
		raw_role = message.get("role", "user")
		role = role_map.get(raw_role, "user")

		step: Dict[str, Any] = {
			"message": {
				"role": role,
				"content": message.get("content") or "",
			}
		}

		raw_tcs = message.get("tool_calls") or []
		if raw_tcs:
			step["tool_calls"] = [
				{
					"id": tc.get("id", f"tc_{i}_{j}"),
					"name": (tc.get("function") or {}).get("name", "unknown"),
					"arguments": _decode_args((tc.get("function") or {}).get("arguments", "{}")),
				}
				for j, tc in enumerate(raw_tcs)
			]

		steps.append(step)

	return {
		"schema_version": "1.5",
		"session_id": raw.get("id", "converted"),
		"agent": {"name": raw.get("model", "unknown")},
		"steps": steps,
	}


def _decode_args(value: Any) -> Any:
	if isinstance(value, str):
		try:
			return json.loads(value)
		except json.JSONDecodeError:
			return value
	return value


def _strip_attachment_content(value: Any) -> Any:
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
	if not isinstance(value, str):
		return value

	matches = [match.strip() for match in _USER_REQUEST_RE.findall(value) if match.strip()]
	if not matches:
		return value
	return "\n\n".join(matches)


def _normalize_message_body(value: Any, role: str) -> Any:
	body = _strip_attachment_content(value)
	if role == "user":
		body = _extract_user_request_content(body)
	return body


def _load_atif_json(path: Union[str, Path]) -> dict:
	with open(path, encoding="utf-8") as fh:
		raw = json.load(fh)

	doc = copilot_to_atif(raw)
	_validate(doc)
	return doc


def normalize(doc: dict) -> List[Event]:
	events: List[Event] = []

	for step_index, step in enumerate(doc["steps"]):
		step_id = int(step.get("step_id", step_index + 1))
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
	table = {
		"user": "user",
		"human": "user",
		"assistant": "agent",
		"agent": "agent",
		"system": "system",
		"tool": "tool",
		"function": "tool",
	}
	return table.get(raw.lower(), "user")


def _is_candidate_path(path: Path) -> bool:
	name = path.name.lower()
	return any(name.endswith(suffix) for suffix in _TRAJECTORY_SUFFIXES)


def _looks_like_trajectory_file(path: Path) -> bool:
	if not path.is_file() or not _is_candidate_path(path):
		return False
	try:
		with open(path, encoding="utf-8") as fh:
			raw = json.load(fh)
	except (OSError, json.JSONDecodeError, UnicodeDecodeError):
		return False
	return isinstance(raw, dict) and ("steps" in raw or "messages" in raw)


def discover_trajectory_files(root: Union[str, Path] = ".") -> List[Path]:
	root_path = Path(root).expanduser().resolve()
	candidates: List[Path] = []

	def _walk(directory: Path) -> None:
		try:
			entries = sorted(directory.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
		except OSError:
			return

		for entry in entries:
			if entry.name.startswith(".") and entry.name not in {".json"}:
				if entry.is_dir():
					continue
			if entry.is_dir():
				if entry.name in _IGNORED_SCAN_DIRS:
					continue
				_walk(entry)
			elif _looks_like_trajectory_file(entry):
				candidates.append(entry)

	_walk(root_path)
	return candidates


def _is_copilot_session_dir(path: Path) -> bool:
	return path.is_dir() and path.joinpath("main.jsonl").is_file()


def _default_workspace_storage_roots() -> List[Path]:
	home = Path.home()
	roots: List[Path] = []

	if sys.platform == "darwin":
		roots.append(home / "Library" / "Application Support" / "Code" / "User" / "workspaceStorage")
	elif sys.platform.startswith("linux"):
		roots.extend([
			home / ".config" / "Code" / "User" / "workspaceStorage",
			home / ".config" / "Code - OSS" / "User" / "workspaceStorage",
			home / ".var" / "app" / "com.visualstudio.code" / "config" / "Code" / "User" / "workspaceStorage",
		])
	elif sys.platform == "win32":
		appdata = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
		roots.extend([
			appdata / "Code" / "User" / "workspaceStorage",
			appdata / "Code - OSS" / "User" / "workspaceStorage",
		])

	return [root for root in roots if root.is_dir()]


def _discover_latest_copilot_session() -> Optional[Path]:
	sessions: List[Path] = []
	for workspace_root in _default_workspace_storage_roots():
		try:
			for main in workspace_root.glob("**/GitHub.copilot-chat/debug-logs/*/main.jsonl"):
				if main.is_file():
					sessions.append(main)
		except OSError:
			continue

	if not sessions:
		return None

	return max(sessions, key=lambda p: p.stat().st_mtime)


def choose_trajectory_file(root: Union[str, Path] = ".", *, use_latest: bool = True) -> Path:
	root_path = Path(root).expanduser().resolve()

	# Explicit .jsonl file (any name, including main.jsonl)
	if root_path.is_file() and root_path.suffix == ".jsonl":
		print(f"Using Copilot session: {root_path}")
		return root_path

	# Explicit session directory containing main.jsonl
	if _is_copilot_session_dir(root_path):
		selected = root_path / "main.jsonl"
		print(f"Using Copilot session: {selected}")
		return selected

	# Auto-discover the most recently modified session from workspaceStorage
	if use_latest:
		recent_session = _discover_latest_copilot_session()
		if recent_session is not None:
			selected = recent_session
			print(f"Using Copilot session: {selected}")
			return selected

	# Fallback to legacy discovery of ATIF-style files
	choices = discover_trajectory_files(root)
	if not choices:
		print(
			"Error: no trajectory file was provided and no ATIF/Copilot-style JSON files were found under the current directory.",
			file=sys.stderr,
		)
		print("Tip: pass a file path explicitly, or run `serve` and open a file in the browser.", file=sys.stderr)
		sys.exit(1)

	if len(choices) == 1 or not sys.stdin.isatty():
		selected = choices[0]
		print(f"Using trajectory: {selected}")
		return selected

	print("Select a trajectory to replay:")
	for index, path in enumerate(choices, start=1):
		try:
			rel = path.relative_to(Path.cwd())
		except ValueError:
			rel = path
		print(f"  {index:>2}. {rel}")

	while True:
		answer = input(f"Enter a number [1-{len(choices)}] or 'q' to quit: ").strip().lower()
		if answer in {"q", "quit", "exit"}:
			sys.exit(0)
		if answer.isdigit():
			index = int(answer)
			if 1 <= index <= len(choices):
				return choices[index - 1]
		print("Please enter a valid selection.")


__all__ = [
	"ATIFValidationError",
	"Event",
	"ToolCall",
	"ToolResult",
	"choose_trajectory_file",
	"copilot_to_atif",
	"discover_trajectory_files",
	"load",
	"normalize",
	"parse_tool_call",
	"parse_tool_result",
]


def _ms_to_iso(ts_ms: Optional[Union[int, float]]) -> Optional[str]:
	"""Convert epoch milliseconds to an ISO8601 Zulu string, or return None."""
	if ts_ms is None:
		return None
	try:
		return datetime.fromtimestamp(float(ts_ms) / 1000.0, UTC).isoformat().replace("+00:00", "Z")
	except Exception:
		return None


def _extract_text_from_truncated_response(raw: str) -> str:
	"""Best-effort text extraction from a truncated Copilot response JSON string.

	Copilot debug logs truncate the ``response`` field when it exceeds their
	size limit, leaving an incomplete JSON string.  This function uses regex to
	recover any text parts that were fully serialised before the cut-off.
	"""
	texts: List[str] = []
	for m in _RESPONSE_TEXT_PART_RE.finditer(raw):
		raw_value = m.group(1) or m.group(2) or ""
		if not raw_value:
			continue
		try:
			# Attempt a proper JSON string decode (handles all escape sequences).
			decoded = json.loads('"' + raw_value + '"')
		except Exception:
			# The captured value may itself be truncated mid-escape; fall back to
			# a manual replacement for the most common JSON escapes.
			decoded = (
				raw_value
				.replace('\\n', '\n')
				.replace('\\r', '\r')
				.replace('\\t', '\t')
				.replace('\\"', '"')
				.replace('\\/', '/')
				.replace('\\\\', '\\')
			)
		if decoded:
			texts.append(decoded)
	return "\n\n".join(texts)


def _parse_agent_response_parts(raw: Any, outer_span_id: Optional[str] = None) -> tuple:
	"""Extract assistant text and tool calls from an agent_response.attrs.response value.

	The response value is usually a JSON-encoded string representing an array
	of message objects with `parts`. Returns a tuple of (text: str, tool_calls: list[dict]).
	Tool call dicts have keys: id, name, arguments, timestamp (None).
	"""
	if not raw:
		return "", []
	try:
		obj = json.loads(raw) if isinstance(raw, str) else raw
		texts: List[str] = []
		tool_calls: List[dict] = []
		if isinstance(obj, list):
			for msg in obj:
				parts = msg.get("parts") if isinstance(msg, dict) else None
				if not parts:
					continue
				for p in parts:
					if not isinstance(p, dict):
						continue
					if p.get("type") == "text":
						texts.append(p.get("content", ""))
					elif p.get("type") == "tool_call":
						call_id = p.get("id") or outer_span_id or f"call_{len(tool_calls)}"
						tool_calls.append({
							"id": call_id,
							"name": p.get("name", "unknown"),
							"arguments": _decode_args(p.get("arguments", {})),
							"timestamp": None,
						})
		return "\n\n".join(t for t in texts if t), tool_calls
	except Exception:
		# Fallback: the response string was likely truncated by the debug log
		# size limit.  Attempt regex-based extraction of any text parts that
		# were fully written before the cut-off.
		if isinstance(raw, str):
			recovered = _extract_text_from_truncated_response(raw)
			if recovered:
				return recovered, []
		# Nothing recoverable – surface the raw value so it is at least visible.
		return str(raw), []


def load_copilot_jsonl(path: Union[str, Path]) -> dict:
	"""Load a Copilot debug-log main.jsonl and synthesize an ATIF-v1.5 doc.

	This is a pragmatic importer: it creates a sequence of ATIF steps from
	`user_message` and `agent_response` events and attaches nearby tool calls
	and results to the most recent step. It is intentionally permissive so the
	viewer can display a faithful, debuggable conversation even when logs are
	partially complete.
	"""
	path = Path(path)
	if not path.is_file():
		raise ATIFValidationError(f"Not a file: {path}")

	events: List[dict] = []
	with open(path, encoding="utf-8") as fh:
		for line in fh:
			line = line.strip()
			if not line:
				continue
			try:
				events.append(json.loads(line))
			except Exception:
				# skip unparseable lines but keep going
				continue

	steps: List[dict] = []
	step_id = 1
	call_map: Dict[str, dict] = {}
	last_call_id: Optional[str] = None
	last_step: Optional[dict] = None
	# Ordered queue of declared (but not yet executed) tool calls extracted from
	# agent_response parts.  Real Copilot logs use separate ID namespaces for the
	# LLM API call-id (toolu_bdrk_*) and the Copilot span-id.  We match them by
	# tool name in FIFO order within each agent turn rather than by ID equality.
	unresolved_part_calls: List[dict] = []  # each item: {"step": ..., "call": ...}

	for ev in events:
		type_ = ev.get("type")
		ts_iso = _ms_to_iso(ev.get("ts"))
		attrs = ev.get("attrs") or {}

		if type_ == "user_message":
			content = attrs.get("content", "")
			step = {"step_id": step_id, "timestamp": ts_iso, "message": {"role": "user", "content": content, "timestamp": ts_iso}}
			step["tool_calls"] = []
			step["observation"] = {"results": []}
			steps.append(step)
			last_step = step
			step_id += 1
			continue

		if type_ == "agent_response":
			resp = attrs.get("response")
			outer_span_id = ev.get("spanId")
			content, extracted_calls = _parse_agent_response_parts(resp, outer_span_id)
			step = {"step_id": step_id, "timestamp": ts_iso, "message": {"role": "agent", "content": content, "timestamp": ts_iso}}
			step["tool_calls"] = []
			step["observation"] = {"results": []}
			steps.append(step)
			last_step = step
			step_id += 1
			# Register extracted tool calls.  Also enqueue them for FIFO name-based
			# result pairing (real Copilot logs use a different ID namespace for the
			# LLM API call-ids vs the instrumentation span-ids).
			for tc in extracted_calls:
				tc_with_ts = dict(tc)
				if tc_with_ts.get("timestamp") is None and ts_iso:
					tc_with_ts["timestamp"] = ts_iso
				last_step["tool_calls"].append(tc_with_ts)
				call_map[tc_with_ts["id"]] = {"step": last_step, "call": tc_with_ts}
				last_call_id = tc_with_ts["id"]
				unresolved_part_calls.append({"step": last_step, "call": tc_with_ts})
			continue

		if type_ == "tool_call":
			name = ev.get("name") or attrs.get("name") or "unknown"
			args_raw = attrs.get("args") or attrs.get("arguments")
			arguments = _decode_args(args_raw)
			call_id = ev.get("spanId") or attrs.get("id") or attrs.get("call_id") or f"call_{step_id}_{len(steps)}"
			# Compute result timestamp: ts + dur gives the wall-clock end time when both are present
			ts_ms = ev.get("ts")
			dur_ms = ev.get("dur")
			result_ts_iso = _ms_to_iso((ts_ms or 0) + (dur_ms or 0)) if ts_ms is not None else ts_iso

			if last_step is None:
				# synthesize an empty system step to hang the call on
				last_step = {"step_id": step_id, "timestamp": ts_iso, "message": {"role": "system", "content": "", "timestamp": ts_iso}, "tool_calls": [], "observation": {"results": []}}
				steps.append(last_step)
				step_id += 1

			# --- Resolve which call record / step owns this invocation ---
			# Priority 1: exact call_id match (e.g. fixture with shared IDs)
			if call_id in call_map:
				target_entry = call_map[call_id]
				target_step = target_entry["step"]
				existing_call = target_entry["call"]
				if arguments:
					existing_call["arguments"] = arguments
			else:
				# Priority 2: FIFO name-based match against declared-but-unresolved
				# parts (handles the real Copilot API-id vs span-id mismatch).
				matched_idx: Optional[int] = None
				for idx, entry in enumerate(unresolved_part_calls):
					if entry["call"].get("name") == name:
						matched_idx = idx
						break

				if matched_idx is not None:
					matched = unresolved_part_calls.pop(matched_idx)
					target_step = matched["step"]
					existing_call = matched["call"]
					# Overwrite arguments with the actual invocation record (more accurate)
					if arguments:
						existing_call["arguments"] = arguments
				else:
					# No matching declaration — create a new call entry on last_step
					new_call = {"id": call_id, "name": name, "arguments": arguments, "timestamp": ts_iso}
					last_step.setdefault("tool_calls", []).append(new_call)
					call_map[call_id] = {"step": last_step, "call": new_call}
					target_step = last_step
					existing_call = new_call

			last_call_id = existing_call["id"]

			# Extract the inline result (real Copilot logs embed result in the same event)
			result_raw = attrs.get("result")
			if result_raw is not None:
				result_content = result_raw
				try:
					if isinstance(result_content, str):
						result_content = json.loads(result_content)
				except Exception:
					pass
				res = {"source_call_id": existing_call["id"], "content": result_content, "timestamp": result_ts_iso}
				target_step.setdefault("observation", {}).setdefault("results", []).append(res)
			continue

		if type_ == "tool_result":
			# result payloads vary; prefer attrs.result but accept other keys
			result_raw = attrs.get("result") or attrs.get("output") or attrs.get("result_text") or attrs.get("resultBody") or attrs.get("resultBodySerialized") or attrs.get("result_string")
			# fallback to attrs itself
			result_content = result_raw if result_raw is not None else attrs
			# try to coerce JSON strings
			try:
				if isinstance(result_content, str):
					result_content = json.loads(result_content)
			except Exception:
				pass
			source_call_id = attrs.get("source_call_id") or attrs.get("ref") or attrs.get("call_id") or last_call_id or "unknown"
			res = {"source_call_id": source_call_id, "content": result_content, "timestamp": ts_iso}
			# attach to the matching call's step if possible
			if source_call_id in call_map:
				call_map[source_call_id]["step"].setdefault("observation", {}).setdefault("results", []).append(res)
			elif last_step is not None:
				last_step.setdefault("observation", {}).setdefault("results", []).append(res)
			else:
				# synthesize a step to hold the orphaned result
				step = {"step_id": step_id, "timestamp": ts_iso, "message": {"role": "system", "content": "", "timestamp": ts_iso}, "tool_calls": [], "observation": {"results": [res]}}
				steps.append(step)
				step_id += 1
			continue

		if type_ == "child_session_ref":
			# If the child log is present, recursively load and flatten it here.
			child_fn = attrs.get("childLogFile") or attrs.get("child_log_file")
			if child_fn and path.parent.joinpath(child_fn).is_file():
				try:
					child_steps_doc = load_copilot_jsonl(path.parent.joinpath(child_fn))
					# child_steps_doc returns a full ATIF-like doc; append its steps
					for s in child_steps_doc.get("steps", []):
						s["step_id"] = step_id
						steps.append(s)
						step_id += 1
				except Exception:
					# ignore failures to read child logs
					pass
			continue

	# synthesize an ATIF doc
	doc = {"schema_version": "1.5", "session_id": str(path.stem), "agent": {"name": "copilot"}, "steps": []}
	for s in steps:
		# normalize step dict shape to the ATIF expectation
		step_obj: Dict[str, Any] = {"step_id": s.get("step_id")}
		if s.get("timestamp"):
			step_obj["timestamp"] = s.get("timestamp")
		msg = s.get("message")
		if msg:
			step_obj["message"] = {"role": msg.get("role"), "content": msg.get("content"), "timestamp": msg.get("timestamp")}
		if s.get("tool_calls"):
			step_obj["tool_calls"] = s.get("tool_calls")
		obs = s.get("observation")
		if obs and obs.get("results"):
			step_obj["observation"] = {"results": obs.get("results")}
		doc["steps"].append(step_obj)

	_validate(doc)
	return doc


def load(path: Union[str, Path]) -> dict:
	"""Load either an ATIF JSON file or a Copilot debug-log session (.jsonl or directory).

	This wrapper prefers the existing ATIF JSON loader but falls back to the
	Copilot JSONL importer when a directory or `.jsonl` file is provided.
	"""
	p = Path(path)
	if p.is_dir():
		# Directory containing Copilot debug-log files
		main = p.joinpath("main.jsonl")
		if main.is_file():
			return load_copilot_jsonl(main)
		# no main.jsonl — fall back to choose_trajectory_file behaviour
		raise ATIFValidationError(f"No main.jsonl found in directory: {p}")

	if p.is_file() and p.suffix == ".jsonl":
		return load_copilot_jsonl(p)

	return _load_atif_json(p)

