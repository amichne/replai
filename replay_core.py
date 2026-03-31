"""Core ATIF replay models, normalization, and file discovery."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


_ATTACHMENTS_BLOCK_RE = re.compile(r"<attachments>.*?</attachments>", re.DOTALL | re.IGNORECASE)
_ATTACHMENT_RE = re.compile(r"<attachment\b[^>]*>.*?</attachment>", re.DOTALL | re.IGNORECASE)
_USER_REQUEST_RE = re.compile(r"<userRequest>\s*(.*?)\s*</userRequest>", re.DOTALL | re.IGNORECASE)

_TRAJECTORY_SUFFIXES = (".json", ".atif", ".atif.json")
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


def load(path: Union[str, Path]) -> dict:
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


def choose_trajectory_file(root: Union[str, Path] = ".") -> Path:
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
