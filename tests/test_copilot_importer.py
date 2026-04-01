import json
from pathlib import Path

import pytest

from replay_core import choose_trajectory_file, load, load_copilot_jsonl

_FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "copilot_debug_logs"
_FULL_SESSION_DIR = _FIXTURES / "full_session"
_REAL_SESSION_DIR = _FIXTURES / "real_session"


def test_load_sample_session(tmp_path):
    # Use the real debug-logs sample checked into the repo workspace
    # project root is two levels up from tests file (repo root is parents[1])
    session_dir = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "copilot_debug_logs" / "sample_session"
    assert session_dir.exists(), "fixture session directory missing"

    doc = load(session_dir)
    assert isinstance(doc, dict)
    assert doc.get("schema_version")
    assert isinstance(doc.get("steps"), list)
    # expect at least one user/agent step pair
    roles = [s.get("message", {}).get("role") for s in doc.get("steps", [])]
    assert any(r == "user" for r in roles) or any(r == "agent" for r in roles)


def test_choose_trajectory_file_uses_explicit_session_dir(monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    session_dir = project_root / "tests" / "fixtures" / "copilot_debug_logs" / "sample_session"
    monkeypatch.setattr("replay_core._discover_latest_copilot_session", lambda: None)

    selected = choose_trajectory_file(session_dir)

    assert selected == session_dir / "main.jsonl"


def test_choose_trajectory_file_falls_back_to_project_trajectory(monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr("replay_core._discover_latest_copilot_session", lambda: None)

    selected = choose_trajectory_file(project_root)

    assert selected == project_root / "trajectory.json"


# ---------------------------------------------------------------------------
# Phase 1: agent_response tool-call extraction
# ---------------------------------------------------------------------------

def test_tool_calls_extracted_from_agent_response_parts():
    """Tool calls embedded as parts in agent_response must appear in the step."""
    doc = load_copilot_jsonl(_FULL_SESSION_DIR / "main.jsonl")
    steps = doc["steps"]
    # First event is an agent_response with one text part and one tool_call part
    agent_step = next(s for s in steps if s.get("message", {}).get("role") == "agent")
    assert agent_step["tool_calls"], "expected at least one extracted tool call"
    tc = agent_step["tool_calls"][0]
    assert tc["id"] == "call_1"
    assert tc["name"] == "read_file"
    assert isinstance(tc["arguments"], dict)
    assert tc["arguments"].get("filePath") == "/tmp/foo.py"


def test_agent_response_text_still_extracted():
    """Text parts must still be extracted alongside tool_call parts."""
    doc = load_copilot_jsonl(_FULL_SESSION_DIR / "main.jsonl")
    agent_step = next(s for s in doc["steps"] if s.get("message", {}).get("role") == "agent")
    assert "I will read the file first" in agent_step["message"]["content"]


# ---------------------------------------------------------------------------
# Phase 2: tool_result pairing
# ---------------------------------------------------------------------------

def test_tool_result_paired_to_call():
    """A tool_result event referencing call_1 must be attached to the correct step."""
    doc = load_copilot_jsonl(_FULL_SESSION_DIR / "main.jsonl")
    # Find the step that owns call_1
    call_step = next(
        s for s in doc["steps"]
        if any(tc.get("id") == "call_1" for tc in s.get("tool_calls", []))
    )
    results = call_step.get("observation", {}).get("results", [])
    assert results, "expected tool_result to be attached to call_1 step"
    assert results[0]["source_call_id"] == "call_1"


# ---------------------------------------------------------------------------
# Phase 3: child session flattening
# ---------------------------------------------------------------------------

def test_child_session_flattened():
    """Steps from a child.jsonl referenced via child_session_ref must be included."""
    doc = load_copilot_jsonl(_FULL_SESSION_DIR / "main.jsonl")
    all_content = " ".join(
        s.get("message", {}).get("content", "") for s in doc["steps"]
    )
    assert "Child sub-agent finished" in all_content, (
        "expected child session content to be flattened into steps"
    )


# ---------------------------------------------------------------------------
# Phase 4: malformed lines
# ---------------------------------------------------------------------------

def test_malformed_lines_skipped(tmp_path):
    """Unparseable lines in a .jsonl file must be silently skipped."""
    jsonl = tmp_path / "main.jsonl"
    # Build valid lines programmatically to avoid manual escaping mistakes
    response_payload = json.dumps(
        [{"role": "assistant", "parts": [{"type": "text", "content": "hi"}]}]
    )
    user_line = json.dumps({"ts": 1000, "sid": "s", "type": "user_message", "attrs": {"content": "hello"}})
    agent_line = json.dumps({"ts": 2000, "sid": "s", "type": "agent_response", "attrs": {"response": response_payload}})
    lines = [
        "NOT JSON AT ALL\n",
        "{incomplete json\n",
        user_line + "\n",
        "\n",
        agent_line + "\n",
    ]
    jsonl.write_text("".join(lines), encoding="utf-8")

    doc = load_copilot_jsonl(jsonl)
    roles = [s.get("message", {}).get("role") for s in doc["steps"]]
    assert "user" in roles
    assert "agent" in roles


# ---------------------------------------------------------------------------
# Phase 5: choose_trajectory_file - --no-latest and explicit .jsonl
# ---------------------------------------------------------------------------

def test_no_latest_skips_auto_discovery(monkeypatch):
    """When use_latest=False, _discover_latest_copilot_session must not be called."""
    called = []

    def _fake_discover():
        called.append(True)
        return None

    monkeypatch.setattr("replay_core._discover_latest_copilot_session", _fake_discover)
    project_root = Path(__file__).resolve().parents[1]
    # use_latest=False; falls back to ATIF scan which finds trajectory.json
    result = choose_trajectory_file(project_root, use_latest=False)

    assert not called, "_discover_latest_copilot_session should not be called when use_latest=False"
    assert result.name == "trajectory.json"


def test_choose_file_explicit_jsonl_passthrough(tmp_path):
    """An explicit .jsonl path (not named main.jsonl) should be returned as-is."""
    jsonl = tmp_path / "session.jsonl"
    jsonl.write_text(
        '{"ts":1000,"sid":"s","type":"user_message","attrs":{"content":"hi"}}\n',
        encoding="utf-8",
    )

    result = choose_trajectory_file(jsonl)

    assert result == jsonl


# ---------------------------------------------------------------------------
# Phase 6: real Copilot log format — inline results in tool_call events
# ---------------------------------------------------------------------------

def test_inline_result_extracted_from_tool_call_event():
    """When a tool_call event carries attrs.result, it must appear as an observation result."""
    doc = load_copilot_jsonl(_REAL_SESSION_DIR / "main.jsonl")
    # Find the agent step that owns the read_file call
    call_step = next(
        s for s in doc["steps"]
        if any(tc.get("name") == "read_file" for tc in s.get("tool_calls", []))
    )
    results = call_step.get("observation", {}).get("results", [])
    assert results, "expected observation result from inline attrs.result"
    assert results[0]["source_call_id"] == "spanid-tc-001"
    assert results[0]["content"]  # some content extracted


def test_inline_result_no_duplicate_tool_calls():
    """When agent_response parts AND a top-level tool_call share the same ID, only one tool_call entry must exist."""
    doc = load_copilot_jsonl(_REAL_SESSION_DIR / "main.jsonl")
    # The real_session fixture has agent_response part with id="spanid-tc-001"
    # AND a top-level tool_call with spanId="spanid-tc-001"
    all_tool_call_ids = [
        tc.get("id")
        for s in doc["steps"]
        for tc in s.get("tool_calls", [])
    ]
    assert all_tool_call_ids.count("spanid-tc-001") == 1, (
        "duplicate tool_call entries detected for the same call_id"
    )


def test_inline_result_for_standalone_tool_call():
    """A top-level tool_call with no preceding agent_response part must still get its result attached."""
    doc = load_copilot_jsonl(_REAL_SESSION_DIR / "main.jsonl")
    # run_in_terminal (spanid-tc-002) is NOT declared in agent_response parts
    call_step = next(
        (s for s in doc["steps"]
         if any(tc.get("id") == "spanid-tc-002" for tc in s.get("tool_calls", []))),
        None,
    )
    assert call_step is not None, "run_in_terminal tool_call not found in any step"
    results = call_step.get("observation", {}).get("results", [])
    assert results, "expected inline result for standalone tool_call"
    assert results[0]["source_call_id"] == "spanid-tc-002"

