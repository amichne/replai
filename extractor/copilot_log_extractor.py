#!/usr/bin/env python3
"""copilot_log_extractor.py - simple extractor for Copilot CLI logs into a canonical event stream.

Usage: python3 scripts/copilot_log_extractor.py --input path/to/log.ndjson --output out.json
"""

import json
import os
import sys
import argparse
from typing import Iterator, Any, Dict


def read_ndjson(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except json.JSONDecodeError as e:
                # Re-raise with file context for easier debugging
                raise


def read_json(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            for item in data:
                yield item
        elif isinstance(data, dict):
            yield data
        else:
            yield {"raw": data}


def load_file(path: str) -> Iterator[Dict[str, Any]]:
    """Load events from a file or directory. Supports .ndjson/.jsonl and .json files."""
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if name.endswith('.ndjson') or name.endswith('.jsonl') or name.endswith('.log'):
                yield from read_ndjson(full)
            elif name.endswith('.json'):
                yield from read_json(full)
    else:
        if path.endswith('.ndjson') or path.endswith('.jsonl') or path.endswith('.log'):
            yield from read_ndjson(path)
        elif path.endswith('.json'):
            yield from read_json(path)
        else:
            # try ndjson first, then json, then fallback to raw lines
            try:
                yield from read_ndjson(path)
            except Exception:
                try:
                    yield from read_json(path)
                except Exception:
                    with open(path, 'r', encoding='utf-8') as f:
                        for ln in f:
                            ln = ln.strip()
                            if not ln:
                                continue
                            try:
                                yield json.loads(ln)
                            except Exception:
                                yield {"raw_line": ln}


def _get_timestamp(raw: Dict[str, Any]):
    for k in ('ts', 'timestamp', 'time', 'created_at', 'date'):
        if k in raw:
            return raw[k]
    return None


def _get_text(raw: Dict[str, Any]):
    if 'content' in raw:
        c = raw['content']
        if isinstance(c, str):
            return c
        elif isinstance(c, dict) and 'text' in c:
            return c['text']
        else:
            return json.dumps(c, ensure_ascii=False)
    if 'text' in raw:
        return raw['text'] if isinstance(raw['text'], str) else json.dumps(raw['text'], ensure_ascii=False)
    for k in ('message', 'body'):
        if k in raw:
            v = raw[k]
            if isinstance(v, str):
                return v
            else:
                return json.dumps(v, ensure_ascii=False)
    return None


def _get_actor(raw: Dict[str, Any]):
    if 'role' in raw:
        return raw.get('role')
    if raw.get('type') in ('tool_call', 'tool', 'tool_event'):
        return 'tool'
    if 'tool' in raw:
        return 'tool'
    if raw.get('actor'):
        return raw.get('actor')
    return None


def _extract_tokens(raw: Dict[str, Any]):
    keys = ['tokens_in', 'tokens_out', 'tokens', 'token_in', 'token_out', 'input_tokens', 'output_tokens']
    out = {}
    for k in keys:
        if k in raw:
            out[k] = raw[k]
    if 'usage' in raw and isinstance(raw['usage'], dict):
        for sub in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            if sub in raw['usage']:
                out[sub] = raw['usage'][sub]
    return out


def normalize_event(raw: Dict[str, Any], seq: int, session_id: str = None) -> Dict[str, Any]:
    actor = _get_actor(raw) or ('user' if raw.get('role') == 'user' else 'assistant' if raw.get('role') == 'assistant' else 'system')
    event_type = raw.get('type')
    if not event_type:
        if actor == 'tool':
            event_type = 'tool_call'
        elif 'role' in raw:
            event_type = 'message'
        elif 'event' in raw:
            event_type = raw.get('event')
        else:
            event_type = 'unknown'
    canon = {
        "event_type": event_type,
        "session_id": session_id or raw.get('session_id') or raw.get('session') or None,
        "ts": _get_timestamp(raw),
        "seq": seq,
        "actor": actor,
        "text": _get_text(raw),
        "model": raw.get('model') or None,
        "tokens": _extract_tokens(raw),
        "tool": None,
        "raw": raw,
    }
    if actor == 'tool' or event_type == 'tool_call':
        tool = {}
        tool['name'] = raw.get('tool') or raw.get('name') or raw.get('command')
        tool['cmd'] = raw.get('cmd') or raw.get('command') or raw.get('args')
        for k in ('stdout', 'stderr', 'exit_code', 'return_code', 'rc'):
            if k in raw:
                tool[k] = raw[k]
        if 'result' in raw:
            tool['result'] = raw['result']
        canon['tool'] = tool
    return canon


def extract_events(path: str):
    events = []
    seq = 0
    for raw in load_file(path):
        seq += 1
        events.append(normalize_event(raw, seq))
    return events


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Copilot CLI log extractor -> canonical event stream (JSON lines)')
    parser.add_argument('--input', '-i', required=True, help='Path to a log file (.ndjson, .json) or directory')
    parser.add_argument('--output', '-o', help='Output file (defaults to stdout)')
    args = parser.parse_args()
    ev = extract_events(args.input)
    out = args.output
    if out:
        with open(out, 'w', encoding='utf-8') as f:
            for e in ev:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"Wrote {len(ev)} events to {out}")
    else:
        for e in ev:
            print(json.dumps(e, ensure_ascii=False))
