#!/usr/bin/env python3
"""copilot_timeline_parser.py - parse Copilot CLI timeline JSON into canonical event stream.

Usage:
  python3 scripts/copilot_timeline_parser.py --input path/to/timeline.json --output out.ndjson
"""

import json
from typing import Any, Dict, List

# Re-use the normalizer from the NDJSON extractor when available
try:
    from copilot_log_extractor import normalize_event
except Exception:
    # Minimal fallback normalizer to keep the parser usable even if the other module
    # isn't importable (useful for isolated tests).
    def normalize_event(raw: Dict[str, Any], seq: int, session_id: str = None) -> Dict[str, Any]:
        actor = raw.get('role') or raw.get('actor') or raw.get('speaker') or ('system' if raw.get('type') == 'session_start' else None)
        event_type = raw.get('type') or ('message' if actor in ('user', 'assistant', 'system') else 'unknown')
        text = raw.get('content') or raw.get('text') or raw.get('message') or raw.get('body')
        return {
            "event_type": event_type,
            "session_id": session_id or raw.get('session_id') or raw.get('session'),
            "ts": raw.get('ts') or raw.get('timestamp') or raw.get('time'),
            "seq": seq,
            "actor": actor,
            "text": text if isinstance(text, str) else (json.dumps(text, ensure_ascii=False) if text is not None else None),
            "model": raw.get('model'),
            "tokens": raw.get('usage') or raw.get('tokens') or {},
            "tool": None,
            "raw": raw,
        }


def find_timeline_items(data: Any) -> List[Dict[str, Any]]:
    """Return a list of timeline items from common timeline JSON shapes.

    Supports top-level arrays, or dicts with keys: timeline, events, items, messages, turns.
    For `turns` that contain `messages` the messages are flattened and merged with turn-level metadata.
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Prefer well-known keys
        for key in ("timeline", "events", "items", "messages", "turns"):
            v = data.get(key)
            if isinstance(v, list):
                if key == "turns":
                    flattened: List[Dict[str, Any]] = []
                    for item in v:
                        if isinstance(item, dict) and isinstance(item.get('messages'), list):
                            for msg in item['messages']:
                                new = dict(item)
                                new.pop('messages', None)
                                if isinstance(msg, dict):
                                    merged = dict(new)
                                    merged.update(msg)
                                    flattened.append(merged)
                                else:
                                    flattened.append(msg)
                        else:
                            flattened.append(item)
                    return flattened
                return v
        # fallback: pick the first nested list of dicts
        for v in data.values():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                return v
    # fallback: wrap single dict into list
    return [data]


def parse_timeline_file(path: str) -> List[Dict[str, Any]]:
    """Parse a timeline JSON file and return normalized events (canonical schema).

    The canonical event schema matches the output of the NDJSON extractor's normalize_event:
    {event_type, session_id, ts, seq, actor, text, model, tokens, tool, raw}
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    items = find_timeline_items(data)
    session_id = data.get('session_id') if isinstance(data, dict) else None
    events: List[Dict[str, Any]] = []
    seq = 0
    for raw in items:
        seq += 1
        if isinstance(raw, str):
            raw = {'content': raw}
        if isinstance(raw, dict):
            # normalize common shorthand keys
            if 'message' in raw and 'content' not in raw:
                raw['content'] = raw['message']
            if 'body' in raw and 'content' not in raw:
                raw['content'] = raw['body']
            if 'speaker' in raw and 'role' not in raw:
                raw['role'] = raw['speaker']
        ev = normalize_event(raw, seq, session_id=session_id)
        events.append(ev)
    return events


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Parse Copilot CLI timeline JSON into canonical event stream (NDJSON)')
    parser.add_argument('--input', '-i', required=True, help='Path to timeline JSON file')
    parser.add_argument('--output', '-o', help='Output NDJSON file (defaults to stdout)')
    args = parser.parse_args()
    events = parse_timeline_file(args.input)
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as out:
            for e in events:
                out.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f'Wrote {len(events)} events to {args.output}')
    else:
        for e in events:
            print(json.dumps(e, ensure_ascii=False))
