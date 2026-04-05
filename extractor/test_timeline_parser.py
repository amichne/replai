#!/usr/bin/env python3
import os
import sys
from copilot_timeline_parser import parse_timeline_file


def main():
    base = os.path.dirname(__file__)
    sample = os.path.join(base, "sample_copilot_timeline.json")
    if not os.path.exists(sample):
        print("Sample file not found:", sample)
        sys.exit(2)
    events = parse_timeline_file(sample)
    print("Parsed", len(events), "events")
    # Basic sanity checks
    if len(events) < 10:
        print("FAIL: expected at least 10 events")
        sys.exit(1)
    # ensure there's a user message about debug logs
    found = any((e.get('actor') == 'user' and e.get('text') and 'debug logs' in e.get('text')) for e in events)
    if not found:
        print("FAIL: did not find user message about debug logs")
        print("Sample event texts:")
        for e in events:
            print(" -", e.get('actor'), e.get('text')[:80] if e.get('text') else '')
        sys.exit(1)
    # ensure a tool call normalized
    tool_found = any(e.get('event_type') == 'tool_call' or e.get('actor') == 'tool' for e in events)
    if not tool_found:
        print("FAIL: did not find tool call event")
        sys.exit(1)

    # Print first few canonical events for inspection
    for i, e in enumerate(events[:5], start=1):
        print(f"Event {i}: seq={e.get('seq')} actor={e.get('actor')} type={e.get('event_type')} ts={e.get('ts')}")
    print("All checks passed.")
    sys.exit(0)


if __name__ == '__main__':
    main()
