# browser_view.py
"""
Local HTTP server for the ATIF browser viewer.

Serves a single self-contained HTML page.  All CSS and JS is inlined;
no external network requests are made at runtime.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from replay_core import ATIFValidationError, Event, load, normalize


# ---------------------------------------------------------------------------
# JSON serialization helpers
# ---------------------------------------------------------------------------

def _js_safe_json(obj: Any) -> str:
    """
    Serialize obj to a JSON string safe for embedding inside a <script> tag.
    Escapes '</' to prevent early script-tag termination.
    """
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


def _esc_html(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ---------------------------------------------------------------------------
# HTML template — split at the data injection point
# ---------------------------------------------------------------------------

# Everything from DOCTYPE through the closing </style> and body markup.
# JS data constants are injected between _HTML_HEAD and _HTML_SCRIPT.
_HTML_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ATIF Replay \u00b7 SESSION_ID_PLACEHOLDER</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body {
  height: 100%;
  overflow: hidden;
}

body {
  display: flex;
  flex-direction: column;
  background: #1e1e1e;
  color: #d4d4d4;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  font-size: 14px;
  line-height: 1.5;
}

/* ── Header ──────────────────────────────────────────────── */
header {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 9px 16px;
  background: #252526;
  border-bottom: 1px solid #3e3e3e;
  flex-shrink: 0;
  min-height: 42px;
}

.brand {
  font-weight: 700;
  font-size: 13px;
  letter-spacing: 0.04em;
  color: #fff;
  white-space: nowrap;
}

.session-meta {
  font-size: 12px;
  color: #666;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.session-meta em { font-style: normal; color: #9e9e9e; }

.controls {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-left: auto;
  flex-shrink: 0;
}

#filter-input {
  background: #3c3c3c;
  border: 1px solid #555;
  color: #d4d4d4;
  padding: 4px 10px;
  border-radius: 4px;
  font-size: 12px;
  width: 150px;
  outline: none;
}
#filter-input:focus { border-color: #007acc; }
#filter-input::placeholder { color: #5a5a5a; }

.ctrl-btn {
  background: #3c3c3c;
  border: 1px solid #555;
  color: #9e9e9e;
  padding: 4px 11px;
  border-radius: 4px;
  font-size: 12px;
  cursor: pointer;
  white-space: nowrap;
  user-select: none;
  transition: background 0.1s, color 0.1s, border-color 0.1s;
}
.ctrl-btn:hover  { background: #4a4a4a; color: #d4d4d4; }
.ctrl-btn.on     { background: #0d3a5c; border-color: #007acc; color: #4fc3f7; }

/* ── Two-column layout ───────────────────────────────────── */
#app {
  display: grid;
  grid-template-columns: 230px 1fr;
  flex: 1;
  overflow: hidden;
  min-height: 0;
}

/* ── Timeline (left column) ──────────────────────────────── */
#timeline {
  background: #252526;
  border-right: 1px solid #3e3e3e;
  overflow-y: auto;
  padding: 4px 0;
}

.step-item {
  display: flex;
  align-items: baseline;
  gap: 7px;
  padding: 7px 10px 7px 13px;
  cursor: pointer;
  border-left: 3px solid transparent;
  user-select: none;
  min-width: 0;
}
.step-item:hover  { background: #2a2d2e; }
.step-item.active { background: #37373d; border-left-color: #007acc; }

.sn {
  font-family: 'SF Mono', 'Cascadia Code', Consolas, monospace;
  font-size: 11px;
  color: #5a5a5a;
  min-width: 18px;
  text-align: right;
  flex-shrink: 0;
}

.sr {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.06em;
  flex-shrink: 0;
  min-width: 48px;
}

.sp {
  font-size: 11px;
  color: #5a5a5a;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1;
  min-width: 0;
}

/* Role colours — shared across timeline and detail */
.r-user   { color: #4ec9b0; }
.r-agent  { color: #4fc1ff; }
.r-system { color: #dcdcaa; }
.r-tool   { color: #c586c0; }

/* ── Detail pane (right column) ──────────────────────────── */
#detail {
  overflow-y: auto;
  padding: 20px 28px;
}

.empty-state {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #3a3a3a;
  font-size: 14px;
}

.step-heading {
  display: flex;
  align-items: baseline;
  gap: 10px;
  margin-bottom: 18px;
  padding-bottom: 12px;
  border-bottom: 1px solid #2e2e2e;
}

.sh-num  { font-family: monospace; font-size: 13px; color: #4a4a4a; }
.sh-role { font-size: 17px; font-weight: 700; letter-spacing: 0.06em; }
.sh-ts   { font-size: 11px; color: #4a4a4a; font-family: monospace; }

/* message */
.ev-message { margin-bottom: 6px; }
.ev-body {
  white-space: pre-wrap;
  word-break: break-word;
  line-height: 1.7;
  color: #d4d4d4;
}

/* reasoning */
.ev-reasoning {
  display: none;
  margin: 14px 0;
  padding: 10px 14px;
  background: #1a1a1a;
  border-left: 3px solid #3a3a3a;
  border-radius: 0 4px 4px 0;
}
.ev-reasoning.visible { display: block; }
.ev-reasoning-label {
  font-size: 10px;
  font-weight: 700;
  color: #5a5a5a;
  letter-spacing: 0.1em;
  margin-bottom: 7px;
  text-transform: uppercase;
}
.ev-reasoning-body {
  font-size: 13px;
  color: #8a8a8a;
  white-space: pre-wrap;
  font-style: italic;
  line-height: 1.65;
}

/* tool call */
.ev-tool-call {
  margin: 12px 0 3px;
  padding: 8px 12px;
  background: #252526;
  border-radius: 6px;
  font-family: 'SF Mono', 'Cascadia Code', Consolas, monospace;
  font-size: 12px;
  border-left: 3px solid rgba(79, 193, 255, 0.25);
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 3px;
  word-break: break-all;
}
.tc-arrow { color: #4fc1ff; margin-right: 2px; }
.tc-kw    { color: #6a6a6a; }
.tc-name  { color: #4fc1ff; font-weight: 600; }
.tc-args  { color: #ce9178; }
.tc-ref   { color: #4a4a4a; font-size: 10px; margin-left: 6px; align-self: center; }

/* tool result */
.ev-tool-result {
  margin: 0 0 12px 18px;
  padding: 8px 12px;
  background: #1a1a1a;
  border: 1px solid #2e2e2e;
  border-radius: 0 0 6px 6px;
  font-family: 'SF Mono', 'Cascadia Code', Consolas, monospace;
  font-size: 12px;
  border-left: 3px solid rgba(197, 134, 192, 0.25);
}
.tr-header {
  font-size: 10px;
  font-weight: 700;
  color: #c586c0;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 7px;
}
.tr-body {
  color: #b5cea8;
  white-space: pre-wrap;
  word-break: break-word;
  line-height: 1.55;
}

/* metrics */
.ev-metrics {
  display: none;
  margin-top: 14px;
  padding: 6px 12px;
  background: #1a1a1a;
  border: 1px solid #2a2a2a;
  border-radius: 4px;
  font-family: 'SF Mono', 'Cascadia Code', Consolas, monospace;
  font-size: 11px;
  color: #5a5a5a;
}
.ev-metrics.visible { display: flex; align-items: center; gap: 0; flex-wrap: wrap; }
.ev-metrics-icon { margin-right: 6px; }
.ev-metrics-kv   { margin-right: 16px; }
.ev-metrics-key  { color: #5a5a5a; }
.ev-metrics-val  { color: #9e9e9e; }

/* ── Raw JSON modal ───────────────────────────────────────── */
#raw-modal {
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.65);
  z-index: 100;
  align-items: center;
  justify-content: center;
}
#raw-modal.open { display: flex; }

#raw-pane {
  background: #1e1e1e;
  border: 1px solid #4e4e4e;
  border-radius: 8px;
  width: 74vw;
  max-height: 78vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  box-shadow: 0 12px 40px rgba(0,0,0,0.6);
}

#raw-pane-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 16px;
  background: #252526;
  border-bottom: 1px solid #3e3e3e;
  font-size: 12px;
  font-weight: 600;
  color: #9e9e9e;
  letter-spacing: 0.04em;
  flex-shrink: 0;
}

#close-raw {
  background: none;
  border: none;
  color: #666;
  cursor: pointer;
  font-size: 18px;
  line-height: 1;
  padding: 0 2px;
  transition: color 0.1s;
}
#close-raw:hover { color: #d4d4d4; }

#raw-content {
  overflow: auto;
  padding: 16px 20px;
  font-family: 'SF Mono', 'Cascadia Code', Consolas, monospace;
  font-size: 12px;
  color: #d4d4d4;
  white-space: pre;
  flex: 1;
  line-height: 1.55;
}

/* ── Scrollbars ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #3e3e3e; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #5a5a5a; }
</style>
</head>
<body>

<header>
  <div class="brand">ATIF Replay</div>
  <div class="session-meta">
    session <em>SESSION_ID_PLACEHOLDER</em>
    &nbsp;&middot;&nbsp;
    agent <em>AGENT_NAME_PLACEHOLDER</em>
    &nbsp;&middot;&nbsp;
    <em>STEP_COUNT_PLACEHOLDER steps</em>
  </div>
  <div class="controls">
    <input id="filter-input" type="text" placeholder="Filter steps&hellip;" autocomplete="off">
    <button id="btn-reasoning" class="ctrl-btn"       title="Toggle reasoning blocks">Show Reasoning</button>
    <button id="btn-metrics"   class="ctrl-btn on"    title="Toggle metrics blocks">Show Metrics</button>
    <button id="btn-raw"       class="ctrl-btn"       title="View raw ATIF JSON">Raw JSON</button>
  </div>
</header>

<div id="app">
  <aside id="timeline"></aside>
  <main  id="detail"><div class="empty-state">Select a step</div></main>
</div>

<div id="raw-modal">
  <div id="raw-pane">
    <div id="raw-pane-header">
      <span>Raw ATIF JSON &mdash; SESSION_ID_PLACEHOLDER</span>
      <button id="close-raw" title="Close">&times;</button>
    </div>
    <pre id="raw-content"></pre>
  </div>
</div>

<script>
/* Data injected by Python — do not edit the constants below by hand. */
"""

# Pure JS logic — no data references until EVENTS and RAW_DOC are declared above.
_HTML_SCRIPT = """\
(function () {
  'use strict';

  /* ── Index ───────────────────────────────────────────────── */
  const stepMap   = new Map();  // step_id → Event[]
  const stepOrder = [];         // step_ids in document order

  for (const ev of EVENTS) {
    if (!stepMap.has(ev.step_id)) {
      stepMap.set(ev.step_id, []);
      stepOrder.push(ev.step_id);
    }
    stepMap.get(ev.step_id).push(ev);
  }

  /* ── Application state ────────────────────────────────────── */
  let selectedStepId = stepOrder.length > 0 ? stepOrder[0] : null;
  let showReasoning  = false;
  let showMetrics    = true;
  let filterText     = '';

  /* ── Lookup tables ────────────────────────────────────────── */
  const ROLE_LABEL = { user: 'USER', agent: 'AGENT', system: 'SYSTEM', tool: 'TOOL' };
  const ROLE_CLASS = { user: 'r-user', agent: 'r-agent', system: 'r-system', tool: 'r-tool' };

  /* ── Utilities ────────────────────────────────────────────── */
  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function primaryRole(events) {
    for (const ev of events) {
      if (ev.kind === 'message') return ev.role;
    }
    return (events[0] && events[0].role) || 'agent';
  }

  function stepPreview(events) {
    for (const ev of events) {
      if (ev.kind === 'message' && ev.body != null) {
        const t = String(ev.body);
        return t.length > 36 ? t.slice(0, 36) + '\u2026' : t;
      }
    }
    for (const ev of events) {
      if (ev.kind === 'tool_call') return 'tool: ' + (ev.title || '?') + '(\u2026)';
    }
    return '';
  }

  function primaryTs(events) {
    for (const ev of events) {
      if (ev.ts) return ev.ts.replace('T', ' ').replace('Z', '').slice(0, 19);
    }
    return null;
  }

  /* ── Render: timeline ─────────────────────────────────────── */
  function renderTimeline() {
    const f  = filterText.toLowerCase();
    let html = '';

    for (const sid of stepOrder) {
      const events  = stepMap.get(sid);
      const role    = primaryRole(events);
      const preview = stepPreview(events);

      if (f && !String(sid).includes(f) && !preview.toLowerCase().includes(f)) continue;

      const active = sid === selectedStepId ? ' active' : '';
      const rc     = ROLE_CLASS[role] || 'r-agent';
      const rl     = ROLE_LABEL[role] || role.toUpperCase();

      html +=
        '<div class="step-item' + active + '" data-sid="' + sid + '">' +
        '<span class="sn">' + sid + '</span>' +
        '<span class="sr ' + rc + '">' + rl + '</span>' +
        '<span class="sp">' + esc(preview) + '</span>' +
        '</div>';
    }

    const tl = document.getElementById('timeline');
    tl.innerHTML = html;

    tl.querySelectorAll('.step-item').forEach(function (el) {
      el.addEventListener('click', function () {
        selectedStepId = parseInt(el.dataset.sid, 10);
        renderTimeline();
        renderDetail();
        scrollTimelineToActive();
      });
    });
  }

  function scrollTimelineToActive() {
    const el = document.querySelector('.step-item.active');
    if (el) el.scrollIntoView({ block: 'nearest' });
  }

  /* ── Render: detail ───────────────────────────────────────── */
  function renderDetail() {
    const detail = document.getElementById('detail');

    if (selectedStepId === null) {
      detail.innerHTML = '<div class="empty-state">Select a step</div>';
      return;
    }

    const events = stepMap.get(selectedStepId) || [];
    const role   = primaryRole(events);
    const rc     = ROLE_CLASS[role] || 'r-agent';
    const rl     = ROLE_LABEL[role] || role.toUpperCase();
    const ts     = primaryTs(events);

    let html =
      '<div class="step-heading">' +
      '<span class="sh-num">[' + selectedStepId + ']</span>' +
      '<span class="sh-role ' + rc + '">' + rl + '</span>' +
      (ts ? '<span class="sh-ts">' + esc(ts) + '</span>' : '') +
      '</div>';

    for (const ev of events) {
      if (ev.kind === 'message') {
        const body = typeof ev.body === 'string' ? ev.body : JSON.stringify(ev.body, null, 2);
        html +=
          '<div class="ev-message">' +
          '<div class="ev-body">' + esc(body) + '</div>' +
          '</div>';

      } else if (ev.kind === 'reasoning') {
        const body = typeof ev.body === 'string' ? ev.body : JSON.stringify(ev.body, null, 2);
        const vis  = showReasoning ? ' visible' : '';
        html +=
          '<div class="ev-reasoning' + vis + '">' +
          '<div class="ev-reasoning-label">\u25c6 Reasoning</div>' +
          '<div class="ev-reasoning-body">' + esc(body) + '</div>' +
          '</div>';

      } else if (ev.kind === 'tool_call') {
        const args   = ev.body != null ? JSON.stringify(ev.body) : '';
        const refHtml = ev.ref
          ? '<span class="tc-ref">#' + esc(ev.ref) + '</span>'
          : '';
        html +=
          '<div class="ev-tool-call">' +
          '<span class="tc-arrow">\u25b6</span>' +
          '<span class="tc-kw">tool:</span>' +
          '<span class="tc-name">' + esc(ev.title || '?') + '</span>' +
          '<span class="tc-args">(' + esc(args) + ')</span>' +
          refHtml +
          '</div>';

      } else if (ev.kind === 'tool_result') {
        const content = typeof ev.body === 'string'
          ? ev.body
          : JSON.stringify(ev.body, null, 2);
        const refHtml = ev.ref
          ? ' <span class="tc-ref">#' + esc(ev.ref) + '</span>'
          : '';
        html +=
          '<div class="ev-tool-result">' +
          '<div class="tr-header">\u25c4 Result' + refHtml + '</div>' +
          '<div class="tr-body">' + esc(content) + '</div>' +
          '</div>';

      } else if (ev.kind === 'metrics') {
        const m = ev.body || {};
        const pairs = [];
        if (m.prompt_tokens     != null) pairs.push(['prompt',     m.prompt_tokens]);
        if (m.completion_tokens != null) pairs.push(['completion', m.completion_tokens]);
        if (m.cost_usd          != null) pairs.push(['cost',       '$' + m.cost_usd.toFixed(4)]);
        const skip = new Set(['prompt_tokens', 'completion_tokens', 'cost_usd']);
        for (const k in m) {
          if (!skip.has(k)) pairs.push([k, m[k]]);
        }
        const vis    = showMetrics ? ' visible' : '';
        const kvHtml = pairs.map(function (p) {
          return '<span class="ev-metrics-kv">' +
                 '<span class="ev-metrics-key">' + esc(p[0]) + '=</span>' +
                 '<span class="ev-metrics-val">' + esc(String(p[1])) + '</span>' +
                 '</span>';
        }).join('');
        html +=
          '<div class="ev-metrics' + vis + '">' +
          '<span class="ev-metrics-icon">\u2b21</span>' +
          '<span class="ev-metrics-key" style="margin-right:12px">metrics</span>' +
          kvHtml +
          '</div>';
      }
    }

    detail.innerHTML = html;
  }

  /* ── Controls ─────────────────────────────────────────────── */
  document.getElementById('filter-input').addEventListener('input', function () {
    filterText = this.value;
    renderTimeline();
  });

  document.getElementById('btn-reasoning').addEventListener('click', function () {
    showReasoning = !showReasoning;
    this.classList.toggle('on', showReasoning);
    renderDetail();
  });

  document.getElementById('btn-metrics').addEventListener('click', function () {
    showMetrics = !showMetrics;
    this.classList.toggle('on', showMetrics);
    renderDetail();
  });

  document.getElementById('btn-raw').addEventListener('click', function () {
    document.getElementById('raw-content').textContent =
      JSON.stringify(RAW_DOC, null, 2);
    document.getElementById('raw-modal').classList.add('open');
  });

  document.getElementById('close-raw').addEventListener('click', function () {
    document.getElementById('raw-modal').classList.remove('open');
  });

  document.getElementById('raw-modal').addEventListener('click', function (e) {
    if (e.target === this) this.classList.remove('open');
  });

  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') {
      document.getElementById('raw-modal').classList.remove('open');
    }
  });

  /* ── Keyboard step navigation ─────────────────────────────── */
  document.addEventListener('keydown', function (e) {
    if (document.getElementById('raw-modal').classList.contains('open')) return;
    if (document.activeElement === document.getElementById('filter-input')) return;

    const idx = stepOrder.indexOf(selectedStepId);
    if (e.key === 'ArrowDown' || e.key === 'j') {
      if (idx < stepOrder.length - 1) {
        selectedStepId = stepOrder[idx + 1];
        renderTimeline();
        renderDetail();
        scrollTimelineToActive();
      }
    } else if (e.key === 'ArrowUp' || e.key === 'k') {
      if (idx > 0) {
        selectedStepId = stepOrder[idx - 1];
        renderTimeline();
        renderDetail();
        scrollTimelineToActive();
      }
    }
  });

  /* ── Bootstrap ────────────────────────────────────────────── */
  renderTimeline();
  renderDetail();
  scrollTimelineToActive();
}());
"""

_HTML_TAIL = """\
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

def _build_html(doc: dict, events: list[Event]) -> str:
    """
    Produce a fully self-contained HTML string by injecting event data and
    session metadata into the page template.

    The data constants (EVENTS, RAW_DOC) are injected as literal JSON between
    the CSS/markup block and the JS logic block, avoiding any placeholder
    collision with the embedded data content.
    """
    session_id = _esc_html(str(doc.get("session_id", "unknown")))
    agent_name = _esc_html(str((doc.get("agent") or {}).get("name", "unknown")))
    step_count = str(len(set(ev.step_id for ev in events)))

    head = (
        _HTML_HEAD
        .replace("SESSION_ID_PLACEHOLDER", session_id)
        .replace("AGENT_NAME_PLACEHOLDER", agent_name)
        .replace("STEP_COUNT_PLACEHOLDER", step_count)
    )

    events_json = _js_safe_json([dataclasses.asdict(ev) for ev in events])
    raw_json    = _js_safe_json(doc)

    data_block = (
        "const EVENTS  = " + events_json + ";\n"
        "const RAW_DOC = " + raw_json    + ";\n\n"
    )

    return head + data_block + _HTML_SCRIPT + _HTML_TAIL


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

def run_serve(path: str, port: int) -> None:
    """
    Load an ATIF file, build the self-contained viewer page, and serve it on
    localhost:{port}.  Opens the default browser automatically if possible.
    """
    try:
        doc = load(path)
    except (ATIFValidationError, Exception) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    events = normalize(doc)
    html   = _build_html(doc, events)
    body   = html.encode("utf-8")

    session_id = doc.get("session_id", "?")
    agent_name = (doc.get("agent") or {}).get("name", "?")
    step_count = len(set(ev.step_id for ev in events))

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
            pass  # suppress default per-request log noise

    url = f"http://localhost:{port}/"

    print(f"ATIF Replay  \u00b7  session: {session_id}  \u00b7  "
          f"agent: {agent_name}  \u00b7  {step_count} steps")
    print(f"Serving at {url}")
    print("Press Ctrl-C to stop.")

    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass

    server = HTTPServer(("127.0.0.1", port), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
