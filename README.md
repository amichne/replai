# replai

Portable local replayer for ATIF-v1.5 agent session files and Copilot debug logs.

## Install

```bash
pip install -r requirements.txt   # only Jinja2 is required
```

Python 3.10+ required. No network access needed.

## Usage

### Open the latest Copilot session in the browser

```bash
python replai.py serve
```

With no arguments, `serve` (and `html`/`play`) automatically discover the most recently
modified `main.jsonl` under your VS Code workspace storage directory. On macOS this is
`~/Library/Application Support/Code/User/workspaceStorage`.

### Open a specific session directory or file

```bash
python replai.py serve /path/to/session_dir          # directory containing main.jsonl
python replai.py serve /path/to/session/main.jsonl   # explicit file
python replai.py serve /path/to/trajectory.json      # ATIF-v1.5 JSON
```

### Export static HTML

```bash
python replai.py html /path/to/session_dir --output /tmp/out.html
python replai.py html /path/to/session_dir --output /tmp/out.html --open
```

Without `--output` the HTML is written to stdout.

### Terminal playback

```bash
python replai.py play /path/to/session_dir
python replai.py play /path/to/session_dir --show-reasoning --speed 2.0
python replai.py play /path/to/session_dir --tool-visibility hidden
```

## Flags

| Flag | Commands | Description |
|---|---|---|
| `--no-latest` | all | Skip workspaceStorage auto-discovery and require an explicit path or interactive selection |
| `--output PATH` | serve, html | Write HTML to this file instead of a temp file / stdout |
| `--open` | html | Open the generated HTML file in the default browser |
| `--http` | serve | Use a localhost HTTP server instead of a static file |
| `--port N` | serve | HTTP port when `--http` is used (default: 8000) |
| `--show-reasoning` | play | Render reasoning_content blocks |
| `--speed FACTOR` | play | Playback speed multiplier (default: 1.0) |
| `--tool-visibility` | play | `full`, `minimal` (default), or `hidden` |

### `--no-latest`

By default, when no file argument is given, replai picks the most recently modified
Copilot debug-log session from your workspaceStorage. Pass `--no-latest` to skip this
and fall back to searching the current directory for ATIF-style JSON files (or an
interactive prompt when multiple are found).

```bash
python replai.py serve --no-latest                  # interactive / scans cwd
python replai.py serve --no-latest trajectory.json  # explicit, no auto-pick
```

## Supported input formats

| Format | Detection |
|---|---|
| ATIF-v1.5 JSON | `.json` / `.atif` / `.atif.json` with `schema_version: "1.5"` |
| Copilot debug log | directory containing `main.jsonl`, or a `.jsonl` file directly |
| OpenAI messages array | JSON with top-level `messages` array (auto-converted) |

## Smoke test

```bash
python -m pytest -q
python replai.py html tests/fixtures/copilot_debug_logs/full_session --output /tmp/replai-smoke.html
python replai.py html tests/fixtures/copilot_debug_logs/sample_session --output /tmp/replai-sample.html
python replai.py serve   # opens latest live session
```
