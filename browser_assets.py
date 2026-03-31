"""Browser viewer static asset loading."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_ASSET_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def load_browser_asset(filename: str) -> str:
    return (_ASSET_DIR / filename).read_text(encoding="utf-8")


BROWSER_CSS = load_browser_asset("browser_styles.css")
BROWSER_JS = load_browser_asset("browser_app.js")


__all__ = ["BROWSER_CSS", "BROWSER_JS", "load_browser_asset"]
