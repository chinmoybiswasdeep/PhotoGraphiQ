"""Self-contained HTML helpers: safe JSON embedding, pinned CDN scripts, notebook iframes."""

from __future__ import annotations

import html
import json
import math
import tempfile
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np

CYTOSCAPE_URL = "https://cdn.jsdelivr.net/npm/cytoscape@3.30.2/dist/cytoscape.min.js"
PLOTLY_URL = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"


def jsonable(value: Any) -> Any:
    """Convert tuples, sets, NumPy values and enums to JSON types; other objects to repr."""
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.generic, np.ndarray)):
        return jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in value]
    if hasattr(value, "value") and isinstance(getattr(value, "value"), str):  # enums
        return value.value
    return repr(value)


def script_json(value: Any) -> str:
    """JSON safe to place inside ``<script>``: no ``</script>`` or ``<!--`` sequences."""
    text = json.dumps(jsonable(value), allow_nan=False)
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def script_tag(url: str, local: str | Path | None) -> str:
    """CDN ``<script src>`` or, when ``local`` is a file path, the inlined library."""
    if local is None:
        return f'<script src="{url}"></script>'
    code = Path(local).read_text(encoding="utf-8").replace("</script", "<\\/script")
    return f"<script>{code}</script>"


def iframe(document: str, height: int) -> str:
    """Notebook display: ``srcdoc`` iframe so the page's scripts execute in isolation."""
    return (
        f'<iframe srcdoc="{html.escape(document, quote=True)}" '
        f'style="width:100%;height:{int(height)}px;border:0" loading="lazy"></iframe>'
    )


def write(document: str, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(document, encoding="utf-8")
    return path


def open_in_browser(document: str, stem: str) -> Path:
    """Write to a temporary ``.html`` file and open it in the default browser."""
    handle = tempfile.NamedTemporaryFile("w", suffix=".html", prefix=stem, delete=False)
    with handle:
        handle.write(document)
    webbrowser.open(Path(handle.name).as_uri())
    return Path(handle.name)
