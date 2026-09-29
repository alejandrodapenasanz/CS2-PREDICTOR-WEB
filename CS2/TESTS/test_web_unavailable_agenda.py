"""Execute the actual vanilla-JS table renderer for unavailable agenda rows."""

import json
from pathlib import Path
import shutil
import subprocess

import pytest


def test_unavailable_row_is_visible_without_fabricated_favorite_or_probability():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is only needed for the additional JS renderer check")
    html = (Path(__file__).resolve().parents[2] / "WEB" / "index.html").read_text(encoding="utf-8")
    helpers = "const hasPrediction" + html.split("const hasPrediction", 1)[1].split("function estimateView", 1)[0]
    renderer = "function renderRows()" + html.split("function renderRows()", 1)[1].split("// ---------- detail", 1)[0]
    row = {
        "id": "pending",
        "team1": {"name": "Alpha"},
        "team2": {"name": "TBD"},
        "prediction": {"status": "unavailable"},
    }
    harness = f"""
const row = {json.dumps(row)};
const elements = {{rows:{{innerHTML:''}}, rowCount:{{textContent:''}}}};
const $ = id => elements[id];
const state = {{selected:'pending'}};
const filtered = () => [row]; const renderKpis = () => {{}};
const updateSortArrows = () => {{}};
const esc = value => String(value ?? ''); const tn = (m,s) => m[s]?.name || 'TBD';
const sideName = (m,s) => tn(m,s);
const document = {{querySelectorAll: () => []}};
{helpers}
{renderer}
renderRows();
console.log(JSON.stringify({{html:elements.rows.innerHTML, confidence:decisionConfidence(row), favorite:decisionFavorite(row)}}));
"""
    completed = subprocess.run([node, "-e", harness], check=True, text=True, encoding="utf-8", capture_output=True)
    result = json.loads(completed.stdout)
    assert "Alpha VS TBD" in result["html"] and "Sin predicción" in result["html"]
    assert "50.0%" not in result["html"] and "100.0%" not in result["html"]
    assert result["confidence"] is None and result["favorite"] is None
