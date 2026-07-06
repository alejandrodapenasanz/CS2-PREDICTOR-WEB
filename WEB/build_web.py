from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
DAILY_ROOT = ROOT / "DAILY_SNAPSHOTS"
MODEL_ROOT = ROOT / "MODEL"
WEB_ROOT = ROOT / "WEB"


def read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def latest_run() -> Path:
    manifest = read_json(DAILY_ROOT / "master" / "manifest.json", {})
    if manifest.get("last_run_id"):
        return DAILY_ROOT / "runs" / manifest["last_run_id"]
    runs = sorted([path for path in (DAILY_ROOT / "runs").iterdir() if path.is_dir()])
    if not runs:
        raise FileNotFoundError("No DAILY_SNAPSHOTS runs found.")
    return runs[-1]


def model_payload() -> dict:
    """Métricas del modelo (walk-forward), SHAP y metadatos para el panel del modelo."""
    metrics = read_json(MODEL_ROOT / "results" / "metrics.json", {})
    shap = read_json(MODEL_ROOT / "results" / "shap_importance.json", [])
    seg = read_json(MODEL_ROOT / "results" / "segmented_eval.json", {})
    return {
        "metrics": metrics,
        "shap_top": shap[:15] if isinstance(shap, list) else [],
        "segments": seg.get("segments", []),
        "market": seg.get("market", {}),
        "production_model": _production_model(),
    }


def _production_model() -> str:
    """Lee el modelo de producción del artefacto si está disponible (import perezoso)."""
    try:
        import sys

        sys.path.insert(0, str(MODEL_ROOT))
        from cs2model.artifacts import load_artifact

        art = load_artifact()
        if art:
            return str(art.metadata.get("production_model", "")) + " · " + str(art.metadata.get("model", ""))
    except Exception:
        pass
    return ""


def parse_match_datetime(date_text: str | None, hour_text: str | None = None) -> datetime | None:
    if not date_text:
        return None
    try:
        date_obj = datetime.strptime(str(date_text), "%Y-%m-%d")
    except ValueError:
        return None
    if hour_text:
        match = re.search(r"(\d{1,2}):(\d{2})", str(hour_text))
        if match:
            return date_obj.replace(hour=int(match.group(1)), minute=int(match.group(2)))
    return date_obj.replace(hour=12, minute=0)


def publishable_matches(matches: list[dict], grace_minutes: int = 15) -> tuple[list[dict], dict]:
    now_local = datetime.now(ZoneInfo("Europe/Madrid")).replace(tzinfo=None)
    cutoff = now_local - timedelta(minutes=grace_minutes)
    kept = []
    skipped = {"past_or_started": 0, "completed": 0}
    for match in matches:
        if match.get("status") == "completed":
            skipped["completed"] += 1
            continue
        match_dt = parse_match_datetime(match.get("date"), match.get("hour"))
        if match_dt and match_dt < cutoff:
            skipped["past_or_started"] += 1
            continue
        kept.append(match)
    return kept, {key: value for key, value in skipped.items() if value}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build local static web dashboard.")
    parser.add_argument("--run-dir", default="")
    args = parser.parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else latest_run()
    raw_data = read_json(run_dir / "predictions_enriched.json", [])
    data, skipped_web = publishable_matches(raw_data)
    manifest = read_json(run_dir / "manifest.json", {})
    master_manifest = read_json(DAILY_ROOT / "master" / "manifest.json", {})
    calibration = read_json(run_dir / "calibration.json", {})

    payload = {
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runDir": str(run_dir),
        "manifest": manifest,
        "masterManifest": master_manifest,
        "calibration": calibration,
        "model": model_payload(),
        "webFilter": {
            "sourceMatches": len(raw_data),
            "publishedMatches": len(data),
            "skipped": skipped_web,
            "timezone": "Europe/Madrid",
        },
        "matches": data,
    }
    WEB_ROOT.mkdir(parents=True, exist_ok=True)
    (WEB_ROOT / "data.js").write_text(
        "window.__CS2_PREDICTOR_DATA__ = " + json.dumps(payload, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )
    model_label = str(payload["model"].get("production_model") or "").encode("ascii", "replace").decode("ascii")
    print(f"Built {WEB_ROOT / 'data.js'} with {len(data)} matches - model={model_label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
