"""Evalua si el contexto HLTV puede ayudar a calibrar el modelo.

No entrena ni promueve ningun artefacto productivo. Usa las predicciones
pre-partido ya cerradas, recupera el bloque `Maps` desde snapshots/assets y
compara la calibracion del modelo por subgrupos.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL_ROOT = ROOT / "MODEL"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from DAILY_SNAPSHOTS.match_context import parse_match_context_meta
from cs2model.metrics import metric_dict

DEFAULT_MASTER = ROOT / "DAILY_SNAPSHOTS" / "master" / "matches.json"
DEFAULT_RUNS = ROOT / "DAILY_SNAPSHOTS" / "runs"
DEFAULT_OUT = ROOT / "MODEL" / "results"
ROOT_REPORT_MD = ROOT / "context_calibration.md"
ROOT_REPORT_JSON = ROOT / "context_calibration.json"
PRODUCTION_MIN_SAMPLES = 200
PRODUCTION_MIN_GROUP_SAMPLES = 30
DIAGNOSTIC_MIN_TRAIN = 40
STAGE_COLUMNS = ["group", "swiss", "ro16", "quarter", "semi", "final", "other"]


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def clamp(value: float, low: float = 1e-6, high: float = 1 - 1e-6) -> float:
    return min(max(float(value), low), high)


def logit(value: float) -> float:
    value = clamp(value)
    return math.log(value / (1.0 - value))


def latest_predictions(runs_dir: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for path in sorted(runs_dir.glob("*/predictions_enriched.json")):
        run_id = path.parent.name
        for entry in read_json(path, []):
            if not (entry.get("data_quality") or {}).get("real_pre_match_snapshot", True):
                continue
            match_id = str(entry.get("id") or "")
            if not match_id:
                continue
            entry = {**entry, "run_id": run_id}
            marker = (entry.get("captured_at") or "", run_id)
            current = latest.get(match_id)
            current_marker = ((current or {}).get("captured_at") or "", (current or {}).get("run_id") or "")
            if current is None or marker > current_marker:
                latest[match_id] = entry
    return latest


def context_from_record(record: dict[str, Any]) -> dict[str, Any]:
    # Prefer re-parsing raw meta so parser fixes apply to old assets.
    meta = None
    source_file = ((record.get("hltv_assets") or {}).get("source_file") or record.get("hltv_assets_file"))
    if source_file:
        path = ROOT / "DAILY_SNAPSHOTS" / source_file
        asset = read_json(path, {}) if path.exists() else {}
        meta = ((asset.get("veto") or {}).get("meta") if isinstance(asset, dict) else None)
    if meta:
        return parse_match_context_meta(meta)
    context = record.get("match_context")
    if isinstance(context, dict) and context:
        raw_meta = context.get("raw_meta")
        return parse_match_context_meta(raw_meta) if raw_meta else context
    return {}


def context_from_prediction(entry: dict[str, Any]) -> dict[str, Any]:
    return ((entry.get("controls") or {}).get("tournament_context") or {})


def build_samples(master: dict[str, Any], predictions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for match_id, entry in predictions.items():
        record = master.get(str(match_id), {})
        if record.get("status") != "completed" or not record.get("score"):
            continue
        pred = entry.get("prediction") or {}
        p = pred.get("model_prob_team1")
        if p is None:
            continue
        score = record.get("score") or {}
        try:
            y = 1 if int(score.get("team1", 0)) > int(score.get("team2", 0)) else 0
        except (TypeError, ValueError):
            continue
        context = context_from_record(record) or context_from_prediction(entry)
        context = context if isinstance(context, dict) else {}
        p = clamp(float(p))
        favorite_side = "team1" if p >= 0.5 else "team2"
        favorite_correct = int((p >= 0.5) == bool(y))
        samples.append(
            {
                "match_id": match_id,
                "captured_at": entry.get("captured_at"),
                "completed_at": record.get("completed_at"),
                "event": record.get("event") or entry.get("event"),
                "format": record.get("format") or entry.get("format"),
                "team1": ((entry.get("team1") or {}).get("name") or ""),
                "team2": ((entry.get("team2") or {}).get("name") or ""),
                "y_team1": y,
                "p_team1": p,
                "confidence": max(p, 1.0 - p),
                "favorite_side": favorite_side,
                "favorite_correct": favorite_correct,
                "context": context,
                "environment": context.get("environment") or "unknown",
                "stage": context.get("stage") or "unknown",
                "high_stakes": bool(context.get("high_stakes")),
                "winner_advances": bool(context.get("winner_advances")),
                "loser_eliminated": bool(context.get("loser_eliminated")),
                "incentive_label": context.get("incentive_label") or "unknown",
                "has_substitution_note": bool(context.get("has_substitution_note")),
            }
        )
    samples.sort(key=lambda row: (row.get("completed_at") or row.get("captured_at") or "", row["match_id"]))
    return samples


def active_context_rows(master: dict[str, Any], predictions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match_id, entry in predictions.items():
        record = master.get(str(match_id), {})
        if record.get("status") == "completed":
            continue
        context = context_from_prediction(entry) or context_from_record(record)
        context = context if isinstance(context, dict) else {}
        team1 = ((entry.get("team1") or {}).get("name") or (record.get("team1") or {}).get("name") or "")
        team2 = ((entry.get("team2") or {}).get("name") or (record.get("team2") or {}).get("name") or "")
        prediction = entry.get("prediction") or {}
        rows.append(
            {
                "match_id": str(match_id),
                "date": entry.get("date") or record.get("date"),
                "hour": entry.get("hour") or record.get("hour"),
                "team1": team1,
                "team2": team2,
                "event": entry.get("event") or record.get("event"),
                "format": entry.get("format") or record.get("format"),
                "environment": context.get("environment") or "unknown",
                "stage": context.get("stage") or "unknown",
                "stage_detail": context.get("stage_detail"),
                "incentive_label": context.get("incentive_label"),
                "source": context.get("source") or ("hltv_maps_box" if context.get("raw_meta") else "unknown"),
                "favorite": prediction.get("decision_favorite") or prediction.get("favorite"),
                "confidence": prediction.get("decision_confidence") or prediction.get("confidence"),
                "hltv_url": "https://www.hltv.org" + (entry.get("link") or record.get("link") or ""),
            }
        )
    rows.sort(key=lambda row: (str(row.get("date") or ""), str(row.get("hour") or ""), row["match_id"]))
    return rows


def metrics_for(samples: list[dict[str, Any]], prob_key: str = "p_team1", y_key: str = "y_team1") -> dict[str, Any]:
    if not samples:
        return {"n": 0}
    y = np.array([int(row[y_key]) for row in samples], dtype=int)
    p = np.array([float(row[prob_key]) for row in samples], dtype=float)
    metrics = metric_dict(y, p)
    metrics["avg_pred"] = float(np.mean(p))
    metrics["observed"] = float(np.mean(y))
    metrics["calibration_gap"] = metrics["avg_pred"] - metrics["observed"]
    metrics["avg_confidence"] = float(np.mean([max(float(row[prob_key]), 1.0 - float(row[prob_key])) for row in samples]))
    return round_floats(metrics)


def favorite_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {"n": 0}
    y = np.array([int(row["favorite_correct"]) for row in samples], dtype=int)
    p = np.array([float(row["confidence"]) for row in samples], dtype=float)
    metrics = metric_dict(y, p)
    metrics["avg_confidence"] = float(np.mean(p))
    metrics["favorite_accuracy"] = float(np.mean(y))
    metrics["confidence_gap"] = metrics["avg_confidence"] - metrics["favorite_accuracy"]
    return round_floats(metrics)


def round_floats(payload: Any) -> Any:
    if isinstance(payload, float):
        if math.isnan(payload) or math.isinf(payload):
            return None
        return round(payload, 6)
    if isinstance(payload, dict):
        return {key: round_floats(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [round_floats(value) for value in payload]
    return payload


def grouped(samples: list[dict[str, Any]], name: str, key_func: Callable[[dict[str, Any]], str]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in samples:
        buckets[key_func(row)].append(row)
    rows = []
    for key, values in sorted(buckets.items(), key=lambda item: (-len(item[1]), item[0])):
        item = {
            "segment": name,
            "group": key,
            "model": metrics_for(values),
            "favorite_confidence": favorite_metrics(values),
        }
        rows.append(item)
    return rows


def context_feature_row(sample: dict[str, Any]) -> list[float]:
    stage = sample.get("stage") or "unknown"
    env = sample.get("environment") or "unknown"
    return [
        logit(sample["confidence"]),
        1.0 if env == "lan" else 0.0,
        1.0 if env == "online" else 0.0,
        1.0 if sample.get("high_stakes") else 0.0,
        1.0 if sample.get("winner_advances") else 0.0,
        1.0 if sample.get("loser_eliminated") else 0.0,
        1.0 if sample.get("has_substitution_note") else 0.0,
        *[1.0 if stage == col else 0.0 for col in STAGE_COLUMNS],
    ]


def context_calibrator_walk_forward(samples: list[dict[str, Any]], min_train: int = DIAGNOSTIC_MIN_TRAIN) -> dict[str, Any]:
    """Calibrador diagnostico de confianza del favorito con contexto.

    Entrena solo con partidos anteriores y predice el siguiente bloque. El output
    NO se usa en produccion; sirve para ver si hay senal.
    """

    if len(samples) <= min_train:
        return {
            "available": False,
            "n_samples": len(samples),
            "min_train": min_train,
            "note": "No hay muestra suficiente ni para diagnostico walk-forward.",
        }
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        return {"available": False, "n_samples": len(samples), "error": str(exc)}

    eval_rows = []
    feature_names = [
        "logit_model_confidence",
        "env_lan",
        "env_online",
        "high_stakes",
        "winner_advances",
        "loser_eliminated",
        "has_substitution_note",
        *[f"stage_{stage}" for stage in STAGE_COLUMNS],
    ]
    for index in range(min_train, len(samples)):
        train = samples[:index]
        test = samples[index]
        y_train = np.array([row["favorite_correct"] for row in train], dtype=int)
        if len(set(y_train.tolist())) < 2:
            continue
        X_train = np.array([context_feature_row(row) for row in train], dtype=float)
        X_test = np.array([context_feature_row(test)], dtype=float)
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("logistic", LogisticRegression(C=0.35, max_iter=1000)),
            ]
        )
        model.fit(X_train, y_train)
        calibrated_conf = clamp(float(model.predict_proba(X_test)[0, 1]))
        # Calibration layer should not flip sides until validated with a large sample.
        calibrated_conf = max(0.5, calibrated_conf)
        p_context = calibrated_conf if test["favorite_side"] == "team1" else 1.0 - calibrated_conf
        eval_rows.append({**test, "p_context_calibrated": p_context})

    if not eval_rows:
        return {"available": False, "n_samples": len(samples), "min_train": min_train, "note": "Sin folds validos."}

    base = metrics_for(eval_rows, "p_team1")
    context = metrics_for(eval_rows, "p_context_calibrated")
    return {
        "available": True,
        "n_samples": len(samples),
        "n_eval": len(eval_rows),
        "min_train": min_train,
        "feature_names": feature_names,
        "base_model": base,
        "context_calibrated": context,
        "delta_log_loss": round(context["log_loss"] - base["log_loss"], 6),
        "delta_brier": round(context["brier"] - base["brier"], 6),
        "note": "Diagnostico. No activar en produccion sin 200+ muestras y grupos con 30+ partidos.",
    }


def markdown_report(report: dict[str, Any]) -> str:
    def fmt(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = [
        "# Context Calibration Diagnostic\n\n",
        f"Generado: {report['generated_at']}  \n",
        f"Muestras cerradas con prediccion pre-partido: **{report['sample_counts']['total']}**  \n",
        f"Muestras con contexto HLTV: **{report['sample_counts']['with_context']}**  \n\n",
        "## Resultado\n\n",
        f"- Estado: **{report['recommendation']['status']}**\n",
        f"- Decision: {report['recommendation']['decision']}\n",
        f"- Motivo: {report['recommendation']['reason']}\n\n",
        "## Global\n\n",
        "| Metrica | Modelo |\n|---|---:|\n",
    ]
    for key in ("n", "accuracy", "log_loss", "brier", "ece_10", "avg_confidence"):
        lines.append(f"| {key} | {fmt(report['overall_model'].get(key))} |\n")

    active_rows = report.get("active_matches") or []
    lines.append("\n## Partidos Activos Por LAN/Online\n\n")
    if active_rows:
        lines.append("| Hora | Partido | Env | Stage | Favorito | Conf | HLTV |\n|---|---|---|---|---|---:|---|\n")
        for row in active_rows:
            match_name = f"{row.get('team1') or '?'} vs {row.get('team2') or '?'}"
            when = f"{row.get('date') or '?'} {row.get('hour') or '?'}"
            link = f"[link]({row.get('hltv_url')})" if row.get("hltv_url") else "-"
            lines.append(
                f"| {when} | {match_name} | {row.get('environment')} | "
                f"{row.get('stage_detail') or row.get('stage')} | {row.get('favorite') or '-'} | "
                f"{fmt(row.get('confidence'))} | {link} |\n"
            )
    else:
        lines.append("No hay predicciones activas pendientes en el ultimo snapshot.\n")

    wf = report["context_calibrator_walk_forward"]
    lines.append("\n## Calibrador Contextual Diagnostico\n\n")
    if wf.get("available"):
        lines.append("| Metrica | Base | Contexto | Delta |\n|---|---:|---:|---:|\n")
        for key in ("log_loss", "brier", "accuracy", "ece_10"):
            base = wf["base_model"].get(key)
            ctx = wf["context_calibrated"].get(key)
            delta = (ctx - base) if isinstance(base, (int, float)) and isinstance(ctx, (int, float)) else None
            lines.append(f"| {key} | {fmt(base)} | {fmt(ctx)} | {fmt(delta)} |\n")
        lines.append(f"\nEval folds: {wf.get('n_eval')} con min_train={wf.get('min_train')}. {wf.get('note')}\n")
    else:
        lines.append(f"{wf.get('note') or wf.get('error')}\n")

    for segment_name, rows in report["segments"].items():
        lines.append(f"\n## {segment_name}\n\n")
        lines.append("| Grupo | n | acc | log_loss | brier | fav_acc | conf_gap |\n|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            model = row["model"]
            fav = row["favorite_confidence"]
            lines.append(
                f"| {row['group']} | {model.get('n', 0)} | {fmt(model.get('accuracy'))} | "
                f"{fmt(model.get('log_loss'))} | {fmt(model.get('brier'))} | "
                f"{fmt(fav.get('favorite_accuracy'))} | {fmt(fav.get('confidence_gap'))} |\n"
            )
    return "".join(lines)


def build_report(master_path: Path, runs_dir: Path) -> dict[str, Any]:
    master = read_json(master_path, {})
    predictions = latest_predictions(runs_dir)
    samples = build_samples(master, predictions)
    with_context = [row for row in samples if row.get("environment") != "unknown" or row.get("stage") != "unknown"]

    report = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sample_counts": {
            "total": len(samples),
            "with_context": len(with_context),
            "environment": dict(Counter(row["environment"] for row in with_context)),
            "stage": dict(Counter(row["stage"] for row in with_context)),
        },
        "active_matches": active_context_rows(master, predictions),
        "production_policy": {
            "min_samples": PRODUCTION_MIN_SAMPLES,
            "min_group_samples": PRODUCTION_MIN_GROUP_SAMPLES,
            "activation_rule": "Only enable a context calibration layer after expanding walk-forward improves log loss/Brier with enough closed point-in-time samples.",
        },
        "overall_model": metrics_for(samples),
        "overall_favorite_confidence": favorite_metrics(samples),
        "context_calibrator_walk_forward": context_calibrator_walk_forward(with_context),
        "segments": {
            "environment": grouped(with_context, "environment", lambda row: row["environment"]),
            "stage": grouped(with_context, "stage", lambda row: row["stage"]),
            "high_stakes": grouped(with_context, "high_stakes", lambda row: "yes" if row["high_stakes"] else "no"),
            "incentive_label": grouped(with_context, "incentive_label", lambda row: row["incentive_label"]),
            "winner_advances": grouped(with_context, "winner_advances", lambda row: "yes" if row["winner_advances"] else "no"),
            "loser_eliminated": grouped(with_context, "loser_eliminated", lambda row: "yes" if row["loser_eliminated"] else "no"),
            "format": grouped(with_context, "format", lambda row: str(row.get("format") or "unknown")),
        },
    }

    wf = report["context_calibrator_walk_forward"]
    if len(with_context) < PRODUCTION_MIN_SAMPLES:
        status = "insufficient_sample"
        decision = "No activar calibracion contextual en produccion."
        reason = f"Solo hay {len(with_context)} muestras con contexto; minimo recomendado {PRODUCTION_MIN_SAMPLES}."
    elif wf.get("available") and wf.get("delta_log_loss", 0.0) < -0.005 and wf.get("delta_brier", 0.0) <= 0:
        status = "candidate"
        decision = "Candidato a calibrador contextual; requiere revision de estabilidad por subgrupo."
        reason = "El diagnostico walk-forward mejora log loss y no empeora Brier."
    else:
        status = "not_useful_yet"
        decision = "Mantener solo como flags/reporting."
        reason = "No hay mejora probabilistica robusta en walk-forward."
    report["recommendation"] = {"status": status, "decision": decision, "reason": reason}
    return round_floats(report)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analiza calibracion del modelo por contexto HLTV Maps.")
    parser.add_argument("--master", default=str(DEFAULT_MASTER))
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    report = build_report(Path(args.master), Path(args.runs_dir))
    markdown = markdown_report(report)
    write_json(out_dir / "context_calibration.json", report)
    (out_dir / "CONTEXT_CALIBRATION.md").write_text(markdown, encoding="utf-8")
    write_json(ROOT_REPORT_JSON, report)
    ROOT_REPORT_MD.write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {
                "samples": report["sample_counts"],
                "recommendation": report["recommendation"],
                "json": str(out_dir / "context_calibration.json"),
                "report": str(out_dir / "CONTEXT_CALIBRATION.md"),
                "root_json": str(ROOT_REPORT_JSON),
                "root_report": str(ROOT_REPORT_MD),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
