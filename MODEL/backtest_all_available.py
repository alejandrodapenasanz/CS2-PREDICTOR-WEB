"""Backtest experimental con toda la informacion point-in-time disponible.

No entrena ni registra el modelo de produccion. El objetivo es responder:

1. Cuanta muestra real tenemos para stats individuales/mapstats/veto/analytics.
2. Si las features adicionales mejoran en walk-forward sin fuga temporal.
3. Si un modo post-veto (veto real ya conocido, antes de jugar) promete.

El boxscore del mismo partido nunca se usa como predictor de ese partido: solo
se usan historiales previos. El veto real del mismo partido se evalua separado
como escenario post-veto, porque no equivale a una prediccion pre-veto.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "MODEL") not in sys.path:
    sys.path.insert(0, str(ROOT / "MODEL"))

from cs2model import dataio
from cs2model.features import (  # noqa: E402
    ANALYTICS_DIFF_COLUMNS,
    ANALYTICS_FEATURE_COLUMNS,
    DIFF_COLUMNS,
    FEATURE_COLUMNS,
    MAP_ASSET_DIFF_COLUMNS,
    MAP_ASSET_FEATURE_COLUMNS,
    ChronologicalState,
    _clean_team,
    _period_index,
    analytics_match_features,
)
from cs2model.metrics import metric_dict  # noqa: E402
from train import DEFAULT_RAW, ODDS_FEATURE_COLUMNS, add_odds_features  # noqa: E402


OUTPUT_JSON = ROOT / "MODEL" / "results" / "all_available_backtest.json"
OUTPUT_MD = ROOT / "MODEL" / "results" / "ALL_AVAILABLE_BACKTEST.md"

POSTVETO_DIFF_COLUMNS = [
    "postveto_played_map_edge_diff",
    "postveto_pick_edge_diff",
    "postveto_decider_edge_diff",
    "postveto_ban_edge_diff",
]
POSTVETO_SYM_COLUMNS = [
    "postveto_available",
    "postveto_played_maps",
    "postveto_common_history_maps",
    "postveto_min_map_samples",
]
POSTVETO_COLUMNS = POSTVETO_DIFF_COLUMNS + POSTVETO_SYM_COLUMNS
EXTRA_DIFF_COLUMNS = set(MAP_ASSET_DIFF_COLUMNS) | set(ANALYTICS_DIFF_COLUMNS) | set(POSTVETO_DIFF_COLUMNS) | {"opening_odds_prob_centered"}


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _avg(values: list[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def _map_edge(state: ChronologicalState, team1_key: str, team2_key: str, map_name: str) -> tuple[float, int]:
    hist1 = state.asset_map_hist.get((team1_key, map_name), [])
    hist2 = state.asset_map_hist.get((team2_key, map_name), [])
    wr1 = (sum(hist1) + 1.0) / (len(hist1) + 2.0)
    wr2 = (sum(hist2) + 1.0) / (len(hist2) + 2.0)
    return wr1 - wr2, min(len(hist1), len(hist2))


def postveto_features(state: ChronologicalState, match: dict[str, Any]) -> dict[str, float]:
    feats = {col: 0.0 for col in POSTVETO_COLUMNS}
    asset = match.get("asset")
    if not isinstance(asset, dict):
        return feats
    team1_key = match["team1_key"]
    team2_key = match["team2_key"]
    feats["postveto_available"] = 1.0

    played_edges: list[float] = []
    history_counts: list[int] = []
    for map_payload in asset.get("mapstats") or []:
        info = map_payload.get("info") or {}
        map_name = info.get("map_name")
        if not map_name:
            continue
        edge, n = _map_edge(state, team1_key, team2_key, map_name)
        played_edges.append(edge)
        history_counts.append(n)

    veto = asset.get("veto") or {}
    pick_edges: list[float] = []
    decider_edges: list[float] = []
    ban_edges: list[float] = []
    for step in veto.get("steps") or []:
        map_name = step.get("map_name")
        action = str(step.get("action") or "").lower()
        team = _clean_team(step.get("team_name"))
        if not map_name:
            continue
        edge, n = _map_edge(state, team1_key, team2_key, map_name)
        if n:
            history_counts.append(n)
        if action == "pick":
            # Edge positivo siempre significa mapa favorable a team1.
            # Si lo pickea team2 y el edge es negativo, tambien favorece a team2.
            pick_edges.append(edge if team == team1_key else -edge if team == team2_key else edge)
        elif action in {"decider", "left", "leftover"}:
            decider_edges.append(edge)
        elif action == "ban":
            # Buena senal para team1: team2 elimina un mapa favorable a team1,
            # o team1 elimina uno favorable a team2.
            ban_edges.append(edge if team == team2_key else -edge if team == team1_key else 0.0)

    feats["postveto_played_maps"] = float(len(played_edges))
    feats["postveto_common_history_maps"] = float(sum(1 for n in history_counts if n > 0))
    feats["postveto_min_map_samples"] = float(min(history_counts) if history_counts else 0)
    feats["postveto_played_map_edge_diff"] = _avg(played_edges)
    feats["postveto_pick_edge_diff"] = _avg(pick_edges)
    feats["postveto_decider_edge_diff"] = _avg(decider_edges)
    feats["postveto_ban_edge_diff"] = _avg(ban_edges)
    return feats


def build_frame(rows: list[dict[str, Any]]) -> tuple[list[dict[str, float]], np.ndarray, list[dict[str, Any]]]:
    state = ChronologicalState()
    X: list[dict[str, float]] = []
    y: list[int] = []
    meta: list[dict[str, Any]] = []
    for m in rows:
        feats = state.emit_features(
            m["team1_key"],
            m["team2_key"],
            m.get("date_obj"),
            m.get("event") or "",
            m.get("format") or "bo3",
        )
        feats.update(analytics_match_features(m))
        feats.update(postveto_features(state, m))
        X.append(feats)
        y.append(int(m["team1_win"]))
        meta.append(
            {
                "id": m["id"],
                "date": m["date"],
                "event": m.get("event"),
                "format": m.get("format"),
                "team1": m["team1"],
                "team2": m["team2"],
            }
        )
        state.observe(m)
    return X, np.asarray(y, dtype=int), meta


def matrix(rows: list[dict[str, float]], cols: list[str]) -> np.ndarray:
    return np.asarray([[row.get(col, np.nan) for col in cols] for row in rows], dtype=float)


def augment(X: np.ndarray, y: np.ndarray, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    diff_idx = [i for i, col in enumerate(cols) if col in DIFF_COLUMNS or col in EXTRA_DIFF_COLUMNS]
    X_rev = X.copy()
    X_rev[:, diff_idx] *= -1.0
    return np.vstack([X, X_rev]), np.concatenate([y, 1 - y])


def fit_logistic(X: np.ndarray, y: np.ndarray, cols: list[str]):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X_aug, y_aug = augment(X, y, cols)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, C=0.5)),
        ]
    )
    model.fit(X_aug, y_aug)
    return model


def predict_proba(model, X: np.ndarray) -> np.ndarray:
    return np.clip(model.predict_proba(X)[:, 1], 1e-4, 1 - 1e-4)


def walk_forward_logistic(
    X: np.ndarray,
    y: np.ndarray,
    periods: np.ndarray,
    meta: list[dict[str, Any]],
    cols: list[str],
    *,
    min_train: int,
    warmup_weeks: int,
    test_filter: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    preds: list[dict[str, Any]] = []
    uniq_periods = sorted(set(periods.tolist()))
    for period in uniq_periods[warmup_weeks:]:
        train_mask = periods < period
        test_mask = periods == period
        if test_filter is not None:
            test_mask = test_mask & test_filter
        if train_mask.sum() < min_train or test_mask.sum() == 0 or len(np.unique(y[train_mask])) < 2:
            continue
        model = fit_logistic(X[train_mask], y[train_mask], cols)
        test_idx = np.where(test_mask)[0]
        probs = predict_proba(model, X[test_idx])
        for local_i, global_i in enumerate(test_idx):
            preds.append({**meta[global_i], "actual": int(y[global_i]), "prob_team1": float(probs[local_i])})
    return preds


def expanding_asset_only(
    X: np.ndarray,
    y: np.ndarray,
    meta: list[dict[str, Any]],
    cols: list[str],
    eligible: np.ndarray,
    *,
    min_train: int = 30,
) -> list[dict[str, Any]]:
    idx = np.where(eligible)[0]
    preds: list[dict[str, Any]] = []
    if len(idx) <= min_train:
        return preds
    for pos in range(min_train, len(idx)):
        train_idx = idx[:pos]
        test_idx = idx[pos : pos + 1]
        if len(np.unique(y[train_idx])) < 2:
            continue
        model = fit_logistic(X[train_idx], y[train_idx], cols)
        prob = float(predict_proba(model, X[test_idx])[0])
        global_i = int(test_idx[0])
        preds.append({**meta[global_i], "actual": int(y[global_i]), "prob_team1": prob})
    return preds


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    y = np.asarray([row["actual"] for row in rows], dtype=int)
    p = np.asarray([row["prob_team1"] for row in rows], dtype=float)
    return metric_dict(y, p)


def coverage_summary(rows: list[dict[str, Any]], X: list[dict[str, float]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "rows_total": len(rows),
        "asset_payload_same_match": sum(1 for row in rows if isinstance(row.get("asset"), dict)),
        "analytics_point_in_time": sum(1 for row in rows if isinstance(row.get("analytics"), dict) and row["analytics"].get("available")),
        "opening_odds": sum(1 for row in rows if row.get("opening_odds_t1") is not None),
        "prior_asset_both_teams": sum(1 for feats in X if (feats.get("asset_maps_played_min") or 0) > 0),
        "prior_asset_any_team": sum(1 for feats in X if (feats.get("asset_maps_played_total") or 0) > 0),
        "postveto_available": sum(1 for feats in X if (feats.get("postveto_available") or 0) >= 0.5),
    }
    for name, predicate in {
        "prior_asset_both_teams": lambda f, r: (f.get("asset_maps_played_min") or 0) > 0,
        "postveto_available": lambda f, r: (f.get("postveto_available") or 0) >= 0.5,
        "analytics_point_in_time": lambda f, r: (f.get("analytics_available") or 0) >= 0.5,
        "opening_odds": lambda f, r: r.get("opening_odds_t1") is not None,
    }.items():
        idx = [i for i, (f, r) in enumerate(zip(X, rows)) if predicate(f, r)]
        if idx:
            out[f"{name}_date_min"] = rows[idx[0]].get("date")
            out[f"{name}_date_max"] = rows[idx[-1]].get("date")
    return out


def write_report(payload: dict[str, Any], path: Path) -> None:
    cov = payload["coverage"]
    lines = [
        "# Backtest experimental: toda informacion disponible",
        "",
        f"Generado: {payload['generated_at']}",
        "",
        "## Cobertura",
        "",
        f"- Total series CS2: {cov['rows_total']}",
        f"- Partidos con mapstats/player stats del mismo partido archivados: {cov['asset_payload_same_match']}",
        f"- Partidos con historial previo de assets para ambos equipos: {cov['prior_asset_both_teams']}",
        f"- Partidos con Analytics point-in-time: {cov['analytics_point_in_time']}",
        f"- Partidos con odds de apertura/snapshot: {cov['opening_odds']}",
        "",
        "## Resultados",
        "",
        "| Experimento | n | Accuracy | Log loss | Nota |",
        "|---|---:|---:|---:|---|",
    ]
    for name, result in payload["experiments"].items():
        metrics = result.get("metrics") or {}
        note = result.get("note") or ""
        n = metrics.get("n", 0)
        acc = metrics.get("accuracy")
        ll = metrics.get("log_loss")
        lines.append(
            f"| {name} | {n} | {acc:.4f} | {ll:.4f} | {note} |"
            if acc is not None and ll is not None
            else f"| {name} | {n} | - | - | {note} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            payload["decision"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest experimental con stats/mapstats/veto/analytics disponibles.")
    parser.add_argument("--raw", default=str(DEFAULT_RAW))
    parser.add_argument("--master", default=str(ROOT / "DAILY_SNAPSHOTS" / "master" / "matches.json"))
    parser.add_argument("--warmup-weeks", type=int, default=10)
    parser.add_argument("--min-train", type=int, default=800)
    args = parser.parse_args()

    rows = dataio.load_training_rows(args.raw, args.master, cs2_only=True)
    X_dicts, y, meta = build_frame(rows)
    X_with_odds = add_odds_features(X_dicts, rows)
    periods = np.asarray([_period_index(row.get("date_obj")) for row in rows], dtype=int)

    no_asset_cols = list(FEATURE_COLUMNS)
    asset_cols = list(FEATURE_COLUMNS) + list(MAP_ASSET_FEATURE_COLUMNS)
    analytics_cols = asset_cols + list(ANALYTICS_FEATURE_COLUMNS)
    odds_cols = analytics_cols + list(ODDS_FEATURE_COLUMNS)
    postveto_cols = odds_cols + list(POSTVETO_COLUMNS)

    masks = {
        "prior_asset_both": np.asarray([(f.get("asset_maps_played_min") or 0) > 0 for f in X_dicts], dtype=bool),
        "postveto": np.asarray([(f.get("postveto_available") or 0) >= 0.5 for f in X_dicts], dtype=bool),
        "analytics": np.asarray([(f.get("analytics_available") or 0) >= 0.5 for f in X_dicts], dtype=bool),
        "odds": np.asarray([row.get("opening_odds_t1") is not None for row in rows], dtype=bool),
    }

    experiments: dict[str, Any] = {}

    for name, source, cols, note in [
        ("pre_match_sin_assets", X_dicts, no_asset_cols, "Modelo logistico walk-forward sin stats/mapstats."),
        ("pre_match_con_assets_rolling", X_dicts, asset_cols, "Incluye historiales previos de mapstats/player stats; no usa boxscore del mismo partido."),
        ("pre_match_assets_analytics_forzado", X_dicts, analytics_cols, "Analytics forzado aunque no llega al umbral productivo de 120 filas."),
        ("pre_match_assets_analytics_odds_forzado", X_with_odds, odds_cols, "Incluye odds point-in-time; diagnostico, no modelo estadistico puro."),
    ]:
        preds = walk_forward_logistic(
            matrix(source, cols),
            y,
            periods,
            meta,
            cols,
            min_train=args.min_train,
            warmup_weeks=args.warmup_weeks,
        )
        experiments[name] = {"metrics": summarize(preds), "note": note}

    # Mismas predicciones, medidas solo donde de verdad habia asset history.
    preds_asset_subset = walk_forward_logistic(
        matrix(X_dicts, asset_cols),
        y,
        periods,
        meta,
        asset_cols,
        min_train=args.min_train,
        warmup_weeks=args.warmup_weeks,
        test_filter=masks["prior_asset_both"],
    )
    experiments["pre_match_con_assets_solo_filas_con_cobertura"] = {
        "metrics": summarize(preds_asset_subset),
        "note": "Subset honesto: solo partidos donde ambos equipos tenian asset history previo.",
    }

    postveto_matrix = matrix(X_with_odds, postveto_cols)
    postveto_preds = walk_forward_logistic(
        postveto_matrix,
        y,
        periods,
        meta,
        postveto_cols,
        min_train=args.min_train,
        warmup_weeks=args.warmup_weeks,
        test_filter=masks["postveto"],
    )
    experiments["post_veto_walk_forward"] = {
        "metrics": summarize(postveto_preds),
        "note": "Escenario post-veto con veto/mapas reales; entrenamiento cronologico, muestra muy pequena.",
    }

    postveto_small = expanding_asset_only(
        postveto_matrix,
        y,
        meta,
        postveto_cols,
        masks["postveto"],
        min_train=30,
    )
    experiments["post_veto_asset_only_expanding_min30"] = {
        "metrics": summarize(postveto_small),
        "note": "Diagnostico solo sobre 92 filas con assets; min_train=30, no robusto.",
    }

    coverage = coverage_summary(rows, X_dicts)
    enough = (
        coverage["prior_asset_both_teams"] >= 120
        and coverage["analytics_point_in_time"] >= 120
        and coverage["opening_odds"] >= 120
    )
    decision = (
        "No se promueve el entrenamiento experimental. La muestra point-in-time para assets/Analytics/odds "
        "esta por debajo del umbral minimo razonable (120 filas cerradas por bloque) y el modo post-veto "
        "solo sirve como senal exploratoria. El modelo productivo queda como estaba."
        if not enough
        else
        "Hay muestra suficiente para plantear una promocion, pero debe revisarse manualmente antes de tocar produccion."
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "coverage": coverage,
        "thresholds": {
            "min_closed_rows_for_training_block": 120,
            "production_min_train": args.min_train,
            "postveto_diagnostic_min_train": 30,
        },
        "experiments": experiments,
        "decision": decision,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(payload, OUTPUT_MD)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
