from __future__ import annotations

from pathlib import Path


WEB_PATH = Path(__file__).resolve().parents[2] / "WEB" / "index.html"
ENRICH_PATH = Path(__file__).resolve().parents[1] / "PIPELINE" / "enrich_predictions.py"


def _function(html: str, name: str, next_marker: str) -> str:
    return html.split(f"function {name}", 1)[1].split(next_marker, 1)[0]


def test_dashboard_renders_estimate_band_level_and_history_coverage() -> None:
    html = WEB_PATH.read_text(encoding="utf-8")
    estimate_helper = _function(html, "estimateView(m){", "const estimatePp")
    detail = _function(html, "detailHTML(m){", "// ---------- best opportunity ----------")
    best = _function(html, "renderBest(){", "// ---------- model view ----------")

    for field in (
        "ensemble_disagreement",
        "estimate_band_half_width",
        "estimate_confidence_level",
        "estimate_history_coverage",
    ):
        assert field in estimate_helper

    assert 'high: { label: "ALTA", color: "var(--green)" }' in estimate_helper
    assert 'medium: { label: "MEDIA", color: "var(--amber)" }' in estimate_helper
    assert 'low: { label: "BAJA", color: "var(--red-soft)" }' in estimate_helper
    assert "Estimación T1:" in detail
    assert "estimatePp(estimate.bandHalfWidth)" in detail
    assert "pct(estimate.historyCoverage)" in detail
    assert "plainPp(estimate.disagreement)" in detail
    assert 'favSide === "team1" ? modelTeam1 : 1-modelTeam1' in best
    assert "p modelo del favorito" in best
    assert "pct(modelFavProb)" in best
    assert "estimatePp(estimate.bandHalfWidth)" in best
    assert "pct(estimate.historyCoverage)" in best
    assert "plainPp(estimate.disagreement)" in best


def test_estimate_display_does_not_change_best_opportunity_policy_or_staking() -> None:
    html = WEB_PATH.read_text(encoding="utf-8")
    best = _function(html, "renderBest(){", "// ---------- model view ----------")
    best_selection = best.split("const estimate=estimateView(m);", 1)[0]
    stake = _function(html, "stakeView(st,m){", "function valueOddsView(m){")

    assert "x.m.prediction?.opportunity_eligible===true" in best_selection
    assert "decisionConfidence(x.m)>=probMin && x.rel>=relMin" in best_selection
    assert ".sort((a,b)=>b.opp-a.opp)" in best_selection
    assert "estimateView" not in best_selection
    assert "estimate_" not in stake
    assert "ensemble_disagreement" not in stake


def test_new_band_is_not_wired_into_legacy_kelly_or_bankroll() -> None:
    source = ENRICH_PATH.read_text(encoding="utf-8")
    staking = source.split("def staking_recommendation(", 1)[1].split("def build_map_state(", 1)[0]

    assert 'prediction.get("model_epistemic_std")' in staking
    assert 'prediction.get("ensemble_disagreement")' not in staking
    assert 'prediction.get("estimate_band_half_width")' not in staking
    assert 'prediction.get("estimate_confidence_level")' not in staking
