"""Load and validate daily CS2 predictions from the canonical pipeline run."""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .models import DailyPredictions, PredictionPick

MIN_OPPORTUNITY_PROBABILITY = 0.65


class PredictionSourceError(RuntimeError):
    """Report an unavailable or malformed canonical prediction source."""


class PredictionValidationError(PredictionSourceError):
    """Report a prediction row that is unsafe or dishonest to publish."""


def _canonical_hltv_url(value: Any, row_label: str) -> str:
    """Validate one HLTV match link and return its canonical absolute URL."""

    link = _required_nonempty_text(value, "link", row_label)
    if any(character.isspace() for character in link) or "\\" in link:
        raise PredictionValidationError(f"{row_label}: link is not a valid HLTV match URL")

    candidate = f"https://www.hltv.org{link}" if link.startswith("/") else link
    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError as exc:
        raise PredictionValidationError(f"{row_label}: link is not a valid HLTV match URL") from exc

    if parsed.scheme.lower() != "https":
        raise PredictionValidationError(f"{row_label}: HLTV link must use https")
    if parsed.hostname is None or parsed.hostname.lower() != "www.hltv.org":
        raise PredictionValidationError(f"{row_label}: HLTV link host must be www.hltv.org")
    if parsed.username is not None or parsed.password is not None or port is not None:
        raise PredictionValidationError(f"{row_label}: HLTV link must not contain credentials or a port")
    if parsed.fragment:
        raise PredictionValidationError(f"{row_label}: HLTV link must not contain a fragment")
    if not parsed.path.startswith("/matches/"):
        raise PredictionValidationError(f"{row_label}: HLTV link path must start with /matches/")

    return urlunsplit(("https", "www.hltv.org", parsed.path, parsed.query, ""))


def _read_json(path: Path, description: str) -> Any:
    """Read one UTF-8 JSON document and wrap filesystem/parser failures clearly."""

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PredictionSourceError(f"{description} not found: {path}") from exc
    except OSError as exc:
        raise PredictionSourceError(f"Could not read {description} at {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PredictionSourceError(f"Invalid JSON in {description} at {path}: {exc}") from exc


def _parse_target_date(value: date | str) -> date:
    """Normalize a date object or an ISO ``YYYY-MM-DD`` string."""

    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise TypeError("target_date must be a datetime.date or an ISO YYYY-MM-DD string")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid target_date {value!r}; expected YYYY-MM-DD") from exc


def _required_nonempty_text(value: Any, field: str, row_label: str) -> str:
    """Validate and return one mandatory non-empty text field."""

    if not isinstance(value, str) or not value.strip():
        raise PredictionValidationError(f"{row_label}: {field} must be non-empty text")
    return value.strip()


def _row_label(row: dict[str, Any], index: int) -> str:
    """Build a stable diagnostic label without trusting optional input fields."""

    match_id = row.get("id")
    if isinstance(match_id, (str, int)) and str(match_id).strip():
        return f"prediction {str(match_id).strip()}"
    return f"prediction row {index}"


def _validate_pick(row: Any, index: int, target_date: date) -> PredictionPick:
    """Validate one source row and convert it into an immutable publication pick."""

    if not isinstance(row, dict):
        raise PredictionValidationError(f"prediction row {index}: expected a JSON object")
    label = _row_label(row, index)

    team1_block = row.get("team1")
    team2_block = row.get("team2")
    prediction = row.get("prediction")
    model_trace = row.get("model_trace")
    if not isinstance(team1_block, dict) or not isinstance(team2_block, dict):
        raise PredictionValidationError(f"{label}: team1 and team2 must be objects")
    if not isinstance(prediction, dict):
        raise PredictionValidationError(f"{label}: prediction must be an object")
    if not isinstance(model_trace, dict):
        raise PredictionValidationError(f"{label}: model_trace must be an object")

    fallback = model_trace.get("is_fallback")
    if not isinstance(fallback, bool):
        raise PredictionValidationError(f"{label}: model_trace.is_fallback must be boolean")
    if fallback:
        raise PredictionValidationError(f"{label}: fallback-model predictions cannot be published")

    team1 = _required_nonempty_text(team1_block.get("name"), "team1.name", label)
    team2 = _required_nonempty_text(team2_block.get("name"), "team2.name", label)
    hltv_url = _canonical_hltv_url(row.get("link"), label)
    winner = _required_nonempty_text(
        prediction.get("decision_favorite"),
        "prediction.decision_favorite",
        label,
    )
    if winner not in {team1, team2}:
        raise PredictionValidationError(f"{label}: prediction.decision_favorite must equal team1.name or team2.name")

    confidence = prediction.get("decision_confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise PredictionValidationError(f"{label}: prediction.decision_confidence must be numeric")
    probability = float(confidence)
    if not math.isfinite(probability) or not 0.5 <= probability <= 1.0:
        raise PredictionValidationError(f"{label}: prediction.decision_confidence must be between 0.5 and 1")

    confidence_level = prediction.get("estimate_confidence_level")
    if confidence_level is not None:
        if not isinstance(confidence_level, str) or confidence_level not in {"high", "medium", "low"}:
            raise PredictionValidationError(
                f"{label}: prediction.estimate_confidence_level must be high, medium, low or null"
            )

    is_best = prediction.get("is_best_opportunity", False)
    if not isinstance(is_best, bool):
        raise PredictionValidationError(f"{label}: prediction.is_best_opportunity must be boolean when present")
    opportunity_eligible = prediction.get("opportunity_eligible", is_best)
    if not isinstance(opportunity_eligible, bool):
        raise PredictionValidationError(f"{label}: prediction.opportunity_eligible must be boolean when present")
    if is_best:
        if "opportunity_eligible" in prediction and prediction.get("opportunity_eligible") is not True:
            raise PredictionValidationError(
                f"{label}: a best opportunity must have prediction.opportunity_eligible=true"
            )
        if "opportunity_rank_day" in prediction and prediction.get("opportunity_rank_day") != 1:
            raise PredictionValidationError(f"{label}: a best opportunity must have prediction.opportunity_rank_day=1")

    raw_hour = row.get("hour", "")
    if raw_hour is None:
        hour = ""
    elif isinstance(raw_hour, str):
        hour = raw_hour.strip()
    else:
        raise PredictionValidationError(f"{label}: hour must be text or null")

    raw_match_id = row.get("id")
    if not isinstance(raw_match_id, (str, int)) or not str(raw_match_id).strip():
        raise PredictionValidationError(f"{label}: id must be non-empty text or an integer")
    match_id = str(raw_match_id).strip()
    return PredictionPick(
        match_id=match_id,
        match_date=target_date,
        hltv_url=hltv_url,
        hour=hour,
        team1=team1,
        team2=team2,
        winner=winner,
        win_probability=probability,
        is_best_opportunity=opportunity_eligible and probability > MIN_OPPORTUNITY_PROBABILITY,
        confidence_level=confidence_level,
    )


def _hour_sort_key(pick: PredictionPick) -> tuple[str, str]:
    """Return a deterministic chronological key for ISO-like match times."""

    return (pick.hour or "99:99", pick.match_id)


def load_daily_predictions(
    project_root: Path | str,
    target_date: date | str,
) -> DailyPredictions:
    """Load validated picks for a date from the run named by the master manifest.

    The source is always ``CS2/PIPELINE/master/manifest.json`` followed by
    ``CS2/PIPELINE/runs/<last_run_id>/predictions_enriched.json``. Rows for other
    dates are ignored. Include every upstream-eligible opportunity whose
    adjusted winner probability is strictly greater than 65%, not just rank 1.
    """

    root = Path(project_root).expanduser().resolve()
    selected_date = _parse_target_date(target_date)
    manifest_path = root / "CS2" / "PIPELINE" / "master" / "manifest.json"
    manifest = _read_json(manifest_path, "CS2 master manifest")
    if not isinstance(manifest, dict):
        raise PredictionSourceError(f"CS2 master manifest must contain an object: {manifest_path}")

    run_id = manifest.get("last_run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise PredictionSourceError("CS2 master manifest has no valid last_run_id")
    run_id = run_id.strip()
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise PredictionSourceError("CS2 master manifest contains an unsafe last_run_id")

    predictions_path = root / "CS2" / "PIPELINE" / "runs" / run_id / "predictions_enriched.json"
    rows = _read_json(predictions_path, "enriched CS2 predictions")
    if not isinstance(rows, list):
        raise PredictionSourceError(f"Enriched CS2 predictions must contain a JSON array: {predictions_path}")

    target_iso = selected_date.isoformat()
    picks: list[PredictionPick] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise PredictionValidationError(f"prediction row {index}: expected a JSON object")
        if row.get("date") == target_iso:
            picks.append(_validate_pick(row, index, selected_date))

    match_ids = [pick.match_id for pick in picks]
    duplicate_ids = sorted({match_id for match_id in match_ids if match_ids.count(match_id) > 1})
    if duplicate_ids:
        raise PredictionValidationError(f"{target_iso}: duplicate match id(s): {', '.join(duplicate_ids)}")
    opportunities = tuple(sorted((pick for pick in picks if pick.is_best_opportunity), key=_hour_sort_key))
    others = tuple(sorted((pick for pick in picks if not pick.is_best_opportunity), key=_hour_sort_key))
    return DailyPredictions(
        target_date=selected_date,
        source_run_id=run_id,
        opportunities=opportunities,
        others=others,
    )


def load_upcoming_opportunities(
    project_root: Path | str,
    from_date: date | str,
) -> tuple[str, tuple[PredictionPick, ...]]:
    """Load every validated qualified opportunity on or after ``from_date``."""

    root = Path(project_root).expanduser().resolve()
    first_date = _parse_target_date(from_date)
    manifest_path = root / "CS2" / "PIPELINE" / "master" / "manifest.json"
    manifest = _read_json(manifest_path, "CS2 master manifest")
    if not isinstance(manifest, dict):
        raise PredictionSourceError(f"CS2 master manifest must contain an object: {manifest_path}")
    run_id = manifest.get("last_run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise PredictionSourceError("CS2 master manifest has no valid last_run_id")
    run_id = run_id.strip()
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise PredictionSourceError("CS2 master manifest contains an unsafe last_run_id")
    rows = _read_json(
        root / "CS2" / "PIPELINE" / "runs" / run_id / "predictions_enriched.json",
        "enriched CS2 predictions",
    )
    if not isinstance(rows, list):
        raise PredictionSourceError("Enriched CS2 predictions must contain a JSON array")

    picks: list[PredictionPick] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise PredictionValidationError(f"prediction row {index}: expected a JSON object")
        raw_date = row.get("date")
        if not isinstance(raw_date, str):
            raise PredictionValidationError(f"prediction row {index}: date must be ISO text")
        try:
            match_date = date.fromisoformat(raw_date)
        except ValueError as exc:
            raise PredictionValidationError(f"prediction row {index}: date must use YYYY-MM-DD") from exc
        if match_date < first_date:
            continue
        pick = _validate_pick(row, index, match_date)
        if pick.match_id in seen:
            raise PredictionValidationError(f"duplicate upcoming match id: {pick.match_id}")
        seen.add(pick.match_id)
        if pick.is_best_opportunity:
            picks.append(pick)
    return run_id, tuple(sorted(picks, key=lambda pick: (pick.match_date, *_hour_sort_key(pick))))


def load_latest_daily_picks(
    project_root: Path | str,
    target_date: date | str,
) -> DailyPredictions:
    """Compatibility wrapper for :func:`load_daily_predictions`."""

    return load_daily_predictions(project_root=project_root, target_date=target_date)
