"""Read-only policy for deciding when CS2 has enough new labels to retrain.

The production artifact records the newest match date used by its fit in
``metadata["date_max"]``.  With daily source precision, matches from that same
day are deliberately not treated as new: only labels from a strictly later day
may trigger automatic retraining.

A rejected or deferred challenger is still a completed, expensive training
attempt.  Its registered ``date_max`` therefore advances only the *automatic
retrain* watermark; it never changes the live cutoff used by the promotion
comparison.  Registry scanning accepts canonical timestamp directories with a
terminal lifecycle status, plus the status-less full structure written by the
historical trainer.  Malformed, unknown-status, non-canonical, symlinked, or
future-dated entries are ignored conservatively.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_NEW_LABELED_THRESHOLD = 100
_VERSION_PATTERN = re.compile(r"^\d{8}_\d{6}Z$")
_KNOWN_REGISTRY_STATUSES = frozenset(
    {
        "approved_pending_publish",
        "promotion_approved",
        "promote",
        "reject",
        "insufficient",
        "bootstrap",
        "promoted",
        "rejected",
        "deferred",
        "final_health_gate_failed",
        "atomic_publish_failed",
    }
)


@dataclass(frozen=True)
class RetrainDecision:
    """Deterministic, serializable result of the automatic retrain policy."""

    should_train: bool
    n_new_labeled: int
    threshold: int
    live_cutoff: str | None
    attempt_cutoff: str | None
    cutoff: str | None
    newest_date: str | None
    reason: str
    registry_warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if not self.registry_warnings:
            payload.pop("registry_warnings")
        return payload


def _normalized_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _row_date(row: Mapping[str, Any]) -> date | None:
    return _normalized_date(row.get("date")) or _normalized_date(row.get("date_obj"))


def _binary_label(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, int):
        return value in (0, 1)
    if isinstance(value, float):
        return value in (0.0, 1.0)
    if isinstance(value, str):
        return value.strip() in {"0", "1"}
    return False


def _is_labeled(row: Mapping[str, Any]) -> bool:
    if _binary_label(row.get("team1_win")) or _binary_label(row.get("label")):
        return True
    score1 = row.get("score1")
    score2 = row.get("score2")
    try:
        return score1 is not None and score2 is not None and float(score1) != float(score2)
    except (TypeError, ValueError):
        return False


def _canonical_version_date(name: str) -> date | None:
    if not _VERSION_PATTERN.fullmatch(name):
        return None
    try:
        return datetime.strptime(name[:8], "%Y%m%d").date()
    except ValueError:
        return None


def _historical_registry_structure(
    version_dir: Path,
    payload: Mapping[str, Any],
    artifact_metadata: Mapping[str, Any],
) -> bool:
    """Recognize the complete pre-status registry format, not partial JSON."""

    registry = artifact_metadata.get("registry")
    return bool(
        isinstance(payload.get("registered_at"), str)
        and _normalized_date(payload.get("registered_at")) is not None
        and isinstance(payload.get("artifact"), str)
        and isinstance(payload.get("feature_columns"), list)
        and isinstance(payload.get("metrics"), Mapping)
        and isinstance(registry, Mapping)
        and isinstance(registry.get("promoted"), bool)
        and (version_dir / "model.pkl").is_file()
        and not (version_dir / "model.pkl").is_symlink()
        and (version_dir / "experiment_manifest.json").is_file()
    )


def _registered_attempt_date(
    version_dir: Path,
    *,
    newest_labeled: date,
) -> date | None:
    version_date = _canonical_version_date(version_dir.name)
    if version_date is None or version_dir.is_symlink() or not version_dir.is_dir():
        return None
    metadata_path = version_dir / "metadata.json"
    if metadata_path.is_symlink() or not metadata_path.is_file():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, Mapping):
        return None
    artifact_metadata = payload.get("metadata")
    if not isinstance(artifact_metadata, Mapping):
        return None
    model_path = version_dir / "model.pkl"
    if (
        not isinstance(payload.get("registered_at"), str)
        or _normalized_date(payload.get("registered_at")) is None
        or not isinstance(payload.get("artifact"), str)
        or model_path.is_symlink()
        or not model_path.is_file()
    ):
        return None

    registry = artifact_metadata.get("registry")
    status = registry.get("status") if isinstance(registry, Mapping) else None
    if status is not None:
        eligible = str(status) in _KNOWN_REGISTRY_STATUSES
    else:
        eligible = _historical_registry_structure(version_dir, payload, artifact_metadata)
    if not eligible:
        return None

    attempt_date = _normalized_date(artifact_metadata.get("date_max"))
    # A real training attempt cannot contain labels after either its UTC version
    # date or the newest currently available labelled match. Both bounds prevent
    # one corrupt far-future date from suppressing retraining indefinitely.
    if attempt_date is None or attempt_date > version_date or attempt_date > newest_labeled:
        return None
    return attempt_date


def latest_registered_attempt_cutoff(
    registry: str | Path,
    *,
    newest_labeled: date | None,
    diagnostics: list[str] | None = None,
) -> date | None:
    """Return the newest trustworthy registered training-attempt cutoff."""

    root = Path(registry)
    if newest_labeled is None:
        return None
    try:
        if root.is_symlink() or not root.is_dir():
            return None
        children = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        if diagnostics is not None:
            diagnostics.append(f"Auto-retrain: registro no accesible {root}: {type(exc).__name__}: {exc}")
        return None
    dates = []
    for child in children:
        try:
            attempt_date = _registered_attempt_date(child, newest_labeled=newest_labeled)
        except OSError as exc:
            # Rejected cleanup residues may deny even lstat on Windows. They
            # cannot set a watermark, and must not stop prediction publication.
            if diagnostics is not None:
                diagnostics.append(
                    f"Auto-retrain: entrada inaccesible omitida {child.name}: {type(exc).__name__}: {exc}"
                )
            continue
        if attempt_date is not None:
            dates.append(attempt_date)
    return max(dates, default=None)


def decide_retrain(
    rows: Iterable[Mapping[str, Any]],
    live_artifact: Any | None,
    threshold: int = DEFAULT_NEW_LABELED_THRESHOLD,
    *,
    registry: str | Path | None = None,
) -> RetrainDecision:
    """Decide without mutation whether automatic retraining should run.

    ``rows`` must be the chronologically normalized training rows returned by
    :mod:`cs2model.dataio`.  The function still checks labels explicitly so an
    incomplete or synthetic row can never advance the counter.
    """

    if threshold < 1:
        raise ValueError("threshold must be >= 1")

    labeled_dates = [row_date for row in rows if _is_labeled(row) and (row_date := _row_date(row)) is not None]
    newest_date = max(labeled_dates, default=None)
    newest = newest_date.isoformat() if newest_date is not None else None

    if live_artifact is None:
        return RetrainDecision(
            should_train=True,
            n_new_labeled=len(labeled_dates),
            threshold=threshold,
            live_cutoff=None,
            attempt_cutoff=None,
            cutoff=None,
            newest_date=newest,
            reason="production_artifact_missing",
        )

    metadata = getattr(live_artifact, "metadata", None)
    raw_cutoff = metadata.get("date_max") if isinstance(metadata, Mapping) else None
    if raw_cutoff is None or not str(raw_cutoff).strip():
        return RetrainDecision(
            should_train=False,
            n_new_labeled=0,
            threshold=threshold,
            live_cutoff=None,
            attempt_cutoff=None,
            cutoff=None,
            newest_date=newest,
            reason="live_artifact_missing_date_max_fail_closed",
        )

    cutoff_date = _normalized_date(raw_cutoff)
    if cutoff_date is None:
        return RetrainDecision(
            should_train=False,
            n_new_labeled=0,
            threshold=threshold,
            live_cutoff=str(raw_cutoff),
            attempt_cutoff=None,
            cutoff=None,
            newest_date=newest,
            reason="live_artifact_invalid_date_max_fail_closed",
        )

    diagnostics: list[str] = []
    registered_cutoff = (
        latest_registered_attempt_cutoff(registry, newest_labeled=newest_date, diagnostics=diagnostics)
        if registry is not None
        else None
    )
    attempt_cutoff = max(cutoff_date, registered_cutoff) if registered_cutoff is not None else cutoff_date
    n_new = sum(row_date > attempt_cutoff for row_date in labeled_dates)
    should_train = n_new >= threshold
    return RetrainDecision(
        should_train=should_train,
        n_new_labeled=n_new,
        threshold=threshold,
        live_cutoff=cutoff_date.isoformat(),
        attempt_cutoff=attempt_cutoff.isoformat(),
        cutoff=attempt_cutoff.isoformat(),
        newest_date=newest,
        reason=("new_labeled_threshold_reached" if should_train else "new_labeled_threshold_not_reached"),
        registry_warnings=tuple(diagnostics),
    )
