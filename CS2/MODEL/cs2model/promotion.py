"""Leak-safe challenger promotion, model pointers, and rollback.

The promotion comparison deliberately receives the challenger's out-of-sample
predictions separately from the point-in-time feature rows used to score the
incumbent.  The challenger cohort is authoritative: every selected match must
have exactly one incumbent feature row with the same date and label.  Extra
feature rows are ignored, so appending future history cannot silently alter an
already frozen challenger hold-out.

Metric deltas are always ``challenger - incumbent``; negative values are
improvements.  Log loss is primary.  A difference inside the configured log
loss epsilon is a tie and is resolved only by a minimum Brier improvement.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Protocol, Sequence

import numpy as np

from .metrics import metric_dict


DEFAULT_LOG_LOSS_EPSILON = 0.001
DEFAULT_BRIER_EPSILON = 0.0005
METRIC_CONSISTENCY_ATOL = 1e-12
_VERSION_PATTERN = re.compile(r"^\d{8}_\d{6}Z$")
_POINTER_NAMES = frozenset({"latest", "last_good"})


class PromotionValidationError(ValueError):
    """Raised when a cohort, pointer, hash, or registry path is unsafe."""


class DeploymentRecoveryError(PromotionValidationError):
    """Raised when rollback itself is incomplete; the deployment lock remains."""


class ProbabilityModel(Protocol):
    """Small surface required to evaluate the incumbent on frozen features."""

    metadata: dict[str, Any]

    def predict_proba_team1(self, feature_rows: list[dict[str, float]]) -> np.ndarray: ...


@dataclass(frozen=True)
class HoldoutMetrics:
    log_loss: float
    brier: float


@dataclass(frozen=True)
class PromotionDecision:
    """Structured, serializable result of the common-holdout comparison."""

    status: Literal["promote", "reject", "insufficient"]
    promote: bool
    n: int
    cutoff: str
    min_samples: int
    log_loss_epsilon: float
    brier_epsilon: float
    incumbent: HoldoutMetrics | None
    challenger: HoldoutMetrics | None
    delta_log_loss: float | None
    delta_brier: float | None
    reason: str
    holdout_sha256: str
    prediction_sha256: str
    message: str
    cohort: dict[str, Any] = field(default_factory=dict)
    by_regime: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assert_same_artifact_same_cohort_metrics(
    artifact: ProbabilityModel,
    feature_rows: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
    expected_probabilities: Sequence[float],
    *,
    atol: float = METRIC_CONSISTENCY_ATOL,
) -> dict[str, Any]:
    """Re-score one artifact on one cohort and require metric identity.

    The promotion path calls this after creating the challenger's frozen
    predictions. It catches a second evaluator silently applying another
    route, preprocessing path or calibration object. Native estimators may
    differ by minute floating-point noise across machines, hence the explicit
    absolute tolerance.
    """

    if atol < 0.0:
        raise PromotionValidationError("metric consistency tolerance must be non-negative")
    normalized_features = [dict(row) for row in feature_rows]
    expected = np.asarray(expected_probabilities, dtype=float).reshape(-1)
    y_true = np.asarray(labels, dtype=int).reshape(-1)
    if len(normalized_features) != len(expected) or len(y_true) != len(expected):
        raise PromotionValidationError("metric consistency inputs must have equal lengths")
    repeated = np.asarray(artifact.predict_proba_team1(normalized_features), dtype=float).reshape(-1)
    if len(repeated) != len(expected):
        raise PromotionValidationError("artifact returned a different number of repeated predictions")
    max_abs_probability_delta = float(np.max(np.abs(repeated - expected))) if len(expected) else 0.0
    expected_metrics = metric_dict(y_true, expected)
    repeated_metrics = metric_dict(y_true, repeated)
    metric_deltas = {
        key: abs(float(repeated_metrics[key]) - float(expected_metrics[key])) for key in ("log_loss", "brier")
    }
    if max_abs_probability_delta > atol or any(delta > atol for delta in metric_deltas.values()):
        raise PromotionValidationError(
            "same artifact/same cohort metric mismatch: "
            f"probability_delta={max_abs_probability_delta:.3g}, "
            f"log_loss_delta={metric_deltas['log_loss']:.3g}, "
            f"brier_delta={metric_deltas['brier']:.3g}, atol={atol:.3g}"
        )
    return {
        "status": "pass",
        "n": int(len(expected)),
        "atol": float(atol),
        "max_abs_probability_delta": max_abs_probability_delta,
        "log_loss_delta": metric_deltas["log_loss"],
        "brier_delta": metric_deltas["brier"],
    }


@dataclass(frozen=True)
class ModelReference:
    """Validated immutable reference to one registry version."""

    version: str
    artifact: Path
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "version": self.version,
            "artifact": str(self.artifact),
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class DeploymentResult:
    """Outcome of a promotion, rejection, or rollback operation."""

    action: Literal["bootstrapped", "promoted", "rejected", "rolled_back"]
    changed: bool
    latest: ModelReference
    last_good: ModelReference | None
    production_sha256: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "changed": self.changed,
            "latest": self.latest.to_dict(),
            "last_good": self.last_good.to_dict() if self.last_good else None,
            "production_sha256": self.production_sha256,
            "message": self.message,
        }


@dataclass(frozen=True)
class LastGoodUpdateResult:
    """Result of pinning one validated registry artifact as rollback target."""

    changed: bool
    latest: ModelReference
    previous_last_good: ModelReference | None
    last_good: ModelReference
    production_sha256: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "changed": self.changed,
            "latest": self.latest.to_dict(),
            "previous_last_good": self.previous_last_good.to_dict() if self.previous_last_good else None,
            "last_good": self.last_good.to_dict(),
            "production_sha256": self.production_sha256,
            "message": self.message,
        }


def _parse_date(value: Any, field: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        raise PromotionValidationError(f"{field} is required")
    try:
        return date.fromisoformat(text[:10])
    except ValueError as exc:
        raise PromotionValidationError(f"{field} is not an ISO date: {text!r}") from exc


def _match_id(row: Mapping[str, Any], source: str) -> str:
    value = str(row.get("match_id") or "").strip()
    if not value:
        raise PromotionValidationError(f"{source} row has no match_id")
    return value


def _label(row: Mapping[str, Any], source: str) -> int:
    raw_value = row.get("actual")
    if raw_value is None:
        raise PromotionValidationError(f"{source} row has no actual label")
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise PromotionValidationError(f"{source} row has an invalid actual label") from exc
    if value not in (0, 1):
        raise PromotionValidationError(f"{source} actual must be 0 or 1")
    return value


def _probability(value: Any, source: str) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise PromotionValidationError(f"{source} probability is invalid") from exc
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise PromotionValidationError(f"{source} probability must be finite and in [0, 1]")
    return probability


def _cohort_hash(rows: Sequence[tuple[str, date, int]]) -> str:
    payload = [
        {"match_id": match_id, "date": match_date.isoformat(), "actual": actual}
        for match_id, match_date, actual in rows
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _format_decision_message(
    *,
    status: str,
    n: int,
    cutoff: str,
    incumbent: HoldoutMetrics | None,
    challenger: HoldoutMetrics | None,
    delta_log_loss: float | None,
    delta_brier: float | None,
    reason: str,
) -> str:
    if incumbent is None or challenger is None:
        return f"PROMOTION {status.upper()}: n={n} cutoff=>{cutoff} reason={reason}"
    return (
        f"PROMOTION {status.upper()}: n={n} cutoff=>{cutoff} "
        f"log_loss incumbent={incumbent.log_loss:.6f} "
        f"challenger={challenger.log_loss:.6f} delta={delta_log_loss:+.6f}; "
        f"brier incumbent={incumbent.brier:.6f} "
        f"challenger={challenger.brier:.6f} delta={delta_brier:+.6f}; "
        f"reason={reason}"
    )


def compare_candidate_to_incumbent(
    incumbent: ProbabilityModel,
    challenger_oos: Sequence[Mapping[str, Any]],
    incumbent_feature_rows: Sequence[Mapping[str, Any]],
    *,
    min_samples: int,
    log_loss_epsilon: float = DEFAULT_LOG_LOSS_EPSILON,
    brier_epsilon: float = DEFAULT_BRIER_EPSILON,
) -> PromotionDecision:
    """Compare challenger and incumbent on one strictly post-cutoff cohort.

    ``challenger_oos`` rows require ``match_id``, ``date``, ``actual`` and
    ``prob_team1``. ``incumbent_feature_rows`` require the same identity fields
    plus a ``features`` mapping. Only challenger rows with ``date`` strictly
    greater than ``incumbent.metadata['date_max']`` enter the comparison.

    The feature input may be a superset of the challenger cohort. Each selected
    challenger ID must nevertheless resolve to exactly one row whose date and
    label are identical. This makes the incumbent consume the exact same
    point-in-time observations while keeping a frozen hold-out invariant when
    unrelated future feature rows are appended.
    """

    if min_samples < 1:
        raise PromotionValidationError("min_samples must be >= 1")
    if log_loss_epsilon < 0.0 or brier_epsilon < 0.0:
        raise PromotionValidationError("promotion epsilons must be non-negative")

    metadata = getattr(incumbent, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise PromotionValidationError("incumbent metadata is unavailable")
    cutoff_date = _parse_date(metadata.get("date_max"), "incumbent.metadata.date_max")
    cutoff = cutoff_date.isoformat()

    feature_by_id: dict[str, Mapping[str, Any]] = {}
    for row in incumbent_feature_rows:
        match_id = _match_id(row, "incumbent feature")
        if match_id in feature_by_id:
            raise PromotionValidationError(f"duplicate incumbent feature row for match_id={match_id}")
        feature_by_id[match_id] = row

    selected: list[tuple[Mapping[str, Any], Mapping[str, Any], str, date, int]] = []
    seen_challenger: set[str] = set()
    for challenger_row in challenger_oos:
        match_id = _match_id(challenger_row, "challenger")
        if match_id in seen_challenger:
            raise PromotionValidationError(f"duplicate challenger prediction for match_id={match_id}")
        seen_challenger.add(match_id)
        match_date = _parse_date(challenger_row.get("date"), "challenger.date")
        if match_date <= cutoff_date:
            continue
        actual = _label(challenger_row, "challenger")
        feature_row = feature_by_id.get(match_id)
        if feature_row is None:
            raise PromotionValidationError(f"missing incumbent feature row for match_id={match_id}")
        feature_date = _parse_date(feature_row.get("date"), "incumbent feature.date")
        feature_actual = _label(feature_row, "incumbent feature")
        if feature_date != match_date:
            raise PromotionValidationError(
                f"date mismatch for match_id={match_id}: "
                f"challenger={match_date.isoformat()} incumbent={feature_date.isoformat()}"
            )
        if feature_actual != actual:
            raise PromotionValidationError(
                f"label mismatch for match_id={match_id}: challenger={actual} incumbent={feature_actual}"
            )
        selected.append((challenger_row, feature_row, match_id, match_date, actual))

    identities = [(row[2], row[3], row[4]) for row in selected]
    holdout_sha256 = _cohort_hash(identities)
    n_rows = len(selected)
    selected_dates = [identity[1] for identity in identities]
    selected_regimes = [str(row[0].get("prediction_regime") or "unspecified") for row in selected]
    regime_counts = {regime: selected_regimes.count(regime) for regime in sorted(set(selected_regimes))}
    cohort = {
        "n": n_rows,
        "date_min": min(selected_dates).isoformat() if selected_dates else None,
        "date_max": max(selected_dates).isoformat() if selected_dates else None,
        "regime_counts": regime_counts,
        "odds_fraction": (regime_counts.get("odds", 0) / n_rows if n_rows else None),
        "no_odds_fraction": (regime_counts.get("no_odds", 0) / n_rows if n_rows else None),
    }
    if n_rows < min_samples:
        reason = f"insufficient_holdout:{n_rows}<{min_samples}"
        message = _format_decision_message(
            status="insufficient",
            n=n_rows,
            cutoff=cutoff,
            incumbent=None,
            challenger=None,
            delta_log_loss=None,
            delta_brier=None,
            reason=reason,
        )
        return PromotionDecision(
            status="insufficient",
            promote=False,
            n=n_rows,
            cutoff=cutoff,
            min_samples=min_samples,
            log_loss_epsilon=float(log_loss_epsilon),
            brier_epsilon=float(brier_epsilon),
            incumbent=None,
            challenger=None,
            delta_log_loss=None,
            delta_brier=None,
            reason=reason,
            holdout_sha256=holdout_sha256,
            prediction_sha256="",
            message=message,
            cohort=cohort,
        )

    feature_payload: list[dict[str, float]] = []
    labels: list[int] = []
    challenger_probabilities: list[float] = []
    for challenger_row, feature_row, match_id, _match_date, actual in selected:
        features = feature_row.get("features")
        if not isinstance(features, Mapping):
            raise PromotionValidationError(f"features must be a mapping for match_id={match_id}")
        feature_payload.append(dict(features))
        labels.append(actual)
        challenger_probabilities.append(_probability(challenger_row.get("prob_team1"), "challenger"))

    incumbent_raw = np.asarray(incumbent.predict_proba_team1(feature_payload), dtype=float).reshape(-1)
    if len(incumbent_raw) != n_rows:
        raise PromotionValidationError("incumbent returned a different number of predictions than the paired cohort")
    incumbent_probabilities = [_probability(value, "incumbent") for value in incumbent_raw.tolist()]
    prediction_payload = [
        {
            "match_id": match_id,
            "date": match_date.isoformat(),
            "actual": actual,
            "incumbent": incumbent_probability,
            "challenger": challenger_probability,
        }
        for (
            _challenger,
            _features,
            match_id,
            match_date,
            actual,
        ), incumbent_probability, challenger_probability in zip(
            selected,
            incumbent_probabilities,
            challenger_probabilities,
            strict=True,
        )
    ]
    prediction_sha256 = hashlib.sha256(
        json.dumps(prediction_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    y_true = np.asarray(labels, dtype=int)
    incumbent_values = metric_dict(y_true, np.asarray(incumbent_probabilities, dtype=float))
    challenger_values = metric_dict(y_true, np.asarray(challenger_probabilities, dtype=float))
    incumbent_metrics = HoldoutMetrics(
        log_loss=float(incumbent_values["log_loss"]),
        brier=float(incumbent_values["brier"]),
    )
    challenger_metrics = HoldoutMetrics(
        log_loss=float(challenger_values["log_loss"]),
        brier=float(challenger_values["brier"]),
    )
    by_regime: dict[str, Any] = {}
    for regime in ("odds", "no_odds"):
        indices = [index for index, selected_regime in enumerate(selected_regimes) if selected_regime == regime]
        if not indices:
            continue
        regime_labels = y_true[indices]
        incumbent_regime = metric_dict(
            regime_labels,
            np.asarray(incumbent_probabilities, dtype=float)[indices],
        )
        challenger_regime = metric_dict(
            regime_labels,
            np.asarray(challenger_probabilities, dtype=float)[indices],
        )
        by_regime[regime] = {
            "n": len(indices),
            "incumbent": incumbent_regime,
            "challenger": challenger_regime,
        }
    delta_log_loss = challenger_metrics.log_loss - incumbent_metrics.log_loss
    delta_brier = challenger_metrics.brier - incumbent_metrics.brier

    if delta_log_loss <= -log_loss_epsilon:
        status: Literal["promote", "reject"] = "promote"
        promote = True
        reason = "log_loss_improved"
    elif delta_log_loss >= log_loss_epsilon:
        status = "reject"
        promote = False
        reason = "log_loss_regressed"
    elif delta_brier <= -brier_epsilon:
        status = "promote"
        promote = True
        reason = "brier_tiebreak_improved"
    else:
        status = "reject"
        promote = False
        reason = "brier_tiebreak_not_improved"

    message = _format_decision_message(
        status=status,
        n=n_rows,
        cutoff=cutoff,
        incumbent=incumbent_metrics,
        challenger=challenger_metrics,
        delta_log_loss=delta_log_loss,
        delta_brier=delta_brier,
        reason=reason,
    )
    return PromotionDecision(
        status=status,
        promote=promote,
        n=n_rows,
        cutoff=cutoff,
        min_samples=min_samples,
        log_loss_epsilon=float(log_loss_epsilon),
        brier_epsilon=float(brier_epsilon),
        incumbent=incumbent_metrics,
        challenger=challenger_metrics,
        delta_log_loss=delta_log_loss,
        delta_brier=delta_brier,
        reason=reason,
        holdout_sha256=holdout_sha256,
        prediction_sha256=prediction_sha256,
        message=message,
        cohort=cohort,
        by_regime=by_regime,
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_version(version: str) -> str:
    value = str(version or "").strip()
    if not _VERSION_PATTERN.fullmatch(value):
        raise PromotionValidationError("registry version must match YYYYMMDD_HHMMSSZ")
    return value


def _registry_root(registry_dir: str | Path) -> Path:
    path = Path(registry_dir)
    if not path.exists() or not path.is_dir():
        raise PromotionValidationError(f"registry directory does not exist: {path}")
    return path.resolve(strict=True)


def reference_for_version(registry_dir: str | Path, version: str) -> ModelReference:
    """Resolve and hash one direct, timestamped child of ``registry_dir``."""

    root = _registry_root(registry_dir)
    safe_version = _validate_version(version)
    version_dir = root / safe_version
    if not version_dir.exists() or not version_dir.is_dir():
        raise PromotionValidationError(f"registry version does not exist: {safe_version}")
    resolved_version_dir = version_dir.resolve(strict=True)
    if resolved_version_dir.parent != root:
        raise PromotionValidationError("registry version escapes registry root")
    artifact = resolved_version_dir / "model.pkl"
    if not artifact.exists() or not artifact.is_file():
        raise PromotionValidationError(f"registry artifact does not exist: {artifact}")
    resolved_artifact = artifact.resolve(strict=True)
    if resolved_artifact.parent != resolved_version_dir:
        raise PromotionValidationError("registry artifact escapes its version directory")
    return ModelReference(
        version=safe_version,
        artifact=resolved_artifact,
        sha256=sha256_file(resolved_artifact),
    )


def _pointer_path(registry_dir: str | Path, pointer_name: str) -> Path:
    if pointer_name not in _POINTER_NAMES:
        raise PromotionValidationError("pointer_name must be 'latest' or 'last_good'")
    return _registry_root(registry_dir) / f"{pointer_name}.json"


def read_model_pointer(registry_dir: str | Path, pointer_name: Literal["latest", "last_good"]) -> ModelReference:
    """Read a pointer and validate version, canonical path, and stored hash.

    Legacy ``latest.json`` files without a hash are accepted only after their
    canonical artifact has been resolved and hashed. Any stored hash is always
    mandatory evidence and must match exactly.
    """

    root = _registry_root(registry_dir)
    pointer_path = _pointer_path(root, pointer_name)
    if not pointer_path.exists():
        raise PromotionValidationError(f"model pointer does not exist: {pointer_path}")
    try:
        payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PromotionValidationError(f"model pointer is unreadable: {pointer_path}") from exc
    if not isinstance(payload, dict):
        raise PromotionValidationError("model pointer payload must be an object")
    version = payload.get("version") or payload.get(pointer_name)
    if pointer_name == "latest":
        version = version or payload.get("latest")
    reference = reference_for_version(root, str(version or ""))

    artifact_value = payload.get("artifact")
    if artifact_value:
        artifact_path = Path(str(artifact_value))
        if not artifact_path.is_absolute():
            artifact_path = root / artifact_path
        try:
            resolved_pointer_artifact = artifact_path.resolve(strict=True)
        except OSError as exc:
            raise PromotionValidationError("pointer artifact path does not resolve") from exc
        if resolved_pointer_artifact != reference.artifact:
            raise PromotionValidationError("pointer artifact does not match its canonical registry version")

    stored_hash = payload.get("sha256")
    if stored_hash is not None:
        normalized_hash = str(stored_hash).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", normalized_hash):
            raise PromotionValidationError("pointer sha256 is malformed")
        if normalized_hash != reference.sha256:
            raise PromotionValidationError("pointer sha256 does not match artifact")
    return reference


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_model_pointer(
    registry_dir: str | Path,
    pointer_name: Literal["latest", "last_good"],
    reference: ModelReference,
) -> Path:
    """Atomically write a pointer only after revalidating its artifact hash."""

    root = _registry_root(registry_dir)
    canonical = reference_for_version(root, reference.version)
    if canonical.artifact != reference.artifact.resolve(strict=True) or canonical.sha256 != reference.sha256:
        raise PromotionValidationError("model reference changed or does not belong to this registry")
    payload: dict[str, Any] = {
        "version": canonical.version,
        pointer_name: canonical.version,
        "artifact": str(Path(canonical.version) / "model.pkl"),
        "sha256": canonical.sha256,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    pointer_path = _pointer_path(root, pointer_name)
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    _atomic_write_bytes(pointer_path, encoded)
    return pointer_path


def _runtime_model_path(registry_dir: str | Path, production_path: str | Path | None) -> Path:
    root = _registry_root(registry_dir)
    expected = root.parent / "model.pkl"
    supplied = Path(production_path) if production_path is not None else expected
    if supplied.absolute() != expected.absolute():
        raise PromotionValidationError(f"production path must be the registry sibling model.pkl: {expected}")
    if supplied.is_symlink():
        raise PromotionValidationError("production model.pkl must not be a symlink")
    return supplied


def _verified_production_hash(path: Path, expected: ModelReference) -> str:
    if not path.exists() or not path.is_file():
        raise PromotionValidationError(f"production artifact does not exist: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != expected.sha256:
        raise PromotionValidationError("production artifact hash does not match latest pointer")
    return actual_hash


def _atomic_copy_artifact(source: ModelReference, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".model.pkl.", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source.artifact, temporary)
        copied_hash = sha256_file(temporary)
        if copied_hash != source.sha256:
            raise PromotionValidationError("staged production copy hash does not match registry artifact")
        os.replace(temporary, destination)
        final_hash = sha256_file(destination)
        if final_hash != source.sha256:
            raise PromotionValidationError("published production artifact hash does not match registry artifact")
        return final_hash
    finally:
        temporary.unlink(missing_ok=True)


def _optional_pointer(registry_dir: Path, pointer_name: Literal["latest", "last_good"]) -> ModelReference | None:
    if not (registry_dir / f"{pointer_name}.json").exists():
        return None
    return read_model_pointer(registry_dir, pointer_name)


@contextmanager
def _deployment_lock(registry_dir: Path) -> Iterator[None]:
    """Serialize pointer/runtime mutations with a fail-closed lock file."""

    lock_path = registry_dir / ".deployment.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise PromotionValidationError(
            f"another model deployment is active (or left a stale lock): {lock_path}"
        ) from exc
    preserve_lock = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()}\n")
            handle.flush()
            os.fsync(handle.fileno())
        yield
    except DeploymentRecoveryError:
        preserve_lock = True
        raise
    finally:
        if not preserve_lock:
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                # The deployment result is already authoritative. A leftover
                # lock fails future mutations closed for manual inspection.
                pass


@contextmanager
def deployment_lock(registry_dir: str | Path) -> Iterator[None]:
    """Serialize any registry mutation with model deployment operations.

    Retention uses this public wrapper so a pointer cannot move after its
    keep-list is refreshed but before an obsolete version is deleted.
    """

    root = _registry_root(registry_dir)
    with _deployment_lock(root):
        yield


def _validate_expected_incumbent(
    registry_dir: Path,
    actual: ModelReference,
    expected: ModelReference,
) -> None:
    canonical_expected = reference_for_version(registry_dir, expected.version)
    if (
        canonical_expected.artifact != expected.artifact.resolve(strict=True)
        or canonical_expected.sha256 != expected.sha256
    ):
        raise PromotionValidationError("expected incumbent reference is stale or outside this registry")
    if actual.version != canonical_expected.version or actual.sha256 != canonical_expected.sha256:
        raise PromotionValidationError(
            "latest changed after challenger evaluation; refusing promotion (compare-and-swap failed)"
        )


def _validate_expected_optional_pointer(
    registry_dir: Path,
    actual: ModelReference | None,
    expected: ModelReference | None,
    *,
    pointer_name: str,
) -> None:
    if expected is None:
        if actual is not None:
            raise PromotionValidationError(f"{pointer_name} changed before lock (compare-and-swap failed)")
        return
    canonical_expected = reference_for_version(registry_dir, expected.version)
    if (
        canonical_expected.artifact != expected.artifact.resolve(strict=True)
        or canonical_expected.sha256 != expected.sha256
    ):
        raise PromotionValidationError(f"expected {pointer_name} reference is stale or outside this registry")
    if actual is None or actual.version != canonical_expected.version or actual.sha256 != canonical_expected.sha256:
        raise PromotionValidationError(f"{pointer_name} changed before lock (compare-and-swap failed)")


def _restore_pointer_snapshot(path: Path, snapshot: bytes | None) -> None:
    if snapshot is None:
        path.unlink(missing_ok=True)
    else:
        _atomic_write_bytes(path, snapshot)


def _run_recovery_steps(original: Exception, steps: Sequence[tuple[str, Any]]) -> None:
    errors: list[str] = []
    for name, restore in steps:
        try:
            restore()
        except Exception as exc:  # pragma: no cover - exact failures are injected in tests
            errors.append(f"{name}={type(exc).__name__}:{exc}")
    if errors:
        raise DeploymentRecoveryError(
            "deployment recovery incomplete; lock preserved; " + "; ".join(errors)
        ) from original


def promote_candidate(
    registry_dir: str | Path,
    candidate_version: str,
    decision: PromotionDecision,
    *,
    production_path: str | Path | None = None,
    expected_incumbent: ModelReference,
    expected_candidate_sha256: str,
) -> DeploymentResult:
    """Publish an accepted candidate, or return without mutation on rejection.

    The outgoing live version becomes ``last_good``. The runtime ``model.pkl``
    is copied through a same-directory temporary file and replaced atomically.
    On any failure, the previous production copy and pointer bytes are restored.
    """

    root = _registry_root(registry_dir)
    safe_candidate_version = _validate_version(candidate_version)
    production = _runtime_model_path(root, production_path)
    with _deployment_lock(root):
        incumbent = read_model_pointer(root, "latest")
        _validate_expected_incumbent(root, incumbent, expected_incumbent)
        production_hash = _verified_production_hash(production, incumbent)
        previous_last_good = _optional_pointer(root, "last_good")

        if not decision.promote:
            return DeploymentResult(
                action="rejected",
                changed=False,
                latest=incumbent,
                last_good=previous_last_good,
                production_sha256=production_hash,
                message=decision.message,
            )

        candidate = reference_for_version(root, safe_candidate_version)
        if candidate.sha256 != expected_candidate_sha256:
            raise PromotionValidationError(
                "candidate changed after evaluation; refusing promotion (compare-and-swap failed)"
            )
        if candidate.version == incumbent.version:
            raise PromotionValidationError("candidate is already the live version")

        latest_path = _pointer_path(root, "latest")
        last_good_path = _pointer_path(root, "last_good")
        latest_snapshot = latest_path.read_bytes()
        last_good_snapshot = last_good_path.read_bytes() if last_good_path.exists() else None
        try:
            write_model_pointer(root, "last_good", incumbent)
            production_hash = _atomic_copy_artifact(candidate, production)
            write_model_pointer(root, "latest", candidate)
            published_latest = read_model_pointer(root, "latest")
            published_last_good = read_model_pointer(root, "last_good")
            _verified_production_hash(production, published_latest)
        except Exception as exc:
            _run_recovery_steps(
                exc,
                (
                    ("runtime", lambda: _atomic_copy_artifact(incumbent, production)),
                    ("latest", lambda: _restore_pointer_snapshot(latest_path, latest_snapshot)),
                    ("last_good", lambda: _restore_pointer_snapshot(last_good_path, last_good_snapshot)),
                ),
            )
            raise

        return DeploymentResult(
            action="promoted",
            changed=True,
            latest=published_latest,
            last_good=published_last_good,
            production_sha256=production_hash,
            message=decision.message,
        )


def bootstrap_candidate(
    registry_dir: str | Path,
    candidate_version: str,
    *,
    production_path: str | Path | None = None,
    expected_candidate_sha256: str,
) -> DeploymentResult:
    """Publish the first validated model without inventing an incumbent.

    Bootstrap is intentionally fail-closed: neither a runtime artifact nor a
    model pointer may already exist.  The sole validated model initializes both
    pointers; the next real promotion will replace ``last_good`` with the
    outgoing incumbent, making rollback useful from that point onward.
    """

    root = _registry_root(registry_dir)
    production = _runtime_model_path(root, production_path)
    latest_path = _pointer_path(root, "latest")
    last_good_path = _pointer_path(root, "last_good")
    with _deployment_lock(root):
        if production.exists() or latest_path.exists() or last_good_path.exists():
            raise PromotionValidationError("bootstrap requires absent production, latest, and last_good artifacts")

        candidate = reference_for_version(root, candidate_version)
        if candidate.sha256 != expected_candidate_sha256:
            raise PromotionValidationError(
                "candidate changed after validation; refusing bootstrap (compare-and-swap failed)"
            )
        try:
            production_hash = _atomic_copy_artifact(candidate, production)
            write_model_pointer(root, "last_good", candidate)
            write_model_pointer(root, "latest", candidate)
            published_latest = read_model_pointer(root, "latest")
            published_last_good = read_model_pointer(root, "last_good")
            _verified_production_hash(production, published_latest)
        except Exception as exc:
            _run_recovery_steps(
                exc,
                (
                    ("runtime", lambda: production.unlink(missing_ok=True)),
                    ("latest", lambda: latest_path.unlink(missing_ok=True)),
                    ("last_good", lambda: last_good_path.unlink(missing_ok=True)),
                ),
            )
            raise

        message = f"BOOTSTRAP: latest={candidate.version} last_good={candidate.version} sha256={production_hash}"
        return DeploymentResult(
            action="bootstrapped",
            changed=True,
            latest=published_latest,
            last_good=published_last_good,
            production_sha256=production_hash,
            message=message,
        )


def rollback_last_good(
    registry_dir: str | Path,
    *,
    production_path: str | Path | None = None,
) -> DeploymentResult:
    """Restore ``last_good`` and rotate the outgoing live version into it."""

    root = _registry_root(registry_dir)
    production = _runtime_model_path(root, production_path)
    with _deployment_lock(root):
        outgoing = read_model_pointer(root, "latest")
        target = read_model_pointer(root, "last_good")
        _verified_production_hash(production, outgoing)
        if target.version == outgoing.version:
            raise PromotionValidationError("latest and last_good point to the same version")

        latest_path = _pointer_path(root, "latest")
        last_good_path = _pointer_path(root, "last_good")
        latest_snapshot = latest_path.read_bytes()
        last_good_snapshot = last_good_path.read_bytes()
        try:
            production_hash = _atomic_copy_artifact(target, production)
            write_model_pointer(root, "last_good", outgoing)
            write_model_pointer(root, "latest", target)
            latest = read_model_pointer(root, "latest")
            last_good = read_model_pointer(root, "last_good")
            _verified_production_hash(production, latest)
        except Exception as exc:
            _run_recovery_steps(
                exc,
                (
                    ("runtime", lambda: _atomic_copy_artifact(outgoing, production)),
                    ("latest", lambda: _restore_pointer_snapshot(latest_path, latest_snapshot)),
                    ("last_good", lambda: _restore_pointer_snapshot(last_good_path, last_good_snapshot)),
                ),
            )
            raise

        message = f"ROLLBACK: latest={latest.version} last_good={last_good.version} sha256={production_hash}"
        return DeploymentResult(
            action="rolled_back",
            changed=True,
            latest=latest,
            last_good=last_good,
            production_sha256=production_hash,
            message=message,
        )


def set_last_good_version(
    registry_dir: str | Path,
    target_version: str,
    *,
    production_path: str | Path | None = None,
    expected_latest: ModelReference,
    expected_last_good: ModelReference | None,
    expected_target_sha256: str,
) -> LastGoodUpdateResult:
    """Pin one explicitly identified registry artifact as ``last_good``.

    The caller snapshots ``latest`` and the previous ``last_good`` before doing
    any expensive artifact validation. Under the deployment lock this function
    revalidates both snapshots, the runtime copy, and the target SHA-256. It only
    writes ``last_good.json``; ``latest.json`` and runtime ``model.pkl`` remain
    byte-for-byte untouched.
    """

    root = _registry_root(registry_dir)
    safe_target_version = _validate_version(target_version)
    production = _runtime_model_path(root, production_path)
    normalized_target_hash = str(expected_target_sha256 or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized_target_hash):
        raise PromotionValidationError("expected target SHA-256 is malformed")

    with _deployment_lock(root):
        latest = read_model_pointer(root, "latest")
        _validate_expected_incumbent(root, latest, expected_latest)
        production_hash = _verified_production_hash(production, latest)
        previous_last_good = _optional_pointer(root, "last_good")
        _validate_expected_optional_pointer(
            root,
            previous_last_good,
            expected_last_good,
            pointer_name="last_good",
        )

        target = reference_for_version(root, safe_target_version)
        if target.sha256 != normalized_target_hash:
            raise PromotionValidationError("target changed after validation (compare-and-swap failed)")
        if target.version == latest.version:
            raise PromotionValidationError("last_good target must differ from latest")
        if previous_last_good is not None and (
            previous_last_good.version == target.version and previous_last_good.sha256 == target.sha256
        ):
            return LastGoodUpdateResult(
                changed=False,
                latest=latest,
                previous_last_good=previous_last_good,
                last_good=previous_last_good,
                production_sha256=production_hash,
                message=f"LAST_GOOD UNCHANGED: version={target.version} sha256={target.sha256}",
            )

        last_good_path = _pointer_path(root, "last_good")
        last_good_snapshot = last_good_path.read_bytes() if last_good_path.exists() else None
        try:
            write_model_pointer(root, "last_good", target)
            published_last_good = read_model_pointer(root, "last_good")
            current_latest = read_model_pointer(root, "latest")
            if current_latest.version != latest.version or current_latest.sha256 != latest.sha256:
                raise PromotionValidationError("latest changed while setting last_good")
            _verified_production_hash(production, current_latest)
        except Exception as exc:
            _run_recovery_steps(
                exc,
                (("last_good", lambda: _restore_pointer_snapshot(last_good_path, last_good_snapshot)),),
            )
            raise

        return LastGoodUpdateResult(
            changed=True,
            latest=latest,
            previous_last_good=previous_last_good,
            last_good=published_last_good,
            production_sha256=production_hash,
            message=f"LAST_GOOD SET: version={target.version} sha256={target.sha256}",
        )
