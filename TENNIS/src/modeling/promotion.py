"""Puerta temporal champion/challenger y punteros de despliegue de TENNIS.

La comparación usa el último bloque anual OOF común de ambos runs. Cada fold
fue entrenado en años anteriores, calibrado en el año inmediatamente anterior
y puntuado en el año de test; por ello nunca se aplica el modelo final a filas
que ya observó. Champion y challenger se emparejan por ``gender+record_id`` y
deben tener exactamente las mismas fechas, etiquetas y segmentos.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from ..config import PHASE7_MODELS_DIR, PHASE7_PROMOTION_CONFIG_PATH
from .artifacts import (
    ArtifactError,
    PublishedRun,
    _activate_gate_approved_run,
    find_verified_run,
    sha256_file,
    verify_published_run,
)
from .metrics import binary_classification_metrics


class PromotionError(RuntimeError):
    """Indica que la comparación o la conmutación no fue segura."""


@dataclass(frozen=True, slots=True)
class PromotionConfig:
    """Parámetros versionados de decisión y consistencia."""

    log_loss_epsilon: float
    brier_epsilon: float
    metric_consistency_atol: float
    minimum_holdout_rows: int
    holdout_seasons: int
    retention_keep_latest: int


@dataclass(frozen=True, slots=True)
class PromotionMetrics:
    """Métricas probabilísticas decisivas y accuracy de producto."""

    n: int
    accuracy: float
    log_loss: float
    brier: float


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    """Resultado auditable de una comparación sobre soporte idéntico."""

    status: Literal["promote", "reject", "insufficient"]
    promote: bool
    reason: str
    champion_fingerprint: str
    challenger_fingerprint: str
    holdout_seasons: tuple[int, ...]
    period_start: str
    period_end: str
    n: int
    holdout_sha256: str
    champion_prediction_sha256: str
    challenger_prediction_sha256: str
    prediction_column: str
    calibration: Mapping[str, object]
    champion: PromotionMetrics | None
    challenger: PromotionMetrics | None
    delta_log_loss: float | None
    delta_brier: float | None
    log_loss_epsilon: float
    brier_epsilon: float
    consistency: Mapping[str, object]
    by_gender: tuple[Mapping[str, object], ...]
    by_segment: tuple[Mapping[str, object], ...]
    created_at_utc: str

    def as_dict(self) -> dict[str, object]:
        """Convierte la decisión a JSON sin perder deltas ni configuración."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class PromotionResult:
    """Estado final del despliegue tras puerta, bootstrap o reutilización."""

    action: Literal["bootstrapped", "promoted", "rejected", "active_reused", "rolled_back"]
    changed: bool
    active: PublishedRun
    last_good: PublishedRun
    decision: PromotionDecision | None
    message: str


_HOLDOUT_COLUMNS = (
    "record_id",
    "gender",
    "match_date",
    "result_available_date",
    "tour_level",
    "surface",
    "y",
    "test_season",
    "lightgbm_platt",
)
_PAIRING_COLUMNS = (
    "record_id",
    "gender",
    "match_date",
    "result_available_date",
    "tour_level",
    "surface",
    "y",
    "test_season",
)


def load_promotion_config(
    path: Path = PHASE7_PROMOTION_CONFIG_PATH,
) -> PromotionConfig:
    """Carga y valida los márgenes anti-churn y la política de retención."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PromotionError(f"No se pudo leer la configuración {path}.") from exc
    if not isinstance(payload, Mapping):
        raise PromotionError("La configuración de promoción debe ser un objeto JSON.")
    try:
        log_epsilon = float(payload["log_loss_epsilon"])
        brier_epsilon = float(payload["brier_epsilon"])
        tolerance = float(payload["metric_consistency_atol"])
        minimum = int(payload["minimum_holdout_rows"])
        seasons = int(payload["holdout_seasons"])
        retention = int(payload["retention_keep_latest"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PromotionError("La configuración de promoción está incompleta.") from exc
    if (
        not all(
            math.isfinite(value) and value >= 0.0
            for value in (log_epsilon, brier_epsilon, tolerance)
        )
        or minimum < 1
        or seasons < 1
        or retention < 1
    ):
        raise PromotionError("La configuración de promoción contiene límites inválidos.")
    return PromotionConfig(
        log_loss_epsilon=log_epsilon,
        brier_epsilon=brier_epsilon,
        metric_consistency_atol=tolerance,
        minimum_holdout_rows=minimum,
        holdout_seasons=seasons,
        retention_keep_latest=retention,
    )


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    """Publica un JSON lateral mediante reemplazo atómico."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    temporary.write_text(text + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_active_run(output_dir: Path) -> tuple[PublishedRun, str] | None:
    """Resuelve el champion y devuelve también el hash CAS del puntero."""

    output = Path(output_dir).resolve()
    pointer = output / "manifest.json"
    if not pointer.exists():
        return None
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PromotionError(f"No se pudo leer el champion {pointer}.") from exc
    if not isinstance(payload, Mapping):
        raise PromotionError("El manifiesto activo no es un objeto JSON.")
    fingerprint = payload.get("fingerprint")
    active_run = payload.get("active_run")
    if not isinstance(fingerprint, str) or active_run != f"runs/{fingerprint}":
        raise PromotionError("El puntero activo no reconcilia con su fingerprint.")
    published = find_verified_run(fingerprint, output_dir=output)
    if published is None:
        raise PromotionError("El run señalado por el champion no existe.")
    return published, sha256_file(pointer)


def _pointer_payload(run: PublishedRun, *, reason: str) -> dict[str, object]:
    """Construye una referencia last_good verificable e independiente."""

    return {
        "fingerprint": run.fingerprint,
        "run": f"runs/{run.fingerprint}",
        "manifest_sha256": sha256_file(run.run_dir / "manifest.json"),
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "reason": reason,
    }


def _read_last_good(output_dir: Path) -> PublishedRun | None:
    """Carga y verifica el puntero de rollback, si existe."""

    output = Path(output_dir).resolve()
    pointer = output / "last_good.json"
    if not pointer.exists():
        return None
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PromotionError(f"No se pudo leer {pointer}.") from exc
    if not isinstance(payload, Mapping):
        raise PromotionError("last_good debe ser un objeto JSON.")
    fingerprint = payload.get("fingerprint")
    if not isinstance(fingerprint, str) or payload.get("run") != f"runs/{fingerprint}":
        raise PromotionError("last_good contiene una referencia inválida.")
    run = find_verified_run(fingerprint, output_dir=output)
    if run is None:
        raise PromotionError("last_good apunta a un run inexistente.")
    if payload.get("manifest_sha256") != sha256_file(run.run_dir / "manifest.json"):
        raise PromotionError("El hash de last_good no coincide con el run.")
    return run


def _created_at(run: PublishedRun) -> datetime:
    """Extrae el instante UTC de un run verificado."""

    value = run.manifest.get("created_at_utc")
    if not isinstance(value, str):
        raise PromotionError(f"El run {run.fingerprint} no declara created_at_utc.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PromotionError("created_at_utc no es ISO válido.") from exc
    if parsed.tzinfo is None:
        raise PromotionError("created_at_utc debe incluir zona horaria.")
    return parsed.astimezone(UTC)


def ensure_last_good(output_dir: Path = PHASE7_MODELS_DIR) -> PublishedRun:
    """Inicializa last_good sin alterar el champion de una instalación legacy."""

    existing = _read_last_good(output_dir)
    if existing is not None:
        return existing
    active_state = _load_active_run(output_dir)
    if active_state is None:
        raise PromotionError("No se puede inicializar last_good sin champion.")
    active, _pointer_sha = active_state
    candidates: list[PublishedRun] = []
    runs_dir = Path(output_dir).resolve() / "runs"
    for path in runs_dir.iterdir():
        if path.is_dir() and path.name != active.fingerprint:
            candidate = find_verified_run(path.name, output_dir=output_dir)
            if candidate is not None:
                candidates.append(candidate)
    selected = max(candidates, key=_created_at) if candidates else active
    _atomic_json(
        Path(output_dir).resolve() / "last_good.json",
        _pointer_payload(selected, reason="legacy_gate_bootstrap"),
    )
    return selected


def _evaluation_protocol(manifest: Mapping[str, object]) -> tuple[int, int, Mapping[str, object]]:
    """Extrae temporadas y calibración del contrato persistido."""

    identity = manifest.get("identity")
    parameters = identity.get("parameters") if isinstance(identity, Mapping) else None
    temporal = parameters.get("temporal_evaluation") if isinstance(parameters, Mapping) else None
    calibration = parameters.get("calibration") if isinstance(parameters, Mapping) else None
    if not isinstance(temporal, Mapping) or not isinstance(calibration, Mapping):
        raise PromotionError("El run no declara evaluación temporal y calibración.")
    try:
        first = int(temporal["first_test_season"])
        last = int(temporal["last_test_season"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PromotionError("Las temporadas de evaluación son inválidas.") from exc
    return first, last, dict(calibration)


def _load_holdout(run: PublishedRun, seasons: Sequence[int]) -> pd.DataFrame:
    """Carga únicamente las columnas OOF necesarias para las temporadas dadas."""

    frames: list[pd.DataFrame] = []
    selected = set(seasons)
    for gender in ("M", "F"):
        path = run.run_dir / "evaluation" / f"oof_{gender}.parquet"
        try:
            frame = pd.read_parquet(path, columns=list(_HOLDOUT_COLUMNS))
        except Exception as exc:
            raise PromotionError(f"No se pudo cargar el OOF verificado {path}.") from exc
        frame = frame.loc[frame["test_season"].isin(selected)].copy()
        frames.append(frame)
    holdout = pd.concat(frames, ignore_index=True)
    if holdout.empty:
        raise PromotionError("El run no contiene el hold-out temporal solicitado.")
    if holdout.duplicated(["gender", "record_id"]).any():
        raise PromotionError("El hold-out contiene identidades duplicadas.")
    holdout["match_date"] = pd.to_datetime(holdout["match_date"], errors="raise")
    holdout["result_available_date"] = pd.to_datetime(
        holdout["result_available_date"], errors="raise"
    )
    holdout["y"] = pd.to_numeric(holdout["y"], errors="raise").astype("int8")
    holdout["lightgbm_platt"] = pd.to_numeric(holdout["lightgbm_platt"], errors="raise").astype(
        float
    )
    if (
        not holdout["y"].isin([0, 1]).all()
        or not np.isfinite(holdout["lightgbm_platt"]).all()
        or not holdout["lightgbm_platt"].between(0.0, 1.0).all()
    ):
        raise PromotionError("El hold-out contiene targets o probabilidades inválidas.")
    return holdout.sort_values(["gender", "record_id"]).reset_index(drop=True)


def _metrics(frame: pd.DataFrame) -> PromotionMetrics:
    """Puntúa el predictor calibrado con el evaluador canónico."""

    values = binary_classification_metrics(frame["y"], frame["lightgbm_platt"])
    return PromotionMetrics(
        n=int(values["n_evaluated"]),
        accuracy=float(values["accuracy"]),
        log_loss=float(values["log_loss"]),
        brier=float(values["brier"]),
    )


def _canonical_hash(records: object) -> str:
    """Calcula SHA-256 sobre un payload JSON canónico."""

    encoded = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _probability_hash(frame: pd.DataFrame) -> str:
    """Liga orden, identidad y probabilidades de una evaluación."""

    rows = [
        {
            "gender": str(row.gender),
            "record_id": str(row.record_id),
            "probability": float(row.lightgbm_platt),
        }
        for row in frame.itertuples(index=False)
    ]
    return _canonical_hash(rows)


def _holdout_hash(frame: pd.DataFrame) -> str:
    """Liga identidad, fecha y etiqueta del soporte común."""

    rows = [
        {
            "gender": str(row.gender),
            "record_id": str(row.record_id),
            "date": row.match_date.date().isoformat(),
            "y": int(row.y),
        }
        for row in frame.itertuples(index=False)
    ]
    return _canonical_hash(rows)


def _segment_rows(
    champion: pd.DataFrame,
    challenger: pd.DataFrame,
) -> tuple[Mapping[str, object], ...]:
    """Reporta ambos modelos por nivel/superficie sobre el mismo soporte."""

    rows: list[Mapping[str, object]] = []
    grouped = champion.groupby(["gender", "tour_level", "surface"], dropna=False, sort=True)
    for keys, live_subset in grouped:
        gender, level, surface = keys
        level_mask = (
            challenger["tour_level"].isna()
            if pd.isna(level)
            else challenger["tour_level"].eq(level)
        )
        surface_mask = (
            challenger["surface"].isna() if pd.isna(surface) else challenger["surface"].eq(surface)
        )
        mask = challenger["gender"].eq(gender) & level_mask & surface_mask
        new_subset = challenger.loc[mask]
        if len(live_subset) != len(new_subset):
            raise PromotionError("El soporte segmentado dejó de ser común.")
        rows.append(
            {
                "segment": f"gender={gender}|tour_level={level}|surface={surface}",
                "champion": asdict(_metrics(live_subset)),
                "challenger": asdict(_metrics(new_subset)),
            }
        )
    return tuple(rows)


def compare_runs_on_common_holdout(
    champion: PublishedRun,
    challenger: PublishedRun,
    *,
    config: PromotionConfig,
) -> PromotionDecision:
    """Compara dos runs sobre exactamente las mismas filas OOF temporales."""

    first_live, last_live, live_calibration = _evaluation_protocol(champion.manifest)
    first_new, last_new, new_calibration = _evaluation_protocol(challenger.manifest)
    if (first_live, last_live) != (first_new, last_new):
        raise PromotionError("Champion y challenger no usan el mismo periodo de backtest.")
    if live_calibration != new_calibration:
        raise PromotionError("Champion y challenger no usan la misma calibración.")
    first_holdout = max(first_live, last_live - config.holdout_seasons + 1)
    seasons = tuple(range(first_holdout, last_live + 1))
    live = _load_holdout(champion, seasons)
    new = _load_holdout(challenger, seasons)
    if len(live) != len(new) or not live.loc[:, _PAIRING_COLUMNS].equals(
        new.loc[:, _PAIRING_COLUMNS]
    ):
        raise PromotionError("Champion y challenger no contienen exactamente el mismo hold-out.")
    repeated = _load_holdout(challenger, seasons)
    repeated_metrics = _metrics(repeated)
    new_metrics = _metrics(new)
    probability_delta = float(
        np.max(np.abs(repeated["lightgbm_platt"].to_numpy() - new["lightgbm_platt"].to_numpy()))
    )
    metric_deltas = {
        "accuracy": abs(repeated_metrics.accuracy - new_metrics.accuracy),
        "log_loss": abs(repeated_metrics.log_loss - new_metrics.log_loss),
        "brier": abs(repeated_metrics.brier - new_metrics.brier),
    }
    if probability_delta > config.metric_consistency_atol or any(
        delta > config.metric_consistency_atol for delta in metric_deltas.values()
    ):
        raise PromotionError("El mismo artefacto no produjo métricas repetibles.")
    consistency: Mapping[str, object] = {
        "status": "pass",
        "atol": config.metric_consistency_atol,
        "max_probability_delta": probability_delta,
        "metric_deltas": metric_deltas,
    }
    period_start = live["match_date"].min().date().isoformat()
    period_end = live["match_date"].max().date().isoformat()
    holdout_sha = _holdout_hash(live)
    if holdout_sha != _holdout_hash(new):
        raise PromotionError("El hash del soporte común no coincide.")
    live_metrics = _metrics(live)
    delta_log_loss = new_metrics.log_loss - live_metrics.log_loss
    delta_brier = new_metrics.brier - live_metrics.brier
    if len(new) < config.minimum_holdout_rows:
        status: Literal["promote", "reject", "insufficient"] = "insufficient"
        promote = False
        reason = f"insufficient_holdout:{len(new)}<{config.minimum_holdout_rows}"
    elif delta_log_loss <= -config.log_loss_epsilon:
        status = "promote"
        promote = True
        reason = "log_loss_improved"
    elif delta_log_loss >= config.log_loss_epsilon:
        status = "reject"
        promote = False
        reason = "log_loss_regressed"
    elif delta_brier <= -config.brier_epsilon:
        status = "promote"
        promote = True
        reason = "brier_tiebreak_improved"
    else:
        status = "reject"
        promote = False
        reason = "brier_tiebreak_not_improved"
    by_gender: list[Mapping[str, object]] = []
    for gender in ("M", "F"):
        live_gender = live.loc[live["gender"].eq(gender)]
        new_gender = new.loc[new["gender"].eq(gender)]
        by_gender.append(
            {
                "gender": gender,
                "champion": asdict(_metrics(live_gender)),
                "challenger": asdict(_metrics(new_gender)),
            }
        )
    return PromotionDecision(
        status=status,
        promote=promote,
        reason=reason,
        champion_fingerprint=champion.fingerprint,
        challenger_fingerprint=challenger.fingerprint,
        holdout_seasons=seasons,
        period_start=period_start,
        period_end=period_end,
        n=len(new),
        holdout_sha256=holdout_sha,
        champion_prediction_sha256=_probability_hash(live),
        challenger_prediction_sha256=_probability_hash(new),
        prediction_column="lightgbm_platt",
        calibration={"champion": live_calibration, "challenger": new_calibration},
        champion=live_metrics,
        challenger=new_metrics,
        delta_log_loss=delta_log_loss,
        delta_brier=delta_brier,
        log_loss_epsilon=config.log_loss_epsilon,
        brier_epsilon=config.brier_epsilon,
        consistency=consistency,
        by_gender=tuple(by_gender),
        by_segment=_segment_rows(live, new),
        created_at_utc=datetime.now(UTC).isoformat(),
    )


def _write_decision(output_dir: Path, decision: PromotionDecision, action: str) -> None:
    """Persiste la decisión lateral sin modificar el run inmutable."""

    payload = decision.as_dict()
    payload["deployment_action"] = action
    target = (
        Path(output_dir).resolve()
        / "promotion"
        / "decisions"
        / f"{decision.challenger_fingerprint}.json"
    )
    _atomic_json(target, payload)


def has_equivalent_feature_gate(
    *,
    champion_fingerprint: str,
    feature_fingerprint: str,
    output_dir: Path = PHASE7_MODELS_DIR,
) -> bool:
    """Accept a new feature identity only after exact gate equivalence.

    A serving-only Elo tail can change the feature artifact fingerprint while
    leaving every historical model input unchanged. The anti-churn gate must
    still reject the challenger, but its decision can attest that the live
    champion produced byte-identical probabilities on the same temporal
    hold-out. This function validates that evidence and the referenced
    challenger run; any malformed or missing field fails closed.
    """

    decisions_dir = Path(output_dir).resolve() / "promotion" / "decisions"
    if not decisions_dir.is_dir():
        return False
    for decision_path in sorted(decisions_dir.iterdir(), reverse=True):
        if decision_path.suffix != ".json" or not decision_path.is_file():
            continue
        try:
            payload = json.loads(decision_path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                continue
            challenger_fingerprint = payload.get("challenger_fingerprint")
            consistency = payload.get("consistency")
            champion_metrics = payload.get("champion")
            challenger_metrics = payload.get("challenger")
            if (
                payload.get("status") != "reject"
                or payload.get("promote") is not False
                or payload.get("deployment_action") != "rejected"
                or payload.get("champion_fingerprint") != champion_fingerprint
                or not isinstance(challenger_fingerprint, str)
                or decision_path.stem != challenger_fingerprint
                or not isinstance(consistency, Mapping)
                or consistency.get("status") != "pass"
                or not isinstance(champion_metrics, Mapping)
                or not isinstance(challenger_metrics, Mapping)
                or dict(champion_metrics) != dict(challenger_metrics)
            ):
                continue
            tolerance = consistency.get("atol")
            maximum_delta = consistency.get("max_probability_delta")
            if (
                isinstance(tolerance, bool)
                or not isinstance(tolerance, (int, float))
                or isinstance(maximum_delta, bool)
                or not isinstance(maximum_delta, (int, float))
                or float(tolerance) < 0.0
                or float(maximum_delta) > float(tolerance)
                or payload.get("champion_prediction_sha256")
                != payload.get("challenger_prediction_sha256")
                or not isinstance(payload.get("holdout_sha256"), str)
                or isinstance(payload.get("n"), bool)
                or not isinstance(payload.get("n"), int)
                or int(payload["n"]) < 1
            ):
                continue
            challenger = find_verified_run(
                challenger_fingerprint,
                output_dir=output_dir,
            )
            if (
                challenger is not None
                and challenger.manifest.get("feature_fingerprint") == feature_fingerprint
            ):
                return True
        except (ArtifactError, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return False


def gate_challenger(
    challenger: PublishedRun,
    *,
    output_dir: Path = PHASE7_MODELS_DIR,
    config_path: Path = PHASE7_PROMOTION_CONFIG_PATH,
) -> PromotionResult:
    """Bootstrappea o somete un challenger registrado a la puerta común."""

    verify_published_run(challenger.run_dir)
    config = load_promotion_config(config_path)
    active_state = _load_active_run(output_dir)
    if active_state is None:
        activated = _activate_gate_approved_run(
            challenger,
            output_dir=output_dir,
            expected_active_sha256=None,
        )
        _atomic_json(
            Path(output_dir).resolve() / "last_good.json",
            _pointer_payload(activated, reason="first_valid_model"),
        )
        return PromotionResult(
            action="bootstrapped",
            changed=True,
            active=activated,
            last_good=activated,
            decision=None,
            message=f"BOOTSTRAP: {activated.fingerprint} es el primer modelo válido.",
        )
    champion, active_sha = active_state
    last_good = ensure_last_good(output_dir)
    if champion.fingerprint == challenger.fingerprint:
        return PromotionResult(
            action="active_reused",
            changed=False,
            active=champion,
            last_good=last_good,
            decision=None,
            message=f"REUSE: {champion.fingerprint} ya es el champion activo.",
        )
    decision = compare_runs_on_common_holdout(champion, challenger, config=config)
    if not decision.promote:
        _write_decision(output_dir, decision, "rejected")
        return PromotionResult(
            action="rejected",
            changed=False,
            active=champion,
            last_good=last_good,
            decision=decision,
            message=(
                f"RECHAZADO n={decision.n}: log-loss vivo={decision.champion.log_loss:.6f} "
                f"challenger={decision.challenger.log_loss:.6f}; Brier vivo="
                f"{decision.champion.brier:.6f} challenger={decision.challenger.brier:.6f}."
                if decision.champion is not None and decision.challenger is not None
                else f"RECHAZADO: {decision.reason}."
            ),
        )
    last_good_path = Path(output_dir).resolve() / "last_good.json"
    previous_pointer = last_good_path.read_bytes() if last_good_path.exists() else None
    _atomic_json(last_good_path, _pointer_payload(champion, reason="pre_promotion_champion"))
    try:
        activated = _activate_gate_approved_run(
            challenger,
            output_dir=output_dir,
            expected_active_sha256=active_sha,
        )
    except Exception:
        if previous_pointer is None:
            last_good_path.unlink(missing_ok=True)
        else:
            temporary = last_good_path.with_name(".last_good.restore.tmp")
            temporary.write_bytes(previous_pointer)
            os.replace(temporary, last_good_path)
        raise
    _write_decision(output_dir, decision, "promoted")
    return PromotionResult(
        action="promoted",
        changed=True,
        active=activated,
        last_good=champion,
        decision=decision,
        message=(
            f"PROMOVIDO n={decision.n}: Δlog-loss={decision.delta_log_loss:+.6f}; "
            f"ΔBrier={decision.delta_brier:+.6f}."
        ),
    )


def rollback_last_good(output_dir: Path = PHASE7_MODELS_DIR) -> PromotionResult:
    """Restaura last_good mediante la misma conmutación verificada y CAS."""

    active_state = _load_active_run(output_dir)
    if active_state is None:
        raise PromotionError("No existe champion activo para hacer rollback.")
    active, active_sha = active_state
    last_good = _read_last_good(output_dir)
    if last_good is None:
        raise PromotionError("No existe puntero last_good.")
    if active.fingerprint == last_good.fingerprint:
        return PromotionResult(
            action="rolled_back",
            changed=False,
            active=active,
            last_good=last_good,
            decision=None,
            message="ROLLBACK: el champion ya coincide con last_good.",
        )
    restored = _activate_gate_approved_run(
        last_good,
        output_dir=output_dir,
        expected_active_sha256=active_sha,
    )
    return PromotionResult(
        action="rolled_back",
        changed=True,
        active=restored,
        last_good=last_good,
        decision=None,
        message=f"ROLLBACK: restaurado {last_good.fingerprint}.",
    )


def active_and_last_good(
    output_dir: Path = PHASE7_MODELS_DIR,
) -> tuple[PublishedRun, PublishedRun]:
    """Devuelve ambos punteros después de verificar artefactos y hashes."""

    active_state = _load_active_run(output_dir)
    if active_state is None:
        raise PromotionError("No existe champion activo.")
    return active_state[0], ensure_last_good(output_dir)
