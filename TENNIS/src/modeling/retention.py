"""Retención preview-first de runs con protección de champion y last_good.

La primera poda real exige confirmación exacta de una keep-list. Tras esa
aprobación, el pipeline puede aplicar automáticamente la misma política de N.
Cada error físico de borrado se convierte en un residuo auditable y nunca hace
fallar entrenamiento o inferencia.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Mapping

from ..config import (
    PHASE7_MODELS_DIR,
    PHASE7_PROMOTION_CONFIG_PATH,
)
from .artifacts import ArtifactError, find_verified_run, verify_published_run
from .promotion import active_and_last_good, load_promotion_config


class RetentionError(RuntimeError):
    """Indica que una keep-list es insegura, inválida u obsoleta."""


@dataclass(frozen=True, slots=True)
class RetentionPlan:
    """Snapshot exacto y confirmable de la retención de modelos."""

    output_dir: Path
    keep_latest: int
    active_fingerprint: str
    last_good_fingerprint: str
    keep_fingerprints: tuple[str, ...]
    delete_fingerprints: tuple[str, ...]
    token: str

    def as_dict(self) -> dict[str, object]:
        """Devuelve la keep-list serializable que se muestra al operador."""

        return {
            "output_dir": str(self.output_dir),
            "keep_latest": self.keep_latest,
            "active_fingerprint": self.active_fingerprint,
            "last_good_fingerprint": self.last_good_fingerprint,
            "keep_fingerprints": list(self.keep_fingerprints),
            "delete_fingerprints": list(self.delete_fingerprints),
            "confirmation_token": self.token,
        }


@dataclass(frozen=True, slots=True)
class RetentionResult:
    """Resultado no fatal de una poda manual o automática."""

    mode: str
    applied: bool
    deleted: tuple[str, ...]
    pending: tuple[Mapping[str, object], ...]
    plan: RetentionPlan
    message: str


def _created_at(run_dir: Path) -> datetime:
    """Lee la fecha UTC de un run cuya integridad ya fue validada."""

    payload = verify_published_run(run_dir)
    value = payload.get("created_at_utc")
    if not isinstance(value, str):
        raise RetentionError(f"El run {run_dir.name} no declara created_at_utc.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RetentionError(f"Fecha inválida en {run_dir.name}.") from exc
    if parsed.tzinfo is None:
        raise RetentionError("created_at_utc debe incluir zona horaria.")
    return parsed.astimezone(UTC)


def _plan_token(payload: Mapping[str, object]) -> str:
    """Firma la lista exacta de protección y eliminación."""

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def plan_model_retention(
    output_dir: Path = PHASE7_MODELS_DIR,
    *,
    config_path: Path = PHASE7_PROMOTION_CONFIG_PATH,
) -> RetentionPlan:
    """Calcula últimos N + champion + last_good sin borrar nada."""

    output = Path(output_dir).resolve()
    runs_dir = (output / "runs").resolve()
    if runs_dir.parent != output or not runs_dir.is_dir():
        raise RetentionError("No existe un directorio runs seguro.")
    active, last_good = active_and_last_good(output)
    keep_latest = load_promotion_config(config_path).retention_keep_latest
    ordered: list[tuple[datetime, str]] = []
    for path in runs_dir.iterdir():
        if not path.is_dir() or path.is_symlink():
            continue
        run = find_verified_run(path.name, output_dir=output)
        if run is None:
            continue
        ordered.append((_created_at(run.run_dir), run.fingerprint))
    ordered.sort(reverse=True)
    latest = {fingerprint for _created, fingerprint in ordered[:keep_latest]}
    protected = latest | {active.fingerprint, last_good.fingerprint}
    keep = tuple(fingerprint for _created, fingerprint in ordered if fingerprint in protected)
    delete = tuple(
        fingerprint for _created, fingerprint in reversed(ordered) if fingerprint not in protected
    )
    if active.fingerprint in delete or last_good.fingerprint in delete:
        raise RetentionError("La keep-list intentó eliminar un puntero protegido.")
    unsigned: dict[str, object] = {
        "output_dir": str(output),
        "keep_latest": keep_latest,
        "active_fingerprint": active.fingerprint,
        "last_good_fingerprint": last_good.fingerprint,
        "keep_fingerprints": list(keep),
        "delete_fingerprints": list(delete),
    }
    return RetentionPlan(
        output_dir=output,
        keep_latest=keep_latest,
        active_fingerprint=active.fingerprint,
        last_good_fingerprint=last_good.fingerprint,
        keep_fingerprints=keep,
        delete_fingerprints=delete,
        token=_plan_token(unsigned),
    )


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    """Escribe estado lateral sin exponer un JSON parcial."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_pending(path: Path) -> list[dict[str, object]]:
    """Carga residuos previos sin convertir corrupción en borrado."""

    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetentionError(f"No se pudo leer {path}.") from exc
    rows = payload.get("pending") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise RetentionError("cleanup_pending.json no contiene una lista válida.")
    return [dict(row) for row in rows]


def _save_pending(path: Path, rows: list[dict[str, object]]) -> None:
    """Persiste únicamente residuos que todavía existen."""

    remaining = [row for row in rows if Path(str(row.get("path", ""))).exists()]
    _atomic_json(
        path,
        {
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "pending": remaining,
        },
    )


def _approval_matches(plan: RetentionPlan, approval_path: Path) -> bool:
    """Comprueba si el operador aprobó esta política de N previamente."""

    if not approval_path.is_file():
        return False
    try:
        payload = json.loads(approval_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(payload, Mapping)
        and payload.get("retention_keep_latest") == plan.keep_latest
        and payload.get("scope") == "TENNIS/models/phase7/runs"
    )


def apply_retention_plan(
    plan: RetentionPlan,
    *,
    confirmation_token: str,
    pending_path: Path | None = None,
) -> RetentionResult:
    """Aplica una preview vigente; cada fallo queda pendiente y continúa."""

    if confirmation_token != plan.token:
        raise RetentionError("El token no confirma esta keep-list exacta.")
    current = plan_model_retention(plan.output_dir)
    if current != plan:
        raise RetentionError("La keep-list cambió desde la previsualización.")
    runs_dir = plan.output_dir / "runs"
    protected = set(plan.keep_fingerprints) | {
        plan.active_fingerprint,
        plan.last_good_fingerprint,
    }
    resolved_pending = (
        plan.output_dir / "cleanup_pending.json" if pending_path is None else Path(pending_path)
    )
    deleted: list[str] = []
    pending = _load_pending(resolved_pending)
    pending_by_fingerprint = {str(row.get("fingerprint")): row for row in pending}
    for fingerprint in plan.delete_fingerprints:
        if fingerprint in protected:
            raise RetentionError("La poda intentó tocar champion o last_good.")
        target = (runs_dir / fingerprint).resolve()
        if target.parent != runs_dir.resolve():
            raise RetentionError(f"Ruta de borrado insegura: {target}.")
        try:
            shutil.rmtree(target)
        except OSError as exc:
            pending_by_fingerprint[fingerprint] = {
                "fingerprint": fingerprint,
                "path": str(target),
                "error": f"{type(exc).__name__}: {exc}",
                "last_attempt_at_utc": datetime.now(UTC).isoformat(),
            }
        else:
            deleted.append(fingerprint)
            pending_by_fingerprint.pop(fingerprint, None)
    pending_rows = list(pending_by_fingerprint.values())
    _save_pending(resolved_pending, pending_rows)
    return RetentionResult(
        mode="applied",
        applied=True,
        deleted=tuple(deleted),
        pending=tuple(pending_rows),
        plan=plan,
        message=(
            f"Poda completada: {len(deleted)} eliminados, "
            f"{len(pending_rows)} pendientes de limpieza."
        ),
    )


def confirm_first_retention(
    confirmation_token: str,
    *,
    output_dir: Path = PHASE7_MODELS_DIR,
    approval_path: Path | None = None,
) -> RetentionResult:
    """Confirma la primera keep-list, registra política y ejecuta su poda."""

    plan = plan_model_retention(output_dir)
    if confirmation_token != plan.token:
        raise RetentionError("La confirmación no corresponde a la preview actual.")
    resolved_approval = (
        Path(output_dir).resolve() / "retention_approval.json"
        if approval_path is None
        else Path(approval_path)
    )
    _atomic_json(
        resolved_approval,
        {
            "scope": "TENNIS/models/phase7/runs",
            "retention_keep_latest": plan.keep_latest,
            "first_confirmed_plan_token": plan.token,
            "approved_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    return apply_retention_plan(plan, confirmation_token=plan.token)


def run_automatic_retention(
    *,
    output_dir: Path = PHASE7_MODELS_DIR,
    approval_path: Path | None = None,
) -> RetentionResult:
    """Previsualiza hasta la primera aprobación; después poda sin romper."""

    plan = plan_model_retention(output_dir)
    resolved_approval = (
        Path(output_dir).resolve() / "retention_approval.json"
        if approval_path is None
        else Path(approval_path)
    )
    pending_path = Path(output_dir).resolve() / "cleanup_pending.json"
    if not _approval_matches(plan, resolved_approval):
        pending = _load_pending(pending_path)
        return RetentionResult(
            mode="preview",
            applied=False,
            deleted=(),
            pending=tuple(pending),
            plan=plan,
            message=(
                "Poda no aplicada: se requiere confirmar primero la keep-list "
                f"con token {plan.token}."
            ),
        )
    try:
        return apply_retention_plan(plan, confirmation_token=plan.token)
    except (RetentionError, ArtifactError, OSError) as exc:
        pending = _load_pending(pending_path)
        return RetentionResult(
            mode="warning",
            applied=False,
            deleted=(),
            pending=tuple(pending),
            plan=plan,
            message=f"AVISO: la poda se omitió sin romper el pipeline: {exc}",
        )


def retention_status(
    output_dir: Path = PHASE7_MODELS_DIR,
) -> Mapping[str, object]:
    """Expone keep-list y residuos para logs, paneles o diagnóstico."""

    output = Path(output_dir).resolve()
    plan = plan_model_retention(output)
    pending = _load_pending(output / "cleanup_pending.json")
    return {
        "plan": plan.as_dict(),
        "pending_cleanup": pending,
        "approved": _approval_matches(plan, output / "retention_approval.json"),
    }
