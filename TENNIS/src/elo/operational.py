"""Adapta resultados operativos causales al estado Elo persistido.

El contrato separa dos épocas: un commit Sackmann congelado cubre hasta una
fecha de corte fija y las fuentes operativas solo pueden aportar partidos con
fecha efectiva posterior. TennisRatio tiene precedencia; Tennis Explorer actúa
como respaldo cuando no existe el mismo partido en TennisRatio. Ninguna lectura
modifica las bases operativas ni el triplete sagrado.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Final, Literal, TypeAlias, cast

from ..surface_catalog import SurfaceCatalog, empty_surface_catalog
from ..tennisratio import MappedResult, load_mapped_results
from .events import normalise_surface
from .types import EventProvenance, Gender, MatchEvent, Surface


OperationalSource = Literal["tennisratio", "tennis_explorer"]
OverlayStatus = Literal["loaded", "disabled", "fallback_sackmann_only"]
SOURCE_PRECEDENCE: Final[tuple[OperationalSource, ...]] = (
    "tennisratio",
    "tennis_explorer",
)
SOURCE_SCHEMA_VERSION: Final[str] = "tennis-operational-elo-v2"


class OperationalEloError(RuntimeError):
    """Indica que el overlay operativo no se puede construir con seguridad."""


@dataclass(frozen=True, slots=True)
class EloHandoffConfig:
    """Fija el límite inmutable entre Sackmann y las fuentes operativas."""

    schema_version: int
    base_source_repository: str
    base_source_commit: str
    base_manifest_path: Path
    cutoff_date: date
    tennisratio_database_path: Path
    operations_database_path: Path
    player_mapping_database_path: Path
    source_precedence: tuple[OperationalSource, ...]

    def as_dict(self, *, project_root: Path) -> dict[str, object]:
        """Serializa el handoff con rutas relativas portables."""

        root = project_root.resolve()

        def relative(path: Path) -> str:
            """Convierte una ruta validada a POSIX relativa al proyecto."""

            return path.resolve().relative_to(root).as_posix()

        return {
            "schema_version": self.schema_version,
            "base": {
                "source_repository": self.base_source_repository,
                "source_commit": self.base_source_commit,
                "manifest_path": relative(self.base_manifest_path),
                "cutoff_date": self.cutoff_date.isoformat(),
            },
            "operational": {
                "tennisratio_database_path": relative(self.tennisratio_database_path),
                "operations_database_path": relative(self.operations_database_path),
                "player_mapping_database_path": relative(self.player_mapping_database_path),
                "source_precedence": list(self.source_precedence),
            },
        }


@dataclass(frozen=True, slots=True)
class OperationalOmission:
    """Registra una fila que no puede entrar al Elo sin adivinar identidad."""

    source: OperationalSource
    source_record_id: str
    reason: str
    effective_date: date | None = None

    def as_dict(self) -> dict[str, object]:
        """Devuelve una fila JSON estable para el log fijo de omitidos."""

        return {
            "source": self.source,
            "source_record_id": self.source_record_id,
            "reason": self.reason,
            "effective_date": (
                None if self.effective_date is None else self.effective_date.isoformat()
            ),
        }


@dataclass(frozen=True, slots=True)
class ExplorerOperationalResult:
    """Representa un resultado terminal Explorer con dos IDs resueltos."""

    source_match_id: str
    gender: Gender
    effective_date: date
    available_date: date
    observed_at_utc: datetime
    tournament: str
    tour_level: str
    winner_id: int
    loser_id: int
    score: str | None
    observation_id: str
    payload_sha256: str
    surface: Surface | None = None
    surface_resolution_method: str | None = None
    surface_evidence_id: str | None = None
    surface_resolution_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ExplorerLoadResult:
    """Agrupa resultados Explorer válidos y filas omitidas por identidad."""

    results: tuple[ExplorerOperationalResult, ...]
    eligible_records: int
    omissions: tuple[OperationalOmission, ...]


@dataclass(frozen=True, slots=True)
class OperationalSourceAudit:
    """Resume cobertura, selección y descartes de una fuente operativa."""

    source: OperationalSource
    eligible_records: int
    mapped_records: int
    selected_events: int
    omitted_records: int
    collapsed_within_source: int
    suppressed_by_precedence: int
    max_effective_date: date | None
    max_available_date: date | None

    def as_dict(self) -> dict[str, object]:
        """Serializa contadores y fechas de una fuente."""

        return {
            "source": self.source,
            "eligible_records": self.eligible_records,
            "mapped_records": self.mapped_records,
            "selected_events": self.selected_events,
            "omitted_records": self.omitted_records,
            "collapsed_within_source": self.collapsed_within_source,
            "suppressed_by_precedence": self.suppressed_by_precedence,
            "max_effective_date": (
                None if self.max_effective_date is None else self.max_effective_date.isoformat()
            ),
            "max_available_date": (
                None if self.max_available_date is None else self.max_available_date.isoformat()
            ),
        }


@dataclass(frozen=True, slots=True)
class OperationalEloOverlay:
    """Contiene los eventos seleccionados y su contrato reproducible."""

    status: OverlayStatus
    as_of_date: date
    cutoff_date: date
    base_source_commit: str
    events: tuple[MatchEvent, ...]
    source_audits: tuple[OperationalSourceAudit, ...]
    overlaps_resolved: int
    omissions: tuple[OperationalOmission, ...]
    fingerprint: str
    failure: str | None = None
    surface_catalog_fingerprint: str | None = None
    surface_resolved_events: int = 0
    surface_missing_events: int = 0
    surface_resolution_counts: Mapping[str, int] | None = None

    def events_for_gender(self, gender: Gender) -> tuple[MatchEvent, ...]:
        """Devuelve solo los eventos del universo solicitado."""

        return tuple(event for event in self.events if event.gender == gender)

    def contract_for_gender(self, gender: Gender) -> dict[str, object]:
        """Construye el fragmento persistido y hasheado para un género."""

        events = self.events_for_gender(gender)
        effective_dates = tuple(event.result_source_date for event in events)
        available_dates = tuple(event.date for event in events)
        return {
            "schema": SOURCE_SCHEMA_VERSION,
            "status": self.status,
            "as_of_date": self.as_of_date.isoformat(),
            "cutoff_date": self.cutoff_date.isoformat(),
            "base_source_commit": self.base_source_commit,
            "source_precedence": list(SOURCE_PRECEDENCE),
            "fingerprint": self.fingerprint,
            "gender": gender,
            "selected_events": len(events),
            "selected_by_source": {
                source: sum(event.provenance.source_commit.startswith(source) for event in events)
                for source in SOURCE_PRECEDENCE
            },
            "max_effective_date": (max(effective_dates).isoformat() if effective_dates else None),
            "max_available_date": (max(available_dates).isoformat() if available_dates else None),
            "surface_catalog_fingerprint": self.surface_catalog_fingerprint,
            "surface_resolved_events": sum(event.surface is not None for event in events),
            "surface_missing_events": sum(event.surface is None for event in events),
            "failure": self.failure,
        }

    def as_dict(self) -> dict[str, object]:
        """Serializa el informe operativo completo sin duplicar los eventos."""

        return {
            "schema": SOURCE_SCHEMA_VERSION,
            "status": self.status,
            "as_of_date": self.as_of_date.isoformat(),
            "cutoff_date": self.cutoff_date.isoformat(),
            "base_source_commit": self.base_source_commit,
            "fingerprint": self.fingerprint,
            "selected_events": len(self.events),
            "overlaps_resolved": self.overlaps_resolved,
            "omitted_records": len(self.omissions),
            "surface_catalog_fingerprint": self.surface_catalog_fingerprint,
            "surface_resolved_events": self.surface_resolved_events,
            "surface_missing_events": self.surface_missing_events,
            "surface_resolution_counts": dict(self.surface_resolution_counts or {}),
            "sources": [audit.as_dict() for audit in self.source_audits],
            "failure": self.failure,
        }


RatioLoader = Callable[..., Sequence[MappedResult]]
ExplorerLoader = Callable[..., ExplorerLoadResult]
OperationalResult: TypeAlias = MappedResult | ExplorerOperationalResult


def _required_mapping(value: object, field: str) -> Mapping[str, Any]:
    """Exige un objeto JSON y nombra el campo que incumple."""

    if not isinstance(value, Mapping):
        raise OperationalEloError(f"{field} debe ser un objeto JSON.")
    return value


def _required_text(value: object, field: str) -> str:
    """Exige texto no vacío sin normalizaciones implícitas."""

    if not isinstance(value, str) or not value.strip():
        raise OperationalEloError(f"{field} debe ser texto no vacío.")
    return value.strip()


def _project_path(project_root: Path, value: object, field: str) -> Path:
    """Resuelve una ruta relativa sin permitir salir del proyecto."""

    raw = _required_text(value, field)
    candidate = Path(raw)
    if candidate.is_absolute():
        raise OperationalEloError(f"{field} debe ser una ruta relativa.")
    root = project_root.resolve()
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise OperationalEloError(f"{field} sale de TENNIS/.")
    return resolved


def load_elo_handoff_config(
    path: Path,
    *,
    project_root: Path,
) -> EloHandoffConfig:
    """Carga y valida el handoff versionado entre las dos épocas."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OperationalEloError(f"No se puede leer la configuración Elo {path}.") from exc
    root = _required_mapping(payload, "raíz")
    if root.get("schema_version") != 1:
        raise OperationalEloError("schema_version del handoff debe ser 1.")
    base = _required_mapping(root.get("base"), "base")
    operational = _required_mapping(root.get("operational"), "operational")
    try:
        cutoff = date.fromisoformat(_required_text(base.get("cutoff_date"), "base.cutoff_date"))
    except ValueError as exc:
        raise OperationalEloError("base.cutoff_date debe usar YYYY-MM-DD.") from exc
    raw_precedence = operational.get("source_precedence")
    if not isinstance(raw_precedence, list):
        raise OperationalEloError("operational.source_precedence debe ser una lista.")
    precedence = tuple(raw_precedence)
    if precedence != SOURCE_PRECEDENCE:
        raise OperationalEloError("La precedencia debe ser tennisratio -> tennis_explorer.")
    config = EloHandoffConfig(
        schema_version=1,
        base_source_repository=_required_text(
            base.get("source_repository"), "base.source_repository"
        ),
        base_source_commit=_required_text(base.get("source_commit"), "base.source_commit"),
        base_manifest_path=_project_path(
            project_root, base.get("manifest_path"), "base.manifest_path"
        ),
        cutoff_date=cutoff,
        tennisratio_database_path=_project_path(
            project_root,
            operational.get("tennisratio_database_path"),
            "operational.tennisratio_database_path",
        ),
        operations_database_path=_project_path(
            project_root,
            operational.get("operations_database_path"),
            "operational.operations_database_path",
        ),
        player_mapping_database_path=_project_path(
            project_root,
            operational.get("player_mapping_database_path"),
            "operational.player_mapping_database_path",
        ),
        source_precedence=cast(tuple[OperationalSource, ...], precedence),
    )
    try:
        manifest = json.loads(config.base_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OperationalEloError(
            "El manifiesto Sackmann base congelado no se puede leer."
        ) from exc
    if not isinstance(manifest, Mapping):
        raise OperationalEloError("El manifiesto Sackmann base no es un objeto JSON.")
    if manifest.get("source_commit") != config.base_source_commit:
        raise OperationalEloError("El commit del manifiesto congelado no coincide con el handoff.")
    if manifest.get("source_repository") != config.base_source_repository:
        raise OperationalEloError("El repositorio del manifiesto congelado no coincide.")
    return config


def _read_only_connection(path: Path) -> sqlite3.Connection:
    """Abre una SQLite en modo read-only y con filas nombradas."""

    if not path.is_file():
        raise OperationalEloError(f"No existe la SQLite operativa: {path}.")
    connection = sqlite3.connect(
        f"file:{path.resolve().as_posix()}?mode=ro",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    return connection


def _parse_utc(value: object, field: str) -> datetime:
    """Convierte un instante ISO consciente de zona a UTC."""

    if not isinstance(value, str):
        raise OperationalEloError(f"{field} debe ser texto ISO-8601.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise OperationalEloError(f"{field} no es ISO-8601 válido.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise OperationalEloError(f"{field} debe incluir zona horaria.")
    return parsed.astimezone(UTC)


def _ratio_eligible_keys(
    *,
    database_path: Path,
    cutoff_date: date,
    as_of_date: date,
) -> tuple[tuple[Gender, date, str], ...]:
    """Lista resultados Ratio terminales aunque su mapping esté incompleto."""

    query = """
        SELECT DISTINCT
            observation.gender,
            observation.effective_date,
            observation.source_match_key
        FROM match_observations AS observation
        JOIN published_batches AS published USING (batch_id)
        WHERE observation.effective_date > ?
          AND observation.effective_date < ?
          AND substr(observation.first_seen_at_utc, 1, 10) < ?
          AND substr(published.published_at_utc, 1, 10) < ?
          AND observation.result IN ('Win', 'Lose')
          AND observation.rival_source_key IS NOT NULL
        ORDER BY observation.gender,
                 observation.effective_date,
                 observation.source_match_key
    """
    with _read_only_connection(database_path) as connection:
        rows = connection.execute(
            query,
            (
                cutoff_date.isoformat(),
                as_of_date.isoformat(),
                as_of_date.isoformat(),
                as_of_date.isoformat(),
            ),
        ).fetchall()
    output: list[tuple[Gender, date, str]] = []
    for row in rows:
        gender = str(row["gender"])
        if gender not in {"M", "F"}:
            continue
        output.append(
            (
                cast(Gender, gender),
                date.fromisoformat(str(row["effective_date"])),
                str(row["source_match_key"]),
            )
        )
    return tuple(output)


def load_explorer_operational_results(
    *,
    operations_database_path: Path,
    player_mapping_database_path: Path,
    cutoff_date: date,
    as_of_date: date,
    surface_catalog: SurfaceCatalog | None = None,
) -> ExplorerLoadResult:
    """Lee resultados Explorer terminales y resuelve ambos slugs por su puente."""

    catalog = surface_catalog or empty_surface_catalog(as_of_date)
    with _read_only_connection(player_mapping_database_path) as mapping_connection:
        mapping_rows = mapping_connection.execute(
            """
            SELECT gender, slug, player_id
            FROM player_mappings
            ORDER BY gender, slug
            """
        ).fetchall()
    mappings = {
        (str(row["gender"]), str(row["slug"])): int(row["player_id"]) for row in mapping_rows
    }
    query = """
        SELECT
            observation.observation_id,
            observation.source_match_id,
            observation.observed_at_utc,
            observation.winner_side,
            observation.sets_score,
            observation.payload_sha256,
            observation.payload_json,
            match.match_date,
            match.gender,
            match.tournament,
            match.tour_level,
            match.player_1_slug,
            match.player_2_slug
        FROM observations AS observation
        JOIN matches AS match USING (source_match_id)
        WHERE observation.source_system = 'tennis_explorer'
          AND observation.is_valid = 1
          AND observation.status = 'finished'
          AND observation.winner_side IN ('player_1', 'player_2')
          AND match.match_date > ?
          AND match.match_date < ?
          AND substr(observation.observed_at_utc, 1, 10) < ?
        ORDER BY observation.source_match_id,
                 observation.observed_at_utc,
                 observation.observation_id
    """
    with _read_only_connection(operations_database_path) as connection:
        rows = connection.execute(
            query,
            (
                cutoff_date.isoformat(),
                as_of_date.isoformat(),
                as_of_date.isoformat(),
            ),
        ).fetchall()
    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_match_id"])].append(row)

    output: list[ExplorerOperationalResult] = []
    omissions: list[OperationalOmission] = []
    for source_match_id in sorted(grouped):
        group = grouped[source_match_id]
        first = group[0]
        effective_date = date.fromisoformat(str(first["match_date"]))
        gender_text = str(first["gender"])
        if gender_text not in {"M", "F"}:
            omissions.append(
                OperationalOmission(
                    source="tennis_explorer",
                    source_record_id=source_match_id,
                    reason="invalid_gender",
                    effective_date=effective_date,
                )
            )
            continue
        gender = cast(Gender, gender_text)
        player_1_slug = str(first["player_1_slug"] or "")
        player_2_slug = str(first["player_2_slug"] or "")
        player_1_id = mappings.get((gender, player_1_slug))
        player_2_id = mappings.get((gender, player_2_slug))
        if player_1_id is None or player_2_id is None or player_1_id == player_2_id:
            omissions.append(
                OperationalOmission(
                    source="tennis_explorer",
                    source_record_id=source_match_id,
                    reason="incomplete_or_ambiguous_mapping",
                    effective_date=effective_date,
                )
            )
            continue
        winner_sides = {str(row["winner_side"]) for row in group}
        if len(winner_sides) != 1:
            omissions.append(
                OperationalOmission(
                    source="tennis_explorer",
                    source_record_id=source_match_id,
                    reason="conflicting_terminal_outcome",
                    effective_date=effective_date,
                )
            )
            continue
        chosen = min(
            group,
            key=lambda row: (
                _parse_utc(row["observed_at_utc"], "observed_at_utc"),
                str(row["observation_id"]),
            ),
        )
        observed = _parse_utc(chosen["observed_at_utc"], "observed_at_utc")
        if observed.date() >= as_of_date:
            omissions.append(
                OperationalOmission(
                    source="tennis_explorer",
                    source_record_id=source_match_id,
                    reason="not_available_strictly_before_as_of",
                    effective_date=effective_date,
                )
            )
            continue
        winner_side = str(chosen["winner_side"])
        winner_id, loser_id = (
            (player_1_id, player_2_id) if winner_side == "player_1" else (player_2_id, player_1_id)
        )
        try:
            payload_object = json.loads(str(chosen["payload_json"]))
        except json.JSONDecodeError as exc:
            raise OperationalEloError(
                f"El resultado Explorer {source_match_id} no contiene JSON válido."
            ) from exc
        if not isinstance(payload_object, Mapping):
            raise OperationalEloError(
                f"El resultado Explorer {source_match_id} no contiene un objeto JSON."
            )
        payload = cast(Mapping[str, object], payload_object)
        resolution = catalog.resolve_for_result(
            source_family="tennis_explorer",
            source_match_id=source_match_id,
            gender=gender,
            effective_date=effective_date,
            available_at_utc=observed,
            tournament=str(first["tournament"] or "unknown"),
            tournament_href=(
                None if payload.get("tournament_href") is None else str(payload["tournament_href"])
            ),
            direct_surface=payload.get("surface"),
            direct_source_url=(
                None if payload.get("source_url") is None else str(payload["source_url"])
            ),
            direct_source_sha256=str(chosen["payload_sha256"]),
            direct_tour_level=str(first["tour_level"] or "Other"),
        )
        output.append(
            ExplorerOperationalResult(
                source_match_id=source_match_id,
                gender=gender,
                effective_date=effective_date,
                available_date=observed.date(),
                observed_at_utc=observed,
                tournament=str(first["tournament"] or "unknown"),
                tour_level=str(first["tour_level"] or "Other"),
                winner_id=winner_id,
                loser_id=loser_id,
                score=(None if chosen["sets_score"] is None else str(chosen["sets_score"])),
                observation_id=str(chosen["observation_id"]),
                payload_sha256=str(chosen["payload_sha256"]),
                surface=resolution.surface,
                surface_resolution_method=resolution.method,
                surface_evidence_id=resolution.evidence_id,
                surface_resolution_reason=resolution.reason,
            )
        )
    return ExplorerLoadResult(
        results=tuple(output),
        eligible_records=len(grouped),
        omissions=tuple(omissions),
    )


def _sporting_identity(
    *,
    gender: Gender,
    effective_date: date,
    winner_id: int,
    loser_id: int,
) -> tuple[Gender, date, int, int]:
    """Crea una identidad común entre fuentes sin depender de slugs."""

    lower, upper = sorted((winner_id, loser_id))
    return gender, effective_date, lower, upper


def _stable_hash(payload: Mapping[str, object]) -> str:
    """Calcula SHA-256 JSON determinista para un evento o contrato."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ratio_event(result: MappedResult, ordinal: int) -> MatchEvent:
    """Convierte un resultado Ratio con su superficie directa publicada."""

    try:
        surface = normalise_surface(result.surface)
    except ValueError:
        surface = None

    payload = {
        "schema": SOURCE_SCHEMA_VERSION,
        "source": "tennisratio",
        "canonical_match_id": result.canonical_match_id,
        "gender": result.gender,
        "effective_date": result.effective_date.isoformat(),
        "available_date": result.available_date.isoformat(),
        "winner_id": result.winner_sackmann_id,
        "loser_id": result.loser_sackmann_id,
        "surface": surface,
        "surface_resolution_method": ("direct_result" if surface is not None else None),
        "source_sha256": result.source_sha256,
    }
    return MatchEvent(
        date=result.available_date,
        source_date=result.effective_date,
        gender=result.gender,
        winner_id=result.winner_sackmann_id,
        loser_id=result.loser_sackmann_id,
        surface=surface,
        tour_level=result.tour_level or "Other",
        score=result.score,
        round=result.round,
        provenance=EventProvenance(
            source_commit="tennisratio-operational-v1",
            source_path=result.source_url,
            source_row=ordinal,
        ),
        source_record_hash=_stable_hash(payload),
        tourney_id=result.canonical_match_id,
    )


def _explorer_event(
    result: ExplorerOperationalResult,
    ordinal: int,
) -> MatchEvent:
    """Convierte un resultado Explorer con superficie exacta, si existe."""

    payload = {
        "schema": SOURCE_SCHEMA_VERSION,
        "source": "tennis_explorer",
        "source_match_id": result.source_match_id,
        "observation_id": result.observation_id,
        "gender": result.gender,
        "effective_date": result.effective_date.isoformat(),
        "available_date": result.available_date.isoformat(),
        "winner_id": result.winner_id,
        "loser_id": result.loser_id,
        "surface": result.surface,
        "surface_resolution_method": result.surface_resolution_method,
        "surface_evidence_id": result.surface_evidence_id,
        "payload_sha256": result.payload_sha256,
    }
    return MatchEvent(
        date=result.available_date,
        source_date=result.effective_date,
        gender=result.gender,
        winner_id=result.winner_id,
        loser_id=result.loser_id,
        surface=result.surface,
        tour_level=result.tour_level or "Other",
        score=result.score,
        provenance=EventProvenance(
            source_commit="tennis_explorer-operational-v1",
            source_path=f"operations:observations/{result.observation_id}",
            source_row=ordinal,
        ),
        source_record_hash=_stable_hash(payload),
        tourney_id=result.source_match_id,
    )


def _deduplicate_source_results(
    records: Iterable[tuple[tuple[Gender, date, int, int], OperationalResult]],
    *,
    source: OperationalSource,
) -> tuple[tuple[OperationalResult, ...], tuple[OperationalOmission, ...]]:
    """Colapsa aliases dentro de una fuente y omite resultados en conflicto."""

    grouped: dict[tuple[Gender, date, int, int], list[OperationalResult]] = defaultdict(list)
    for identity, record in records:
        grouped[identity].append(record)
    selected: list[OperationalResult] = []
    omissions: list[OperationalOmission] = []
    for identity in sorted(grouped):
        group = grouped[identity]
        outcomes = {_result_outcome(item) for item in group}
        if len(outcomes) != 1:
            for item in group:
                record_id = _result_record_id(item)
                omissions.append(
                    OperationalOmission(
                        source=source,
                        source_record_id=record_id,
                        reason="conflicting_sporting_identity",
                        effective_date=identity[1],
                    )
                )
            continue
        selected.append(
            min(
                group,
                key=lambda item: (
                    getattr(item, "available_date"),
                    _result_available_instant(item),
                    _result_record_id(item),
                ),
            )
        )
    return tuple(selected), tuple(omissions)


def _result_outcome(result: OperationalResult) -> tuple[int, int]:
    """Devuelve ganador/perdedor sin reflexión con defaults evaluados."""

    if isinstance(result, MappedResult):
        return result.winner_sackmann_id, result.loser_sackmann_id
    return result.winner_id, result.loser_id


def _result_record_id(result: OperationalResult) -> str:
    """Devuelve la identidad nativa del resultado según su fuente."""

    if isinstance(result, MappedResult):
        return result.canonical_match_id
    return result.source_match_id


def _result_available_instant(result: OperationalResult) -> datetime:
    """Devuelve el instante real que completó la disponibilidad."""

    if isinstance(result, MappedResult):
        return result.first_seen_at_utc
    return result.observed_at_utc


def assemble_operational_overlay(
    *,
    config: EloHandoffConfig,
    as_of_date: date,
    ratio_results: Sequence[MappedResult],
    ratio_eligible_keys: Sequence[tuple[Gender, date, str]],
    explorer_load: ExplorerLoadResult,
    surface_catalog: SurfaceCatalog | None = None,
) -> OperationalEloOverlay:
    """Aplica causalidad, deduplicación común y precedencia entre fuentes."""

    if not isinstance(as_of_date, date) or isinstance(as_of_date, datetime):
        raise TypeError("as_of_date debe ser datetime.date estricto.")
    ratio_eligible_by_id = {
        source_id: (gender, effective_date)
        for gender, effective_date, source_id in ratio_eligible_keys
    }
    ratio_valid: list[MappedResult] = []
    omissions = list(explorer_load.omissions)
    for result in ratio_results:
        if (
            result.effective_date <= config.cutoff_date
            or result.effective_date >= as_of_date
            or result.available_date >= as_of_date
            or result.effective_date > result.available_date
            or result.winner_sackmann_id == result.loser_sackmann_id
        ):
            omissions.append(
                OperationalOmission(
                    source="tennisratio",
                    source_record_id=result.canonical_match_id,
                    reason="outside_strict_handoff_or_availability",
                    effective_date=result.effective_date,
                )
            )
            continue
        ratio_valid.append(result)
    loaded_ratio_ids = {item.canonical_match_id for item in ratio_results}
    for source_id, (_, effective_date) in sorted(ratio_eligible_by_id.items()):
        if source_id not in loaded_ratio_ids:
            omissions.append(
                OperationalOmission(
                    source="tennisratio",
                    source_record_id=source_id,
                    reason="incomplete_or_ambiguous_mapping",
                    effective_date=effective_date,
                )
            )

    ratio_records = (
        (
            _sporting_identity(
                gender=item.gender,
                effective_date=item.effective_date,
                winner_id=item.winner_sackmann_id,
                loser_id=item.loser_sackmann_id,
            ),
            item,
        )
        for item in ratio_valid
    )
    ratio_selected_raw, ratio_conflicts = _deduplicate_source_results(
        ratio_records,
        source="tennisratio",
    )
    omissions.extend(ratio_conflicts)
    ratio_selected = tuple(cast(MappedResult, item) for item in ratio_selected_raw)

    explorer_valid = tuple(
        item
        for item in explorer_load.results
        if (
            item.effective_date > config.cutoff_date
            and item.effective_date < as_of_date
            and item.available_date < as_of_date
            and item.effective_date <= item.available_date
            and item.winner_id != item.loser_id
        )
    )
    explorer_records = (
        (
            _sporting_identity(
                gender=item.gender,
                effective_date=item.effective_date,
                winner_id=item.winner_id,
                loser_id=item.loser_id,
            ),
            item,
        )
        for item in explorer_valid
    )
    explorer_selected_raw, explorer_conflicts = _deduplicate_source_results(
        explorer_records,
        source="tennis_explorer",
    )
    omissions.extend(explorer_conflicts)
    explorer_selected = tuple(
        cast(ExplorerOperationalResult, item) for item in explorer_selected_raw
    )

    ratio_identities = {
        _sporting_identity(
            gender=item.gender,
            effective_date=item.effective_date,
            winner_id=item.winner_sackmann_id,
            loser_id=item.loser_sackmann_id,
        )
        for item in ratio_selected
    }
    explorer_after_precedence = tuple(
        item
        for item in explorer_selected
        if _sporting_identity(
            gender=item.gender,
            effective_date=item.effective_date,
            winner_id=item.winner_id,
            loser_id=item.loser_id,
        )
        not in ratio_identities
    )
    overlaps_resolved = len(explorer_selected) - len(explorer_after_precedence)
    events = tuple(
        sorted(
            (
                *(
                    _ratio_event(item, ordinal)
                    for ordinal, item in enumerate(ratio_selected, start=1)
                ),
                *(
                    _explorer_event(item, ordinal)
                    for ordinal, item in enumerate(explorer_after_precedence, start=1)
                ),
            ),
            key=lambda event: event.sort_key,
        )
    )
    if any(
        event.result_source_date <= config.cutoff_date
        or event.result_source_date >= as_of_date
        or event.date >= as_of_date
        for event in events
    ):
        raise OperationalEloError("Un evento operativo cruzó el corte causal.")
    identities = [
        _sporting_identity(
            gender=event.gender,
            effective_date=event.result_source_date,
            winner_id=event.winner_id,
            loser_id=event.loser_id,
        )
        for event in events
    ]
    if len(identities) != len(set(identities)):
        raise OperationalEloError("La precedencia dejó un partido operativo contado dos veces.")

    ratio_events = tuple(
        event for event in events if event.provenance.source_commit.startswith("tennisratio")
    )
    explorer_events = tuple(
        event for event in events if event.provenance.source_commit.startswith("tennis_explorer")
    )
    omission_counts = Counter(item.source for item in omissions)

    def source_audit(
        source: OperationalSource,
        source_events: tuple[MatchEvent, ...],
        eligible: int,
        mapped: int,
        collapsed: int,
        suppressed: int,
    ) -> OperationalSourceAudit:
        """Crea métricas homogéneas para una fuente."""

        return OperationalSourceAudit(
            source=source,
            eligible_records=eligible,
            mapped_records=mapped,
            selected_events=len(source_events),
            omitted_records=int(omission_counts[source]),
            collapsed_within_source=collapsed,
            suppressed_by_precedence=suppressed,
            max_effective_date=max(
                (event.result_source_date for event in source_events),
                default=None,
            ),
            max_available_date=max(
                (event.date for event in source_events),
                default=None,
            ),
        )

    audits = (
        source_audit(
            "tennisratio",
            ratio_events,
            len(ratio_eligible_keys),
            len(ratio_valid),
            max(0, len(ratio_valid) - len(ratio_selected) - len(ratio_conflicts)),
            0,
        ),
        source_audit(
            "tennis_explorer",
            explorer_events,
            explorer_load.eligible_records,
            len(explorer_valid),
            max(
                0,
                len(explorer_valid) - len(explorer_selected) - len(explorer_conflicts),
            ),
            overlaps_resolved,
        ),
    )
    catalog = surface_catalog or empty_surface_catalog(as_of_date)
    resolution_counts: Counter[str] = Counter()
    for item in ratio_selected:
        try:
            ratio_surface = normalise_surface(item.surface)
        except ValueError:
            ratio_surface = None
        resolution_counts["direct_result" if ratio_surface is not None else "unresolved"] += 1
    for item in explorer_after_precedence:
        resolution_counts[
            item.surface_resolution_method
            or f"unresolved:{item.surface_resolution_reason or 'unknown'}"
        ] += 1
    fingerprint = _stable_hash(
        {
            "schema": SOURCE_SCHEMA_VERSION,
            "base_source_commit": config.base_source_commit,
            "cutoff_date": config.cutoff_date.isoformat(),
            "as_of_date": as_of_date.isoformat(),
            "source_precedence": list(config.source_precedence),
            "surface_catalog_fingerprint": catalog.fingerprint,
            "events": [
                {
                    "gender": event.gender,
                    "effective_date": event.result_source_date.isoformat(),
                    "available_date": event.date.isoformat(),
                    "winner_id": event.winner_id,
                    "loser_id": event.loser_id,
                    "surface": event.surface,
                    "source_commit": event.provenance.source_commit,
                    "source_record_hash": event.source_record_hash,
                }
                for event in events
            ],
        }
    )
    return OperationalEloOverlay(
        status="loaded",
        as_of_date=as_of_date,
        cutoff_date=config.cutoff_date,
        base_source_commit=config.base_source_commit,
        events=events,
        source_audits=audits,
        overlaps_resolved=overlaps_resolved,
        omissions=tuple(
            sorted(
                omissions,
                key=lambda item: (
                    item.source,
                    item.effective_date or date.min,
                    item.source_record_id,
                    item.reason,
                ),
            )
        ),
        fingerprint=fingerprint,
        surface_catalog_fingerprint=catalog.fingerprint,
        surface_resolved_events=sum(event.surface is not None for event in events),
        surface_missing_events=sum(event.surface is None for event in events),
        surface_resolution_counts=dict(sorted(resolution_counts.items())),
    )


def build_operational_elo_overlay(
    *,
    config: EloHandoffConfig,
    as_of_date: date,
    ratio_loader: RatioLoader = load_mapped_results,
    explorer_loader: ExplorerLoader = load_explorer_operational_results,
    surface_catalog: SurfaceCatalog | None = None,
) -> OperationalEloOverlay:
    """Carga ambas fuentes y construye el overlay completo o lanza error."""

    ratio_results = ratio_loader(
        as_of_date,
        base_cutoff_by_gender={
            "M": config.cutoff_date,
            "F": config.cutoff_date,
        },
        database_path=config.tennisratio_database_path,
    )
    ratio_keys = _ratio_eligible_keys(
        database_path=config.tennisratio_database_path,
        cutoff_date=config.cutoff_date,
        as_of_date=as_of_date,
    )
    catalog = surface_catalog or empty_surface_catalog(as_of_date)
    explorer = explorer_loader(
        operations_database_path=config.operations_database_path,
        player_mapping_database_path=config.player_mapping_database_path,
        cutoff_date=config.cutoff_date,
        as_of_date=as_of_date,
        surface_catalog=catalog,
    )
    return assemble_operational_overlay(
        config=config,
        as_of_date=as_of_date,
        ratio_results=ratio_results,
        ratio_eligible_keys=ratio_keys,
        explorer_load=explorer,
        surface_catalog=catalog,
    )


def build_operational_elo_overlay_with_fallback(
    *,
    config: EloHandoffConfig,
    as_of_date: date,
    builder: Callable[..., OperationalEloOverlay] = (build_operational_elo_overlay),
    surface_catalog: SurfaceCatalog | None = None,
) -> OperationalEloOverlay:
    """Cae a Sackmann-only ante cualquier fallo de adquisición operativa."""

    try:
        return builder(
            config=config,
            as_of_date=as_of_date,
            surface_catalog=surface_catalog,
        )
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        fingerprint = _stable_hash(
            {
                "schema": SOURCE_SCHEMA_VERSION,
                "status": "fallback_sackmann_only",
                "base_source_commit": config.base_source_commit,
                "cutoff_date": config.cutoff_date.isoformat(),
                "as_of_date": as_of_date.isoformat(),
                "surface_catalog_fingerprint": (
                    None if surface_catalog is None else surface_catalog.fingerprint
                ),
                "failure": failure,
            }
        )
        return OperationalEloOverlay(
            status="fallback_sackmann_only",
            as_of_date=as_of_date,
            cutoff_date=config.cutoff_date,
            base_source_commit=config.base_source_commit,
            events=(),
            source_audits=(),
            overlaps_resolved=0,
            omissions=(),
            fingerprint=fingerprint,
            failure=failure,
            surface_catalog_fingerprint=(
                None if surface_catalog is None else surface_catalog.fingerprint
            ),
            surface_resolved_events=0,
            surface_missing_events=0,
            surface_resolution_counts={},
        )


def publish_operational_omissions(
    path: Path,
    overlay: OperationalEloOverlay,
) -> None:
    """Publica atómicamente un único log fijo, sin acumular snapshots."""

    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SOURCE_SCHEMA_VERSION,
        "overlay": overlay.as_dict(),
        "omissions": [item.as_dict() for item in overlay.omissions],
    }
    temporary = resolved.with_name(f".{resolved.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(resolved)


__all__ = [
    "EloHandoffConfig",
    "ExplorerLoadResult",
    "ExplorerOperationalResult",
    "OperationalEloError",
    "OperationalEloOverlay",
    "OperationalOmission",
    "OperationalSourceAudit",
    "SOURCE_PRECEDENCE",
    "assemble_operational_overlay",
    "build_operational_elo_overlay",
    "build_operational_elo_overlay_with_fallback",
    "load_elo_handoff_config",
    "load_explorer_operational_results",
    "publish_operational_omissions",
]
