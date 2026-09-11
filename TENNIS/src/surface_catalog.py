"""Catálogo causal de superficies con evidencia exacta y procedencia.

El catálogo no adivina una superficie a partir de ciudad, país ni historia del
torneo. Solo acepta valores publicados literalmente por una fuente para un
partido o para una identidad exacta de torneo/edición. Las bases operativas se
abren en modo de solo lectura y el artefacto JSON resultante es una caché
derivada, reemplazable y con fingerprint reproducible.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from tempfile import NamedTemporaryFile
from typing import Final, Literal, cast
from urllib.parse import urlsplit

Gender = Literal["M", "F"]
Surface = Literal["Hard", "Clay", "Grass", "Carpet"]
SurfaceSource = Literal["tennisratio", "tennis_explorer"]
SurfaceEvidenceKind = Literal[
    "direct_agenda",
    "direct_result",
    "exact_tournament_reference",
]
SurfaceResolutionMethod = Literal[
    "direct_pre_date",
    "direct_result",
    "catalog_exact_edition",
    "same_edition_propagation",
]

CATALOG_SCHEMA: Final[str] = "tennis-surface-catalog-v1"
_SURFACES: Final[dict[str, Surface]] = {
    "hard": "Hard",
    "clay": "Clay",
    "grass": "Grass",
    "carpet": "Carpet",
}
_AMBIGUOUS_SURFACES: Final[frozenset[str]] = frozenset(
    {"", "indoor", "indoors", "unknown", "n/a", "none"}
)
_AGGREGATE_TOURNAMENT: Final[re.Pattern[str]] = re.compile(
    r"^(?:futures|challengers?|itf)(?:\s+\d{4})?$",
    re.IGNORECASE,
)


class SurfaceCatalogError(RuntimeError):
    """Indica que el catálogo no puede demostrar su contrato causal."""


@dataclass(frozen=True, slots=True)
class SurfaceEvidence:
    """Una observación exacta de superficie publicada por una fuente."""

    evidence_id: str
    source_family: SurfaceSource
    evidence_kind: SurfaceEvidenceKind
    source_match_id: str
    gender: Gender
    edition_year: int
    tournament: str
    tournament_href: str | None
    tour_level: str | None
    surface: Surface
    effective_date: date
    captured_at_utc: datetime
    source_url: str
    source_sha256: str

    @property
    def tournament_identity(self) -> str | None:
        """Devuelve una identidad de edición exacta, nunca una heurística."""

        return _tournament_identity(
            source_family=self.source_family,
            gender=self.gender,
            edition_year=self.edition_year,
            tournament=self.tournament,
            tournament_href=self.tournament_href,
        )

    def as_dict(self) -> dict[str, object]:
        """Serializa la evidencia en orden de campos estable."""

        return {
            "evidence_id": self.evidence_id,
            "source_family": self.source_family,
            "evidence_kind": self.evidence_kind,
            "source_match_id": self.source_match_id,
            "gender": self.gender,
            "edition_year": self.edition_year,
            "tournament": self.tournament,
            "tournament_href": self.tournament_href,
            "tour_level": self.tour_level,
            "surface": self.surface,
            "effective_date": self.effective_date.isoformat(),
            "captured_at_utc": self.captured_at_utc.isoformat(),
            "source_url": self.source_url,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True, slots=True)
class SurfaceResolution:
    """Superficie resuelta y evidencia concreta que la justifica."""

    surface: Surface | None
    method: SurfaceResolutionMethod | None
    evidence_id: str | None
    captured_at_utc: datetime | None
    source_family: SurfaceSource | None
    source_url: str | None
    source_sha256: str | None
    tour_level: str | None
    reason: str | None

    @property
    def resolved(self) -> bool:
        """Indica si existe una superficie no ambigua."""

        return self.surface is not None


@dataclass(frozen=True, slots=True)
class SurfaceCatalog:
    """Índice inmutable de evidencias exactas anteriores a un corte."""

    as_of_date: date
    evidence: tuple[SurfaceEvidence, ...]
    fingerprint: str
    source_counts: Mapping[str, int]

    def resolve_for_prediction(
        self,
        *,
        source_family: SurfaceSource,
        source_match_id: str,
        gender: Gender,
        match_date: date,
        tournament: str,
        tournament_href: str | None,
        direct_surface: object = None,
        direct_captured_at_utc: datetime | None = None,
        direct_source_url: str | None = None,
        direct_source_sha256: str | None = None,
        direct_tour_level: str | None = None,
    ) -> SurfaceResolution:
        """Resuelve para una predicción usando exclusivamente capturas `< D`."""

        if match_date != self.as_of_date:
            raise SurfaceCatalogError(
                "El catálogo diario debe construirse con as_of_date igual a D."
            )
        normalized = _surface_or_none(direct_surface)
        if normalized is not None and direct_captured_at_utc is not None:
            captured = _aware_utc(direct_captured_at_utc, "direct_captured_at_utc")
            if captured.date() < match_date:
                return SurfaceResolution(
                    surface=normalized,
                    method="direct_pre_date",
                    evidence_id=_stable_id(
                        "direct",
                        source_family,
                        source_match_id,
                        normalized,
                        captured.isoformat(),
                        direct_source_sha256 or "missing",
                    ),
                    captured_at_utc=captured,
                    source_family=source_family,
                    source_url=direct_source_url,
                    source_sha256=direct_source_sha256,
                    tour_level=_optional_text(direct_tour_level),
                    reason=None,
                )
        return self._resolve_catalog(
            source_family=source_family,
            gender=gender,
            match_date=match_date,
            tournament=tournament,
            tournament_href=tournament_href,
            captured_before_date=match_date,
            captured_at_or_before=None,
        )

    def resolve_for_result(
        self,
        *,
        source_family: SurfaceSource,
        source_match_id: str,
        gender: Gender,
        effective_date: date,
        available_at_utc: datetime,
        tournament: str,
        tournament_href: str | None,
        direct_surface: object = None,
        direct_source_url: str | None = None,
        direct_source_sha256: str | None = None,
        direct_tour_level: str | None = None,
    ) -> SurfaceResolution:
        """Resuelve un resultado sin adelantar evidencia al instante disponible."""

        available = _aware_utc(available_at_utc, "available_at_utc")
        normalized = _surface_or_none(direct_surface)
        if normalized is not None:
            return SurfaceResolution(
                surface=normalized,
                method="direct_result",
                evidence_id=_stable_id(
                    "direct-result",
                    source_family,
                    source_match_id,
                    normalized,
                    available.isoformat(),
                    direct_source_sha256 or "missing",
                ),
                captured_at_utc=available,
                source_family=source_family,
                source_url=direct_source_url,
                source_sha256=direct_source_sha256,
                tour_level=_optional_text(direct_tour_level),
                reason=None,
            )
        return self._resolve_catalog(
            source_family=source_family,
            gender=gender,
            match_date=effective_date,
            tournament=tournament,
            tournament_href=tournament_href,
            captured_before_date=None,
            captured_at_or_before=available,
        )

    def _resolve_catalog(
        self,
        *,
        source_family: SurfaceSource,
        gender: Gender,
        match_date: date,
        tournament: str,
        tournament_href: str | None,
        captured_before_date: date | None,
        captured_at_or_before: datetime | None,
    ) -> SurfaceResolution:
        """Aplica identidad exacta, corte temporal y rechazo por conflicto."""

        identity = _tournament_identity(
            source_family=source_family,
            gender=gender,
            edition_year=match_date.year,
            tournament=tournament,
            tournament_href=tournament_href,
        )
        if identity is None:
            return _unresolved("aggregate_or_missing_exact_tournament_identity")
        candidates = [
            item
            for item in self.evidence
            if item.tournament_identity == identity
            and (captured_before_date is None or item.captured_at_utc.date() < captured_before_date)
            and (captured_at_or_before is None or item.captured_at_utc <= captured_at_or_before)
        ]
        if not candidates:
            return _unresolved("no_exact_pre_cutoff_surface_evidence")
        surfaces = {item.surface for item in candidates}
        if len(surfaces) != 1:
            return _unresolved("conflicting_surface_same_tournament_edition")
        chosen = min(
            candidates,
            key=lambda item: (
                item.captured_at_utc,
                item.evidence_id,
            ),
        )
        method: SurfaceResolutionMethod = (
            "catalog_exact_edition"
            if chosen.evidence_kind == "exact_tournament_reference"
            else "same_edition_propagation"
        )
        levels = {item.tour_level for item in candidates if item.tour_level is not None}
        return SurfaceResolution(
            surface=chosen.surface,
            method=method,
            evidence_id=chosen.evidence_id,
            captured_at_utc=chosen.captured_at_utc,
            source_family=chosen.source_family,
            source_url=chosen.source_url,
            source_sha256=chosen.source_sha256,
            tour_level=next(iter(levels)) if len(levels) == 1 else None,
            reason=None,
        )

    def as_dict(self) -> dict[str, object]:
        """Serializa la caché completa con fingerprint verificable."""

        return {
            "schema": CATALOG_SCHEMA,
            "as_of_date": self.as_of_date.isoformat(),
            "fingerprint": self.fingerprint,
            "source_counts": dict(sorted(self.source_counts.items())),
            "evidence": [item.as_dict() for item in self.evidence],
        }


def build_surface_catalog(
    *,
    as_of_date: date,
    tennisratio_database_path: Path,
    operations_database_path: Path,
    cache_path: Path | None = None,
) -> SurfaceCatalog:
    """Construye el catálogo desde sidecars persistentes abiertos read-only."""

    if isinstance(as_of_date, datetime) or not isinstance(as_of_date, date):
        raise TypeError("as_of_date debe ser datetime.date estricto.")
    evidence = _deduplicate_evidence(
        (
            *_load_tennisratio_evidence(
                Path(tennisratio_database_path),
                as_of_date=as_of_date,
            ),
            *_load_tennis_explorer_evidence(
                Path(operations_database_path),
                as_of_date=as_of_date,
            ),
        )
    )
    fingerprint = _catalog_fingerprint(as_of_date, evidence)
    counts = Counter(item.source_family for item in evidence)
    catalog = SurfaceCatalog(
        as_of_date=as_of_date,
        evidence=evidence,
        fingerprint=fingerprint,
        source_counts={str(source): int(count) for source, count in counts.items()},
    )
    if cache_path is not None:
        _publish_cache(Path(cache_path), catalog)
    return catalog


def empty_surface_catalog(as_of_date: date) -> SurfaceCatalog:
    """Devuelve un catálogo vacío válido para tests o fuentes ausentes."""

    return SurfaceCatalog(
        as_of_date=as_of_date,
        evidence=(),
        fingerprint=_catalog_fingerprint(as_of_date, ()),
        source_counts={},
    )


def _load_tennisratio_evidence(
    database_path: Path,
    *,
    as_of_date: date,
) -> tuple[SurfaceEvidence, ...]:
    """Lee agendas publicadas de TennisRatio estrictamente anteriores al corte."""

    if not database_path.is_file():
        return ()
    query = """
        SELECT agenda.observation_id, agenda.source_match_id,
               agenda.match_date, agenda.first_seen_at_utc,
               agenda.source_url, agenda.source_sha256, agenda.payload_json,
               published.published_at_utc
        FROM agenda_observations AS agenda
        JOIN published_batches AS published USING (batch_id)
        WHERE substr(agenda.first_seen_at_utc, 1, 10) < ?
          AND substr(published.published_at_utc, 1, 10) < ?
        ORDER BY agenda.first_seen_at_utc, agenda.observation_id
    """
    with _read_only_connection(database_path) as connection:
        rows = connection.execute(
            query,
            (as_of_date.isoformat(), as_of_date.isoformat()),
        ).fetchall()
    output: list[SurfaceEvidence] = []
    for row in rows:
        payload = _payload_object(row["payload_json"])
        surface = _surface_or_none(payload.get("surface"))
        if surface is None:
            continue
        gender = _gender(payload.get("gender"))
        effective = _date_value(payload.get("match_date"), "match_date")
        captured = max(
            _instant(row["first_seen_at_utc"], "first_seen_at_utc"),
            _instant(row["published_at_utc"], "published_at_utc"),
        )
        output.append(
            _evidence(
                source_family="tennisratio",
                evidence_kind="direct_agenda",
                source_match_id=str(row["source_match_id"]),
                gender=gender,
                effective_date=effective,
                tournament=_required_text(payload.get("tournament"), "tournament"),
                tournament_href=_optional_text(payload.get("tournament_href")),
                tour_level=_optional_text(payload.get("tour_level")),
                surface=surface,
                captured_at_utc=captured,
                source_url=str(row["source_url"]),
                source_sha256=str(row["source_sha256"]),
            )
        )
    return tuple(output)


def _load_tennis_explorer_evidence(
    database_path: Path,
    *,
    as_of_date: date,
) -> tuple[SurfaceEvidence, ...]:
    """Lee snapshots Explorer existentes sin modificar el triplete sagrado."""

    if not database_path.is_file():
        return ()
    observation_query = """
        SELECT observation.observation_id AS record_id,
               observation.source_match_id, observation.observed_at_utc,
               observation.payload_sha256, observation.payload_json
        FROM observations AS observation
        WHERE observation.source_system = 'tennis_explorer'
          AND observation.is_valid = 1
          AND substr(observation.observed_at_utc, 1, 10) < ?
        ORDER BY observation.observed_at_utc, observation.observation_id
    """
    prediction_query = """
        SELECT prediction.prediction_id AS record_id,
               prediction.source_match_id,
               prediction.source_retrieved_at_utc AS observed_at_utc,
               prediction.payload_sha256, prediction.payload_json
        FROM predictions AS prediction
        JOIN matches AS match USING (source_match_id)
        WHERE match.source_system = 'tennis_explorer'
          AND prediction.source_retrieved_at_utc IS NOT NULL
          AND substr(prediction.source_retrieved_at_utc, 1, 10) < ?
        ORDER BY prediction.source_retrieved_at_utc, prediction.prediction_id
    """
    with _read_only_connection(database_path) as connection:
        rows = (
            *connection.execute(
                observation_query,
                (as_of_date.isoformat(),),
            ).fetchall(),
            *connection.execute(
                prediction_query,
                (as_of_date.isoformat(),),
            ).fetchall(),
        )
    output: list[SurfaceEvidence] = []
    for row in rows:
        payload = _payload_object(row["payload_json"])
        surface = _surface_or_none(payload.get("feature_surface", payload.get("surface")))
        if surface is None:
            surface = _surface_or_none(payload.get("surface"))
        if surface is None:
            continue
        tournament_href = _optional_text(payload.get("tournament_href"))
        if tournament_href is None:
            continue
        observed = _instant(row["observed_at_utc"], "observed_at_utc")
        payload_retrieved = payload.get("retrieved_at_utc")
        captured = (
            observed
            if payload_retrieved is None
            else max(observed, _instant(payload_retrieved, "retrieved_at_utc"))
        )
        effective = _date_value(
            payload.get("match_date", payload.get("prediction_date")),
            "match_date",
        )
        source_url = _optional_text(payload.get("source_url")) or (
            "operations:predictions-or-observations"
        )
        source_sha256 = (
            _optional_text(payload.get("snapshot_sha256"))
            or _optional_text(payload.get("source_snapshot_sha256"))
            or str(row["payload_sha256"])
        )
        output.append(
            _evidence(
                source_family="tennis_explorer",
                evidence_kind="direct_agenda",
                source_match_id=str(row["source_match_id"]),
                gender=_gender(payload.get("gender")),
                effective_date=effective,
                tournament=_required_text(payload.get("tournament"), "tournament"),
                tournament_href=tournament_href,
                tour_level=_optional_text(payload.get("tour_level")),
                surface=surface,
                captured_at_utc=captured,
                source_url=source_url,
                source_sha256=source_sha256,
            )
        )
    return tuple(output)


def _evidence(
    *,
    source_family: SurfaceSource,
    evidence_kind: SurfaceEvidenceKind,
    source_match_id: str,
    gender: Gender,
    effective_date: date,
    tournament: str,
    tournament_href: str | None,
    tour_level: str | None,
    surface: Surface,
    captured_at_utc: datetime,
    source_url: str,
    source_sha256: str,
) -> SurfaceEvidence:
    """Construye una observación con ID de contenido reproducible."""

    payload = {
        "schema": CATALOG_SCHEMA,
        "source_family": source_family,
        "evidence_kind": evidence_kind,
        "source_match_id": source_match_id,
        "gender": gender,
        "effective_date": effective_date.isoformat(),
        "tournament": tournament,
        "tournament_href": tournament_href,
        "tour_level": tour_level,
        "surface": surface,
        "captured_at_utc": captured_at_utc.isoformat(),
        "source_url": source_url,
        "source_sha256": source_sha256,
    }
    return SurfaceEvidence(
        evidence_id=_stable_hash(payload),
        source_family=source_family,
        evidence_kind=evidence_kind,
        source_match_id=source_match_id,
        gender=gender,
        edition_year=effective_date.year,
        tournament=tournament,
        tournament_href=tournament_href,
        tour_level=tour_level,
        surface=surface,
        effective_date=effective_date,
        captured_at_utc=captured_at_utc,
        source_url=source_url,
        source_sha256=source_sha256,
    )


def _deduplicate_evidence(
    evidence: Iterable[SurfaceEvidence],
) -> tuple[SurfaceEvidence, ...]:
    """Colapsa únicamente observaciones byte-equivalentes por ID."""

    by_id: dict[str, SurfaceEvidence] = {}
    for item in evidence:
        previous = by_id.setdefault(item.evidence_id, item)
        if previous != item:
            raise SurfaceCatalogError("Un evidence_id identifica dos superficies distintas.")
    return tuple(
        sorted(
            by_id.values(),
            key=lambda item: (
                item.captured_at_utc,
                item.source_family,
                item.evidence_id,
            ),
        )
    )


def _catalog_fingerprint(
    as_of_date: date,
    evidence: tuple[SurfaceEvidence, ...],
) -> str:
    """Hash estable del corte y todas las evidencias que contiene."""

    return _stable_hash(
        {
            "schema": CATALOG_SCHEMA,
            "as_of_date": as_of_date.isoformat(),
            "evidence": [item.as_dict() for item in evidence],
        }
    )


def _publish_cache(path: Path, catalog: SurfaceCatalog) -> None:
    """Publica una única caché JSON atómica sin acumular versiones."""

    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{resolved.name}.",
            suffix=".tmp",
            dir=resolved.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(
                catalog.as_dict(),
                stream,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, resolved)
    except OSError as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise SurfaceCatalogError(f"No se pudo publicar la caché {resolved}.") from exc


def _tournament_identity(
    *,
    source_family: SurfaceSource,
    gender: Gender,
    edition_year: int,
    tournament: str,
    tournament_href: str | None,
) -> str | None:
    """Crea una clave exacta de fuente+edición; nunca clasifica por nombre."""

    label = " ".join(str(tournament).split())
    if not label or _AGGREGATE_TOURNAMENT.fullmatch(label):
        return None
    if source_family == "tennis_explorer":
        href = _optional_text(tournament_href)
        if href is None:
            return None
        parsed = urlsplit(href)
        path = parsed.path.rstrip("/") + "/"
        if parsed.query or parsed.fragment or f"/{edition_year}/" not in path:
            return None
        return f"tennis_explorer|{gender}|{edition_year}|href:{path.casefold()}"
    return f"tennisratio|{gender}|{edition_year}|label:{label.casefold()}"


def _surface_or_none(value: object) -> Surface | None:
    """Acepta solo las cuatro superficies exactas y rechaza `Indoors`."""

    if value is None:
        return None
    text = str(value).strip().casefold()
    if text in _AMBIGUOUS_SURFACES:
        return None
    return _SURFACES.get(text)


def _gender(value: object) -> Gender:
    """Valida el universo ATP/WTA ya normalizado."""

    text = str(value)
    if text not in {"M", "F"}:
        raise SurfaceCatalogError(f"Género de superficie inválido: {value!r}.")
    return cast(Gender, text)


def _payload_object(value: object) -> dict[str, object]:
    """Decodifica un payload SQLite y exige un objeto JSON."""

    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise SurfaceCatalogError("Una evidencia de superficie no contiene JSON válido.") from exc
    if not isinstance(payload, dict):
        raise SurfaceCatalogError("El payload de superficie no es un objeto JSON.")
    return cast(dict[str, object], payload)


def _date_value(value: object, field: str) -> date:
    """Convierte fecha ISO o timestamp sin perder su día civil."""

    text = _required_text(value, field)
    try:
        return date.fromisoformat(text[:10])
    except ValueError as exc:
        raise SurfaceCatalogError(f"{field} no contiene una fecha ISO válida.") from exc


def _instant(value: object, field: str) -> datetime:
    """Convierte un instante ISO SQLite a UTC consciente de zona."""

    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise SurfaceCatalogError(f"{field} no es un instante ISO válido.") from exc
    return _aware_utc(parsed, field)


def _aware_utc(value: datetime, field: str) -> datetime:
    """Exige zona horaria y normaliza a UTC."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise SurfaceCatalogError(f"{field} debe incluir zona horaria.")
    return value.astimezone(UTC)


def _required_text(value: object, field: str) -> str:
    """Valida texto no vacío sin normalización semántica."""

    text = "" if value is None else str(value).strip()
    if not text:
        raise SurfaceCatalogError(f"{field} debe ser texto no vacío.")
    return text


def _optional_text(value: object) -> str | None:
    """Normaliza solo ausencia y espacios de un texto opcional."""

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _stable_hash(payload: Mapping[str, object]) -> str:
    """Calcula SHA-256 JSON determinista sin NaN."""

    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_id(*parts: str) -> str:
    """Deriva un ID de evidencia a partir de campos escalares."""

    return _stable_hash({"parts": list(parts)})


def _unresolved(reason: str) -> SurfaceResolution:
    """Construye una resolución ausente con razón explícita."""

    return SurfaceResolution(
        surface=None,
        method=None,
        evidence_id=None,
        captured_at_utc=None,
        source_family=None,
        source_url=None,
        source_sha256=None,
        tour_level=None,
        reason=reason,
    )


class _ReadOnlyConnection:
    """Context manager mínimo para una conexión SQLite inmutable."""

    def __init__(self, path: Path) -> None:
        """Conserva la ruta resuelta hasta entrar en el contexto."""

        self.path = path.resolve()
        self.connection: sqlite3.Connection | None = None

    def __enter__(self) -> sqlite3.Connection:
        """Abre SQLite con `mode=ro` y filas nombradas."""

        uri = f"{self.path.as_uri()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        self.connection = connection
        return connection

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        """Cierra siempre la conexión sin alterar la excepción activa."""

        del exc_type, exc_value, traceback
        if self.connection is not None:
            self.connection.close()


def _read_only_connection(path: Path) -> _ReadOnlyConnection:
    """Devuelve una conexión que no puede escribir la base fuente."""

    return _ReadOnlyConnection(path)


__all__ = [
    "CATALOG_SCHEMA",
    "Surface",
    "SurfaceCatalog",
    "SurfaceCatalogError",
    "SurfaceEvidence",
    "SurfaceResolution",
    "SurfaceSource",
    "build_surface_catalog",
    "empty_surface_catalog",
]
