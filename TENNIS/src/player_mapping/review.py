"""Persistencia estricta de overrides y de identidades no resueltas.

El módulo mantiene dos contratos CSV separados:

* una tabla manual e inyectable de overrides ``slug -> player_id``;
* una cola agregada de observaciones no resueltas para revisión humana.

La cola se actualiza bajo un lock local exclusivo y se sustituye de forma
atómica. Una repetición de la misma identidad en la misma fecha no aumenta
``occurrences``: esta métrica cuenta fechas observadas distintas, registradas
también de forma explícita en ``observed_dates``. Si faltan tanto un slug como
un nombre abreviado interpretable, la cola deduplica por el texto visible
exacto con un prefijo interno y deja vacías las columnas normalizadas.
"""

from __future__ import annotations

from contextlib import contextmanager
import csv
from dataclasses import dataclass
from datetime import date, datetime
import os
from pathlib import Path
import re
import tempfile
from typing import Final, Iterable, Iterator, Mapping, TypeAlias

from .types import (
    Gender,
    NameKey,
    PlayerMappingError,
    PlayerMappingSchemaError,
    PlayerMappingSourceError,
    PlayerMappingValidationError,
)


OVERRIDE_COLUMNS: Final[tuple[str, ...]] = (
    "gender",
    "slug",
    "player_id",
    "reason",
)

UNRESOLVED_COLUMNS: Final[tuple[str, ...]] = (
    "gender",
    "slug",
    "visible_name",
    "normalized_last_name",
    "first_initial",
    "reason",
    "candidate_count",
    "candidate_player_ids",
    "first_seen_date",
    "last_seen_date",
    "observed_dates",
    "occurrences",
    "last_tour_level",
    "last_tournament",
)

_POSITIVE_INTEGER_PATTERN: Final[re.Pattern[str]] = re.compile(r"[1-9]\d*")
_NONNEGATIVE_INTEGER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?:0|[1-9]\d*)"
)
_LIST_SEPARATOR: Final[str] = "|"

ReviewKey: TypeAlias = tuple[Gender, str]


class PlayerMappingReviewLockError(PlayerMappingError):
    """Indica que otra actualización posee el lock local de la cola."""


@dataclass(frozen=True, slots=True)
class OverrideRecord:
    """Representa una decisión manual, motivada y separada por género."""

    gender: Gender
    slug: str
    player_id: int
    reason: str

    def __post_init__(self) -> None:
        """Valida el contrato estricto de una fila de override."""

        _validate_gender(self.gender, field_context="override")
        _validate_slug(self.slug, allow_none=False, field_context="override")
        _validate_positive_player_id(
            self.player_id,
            field_context="override",
        )
        _validate_nonempty_text(self.reason, field_name="reason")


@dataclass(frozen=True, slots=True)
class UnresolvedObservation:
    """Describe una identidad no resuelta observada en una fecha concreta.

    ``candidate_count`` no se recibe como argumento: se deriva siempre de
    ``candidate_player_ids`` para impedir que ambas representaciones diverjan.
    """

    gender: Gender
    slug: str | None
    visible_name: str
    normalized_last_name: str | None
    first_initial: str | None
    reason: str
    candidate_player_ids: tuple[int, ...]
    observed_date: date
    tour_level: str
    tournament: str

    def __post_init__(self) -> None:
        """Valida que la observación pueda persistirse sin inferencias."""

        _validate_gender(self.gender, field_context="observación")
        _validate_slug(
            self.slug,
            allow_none=True,
            field_context="observación",
        )
        _validate_nonempty_text(
            self.visible_name,
            field_name="visible_name",
        )
        if (self.normalized_last_name is None) != (
            self.first_initial is None
        ):
            raise PlayerMappingValidationError(
                "normalized_last_name y first_initial deben estar ambos "
                "poblados o ambos ausentes."
            )
        if self.normalized_last_name is not None:
            NameKey(
                normalized_last_name=self.normalized_last_name,
                first_initial=self.first_initial,  # type: ignore[arg-type]
            )
        _validate_nonempty_text(self.reason, field_name="reason")
        if not isinstance(self.candidate_player_ids, tuple):
            raise PlayerMappingValidationError(
                "candidate_player_ids debe ser una tupla de player_id."
            )
        if len(set(self.candidate_player_ids)) != len(
            self.candidate_player_ids
        ):
            raise PlayerMappingValidationError(
                "candidate_player_ids no puede contener duplicados."
            )
        for player_id in self.candidate_player_ids:
            _validate_positive_player_id(
                player_id,
                field_context="candidate_player_ids",
            )
        if not isinstance(self.observed_date, date) or isinstance(
            self.observed_date,
            datetime,
        ):
            raise PlayerMappingValidationError(
                "observed_date debe ser datetime.date."
            )
        _validate_nonempty_text(
            self.tour_level,
            field_name="tour_level",
        )
        _validate_nonempty_text(
            self.tournament,
            field_name="tournament",
        )

    @property
    def candidate_count(self) -> int:
        """Devuelve el número de candidatos Sackmann auditados."""

        return len(self.candidate_player_ids)

    @property
    def review_key(self) -> ReviewKey:
        """Devuelve la clave estable usada para deduplicar la cola."""

        return _build_review_key(
            gender=self.gender,
            slug=self.slug,
            visible_name=self.visible_name,
            normalized_last_name=self.normalized_last_name,
            first_initial=self.first_initial,
        )


@dataclass(slots=True)
class _PendingRecord:
    """Estado agregado interno de una identidad aún pendiente."""

    gender: Gender
    slug: str | None
    visible_name: str
    normalized_last_name: str | None
    first_initial: str | None
    reason: str
    candidate_player_ids: tuple[int, ...]
    observed_dates: set[date]
    last_tour_level: str
    last_tournament: str

    @classmethod
    def from_observation(
        cls,
        observation: UnresolvedObservation,
    ) -> _PendingRecord:
        """Crea un pendiente nuevo a partir de una observación validada."""

        return cls(
            gender=observation.gender,
            slug=observation.slug,
            visible_name=observation.visible_name,
            normalized_last_name=observation.normalized_last_name,
            first_initial=observation.first_initial,
            reason=observation.reason,
            candidate_player_ids=observation.candidate_player_ids,
            observed_dates={observation.observed_date},
            last_tour_level=observation.tour_level,
            last_tournament=observation.tournament,
        )

    @property
    def review_key(self) -> ReviewKey:
        """Devuelve la clave con la que se almacena el pendiente."""

        return _build_review_key(
            gender=self.gender,
            slug=self.slug,
            visible_name=self.visible_name,
            normalized_last_name=self.normalized_last_name,
            first_initial=self.first_initial,
        )

    @property
    def first_seen_date(self) -> date:
        """Devuelve la primera fecha observada."""

        return min(self.observed_dates)

    @property
    def last_seen_date(self) -> date:
        """Devuelve la última fecha observada."""

        return max(self.observed_dates)

    def merge(self, observation: UnresolvedObservation) -> None:
        """Agrega una observación sin contar dos veces una misma fecha."""

        if observation.review_key != self.review_key:
            raise PlayerMappingValidationError(
                "No se pueden fusionar observaciones con claves distintas."
            )
        previous_last_seen = self.last_seen_date
        self.observed_dates.add(observation.observed_date)

        if observation.observed_date >= previous_last_seen:
            self.visible_name = observation.visible_name
            self.normalized_last_name = observation.normalized_last_name
            self.first_initial = observation.first_initial
            self.reason = observation.reason
            self.candidate_player_ids = observation.candidate_player_ids
            self.last_tour_level = observation.tour_level
            self.last_tournament = observation.tournament

    def to_csv_row(self) -> dict[str, str]:
        """Serializa el pendiente al esquema CSV canónico."""

        ordered_dates = sorted(self.observed_dates)
        return {
            "gender": self.gender,
            "slug": self.slug or "",
            "visible_name": self.visible_name,
            "normalized_last_name": self.normalized_last_name or "",
            "first_initial": self.first_initial or "",
            "reason": self.reason,
            "candidate_count": str(len(self.candidate_player_ids)),
            "candidate_player_ids": _LIST_SEPARATOR.join(
                str(player_id) for player_id in self.candidate_player_ids
            ),
            "first_seen_date": ordered_dates[0].isoformat(),
            "last_seen_date": ordered_dates[-1].isoformat(),
            "observed_dates": _LIST_SEPARATOR.join(
                observed_date.isoformat() for observed_date in ordered_dates
            ),
            "occurrences": str(len(ordered_dates)),
            "last_tour_level": self.last_tour_level,
            "last_tournament": self.last_tournament,
        }


def load_overrides(
    path: str | os.PathLike[str],
) -> dict[ReviewKey, OverrideRecord]:
    """Carga una tabla manual de overrides con esquema y claves estrictos.

    Args:
        path: CSV inyectable cuyas columnas, en este orden, deben ser
            ``gender,slug,player_id,reason``.

    Returns:
        Diccionario indexado por ``(gender, slug)``.

    Raises:
        PlayerMappingSourceError: Si el archivo no existe o no es un archivo.
        PlayerMappingSchemaError: Si el esquema o alguna fila no son válidos.
    """

    csv_path = _coerce_path(path, field_name="path")
    if not csv_path.exists():
        raise PlayerMappingSourceError(
            f"No existe el CSV de overrides: {csv_path}."
        )
    if not csv_path.is_file():
        raise PlayerMappingSourceError(
            f"La ruta de overrides no es un archivo: {csv_path}."
        )

    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            _validate_header(
                reader.fieldnames,
                expected=OVERRIDE_COLUMNS,
                source=csv_path,
            )
            overrides: dict[ReviewKey, OverrideRecord] = {}
            for line_number, row in enumerate(reader, start=2):
                record = _parse_override_row(
                    row,
                    source=csv_path,
                    line_number=line_number,
                )
                key: ReviewKey = (record.gender, record.slug)
                if key in overrides:
                    raise PlayerMappingSchemaError(
                        f"Override duplicado para {key!r} en "
                        f"{csv_path}, línea {line_number}."
                    )
                overrides[key] = record
    except UnicodeError as error:
        raise PlayerMappingSchemaError(
            f"El CSV de overrides no es UTF-8 válido: {csv_path}."
        ) from error
    except csv.Error as error:
        raise PlayerMappingSchemaError(
            f"El CSV de overrides no puede interpretarse: {csv_path}."
        ) from error
    return overrides


def update_unresolved_queue(
    path: str | os.PathLike[str],
    observations: Iterable[UnresolvedObservation],
    resolved_keys: Iterable[ReviewKey],
) -> int:
    """Actualiza de forma atómica la cola persistente de no resueltos.

    La clave primaria es ``(gender, slug)``. Cuando no hay slug, el segundo
    componente es ``"apellido normalizado inicial"``. Si tampoco se puede
    interpretar el texto, se usa ``"unparsed:<texto visible>"`` únicamente
    para deduplicar la revisión. Los elementos de ``resolved_keys`` deben usar
    exactamente la misma representación.

    Args:
        path: Destino CSV inyectable de la cola.
        observations: Observaciones actuales que siguen sin resolverse.
        resolved_keys: Claves que deben retirarse de pendientes anteriores.

    Returns:
        Número de identidades pendientes después de la actualización.

    Raises:
        PlayerMappingValidationError: Si los argumentos no cumplen el
            contrato.
        PlayerMappingSchemaError: Si una cola preexistente está corrupta.
        PlayerMappingReviewLockError: Si otra actualización posee el lock.
    """

    csv_path = _coerce_path(path, field_name="path")
    validated_observations = _materialize_observations(observations)
    validated_resolved_keys = _materialize_resolved_keys(resolved_keys)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with _exclusive_local_lock(csv_path):
        pending = _load_unresolved_queue(csv_path)
        for key in validated_resolved_keys:
            pending.pop(key, None)
        for observation in validated_observations:
            key = observation.review_key
            if key in validated_resolved_keys:
                continue
            current = pending.get(key)
            if current is None:
                pending[key] = _PendingRecord.from_observation(observation)
            else:
                current.merge(observation)
        _write_unresolved_queue_atomic(csv_path, pending)
    return len(pending)


def _validate_gender(value: object, *, field_context: str) -> None:
    """Valida un género sin normalizar ni corregir silenciosamente."""

    if value not in {"M", "F"}:
        raise PlayerMappingValidationError(
            f"gender debe ser exactamente 'M' o 'F' en {field_context}."
        )


def _validate_slug(
    value: object,
    *,
    allow_none: bool,
    field_context: str,
) -> None:
    """Valida un slug Tennis Explorer o un nulo expresamente permitido."""

    if value is None and allow_none:
        return
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(character.isspace() for character in value)
        or any(character in value for character in "/?#")
    ):
        null_note = " o None" if allow_none else ""
        raise PlayerMappingValidationError(
            f"slug debe ser texto no vacío sin espacios, '/', '?' ni '#'"
            f"{null_note} en {field_context}."
        )


def _validate_positive_player_id(
    value: object,
    *,
    field_context: str,
) -> None:
    """Valida un identificador Sackmann entero y positivo."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise PlayerMappingValidationError(
            f"player_id debe ser entero positivo en {field_context}."
        )


def _validate_nonempty_text(value: object, *, field_name: str) -> None:
    """Valida texto no vacío sin modificar espacios del dato original."""

    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
    ):
        raise PlayerMappingValidationError(
            f"{field_name} debe ser texto no vacío y sin espacios exteriores."
        )


def _coerce_path(
    value: str | os.PathLike[str],
    *,
    field_name: str,
) -> Path:
    """Convierte una ruta inyectada a ``Path`` con error controlado."""

    if not isinstance(value, (str, os.PathLike)):
        raise PlayerMappingValidationError(
            f"{field_name} debe ser str u os.PathLike."
        )
    try:
        path = Path(value)
    except (TypeError, ValueError) as error:
        raise PlayerMappingValidationError(
            f"{field_name} no contiene una ruta válida."
        ) from error
    if not str(path):
        raise PlayerMappingValidationError(
            f"{field_name} no puede ser una ruta vacía."
        )
    return path


def _build_review_key(
    *,
    gender: Gender,
    slug: str | None,
    visible_name: str,
    normalized_last_name: str | None,
    first_initial: str | None,
) -> ReviewKey:
    """Construye la clave por slug, nombre parseado o texto visible exacto."""

    if slug is not None:
        return (gender, slug)
    if normalized_last_name is not None and first_initial is not None:
        return (gender, f"{normalized_last_name} {first_initial}")
    return (gender, f"unparsed:{visible_name}")


def _validate_header(
    actual: list[str] | None,
    *,
    expected: tuple[str, ...],
    source: Path,
) -> None:
    """Exige un encabezado exacto para evitar pérdidas silenciosas."""

    if tuple(actual or ()) != expected:
        raise PlayerMappingSchemaError(
            f"Cabecera inválida en {source}: se esperaba {list(expected)!r} "
            f"y se recibió {actual!r}."
        )


def _parse_override_row(
    row: Mapping[str | None, str | list[str] | None],
    *,
    source: Path,
    line_number: int,
) -> OverrideRecord:
    """Convierte una fila CSV de override en un registro validado."""

    if None in row:
        raise PlayerMappingSchemaError(
            f"Hay columnas adicionales en {source}, línea {line_number}."
        )
    try:
        gender = _csv_text(row, "gender")
        slug = _csv_text(row, "slug")
        player_id_text = _csv_text(row, "player_id")
        reason = _csv_text(row, "reason")
        if _POSITIVE_INTEGER_PATTERN.fullmatch(player_id_text) is None:
            raise PlayerMappingSchemaError(
                "player_id debe ser un entero decimal positivo."
            )
        return OverrideRecord(
            gender=gender,  # type: ignore[arg-type]
            slug=slug,
            player_id=int(player_id_text),
            reason=reason,
        )
    except PlayerMappingValidationError as error:
        raise PlayerMappingSchemaError(
            f"Override inválido en {source}, línea {line_number}: {error}"
        ) from error
    except PlayerMappingSchemaError as error:
        raise PlayerMappingSchemaError(
            f"Override inválido en {source}, línea {line_number}: {error}"
        ) from error


def _csv_text(
    row: Mapping[str | None, str | list[str] | None],
    column: str,
) -> str:
    """Obtiene una celda textual obligatoria de una fila CSV."""

    value = row.get(column)
    if not isinstance(value, str):
        raise PlayerMappingSchemaError(
            f"La columna {column!r} no contiene texto."
        )
    if not value or value != value.strip():
        raise PlayerMappingSchemaError(
            f"La columna {column!r} debe ser texto no vacío "
            "y sin espacios exteriores."
        )
    return value


def _materialize_observations(
    observations: Iterable[UnresolvedObservation],
) -> tuple[UnresolvedObservation, ...]:
    """Materializa y valida observaciones antes de adquirir el lock."""

    if isinstance(observations, (str, bytes)):
        raise PlayerMappingValidationError(
            "observations debe ser un iterable de UnresolvedObservation."
        )
    try:
        materialized = tuple(observations)
    except TypeError as error:
        raise PlayerMappingValidationError(
            "observations debe ser iterable."
        ) from error
    for observation in materialized:
        if not isinstance(observation, UnresolvedObservation):
            raise PlayerMappingValidationError(
                "Cada observación debe ser UnresolvedObservation."
            )
    return materialized


def _materialize_resolved_keys(
    resolved_keys: Iterable[ReviewKey],
) -> set[ReviewKey]:
    """Materializa y valida claves resueltas antes de tocar el CSV."""

    if isinstance(resolved_keys, (str, bytes)):
        raise PlayerMappingValidationError(
            "resolved_keys debe ser un iterable de pares."
        )
    try:
        materialized = tuple(resolved_keys)
    except TypeError as error:
        raise PlayerMappingValidationError(
            "resolved_keys debe ser iterable."
        ) from error

    validated: set[ReviewKey] = set()
    for key in materialized:
        if not isinstance(key, tuple) or len(key) != 2:
            raise PlayerMappingValidationError(
                "Cada resolved_key debe ser una tupla (gender, identity)."
            )
        gender, identity = key
        _validate_gender(gender, field_context="resolved_keys")
        if (
            not isinstance(identity, str)
            or not identity
            or identity != identity.strip()
        ):
            raise PlayerMappingValidationError(
                "La identidad de resolved_keys debe ser texto no vacío."
            )
        validated.add((gender, identity))  # type: ignore[arg-type]
    return validated


def _load_unresolved_queue(
    path: Path,
) -> dict[ReviewKey, _PendingRecord]:
    """Carga y valida una cola existente; una ruta ausente equivale a vacía."""

    if not path.exists():
        return {}
    if not path.is_file():
        raise PlayerMappingSourceError(
            f"La ruta de no resueltos no es un archivo: {path}."
        )

    pending: dict[ReviewKey, _PendingRecord] = {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            _validate_header(
                reader.fieldnames,
                expected=UNRESOLVED_COLUMNS,
                source=path,
            )
            for line_number, row in enumerate(reader, start=2):
                record = _parse_pending_row(
                    row,
                    source=path,
                    line_number=line_number,
                )
                if record.review_key in pending:
                    raise PlayerMappingSchemaError(
                        f"Clave pendiente duplicada {record.review_key!r} "
                        f"en {path}, línea {line_number}."
                    )
                pending[record.review_key] = record
    except UnicodeError as error:
        raise PlayerMappingSchemaError(
            f"El CSV de no resueltos no es UTF-8 válido: {path}."
        ) from error
    except csv.Error as error:
        raise PlayerMappingSchemaError(
            f"El CSV de no resueltos no puede interpretarse: {path}."
        ) from error
    return pending


def _parse_pending_row(
    row: Mapping[str | None, str | list[str] | None],
    *,
    source: Path,
    line_number: int,
) -> _PendingRecord:
    """Convierte y valida una fila agregada de la cola."""

    if None in row:
        raise PlayerMappingSchemaError(
            f"Hay columnas adicionales en {source}, línea {line_number}."
        )
    try:
        gender = _csv_text(row, "gender")
        slug_text = row.get("slug")
        if not isinstance(slug_text, str):
            raise PlayerMappingSchemaError(
                "La columna 'slug' debe ser texto, aunque esté vacía."
            )
        slug = slug_text or None
        visible_name = _csv_text(row, "visible_name")
        normalized_last_name_text = row.get("normalized_last_name")
        first_initial_text = row.get("first_initial")
        if not isinstance(normalized_last_name_text, str) or not isinstance(
            first_initial_text,
            str,
        ):
            raise PlayerMappingSchemaError(
                "normalized_last_name y first_initial deben ser texto, "
                "aunque estén vacíos."
            )
        normalized_last_name = normalized_last_name_text or None
        first_initial = first_initial_text or None
        reason = _csv_text(row, "reason")
        candidate_count = _parse_nonnegative_integer_cell(
            row,
            "candidate_count",
        )
        candidate_player_ids = _parse_player_id_list(
            row.get("candidate_player_ids")
        )
        first_seen_date = _parse_iso_date_cell(row, "first_seen_date")
        last_seen_date = _parse_iso_date_cell(row, "last_seen_date")
        observed_dates = _parse_date_list(row.get("observed_dates"))
        occurrences = _parse_nonnegative_integer_cell(row, "occurrences")
        last_tour_level = _csv_text(row, "last_tour_level")
        last_tournament = _csv_text(row, "last_tournament")

        _validate_gender(gender, field_context="cola de no resueltos")
        _validate_slug(
            slug,
            allow_none=True,
            field_context="cola de no resueltos",
        )
        _validate_nonempty_text(
            visible_name,
            field_name="visible_name",
        )
        if (normalized_last_name is None) != (first_initial is None):
            raise PlayerMappingSchemaError(
                "normalized_last_name y first_initial deben estar ambos "
                "poblados o ambos vacíos."
            )
        if normalized_last_name is not None:
            NameKey(
                normalized_last_name=normalized_last_name,
                first_initial=first_initial,  # type: ignore[arg-type]
            )
        if candidate_count != len(candidate_player_ids):
            raise PlayerMappingSchemaError(
                "candidate_count no coincide con candidate_player_ids."
            )
        if not observed_dates:
            raise PlayerMappingSchemaError(
                "observed_dates debe contener al menos una fecha."
            )
        if first_seen_date != min(observed_dates):
            raise PlayerMappingSchemaError(
                "first_seen_date no coincide con observed_dates."
            )
        if last_seen_date != max(observed_dates):
            raise PlayerMappingSchemaError(
                "last_seen_date no coincide con observed_dates."
            )
        if occurrences != len(observed_dates):
            raise PlayerMappingSchemaError(
                "occurrences no coincide con las fechas observadas únicas."
            )
        return _PendingRecord(
            gender=gender,  # type: ignore[arg-type]
            slug=slug,
            visible_name=visible_name,
            normalized_last_name=normalized_last_name,
            first_initial=first_initial,
            reason=reason,
            candidate_player_ids=candidate_player_ids,
            observed_dates=observed_dates,
            last_tour_level=last_tour_level,
            last_tournament=last_tournament,
        )
    except PlayerMappingValidationError as error:
        raise PlayerMappingSchemaError(
            f"Pendiente inválido en {source}, línea {line_number}: {error}"
        ) from error
    except PlayerMappingSchemaError as error:
        raise PlayerMappingSchemaError(
            f"Pendiente inválido en {source}, línea {line_number}: {error}"
        ) from error


def _parse_nonnegative_integer_cell(
    row: Mapping[str | None, str | list[str] | None],
    column: str,
) -> int:
    """Interpreta un entero decimal no negativo de una fila CSV."""

    text = _csv_text(row, column)
    if _NONNEGATIVE_INTEGER_PATTERN.fullmatch(text) is None:
        raise PlayerMappingSchemaError(
            f"La columna {column!r} debe ser un entero no negativo."
        )
    return int(text)


def _parse_player_id_list(value: str | list[str] | None) -> tuple[int, ...]:
    """Interpreta la lista canónica de player_id separada por ``|``."""

    if not isinstance(value, str):
        raise PlayerMappingSchemaError(
            "candidate_player_ids debe ser texto."
        )
    if not value:
        return ()
    components = value.split(_LIST_SEPARATOR)
    if any(
        _POSITIVE_INTEGER_PATTERN.fullmatch(component) is None
        for component in components
    ):
        raise PlayerMappingSchemaError(
            "candidate_player_ids contiene un player_id inválido."
        )
    player_ids = tuple(int(component) for component in components)
    if len(set(player_ids)) != len(player_ids):
        raise PlayerMappingSchemaError(
            "candidate_player_ids contiene duplicados."
        )
    return player_ids


def _parse_iso_date_cell(
    row: Mapping[str | None, str | list[str] | None],
    column: str,
) -> date:
    """Interpreta una fecha ISO estricta de una fila CSV."""

    return _parse_iso_date(_csv_text(row, column), field_name=column)


def _parse_iso_date(value: str, *, field_name: str) -> date:
    """Interpreta ``YYYY-MM-DD`` sin aceptar representaciones alternativas."""

    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise PlayerMappingSchemaError(
            f"{field_name} no es una fecha ISO válida: {value!r}."
        ) from error
    if parsed.isoformat() != value:
        raise PlayerMappingSchemaError(
            f"{field_name} debe usar exactamente YYYY-MM-DD."
        )
    return parsed


def _parse_date_list(value: str | list[str] | None) -> set[date]:
    """Interpreta fechas ISO únicas separadas por ``|``."""

    if not isinstance(value, str) or not value:
        raise PlayerMappingSchemaError(
            "observed_dates debe ser una lista textual no vacía."
        )
    components = value.split(_LIST_SEPARATOR)
    dates = {
        _parse_iso_date(component, field_name="observed_dates")
        for component in components
    }
    if len(dates) != len(components):
        raise PlayerMappingSchemaError(
            "observed_dates contiene fechas duplicadas."
        )
    return dates


@contextmanager
def _exclusive_local_lock(path: Path) -> Iterator[None]:
    """Adquiere mediante creación exclusiva un lock vecino al CSV."""

    lock_path = path.with_name(f"{path.name}.lock")
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as error:
        raise PlayerMappingReviewLockError(
            f"La cola {path} está bloqueada por {lock_path}. "
            "Si ningún proceso sigue activo, revise y retire manualmente "
            "ese lock."
        ) from error
    except OSError as error:
        raise PlayerMappingReviewLockError(
            f"No se pudo crear el lock exclusivo {lock_path}: {error}."
        ) from error

    try:
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as lock:
            lock.write(f"pid={os.getpid()}\n")
            lock.flush()
            os.fsync(lock.fileno())
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _write_unresolved_queue_atomic(
    path: Path,
    pending: Mapping[ReviewKey, _PendingRecord],
) -> None:
    """Escribe el CSV completo y lo sustituye atómicamente en el mismo disco."""

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(
                handle,
                fieldnames=list(UNRESOLVED_COLUMNS),
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            for key in sorted(pending):
                writer.writerow(pending[key].to_csv_row())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    except OSError as error:
        raise PlayerMappingSourceError(
            f"No se pudo actualizar atómicamente la cola {path}: {error}."
        ) from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
