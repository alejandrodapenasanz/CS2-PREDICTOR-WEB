"""Detecta y persiste identidades Sackmann incompatibles con la biografía.

Una misma clave ``(gender, player_id)`` no puede representar con seguridad a
un jugador cuyo histórico contiene partidos antes de cumplir diez años. El
módulo no intenta adivinar dónde cambia la identidad ni crea IDs sintéticos.
La fecha de nacimiento procede del maestro actual y no acredita cuándo se
conoció el conflicto; por ello la cuarentena NO selecciona ni elimina filas de
un backtest histórico. Su lista de claves solo sirve para degradar o bloquear
la inferencia operativa actual, donde la incompatibilidad ya es conocida.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import MappingProxyType
from typing import Callable, Final, Iterable, Mapping

import pandas as pd

from .config import (
    IDENTITY_QUARANTINE_MANIFEST_PATH,
    IDENTITY_QUARANTINE_PATH,
    RAW_DATA_DIR,
    SACKMANN_MANIFEST_PATH,
)
from .data_loaders import load_players
from .elo.build import VerifiedManifest, load_verified_manifest
IDENTITY_QUARANTINE_SCHEMA_VERSION: Final[str] = (
    "tennis-identity-quarantine-v3"
)
IDENTITY_QUARANTINE_USAGE: Final[str] = (
    "diagnostic_current_inference_only_no_historical_selection"
)
MIN_PLAUSIBLE_MATCH_AGE_YEARS: Final[int] = 10
QUARANTINE_COLUMNS: Final[tuple[str, ...]] = (
    "gender",
    "player_id",
    "player_name",
    "birth_date",
    "first_match_date",
    "last_match_date",
    "earliest_age_years",
    "appearances",
    "reason",
)


class IdentityIntegrityError(RuntimeError):
    """Indica que la cuarentena de identidades no es íntegra o reproducible."""


@dataclass(frozen=True, slots=True)
class IdentityConflict:
    """Describe una clave Sackmann incompatible con su fecha de nacimiento."""

    gender: str
    player_id: int
    player_name: str | None
    birth_date: date
    first_match_date: date
    last_match_date: date
    earliest_age_years: float
    appearances: int
    reason: str = "match_before_tenth_birthday"

    def as_record(self) -> dict[str, object]:
        """Devuelve un registro CSV estable con fechas ISO."""

        record = asdict(self)
        for field_name in ("birth_date", "first_match_date", "last_match_date"):
            record[field_name] = record[field_name].isoformat()
        return record


@dataclass(frozen=True, slots=True)
class IdentityQuarantineReport:
    """Resume una detección publicada o una carga verificada."""

    source_commit: str
    conflicts: tuple[IdentityConflict, ...]
    csv_path: Path
    manifest_path: Path
    csv_sha256: str

    @property
    def keys(self) -> frozenset[tuple[str, int]]:
        """Expone claves completas para degradación conservadora actual."""

        return frozenset(
            (conflict.gender, conflict.player_id)
            for conflict in self.conflicts
        )

    @property
    def exclusion_after_dates(self) -> Mapping[tuple[str, int], date]:
        """Expone la primera aparición como metadato diagnóstico no causal.

        Este mapping se conserva por compatibilidad de esquema y para poder
        auditar los artefactos existentes. No debe pasarse como selector de
        filas históricas: ``first_match_date`` se reconstruye con el snapshot
        completo y la DOB actual, no con evidencia fechada de aquel momento.
        """

        return MappingProxyType(
            {
                (conflict.gender, conflict.player_id): (
                    conflict.first_match_date
                )
                for conflict in self.conflicts
            }
        )


def _sha256_file(path: Path) -> str:
    """Calcula SHA-256 por bloques sin cargar el archivo completo."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    """Publica bytes mediante un temporal adyacente y reemplazo atómico."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise IdentityIntegrityError(
            f"No se pudo publicar atómicamente {path}."
        ) from exc


def _player_metadata(
    genders: Iterable[str],
    *,
    raw_dir: Path,
) -> dict[tuple[str, int], tuple[str | None, date]]:
    """Carga nombres y nacimientos conocidos para las claves solicitadas."""

    metadata: dict[tuple[str, int], tuple[str | None, date]] = {}
    for gender in genders:
        frame = load_players(gender, raw_dir=raw_dir)
        for row in frame.itertuples(index=False):
            if pd.isna(row.dob):
                continue
            player_name = (
                None
                if pd.isna(row.player_name)
                else str(row.player_name)
            )
            metadata[(gender, int(row.player_id))] = (
                player_name,
                pd.Timestamp(row.dob).date(),
            )
    return metadata


def _merge_appearance_stats(
    target: dict[tuple[str, int], tuple[date, date, int]],
    gender: str,
    identifiers: pd.Series,
    dates: pd.Series,
) -> None:
    """Agrega mínimo, máximo y apariciones de un lado de varios partidos."""

    side = pd.DataFrame({"player_id": identifiers, "match_date": dates})
    side["player_id"] = pd.to_numeric(
        side["player_id"], errors="coerce"
    ).astype("Int64")
    side.dropna(subset=["player_id", "match_date"], inplace=True)
    if side.empty:
        return
    grouped = side.groupby("player_id", observed=True)["match_date"].agg(
        ["min", "max", "count"]
    )
    for player_id, row in grouped.iterrows():
        key = (gender, int(player_id))
        minimum = pd.Timestamp(row["min"]).date()
        maximum = pd.Timestamp(row["max"]).date()
        count = int(row["count"])
        previous = target.get(key)
        if previous is None:
            target[key] = (minimum, maximum, count)
        else:
            target[key] = (
                min(previous[0], minimum),
                max(previous[1], maximum),
                previous[2] + count,
            )


def _scan_appearances(
    manifest: VerifiedManifest,
    *,
    chunksize: int,
) -> dict[tuple[str, int], tuple[date, date, int]]:
    """Recorre solo fecha e IDs de los CSV verificados del snapshot."""

    if isinstance(chunksize, bool) or not isinstance(chunksize, int):
        raise TypeError("chunksize debe ser un entero positivo.")
    if chunksize <= 0:
        raise ValueError("chunksize debe ser positivo.")
    stats: dict[tuple[str, int], tuple[date, date, int]] = {}
    for source in manifest.match_files:
        try:
            chunks = pd.read_csv(
                source.local_path,
                usecols=("tourney_date", "winner_id", "loser_id"),
                dtype="string",
                chunksize=chunksize,
                keep_default_na=False,
                na_values=[""],
                low_memory=False,
            )
            for chunk in chunks:
                dates = pd.to_datetime(
                    chunk["tourney_date"],
                    format="%Y%m%d",
                    errors="raise",
                )
                _merge_appearance_stats(
                    stats,
                    source.gender,
                    chunk["winner_id"],
                    dates,
                )
                _merge_appearance_stats(
                    stats,
                    source.gender,
                    chunk["loser_id"],
                    dates,
                )
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            raise IdentityIntegrityError(
                f"No se pudo auditar identidad en {source.local_path}."
            ) from exc
    return stats


def detect_identity_conflicts(
    *,
    manifest_path: Path = SACKMANN_MANIFEST_PATH,
    raw_dir: Path = RAW_DATA_DIR,
    chunksize: int = 100_000,
) -> tuple[str, tuple[IdentityConflict, ...]]:
    """Detecta IDs con partidos anteriores a su décimo cumpleaños.

    El umbral no pretende describir la edad mínima reglamentaria de todos los
    circuitos. Se usa como señal conservadora de que el maestro y el histórico
    no representan inequívocamente a la misma persona.
    """

    verified = load_verified_manifest(Path(manifest_path), Path(raw_dir))
    genders = tuple(sorted({source.gender for source in verified.match_files}))
    players = _player_metadata(genders, raw_dir=Path(raw_dir))
    appearances = _scan_appearances(verified, chunksize=chunksize)
    conflicts: list[IdentityConflict] = []
    for key, (first_match, last_match, count) in appearances.items():
        player = players.get(key)
        if player is None:
            continue
        player_name, birth_date = player
        tenth_birthday = (
            pd.Timestamp(birth_date) + pd.DateOffset(
                years=MIN_PLAUSIBLE_MATCH_AGE_YEARS
            )
        ).date()
        if first_match >= tenth_birthday:
            continue
        conflicts.append(
            IdentityConflict(
                gender=key[0],
                player_id=key[1],
                player_name=player_name,
                birth_date=birth_date,
                first_match_date=first_match,
                last_match_date=last_match,
                earliest_age_years=round(
                    (first_match - birth_date).days / 365.2425,
                    6,
                ),
                appearances=count,
            )
        )
    conflicts.sort(key=lambda item: (item.gender, item.player_id))
    return verified.source_commit, tuple(conflicts)


def publish_identity_quarantine(
    *,
    manifest_path: Path = SACKMANN_MANIFEST_PATH,
    raw_dir: Path = RAW_DATA_DIR,
    csv_path: Path = IDENTITY_QUARANTINE_PATH,
    quarantine_manifest_path: Path = IDENTITY_QUARANTINE_MANIFEST_PATH,
    chunksize: int = 100_000,
    clock: Callable[[], datetime] | None = None,
) -> IdentityQuarantineReport:
    """Detecta conflictos y publica CSV más manifiesto de integridad."""

    source_commit, conflicts = detect_identity_conflicts(
        manifest_path=manifest_path,
        raw_dir=raw_dir,
        chunksize=chunksize,
    )
    rows = [conflict.as_record() for conflict in conflicts]
    frame = pd.DataFrame.from_records(rows, columns=QUARANTINE_COLUMNS)
    csv_bytes = frame.to_csv(
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.6f",
    ).encode("utf-8")
    resolved_csv = Path(csv_path)
    _atomic_write_bytes(resolved_csv, csv_bytes)
    csv_hash = hashlib.sha256(csv_bytes).hexdigest()
    observed = clock() if clock is not None else datetime.now(UTC)
    if (
        not isinstance(observed, datetime)
        or observed.tzinfo is None
        or observed.utcoffset() is None
    ):
        raise IdentityIntegrityError("clock debe devolver datetime con zona.")
    payload = {
        "schema_version": IDENTITY_QUARANTINE_SCHEMA_VERSION,
        "source_commit": source_commit,
        "created_at_utc": observed.astimezone(UTC).isoformat(),
        "minimum_plausible_match_age_years": (
            MIN_PLAUSIBLE_MATCH_AGE_YEARS
        ),
        "historical_exclusion_rule": "disabled_noncausal_current_metadata",
        "usage_contract": IDENTITY_QUARANTINE_USAGE,
        "rows": len(conflicts),
        "csv": {
            "path": resolved_csv.name,
            "size": len(csv_bytes),
            "sha256": csv_hash,
        },
    }
    resolved_manifest = Path(quarantine_manifest_path)
    _atomic_write_bytes(
        resolved_manifest,
        (
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8"),
    )
    return IdentityQuarantineReport(
        source_commit=source_commit,
        conflicts=conflicts,
        csv_path=resolved_csv,
        manifest_path=resolved_manifest,
        csv_sha256=csv_hash,
    )


def _required_manifest_mapping(path: Path) -> Mapping[str, object]:
    """Lee un manifiesto de cuarentena como objeto JSON."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IdentityIntegrityError(
            f"No se pudo leer el manifiesto de identidades {path}."
        ) from exc
    if not isinstance(payload, Mapping):
        raise IdentityIntegrityError(
            "El manifiesto de identidades debe ser un objeto JSON."
        )
    return payload


def load_identity_quarantine(
    *,
    expected_source_commit: str,
    csv_path: Path = IDENTITY_QUARANTINE_PATH,
    manifest_path: Path = IDENTITY_QUARANTINE_MANIFEST_PATH,
) -> IdentityQuarantineReport:
    """Carga la cuarentena y verifica commit, esquema, tamaño y SHA-256."""

    resolved_csv = Path(csv_path)
    resolved_manifest = Path(manifest_path)
    payload = _required_manifest_mapping(resolved_manifest)
    if (
        payload.get("schema_version")
        != IDENTITY_QUARANTINE_SCHEMA_VERSION
        or payload.get("source_commit") != expected_source_commit
        or payload.get("minimum_plausible_match_age_years")
        != MIN_PLAUSIBLE_MATCH_AGE_YEARS
        or payload.get("historical_exclusion_rule")
        != "disabled_noncausal_current_metadata"
        or payload.get("usage_contract") != IDENTITY_QUARANTINE_USAGE
    ):
        raise IdentityIntegrityError(
            "La cuarentena no corresponde al commit o contrato activo."
        )
    csv_meta = payload.get("csv")
    if not isinstance(csv_meta, Mapping):
        raise IdentityIntegrityError("Falta metadata del CSV de cuarentena.")
    if not resolved_csv.is_file():
        raise IdentityIntegrityError(
            f"No existe el CSV de cuarentena {resolved_csv}."
        )
    expected_size = csv_meta.get("size")
    expected_hash = csv_meta.get("sha256")
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or resolved_csv.stat().st_size != expected_size
        or not isinstance(expected_hash, str)
        or _sha256_file(resolved_csv) != expected_hash
    ):
        raise IdentityIntegrityError(
            "El CSV de cuarentena no coincide con su manifiesto."
        )
    try:
        frame = pd.read_csv(
            resolved_csv,
            dtype={
                "gender": "string",
                "player_id": "Int64",
                "player_name": "string",
                "birth_date": "string",
                "first_match_date": "string",
                "last_match_date": "string",
                "earliest_age_years": "float64",
                "appearances": "Int64",
                "reason": "string",
            },
        )
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise IdentityIntegrityError(
            "El CSV de cuarentena no puede interpretarse."
        ) from exc
    if tuple(frame.columns) != QUARANTINE_COLUMNS:
        raise IdentityIntegrityError(
            "El CSV de cuarentena cambió de columnas u orden."
        )
    conflicts: list[IdentityConflict] = []
    for record in frame.to_dict(orient="records"):
        try:
            gender = str(record["gender"])
            player_id = int(record["player_id"])
            birth_date = date.fromisoformat(str(record["birth_date"]))
            first_match_date = date.fromisoformat(
                str(record["first_match_date"])
            )
            last_match_date = date.fromisoformat(
                str(record["last_match_date"])
            )
            if (
                gender not in {"M", "F"}
                or player_id < 0
                or first_match_date > last_match_date
            ):
                raise ValueError("Registro diagnóstico de identidad inválido.")
            conflicts.append(
                IdentityConflict(
                    gender=gender,
                    player_id=player_id,
                    player_name=(
                        None
                        if pd.isna(record["player_name"])
                        else str(record["player_name"])
                    ),
                    birth_date=birth_date,
                    first_match_date=first_match_date,
                    last_match_date=last_match_date,
                    earliest_age_years=float(
                        record["earliest_age_years"]
                    ),
                    appearances=int(record["appearances"]),
                    reason=str(record["reason"]),
                )
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise IdentityIntegrityError(
                "Una fila de cuarentena contiene valores inválidos."
            ) from exc
    declared_rows = payload.get("rows")
    if (
        isinstance(declared_rows, bool)
        or not isinstance(declared_rows, int)
        or declared_rows != len(conflicts)
    ):
        raise IdentityIntegrityError(
            "El recuento declarado de identidades no coincide."
        )
    keys = [(conflict.gender, conflict.player_id) for conflict in conflicts]
    if len(keys) != len(set(keys)):
        raise IdentityIntegrityError(
            "El CSV de cuarentena contiene reglas duplicadas."
        )
    return IdentityQuarantineReport(
        source_commit=expected_source_commit,
        conflicts=tuple(conflicts),
        csv_path=resolved_csv,
        manifest_path=resolved_manifest,
        csv_sha256=str(expected_hash),
    )
