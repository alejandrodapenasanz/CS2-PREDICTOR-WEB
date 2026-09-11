"""Descarga, versiona y parsea los rankings Elo públicos de Tennis Abstract.

La fuente se limita deliberadamente a las dos páginas HTML públicas de Elo
general ATP y WTA. Cada intento HTTP queda registrado en un manifiesto local
y se ejecuta de forma secuencial, sin imponer una cuota local: la frecuencia
la controla el operador. No se consultan páginas de jugadores ni otras rutas.

El parser refleja la estructura observada el 30 de julio de 2026: una única
tabla ``table#reportable`` con 17 columnas, incluidas tres columnas separadoras
vacías. Cualquier cambio de esa firma produce ``EloSchemaError``. La descarga
se conserva antes de parsearse, de modo que un cambio de HTML nunca se oculta
ni se adapta mediante supuestos.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
from html.parser import HTMLParser
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Callable, Final, Literal, Mapping, Protocol
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests

from src.config import TENNIS_ABSTRACT_ELO_RAW_DIR
from src.responsible_http import ResponsibleHttpClient, build_http_client


LOGGER = logging.getLogger(__name__)

Gender = Literal["M", "F"]

ELO_URLS: Final[dict[Gender, str]] = {
    "M": "https://www.tennisabstract.com/reports/atp_elo_ratings.html",
    "F": "https://www.tennisabstract.com/reports/wta_elo_ratings.html",
}
ELO_SOURCE_ORDER: Final[tuple[Gender, Gender]] = ("M", "F")
TOUR_LABELS: Final[dict[Gender, str]] = {"M": "atp", "F": "wta"}
EXPECTED_PLAYER_PATHS: Final[dict[Gender, str]] = {
    "M": "/cgi-bin/player.cgi",
    "F": "/cgi-bin/wplayer.cgi",
}
BLOCKED_PATH_PREFIXES: Final[tuple[str, ...]] = (
    "/jsfrags/",
    "/jsmatches/",
    "/jsplayers/",
)

USER_AGENT: Final[str] = "CS2-Predictor-TENNIS-TennisAbstract-Elo-Ingestion/1.0"
REQUEST_TIMEOUT: Final[tuple[float, float]] = (10.0, 30.0)
FAILURE_BACKOFF: Final[timedelta] = timedelta(hours=24)
MANIFEST_SCHEMA_VERSION: Final[int] = 1
SNAPSHOT_METADATA_SCHEMA_VERSION: Final[int] = 1

EXPECTED_HEADERS: Final[dict[Gender, tuple[str, ...]]] = {
    "M": (
        "Elo Rank",
        "Player",
        "Age",
        "Elo",
        "",
        "hElo Rank",
        "hElo",
        "cElo Rank",
        "cElo",
        "gElo Rank",
        "gElo",
        "",
        "Peak Elo",
        "Peak Month",
        "",
        "ATP Rank",
        "Log diff",
    ),
    "F": (
        "Elo Rank",
        "Player",
        "Age",
        "Elo",
        "",
        "hElo Rank",
        "hElo",
        "cElo Rank",
        "cElo",
        "gElo Rank",
        "gElo",
        "",
        "Peak Elo",
        "Peak Month",
        "",
        "WTA Rank",
        "Log diff",
    ),
}

OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "gender",
    "rating_date",
    "elo_rank",
    "player_name",
    "player_url",
    "age",
    "elo",
    "hard_elo_rank",
    "hard_elo",
    "clay_elo_rank",
    "clay_elo",
    "grass_elo_rank",
    "grass_elo",
    "peak_elo",
    "peak_month",
    "official_rank",
    "log_diff",
    "source_url",
    "retrieved_at_utc",
)

_INTEGER_COLUMNS: Final[tuple[str, ...]] = (
    "elo_rank",
    "hard_elo_rank",
    "clay_elo_rank",
    "grass_elo_rank",
    "official_rank",
)
_FLOAT_COLUMNS: Final[tuple[str, ...]] = (
    "age",
    "elo",
    "hard_elo",
    "clay_elo",
    "grass_elo",
    "peak_elo",
    "log_diff",
)
_REQUIRED_VALUE_COLUMNS: Final[tuple[str, ...]] = (
    "elo_rank",
    "player_name",
    "elo",
)
_RATING_DATE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\bLast\s+update:\s*(\d{4}-\d{2}-\d{2})\b",
    flags=re.IGNORECASE,
)


class TennisAbstractEloError(RuntimeError):
    """Representa un fallo controlado de la ingesta Elo de Tennis Abstract."""


class EloManifestError(TennisAbstractEloError):
    """Indica que el manifiesto o la metadata local no son fiables."""


class EloSchemaError(TennisAbstractEloError):
    """Indica que el HTML ya no coincide con la tabla Elo inspeccionada."""


class EloSnapshotIntegrityError(TennisAbstractEloError):
    """Indica que un snapshot no coincide con el hash de su metadata."""


class HttpSession(Protocol):
    """Define la parte mínima de una sesión HTTP usada por el descargador."""

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: tuple[float, float],
        allow_redirects: bool,
    ) -> Any:
        """Realiza exactamente un ``GET`` sin redirects ni retries automáticos."""

    def close(self) -> None:
        """Libera los recursos asociados a la sesión."""


@dataclass(frozen=True)
class EloFetchResult:
    """Describe el resultado para una de las dos páginas Elo."""

    gender: Gender
    url: str
    outcome: str
    status_code: int | None
    attempted_at_utc: datetime | None
    retry_after_utc: datetime | None
    snapshot_path: Path | None = None
    metadata_path: Path | None = None
    detail: str | None = None


@dataclass(frozen=True)
class EloFetchReport:
    """Resume una ejecución, sus GET secuenciales y la validación local."""

    results: tuple[EloFetchResult, ...]
    manifest_path: Path
    get_count: int
    locally_validated: tuple[Path, ...]

    @property
    def downloaded_count(self) -> int:
        """Devuelve el número de nuevas respuestas ``200`` versionadas."""

        return sum(result.outcome == "downloaded" for result in self.results)

    @property
    def failed_count(self) -> int:
        """Devuelve el número de resultados HTTP, red o almacenamiento fallidos."""

        failure_outcomes = {
            "http_error",
            "network_error",
            "schema_error",
            "storage_error",
            "invalid_not_modified",
        }
        return sum(result.outcome in failure_outcomes for result in self.results)


@dataclass(frozen=True)
class _HtmlCell:
    """Conserva texto, etiqueta y enlaces de una celda HTML inspeccionada."""

    tag: str
    text: str
    hrefs: tuple[str, ...]


class _EloTableExtractor(HTMLParser):
    """Extrae únicamente tablas cuyo identificador sea ``reportable``."""

    def __init__(self) -> None:
        """Inicializa el estado de análisis sin reparar HTML incompleto."""

        super().__init__(convert_charrefs=True)
        self.tables: list[tuple[tuple[_HtmlCell, ...], ...]] = []
        self.document_text: list[str] = []
        self._table_depth = 0
        self._rows: list[tuple[_HtmlCell, ...]] | None = None
        self._row: list[_HtmlCell] | None = None
        self._cell_tag: str | None = None
        self._cell_text: list[str] = []
        self._cell_hrefs: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        """Registra aperturas relevantes dentro de ``table#reportable``."""

        normalized_tag = tag.casefold()
        attributes = {key.casefold(): value for key, value in attrs if value is not None}
        if normalized_tag == "table":
            if self._table_depth > 0:
                self._table_depth += 1
            elif attributes.get("id") == "reportable":
                self._table_depth = 1
                self._rows = []
            return

        if self._table_depth != 1:
            return
        if normalized_tag == "tr":
            if self._row is not None:
                raise EloSchemaError("La tabla Elo contiene filas HTML anidadas o sin cerrar.")
            self._row = []
            return
        if normalized_tag in {"th", "td"} and self._row is not None:
            if self._cell_tag is not None:
                raise EloSchemaError("La tabla Elo contiene celdas HTML anidadas o sin cerrar.")
            self._cell_tag = normalized_tag
            self._cell_text = []
            self._cell_hrefs = []
            return
        if normalized_tag == "a" and self._cell_tag is not None:
            href = attributes.get("href")
            if href is not None:
                self._cell_hrefs.append(href)
        elif normalized_tag == "br" and self._cell_tag is not None:
            self._cell_text.append(" ")

    def handle_endtag(self, tag: str) -> None:
        """Cierra celdas, filas y la tabla objetivo sin inferir elementos."""

        normalized_tag = tag.casefold()
        if normalized_tag == "table" and self._table_depth > 0:
            if self._table_depth == 1:
                if self._cell_tag is not None or self._row is not None:
                    raise EloSchemaError("La tabla Elo termina con una celda o fila sin cerrar.")
                if self._rows is None:
                    raise EloSchemaError("El extractor perdió el contenido de la tabla Elo.")
                self.tables.append(tuple(self._rows))
                self._rows = None
            self._table_depth -= 1
            return

        if self._table_depth != 1:
            return
        if normalized_tag in {"th", "td"} and self._cell_tag is not None:
            if normalized_tag != self._cell_tag:
                raise EloSchemaError("La tabla Elo cierra una celda con una etiqueta distinta.")
            if self._row is None:
                raise EloSchemaError("Se encontró una celda Elo fuera de una fila.")
            self._row.append(
                _HtmlCell(
                    tag=self._cell_tag,
                    text=_normalize_text("".join(self._cell_text)),
                    hrefs=tuple(self._cell_hrefs),
                )
            )
            self._cell_tag = None
            self._cell_text = []
            self._cell_hrefs = []
            return
        if normalized_tag == "tr" and self._row is not None:
            if self._cell_tag is not None:
                raise EloSchemaError("Una fila Elo termina con una celda abierta.")
            if self._rows is None:
                raise EloSchemaError("Se encontró una fila Elo fuera de la tabla.")
            self._rows.append(tuple(self._row))
            self._row = None

    def handle_data(self, data: str) -> None:
        """Conserva texto documental y el texto literal de la celda activa."""

        self.document_text.append(data)
        if self._table_depth == 1 and self._cell_tag is not None:
            self._cell_text.append(data)

    def validate_closed(self) -> None:
        """Comprueba que el documento no dejó abierta la tabla objetivo."""

        if (
            self._table_depth != 0
            or self._rows is not None
            or self._row is not None
            or self._cell_tag is not None
        ):
            raise EloSchemaError("El HTML termina antes de cerrar completamente table#reportable.")


def _utc_now() -> datetime:
    """Devuelve la hora actual como ``datetime`` consciente en UTC."""

    return datetime.now(UTC)


def _as_utc(value: datetime, *, field_name: str) -> datetime:
    """Convierte una fecha consciente a UTC y rechaza fechas sin zona."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} debe incluir una zona horaria explícita.")
    return value.astimezone(UTC)


def _parse_utc_iso(value: Any, *, field_name: str) -> datetime:
    """Parsea un timestamp ISO del manifiesto y exige información de zona."""

    if not isinstance(value, str):
        raise EloManifestError(f"{field_name} no contiene un timestamp ISO.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return _as_utc(parsed, field_name=field_name)
    except ValueError as exc:
        raise EloManifestError(f"{field_name} no contiene un timestamp ISO UTC válido.") from exc


def _iso_utc(value: datetime) -> str:
    """Serializa un ``datetime`` consciente con offset UTC explícito."""

    return _as_utc(value, field_name="timestamp").isoformat()


def _normalize_text(value: str) -> str:
    """Colapsa espacios HTML, incluidos ``nbsp``, sin alterar caracteres."""

    return " ".join(value.replace("\xa0", " ").split())


def _validate_gender(gender: str) -> Gender:
    """Valida un código de género sin inferir alias ni mezclar universos."""

    if gender not in ELO_URLS:
        raise ValueError("gender debe ser exactamente 'M' o 'F'.")
    return gender


def _validate_public_url(gender: Gender, url: str) -> None:
    """Exige la URL pública auditada y rechaza cualquier ruta bloqueada."""

    expected_url = ELO_URLS[gender]
    if url != expected_url:
        raise TennisAbstractEloError(
            f"La URL para {gender} no coincide con la fuente auditada: {url!r}."
        )
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "www.tennisabstract.com":
        raise TennisAbstractEloError(f"La URL Elo no es HTTPS pública: {url!r}.")
    if any(parsed.path.startswith(prefix) for prefix in BLOCKED_PATH_PREFIXES):
        raise TennisAbstractEloError(
            f"La URL Elo entra en una ruta excluida por robots.txt: {url!r}."
        )


def build_http_session() -> ResponsibleHttpClient:
    """Crea una sesión identificada y desactiva reintentos HTTP automáticos."""

    return build_http_client(
        transport_kind="scrapling",
        browser_impersonation=True,
        stealthy_headers=True,
        detect_waf=True,
        default_headers={"User-Agent": USER_AGENT},
        robots_user_agent=USER_AGENT,
        request_retry_attempts=1,
    )


def _new_manifest(created_at: datetime) -> dict[str, Any]:
    """Crea un manifiesto vacío con la política de acceso incorporada."""

    timestamp = _iso_utc(created_at)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source": "Tennis Abstract public Elo reports",
        "created_at_utc": timestamp,
        "updated_at_utc": timestamp,
        "operator_managed_frequency": True,
        "failure_backoff_hours": 24,
        "sources": {},
        "history": [],
    }


def _load_manifest(manifest_path: Path, now: datetime) -> dict[str, Any]:
    """Carga y valida el manifiesto, o crea uno en memoria si aún no existe."""

    if not manifest_path.exists():
        return _new_manifest(now)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EloManifestError(f"No se puede leer el manifiesto Elo: {manifest_path}.") from exc
    if not isinstance(payload, dict):
        raise EloManifestError("El manifiesto Elo no es un objeto JSON.")
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise EloManifestError("La versión del manifiesto Elo no coincide con la implementada.")
    if not isinstance(payload.get("sources"), dict):
        raise EloManifestError("El manifiesto Elo no contiene un objeto sources.")
    if not isinstance(payload.get("history"), list):
        raise EloManifestError("El manifiesto Elo no contiene una lista history.")

    for gender, record in payload["sources"].items():
        if gender not in ELO_URLS or not isinstance(record, Mapping):
            raise EloManifestError(
                f"El manifiesto contiene una fuente Elo desconocida: {gender!r}."
            )
        if record.get("url") != ELO_URLS[gender]:
            raise EloManifestError(f"La URL registrada para {gender} no es la URL auditada.")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Escribe JSON UTF-8 de forma atómica en el directorio solicitado."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.part")
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_bytes_immutable(path: Path, content: bytes) -> None:
    """Crea un snapshot atómico y nunca reemplaza bytes ya versionados."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == content:
            return
        raise EloSnapshotIntegrityError(
            f"Ya existe un snapshot distinto en la ruta versionada {path}."
        )
    temporary_path = path.with_name(f"{path.name}.part")
    try:
        temporary_path.write_bytes(content)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _safe_relative_path(root: Path, path: Path) -> str:
    """Devuelve una ruta POSIX relativa y comprueba que no escape de ``root``."""

    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise EloManifestError(f"La ruta {path} queda fuera de {resolved_root}.")
    return resolved_path.relative_to(resolved_root).as_posix()


def _resolve_manifest_path(root: Path, relative_path: Any) -> Path:
    """Resuelve una ruta del manifiesto sin permitir travesía de directorios."""

    if not isinstance(relative_path, str) or not relative_path:
        raise EloManifestError("El manifiesto no contiene una ruta local válida.")
    resolved_root = root.resolve()
    resolved_path = (resolved_root / Path(relative_path)).resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise EloManifestError(
            f"Una ruta del manifiesto sale del directorio Elo: {relative_path!r}."
        )
    return resolved_path


def _response_headers(headers: Mapping[Any, Any]) -> dict[str, str]:
    """Convierte los headers de respuesta a un objeto JSON reproducible."""

    return {
        str(key): str(value)
        for key, value in sorted(
            headers.items(),
            key=lambda item: str(item[0]).casefold(),
        )
    }


def _header_value(headers: Mapping[str, str], name: str) -> str | None:
    """Busca un header sin distinguir mayúsculas y conserva su valor exacto."""

    expected = name.casefold()
    for key, value in headers.items():
        if key.casefold() == expected:
            return value
    return None


def _source_record(
    manifest: Mapping[str, Any],
    gender: Gender,
) -> Mapping[str, Any] | None:
    """Obtiene el registro de una fuente previamente validada."""

    sources = manifest["sources"]
    record = sources.get(gender)
    if record is None:
        return None
    if not isinstance(record, Mapping):
        raise EloManifestError(f"El registro del manifiesto para {gender} no es un objeto.")
    return record


def _throttle_deadline(
    manifest: Mapping[str, Any],
    now: datetime,
) -> tuple[datetime | None, str | None]:
    """Respeta únicamente el cortacircuitos temporal tras un fallo."""

    deadlines: list[tuple[datetime, str]] = []
    paused_until_value = manifest.get("access_paused_until_utc")
    if paused_until_value is not None:
        paused_until = _parse_utc_iso(
            paused_until_value,
            field_name="access_paused_until_utc",
        )
        if now < paused_until:
            deadlines.append((paused_until, "access_pause_after_failure"))

    if not deadlines:
        return None, None
    deadline, reason = max(deadlines, key=lambda item: item[0])
    return deadline, reason


def _conditional_request_headers(
    record: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Construye headers HTTP condicionales a partir del último ``200``."""

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml",
    }
    if record is None:
        return headers
    etag = record.get("etag")
    last_modified = record.get("last_modified")
    if isinstance(etag, str) and etag:
        headers["If-None-Match"] = etag
    if isinstance(last_modified, str) and last_modified:
        headers["If-Modified-Since"] = last_modified
    return headers


def _record_attempt(
    manifest: dict[str, Any],
    *,
    gender: Gender,
    attempted_at: datetime,
    outcome: str,
    status_code: int | None,
    headers: Mapping[str, str] | None = None,
    snapshot_path: str | None = None,
    metadata_path: str | None = None,
    content_sha256: str | None = None,
    retrieved_at: datetime | None = None,
    detail: str | None = None,
    activate_latest: bool = True,
    activate_validators: bool = True,
) -> None:
    """Actualiza fuente e historial después de un único GET ya consumido."""

    sources = manifest["sources"]
    previous = sources.get(gender, {})
    if not isinstance(previous, Mapping):
        raise EloManifestError(f"El registro previo para {gender} no es un objeto JSON.")
    record = dict(previous)
    record.update(
        {
            "url": ELO_URLS[gender],
            "last_attempted_at_utc": _iso_utc(attempted_at),
            "last_outcome": outcome,
            "last_status_code": status_code,
        }
    )
    if headers is not None:
        record["last_response_headers"] = dict(headers)
        if activate_validators:
            etag = _header_value(headers, "ETag")
            last_modified = _header_value(headers, "Last-Modified")
            if etag:
                record["etag"] = etag
            if last_modified:
                record["last_modified"] = last_modified
    if snapshot_path is not None:
        destination = "latest_snapshot" if activate_latest else "last_rejected_snapshot"
        record[destination] = snapshot_path
    if metadata_path is not None:
        destination = "latest_metadata" if activate_latest else "last_rejected_metadata"
        record[destination] = metadata_path
    if content_sha256 is not None:
        destination = "latest_sha256" if activate_latest else "last_rejected_sha256"
        record[destination] = content_sha256
    if retrieved_at is not None:
        destination = (
            "last_retrieved_at_utc" if activate_latest else "last_rejected_retrieved_at_utc"
        )
        record[destination] = _iso_utc(retrieved_at)
    if detail is None:
        record.pop("last_detail", None)
    else:
        record["last_detail"] = detail
    sources[gender] = record

    event: dict[str, Any] = {
        "event": "http_get",
        "gender": gender,
        "url": ELO_URLS[gender],
        "attempted_at_utc": _iso_utc(attempted_at),
        "outcome": outcome,
        "status_code": status_code,
    }
    if snapshot_path is not None:
        event["snapshot"] = snapshot_path
    if metadata_path is not None:
        event["metadata"] = metadata_path
    if content_sha256 is not None:
        event["sha256"] = content_sha256
    if detail is not None:
        event["detail"] = detail
    manifest["history"].append(event)
    manifest["updated_at_utc"] = _iso_utc(attempted_at)


def _pause_access_after_failure(
    manifest: dict[str, Any],
    *,
    attempted_at: datetime,
    reason: str,
) -> datetime:
    """Bloquea todas las fuentes 24 horas tras un fallo y devuelve el límite."""

    paused_until = attempted_at + FAILURE_BACKOFF
    manifest["access_paused_until_utc"] = _iso_utc(paused_until)
    manifest["access_pause_reason"] = reason
    return paused_until


def _snapshot_paths(
    raw_root: Path,
    gender: Gender,
    retrieved_at: datetime,
    content_sha256: str,
) -> tuple[Path, Path]:
    """Genera nombres versionados por tiempo UTC, género y hash."""

    timestamp = retrieved_at.strftime("%Y%m%dT%H%M%S%fZ")
    stem = f"{timestamp}_{content_sha256[:16]}"
    source_dir = raw_root / TOUR_LABELS[gender]
    snapshot_path = source_dir / f"{stem}.html"
    metadata_path = source_dir / f"{stem}.metadata.json"
    return snapshot_path, metadata_path


def _save_snapshot(
    *,
    raw_root: Path,
    gender: Gender,
    content: bytes,
    retrieved_at: datetime,
    response_headers: Mapping[str, str],
) -> tuple[Path, Path, str]:
    """Versiona bytes y metadata de procedencia para una respuesta ``200``."""

    content_sha256 = hashlib.sha256(content).hexdigest()
    snapshot_path, metadata_path = _snapshot_paths(
        raw_root,
        gender,
        retrieved_at,
        content_sha256,
    )
    snapshot_relative = _safe_relative_path(raw_root, snapshot_path)
    metadata_payload = {
        "schema_version": SNAPSHOT_METADATA_SCHEMA_VERSION,
        "gender": gender,
        "tour": TOUR_LABELS[gender].upper(),
        "retrieved_at_utc": _iso_utc(retrieved_at),
        "url": ELO_URLS[gender],
        "status_code": 200,
        "response_headers": dict(response_headers),
        "sha256": content_sha256,
        "content_length": len(content),
        "snapshot_file": snapshot_path.name,
        "snapshot_path": snapshot_relative,
    }
    _write_bytes_immutable(snapshot_path, content)
    _write_json_atomic(metadata_path, metadata_payload)
    return snapshot_path, metadata_path, content_sha256


def _latest_paths(
    manifest: Mapping[str, Any],
    raw_root: Path,
    gender: Gender,
) -> tuple[Path, Path] | None:
    """Resuelve el último par snapshot/metadata registrado para un género."""

    record = _source_record(manifest, gender)
    if record is None or record.get("latest_snapshot") is None:
        return None
    snapshot_path = _resolve_manifest_path(
        raw_root,
        record.get("latest_snapshot"),
    )
    metadata_path = _resolve_manifest_path(
        raw_root,
        record.get("latest_metadata"),
    )
    return snapshot_path, metadata_path


def download_tennis_abstract_elo(
    *,
    force: bool = False,
    raw_dir: Path = TENNIS_ABSTRACT_ELO_RAW_DIR,
    session: HttpSession | None = None,
    clock: Callable[[], datetime] | None = None,
) -> EloFetchReport:
    """Actualiza secuencialmente los dos snapshots Elo públicos.

    La función intenta ATP y después WTA, pero se detiene ante el primer fallo
    de red, HTTP, almacenamiento o esquema. No hay redirects ni reintentos
    automáticos. La frecuencia ordinaria la controla el operador; un fallo
    inequívoco sí abre un cortacircuitos de 24 horas.

    ``force`` vuelve a parsear localmente cualquier snapshot disponible antes
    de realizar las peticiones condicionales.

    Args:
        force: Si es verdadero, revalida localmente los snapshots disponibles.
        raw_dir: Directorio dedicado a snapshots, metadata y manifiesto Elo.
        session: Cliente HTTP inyectable; producción usa una sesión sin retries.
        clock: Reloj consciente de zona inyectable para tests deterministas.

    Returns:
        Un informe con los géneros alcanzados, GET consumidos y rutas validadas.

    Raises:
        EloManifestError: Si el estado local no es seguro o coherente.
        EloSchemaError: Si ``force`` detecta un cambio en una tabla guardada.
        ValueError: Si el reloj suministra una fecha sin zona horaria.
    """

    active_clock = clock or _utc_now
    execution_time = _as_utc(
        active_clock(),
        field_name="clock()",
    )
    raw_root = raw_dir.resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_root / "manifest.json"
    manifest = _load_manifest(manifest_path, execution_time)

    owned_session = session is None
    active_session = session or build_http_session()
    results: list[EloFetchResult] = []
    get_count = 0
    try:
        for gender in ELO_SOURCE_ORDER:
            url = ELO_URLS[gender]
            _validate_public_url(gender, url)
            retry_after, throttle_reason = _throttle_deadline(
                manifest,
                execution_time,
            )
            if retry_after is not None:
                results.append(
                    EloFetchResult(
                        gender=gender,
                        url=url,
                        outcome="throttled",
                        status_code=None,
                        attempted_at_utc=None,
                        retry_after_utc=retry_after,
                        detail=throttle_reason,
                    )
                )
                continue

            previous_record = _source_record(manifest, gender)
            request_headers = _conditional_request_headers(previous_record)
            get_count += 1
            try:
                response = active_session.get(
                    url,
                    headers=request_headers,
                    timeout=REQUEST_TIMEOUT,
                    allow_redirects=False,
                )
            except requests.RequestException as exc:
                detail = f"{type(exc).__name__}: {exc}"
                _record_attempt(
                    manifest,
                    gender=gender,
                    attempted_at=execution_time,
                    outcome="network_error",
                    status_code=None,
                    detail=detail,
                )
                retry_after = _pause_access_after_failure(
                    manifest,
                    attempted_at=execution_time,
                    reason="network_error",
                )
                _write_json_atomic(manifest_path, manifest)
                results.append(
                    EloFetchResult(
                        gender=gender,
                        url=url,
                        outcome="network_error",
                        status_code=None,
                        attempted_at_utc=execution_time,
                        retry_after_utc=retry_after,
                        detail=detail,
                    )
                )
                LOGGER.warning("GET Elo fallido para %s: %s", gender, detail)
                break

            status_code = int(response.status_code)
            headers = _response_headers(response.headers)
            if status_code == 200:
                try:
                    content = bytes(response.content)
                    snapshot_path, metadata_path, content_sha256 = _save_snapshot(
                        raw_root=raw_root,
                        gender=gender,
                        content=content,
                        retrieved_at=execution_time,
                        response_headers=headers,
                    )
                except (OSError, EloSnapshotIntegrityError) as exc:
                    detail = f"{type(exc).__name__}: {exc}"
                    _record_attempt(
                        manifest,
                        gender=gender,
                        attempted_at=execution_time,
                        outcome="storage_error",
                        status_code=status_code,
                        headers=headers,
                        detail=detail,
                        activate_validators=False,
                    )
                    retry_after = _pause_access_after_failure(
                        manifest,
                        attempted_at=execution_time,
                        reason="storage_error",
                    )
                    _write_json_atomic(manifest_path, manifest)
                    results.append(
                        EloFetchResult(
                            gender=gender,
                            url=url,
                            outcome="storage_error",
                            status_code=status_code,
                            attempted_at_utc=execution_time,
                            retry_after_utc=retry_after,
                            detail=detail,
                        )
                    )
                    break

                snapshot_relative = _safe_relative_path(
                    raw_root,
                    snapshot_path,
                )
                metadata_relative = _safe_relative_path(
                    raw_root,
                    metadata_path,
                )
                try:
                    load_elo_snapshot(
                        snapshot_path,
                        metadata_path=metadata_path,
                    )
                except EloSchemaError as exc:
                    detail = f"{type(exc).__name__}: {exc}"
                    _record_attempt(
                        manifest,
                        gender=gender,
                        attempted_at=execution_time,
                        outcome="schema_error",
                        status_code=status_code,
                        headers=headers,
                        snapshot_path=snapshot_relative,
                        metadata_path=metadata_relative,
                        content_sha256=content_sha256,
                        retrieved_at=execution_time,
                        detail=detail,
                        activate_latest=False,
                        activate_validators=False,
                    )
                    retry_after = _pause_access_after_failure(
                        manifest,
                        attempted_at=execution_time,
                        reason="schema_error",
                    )
                    _write_json_atomic(manifest_path, manifest)
                    results.append(
                        EloFetchResult(
                            gender=gender,
                            url=url,
                            outcome="schema_error",
                            status_code=status_code,
                            attempted_at_utc=execution_time,
                            retry_after_utc=retry_after,
                            snapshot_path=snapshot_path,
                            metadata_path=metadata_path,
                            detail=detail,
                        )
                    )
                    LOGGER.error(
                        "Snapshot Elo conservado pero rechazado para %s: %s",
                        gender,
                        detail,
                    )
                    break
                except (EloManifestError, EloSnapshotIntegrityError) as exc:
                    detail = f"{type(exc).__name__}: {exc}"
                    _record_attempt(
                        manifest,
                        gender=gender,
                        attempted_at=execution_time,
                        outcome="storage_error",
                        status_code=status_code,
                        headers=headers,
                        snapshot_path=snapshot_relative,
                        metadata_path=metadata_relative,
                        content_sha256=content_sha256,
                        retrieved_at=execution_time,
                        detail=detail,
                        activate_latest=False,
                        activate_validators=False,
                    )
                    retry_after = _pause_access_after_failure(
                        manifest,
                        attempted_at=execution_time,
                        reason="storage_error",
                    )
                    _write_json_atomic(manifest_path, manifest)
                    results.append(
                        EloFetchResult(
                            gender=gender,
                            url=url,
                            outcome="storage_error",
                            status_code=status_code,
                            attempted_at_utc=execution_time,
                            retry_after_utc=retry_after,
                            snapshot_path=snapshot_path,
                            metadata_path=metadata_path,
                            detail=detail,
                        )
                    )
                    break

                _record_attempt(
                    manifest,
                    gender=gender,
                    attempted_at=execution_time,
                    outcome="downloaded",
                    status_code=status_code,
                    headers=headers,
                    snapshot_path=snapshot_relative,
                    metadata_path=metadata_relative,
                    content_sha256=content_sha256,
                    retrieved_at=execution_time,
                )
                _write_json_atomic(manifest_path, manifest)
                results.append(
                    EloFetchResult(
                        gender=gender,
                        url=url,
                        outcome="downloaded",
                        status_code=status_code,
                        attempted_at_utc=execution_time,
                        retry_after_utc=(execution_time + FAILURE_BACKOFF),
                        snapshot_path=snapshot_path,
                        metadata_path=metadata_path,
                    )
                )
                continue

            if status_code == 304:
                latest_paths = _latest_paths(manifest, raw_root, gender)
                if latest_paths is None or not all(path.exists() for path in latest_paths):
                    detail = (
                        "El servidor respondió 304, pero no existe un snapshot "
                        "local completo al que aplicar la respuesta."
                    )
                    outcome = "invalid_not_modified"
                    snapshot_path = None
                    metadata_path = None
                else:
                    detail = None
                    outcome = "not_modified"
                    snapshot_path, metadata_path = latest_paths
                _record_attempt(
                    manifest,
                    gender=gender,
                    attempted_at=execution_time,
                    outcome=outcome,
                    status_code=status_code,
                    headers=headers,
                    detail=detail,
                    activate_validators=(outcome == "not_modified"),
                )
                if outcome == "invalid_not_modified":
                    retry_after = _pause_access_after_failure(
                        manifest,
                        attempted_at=execution_time,
                        reason=outcome,
                    )
                else:
                    retry_after = execution_time + FAILURE_BACKOFF
                _write_json_atomic(manifest_path, manifest)
                results.append(
                    EloFetchResult(
                        gender=gender,
                        url=url,
                        outcome=outcome,
                        status_code=status_code,
                        attempted_at_utc=execution_time,
                        retry_after_utc=retry_after,
                        snapshot_path=snapshot_path,
                        metadata_path=metadata_path,
                        detail=detail,
                    )
                )
                if outcome == "invalid_not_modified":
                    break
                continue

            detail = f"Respuesta HTTP {status_code}; no se realizó ningún retry."
            _record_attempt(
                manifest,
                gender=gender,
                attempted_at=execution_time,
                outcome="http_error",
                status_code=status_code,
                headers=headers,
                detail=detail,
                activate_validators=False,
            )
            retry_after = _pause_access_after_failure(
                manifest,
                attempted_at=execution_time,
                reason=f"http_{status_code}",
            )
            _write_json_atomic(manifest_path, manifest)
            results.append(
                EloFetchResult(
                    gender=gender,
                    url=url,
                    outcome="http_error",
                    status_code=status_code,
                    attempted_at_utc=execution_time,
                    retry_after_utc=retry_after,
                    detail=detail,
                )
            )
            LOGGER.warning("Tennis Abstract devolvió %d para %s.", status_code, gender)
            break
    finally:
        if owned_session:
            active_session.close()

    locally_validated: list[Path] = []
    if force:
        for gender in ELO_SOURCE_ORDER:
            latest_paths = _latest_paths(manifest, raw_root, gender)
            if latest_paths is None:
                continue
            snapshot_path, metadata_path = latest_paths
            load_elo_snapshot(
                snapshot_path,
                metadata_path=metadata_path,
            )
            locally_validated.append(snapshot_path)

    return EloFetchReport(
        results=tuple(results),
        manifest_path=manifest_path,
        get_count=get_count,
        locally_validated=tuple(locally_validated),
    )


def _decode_html(html: str | bytes) -> str:
    """Convierte bytes UTF-8 a texto y rechaza codificaciones no inspeccionadas."""

    if isinstance(html, str):
        return html
    if isinstance(html, bytes):
        try:
            return html.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise EloSchemaError("El snapshot Elo ya no se puede decodificar como UTF-8.") from exc
    raise TypeError("html debe ser str o bytes.")


def _extract_table(
    html: str | bytes,
    gender: Gender,
) -> tuple[tuple[tuple[_HtmlCell, ...], ...], str]:
    """Extrae la tabla única y el texto documental del HTML real."""

    parser = _EloTableExtractor()
    try:
        parser.feed(_decode_html(html))
        parser.close()
        parser.validate_closed()
    except EloSchemaError:
        raise
    except Exception as exc:
        raise EloSchemaError("No se pudo recorrer el HTML Elo.") from exc

    if len(parser.tables) != 1:
        raise EloSchemaError(
            "Se esperaba exactamente una tabla table#reportable; "
            f"se encontraron {len(parser.tables)}."
        )
    table = parser.tables[0]
    if len(table) < 2:
        raise EloSchemaError("La tabla Elo no contiene cabecera y filas de datos.")

    header = table[0]
    if any(cell.tag != "th" for cell in header):
        raise EloSchemaError("La primera fila Elo no está formada solo por th.")
    observed_headers = tuple(cell.text for cell in header)
    expected_headers = EXPECTED_HEADERS[gender]
    if observed_headers != expected_headers:
        raise EloSchemaError(
            "La cabecera Elo cambió. "
            f"Esperada={expected_headers!r}; observada={observed_headers!r}."
        )
    document_text = _normalize_text(" ".join(parser.document_text))
    return table, document_text


def _rating_date_from_text(document_text: str) -> datetime:
    """Extrae la única fecha ``Last update`` observada en la página."""

    matches = _RATING_DATE_PATTERN.findall(document_text)
    if len(matches) != 1:
        raise EloSchemaError("La página Elo no contiene exactamente una fecha 'Last update'.")
    try:
        return datetime.strptime(matches[0], "%Y-%m-%d")
    except ValueError as exc:
        raise EloSchemaError(f"La fecha Last update no es válida: {matches[0]!r}.") from exc


def _player_url(
    cell: _HtmlCell,
    *,
    gender: Gender,
    source_url: str,
) -> str:
    """Valida y normaliza el único enlace público de una fila de jugador."""

    if len(cell.hrefs) != 1:
        raise EloSchemaError("Cada jugador Elo debe tener exactamente un enlace en su celda.")
    resolved = urljoin(source_url, cell.hrefs[0])
    parsed = urlparse(resolved)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in {"tennisabstract.com", "www.tennisabstract.com"}
        or parsed.path != EXPECTED_PLAYER_PATHS[gender]
        or not parsed.query
    ):
        raise EloSchemaError(
            f"El enlace de jugador no coincide con el formato inspeccionado: {resolved!r}."
        )
    return resolved


def _raw_rows(
    table: tuple[tuple[_HtmlCell, ...], ...],
    *,
    gender: Gender,
    source_url: str,
) -> list[dict[str, str]]:
    """Transforma posiciones HTML validadas a nombres internos explícitos."""

    rows: list[dict[str, str]] = []
    expected_width = len(EXPECTED_HEADERS[gender])
    for row_number, cells in enumerate(table[1:], start=1):
        if len(cells) != expected_width:
            raise EloSchemaError(
                f"La fila Elo {row_number} tiene {len(cells)} celdas; "
                f"se esperaban {expected_width}."
            )
        if any(cell.tag != "td" for cell in cells):
            raise EloSchemaError(f"La fila Elo {row_number} no está formada solo por td.")
        if any(cells[index].text for index in (4, 11, 14)):
            raise EloSchemaError(f"La fila Elo {row_number} ya no conserva los separadores vacíos.")
        rows.append(
            {
                "elo_rank": cells[0].text,
                "player_name": cells[1].text,
                "player_url": _player_url(
                    cells[1],
                    gender=gender,
                    source_url=source_url,
                ),
                "age": cells[2].text,
                "elo": cells[3].text,
                "hard_elo_rank": cells[5].text,
                "hard_elo": cells[6].text,
                "clay_elo_rank": cells[7].text,
                "clay_elo": cells[8].text,
                "grass_elo_rank": cells[9].text,
                "grass_elo": cells[10].text,
                "peak_elo": cells[12].text,
                "peak_month": cells[13].text,
                "official_rank": cells[15].text,
                "log_diff": cells[16].text,
            }
        )
    if not rows:
        raise EloSchemaError("La tabla Elo no contiene jugadores.")
    return rows


def _numeric_series(
    values: pd.Series,
    *,
    column: str,
    integer: bool,
) -> pd.Series:
    """Convierte una columna numérica y distingue ausencias de valores inválidos."""

    normalized = values.replace("", pd.NA)
    converted = pd.to_numeric(normalized, errors="coerce")
    invalid = normalized.notna() & converted.isna()
    if invalid.any():
        examples = normalized.loc[invalid].astype(str).head(3).tolist()
        raise EloSchemaError(f"La columna {column!r} contiene valores no numéricos: {examples}.")
    if integer:
        non_null = converted.dropna()
        if not non_null.empty and (non_null % 1 != 0).any():
            raise EloSchemaError(f"La columna entera {column!r} contiene valores fraccionarios.")
        return converted.astype("Int64")
    return converted.astype("Float64")


def _peak_month_series(values: pd.Series) -> pd.Series:
    """Convierte ``YYYY-MM`` a fecha del primer día sin ocultar errores."""

    normalized = values.replace("", pd.NA)
    parsed = pd.to_datetime(normalized, format="%Y-%m", errors="coerce")
    invalid = normalized.notna() & parsed.isna()
    if invalid.any():
        examples = normalized.loc[invalid].astype(str).head(3).tolist()
        raise EloSchemaError(f"Peak Month contiene valores fuera de YYYY-MM: {examples}.")
    return parsed


def parse_elo_html(
    html: str | bytes,
    *,
    gender: Gender,
    source_url: str | None = None,
    retrieved_at_utc: datetime | None = None,
) -> pd.DataFrame:
    """Parsea una tabla Elo ATP o WTA con esquema y tipos estrictos.

    Args:
        html: Documento HTML completo como texto o bytes UTF-8.
        gender: Universo independiente, exactamente ``"M"`` o ``"F"``.
        source_url: URL de procedencia; debe ser la pública auditada del género.
        retrieved_at_utc: Momento consciente de zona en que se obtuvo el HTML.

    Returns:
        Un DataFrame con rankings nullable ``Int64``, ratings ``Float64``,
        fechas reales, procedencia y género explícito.

    Raises:
        EloSchemaError: Si tabla, cabecera, filas, enlaces o valores cambiaron.
        ValueError: Si género o fecha de recuperación no son válidos.
    """

    validated_gender = _validate_gender(gender)
    validated_url = source_url or ELO_URLS[validated_gender]
    _validate_public_url(validated_gender, validated_url)
    table, document_text = _extract_table(html, validated_gender)
    rating_date = _rating_date_from_text(document_text)
    raw_rows = _raw_rows(
        table,
        gender=validated_gender,
        source_url=validated_url,
    )
    frame = pd.DataFrame(raw_rows, dtype="string")

    for column in _INTEGER_COLUMNS:
        frame[column] = _numeric_series(
            frame[column],
            column=column,
            integer=True,
        )
    for column in _FLOAT_COLUMNS:
        frame[column] = _numeric_series(
            frame[column],
            column=column,
            integer=False,
        )
    frame["peak_month"] = _peak_month_series(frame["peak_month"])

    for column in _REQUIRED_VALUE_COLUMNS:
        if frame[column].isna().any() or (
            frame[column].dtype == "string" and frame[column].eq("").any()
        ):
            raise EloSchemaError(f"La columna Elo obligatoria {column!r} contiene ausencias.")

    row_count = len(frame)
    frame.insert(
        0,
        "rating_date",
        pd.Series([rating_date] * row_count, dtype="datetime64[ns]"),
    )
    frame.insert(
        0,
        "gender",
        pd.Series([validated_gender] * row_count, dtype="string"),
    )
    frame["source_url"] = pd.Series(
        [validated_url] * row_count,
        dtype="string",
    )
    if retrieved_at_utc is None:
        retrieved_series = pd.Series(
            [pd.NaT] * row_count,
            dtype="datetime64[ns, UTC]",
        )
    else:
        retrieved = _as_utc(
            retrieved_at_utc,
            field_name="retrieved_at_utc",
        )
        retrieved_series = pd.Series(
            [retrieved] * row_count,
            dtype="datetime64[ns, UTC]",
        )
    frame["retrieved_at_utc"] = retrieved_series
    return frame.loc[:, OUTPUT_COLUMNS]


def _read_snapshot_metadata(metadata_path: Path) -> Mapping[str, Any]:
    """Lee y valida los campos obligatorios de una metadata de snapshot."""

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EloManifestError(f"No se puede leer la metadata Elo: {metadata_path}.") from exc
    if not isinstance(payload, Mapping):
        raise EloManifestError("La metadata del snapshot no es un objeto JSON.")
    if payload.get("schema_version") != SNAPSHOT_METADATA_SCHEMA_VERSION:
        raise EloManifestError("La versión de metadata del snapshot Elo no es compatible.")
    required_fields = {
        "gender",
        "retrieved_at_utc",
        "url",
        "sha256",
        "snapshot_file",
        "response_headers",
    }
    missing = sorted(required_fields.difference(payload))
    if missing:
        raise EloManifestError(f"La metadata Elo carece de campos obligatorios: {missing}.")
    if not isinstance(payload.get("response_headers"), Mapping):
        raise EloManifestError("response_headers no es un objeto en la metadata.")
    return payload


def load_elo_snapshot(
    snapshot_path: Path,
    *,
    metadata_path: Path | None = None,
) -> pd.DataFrame:
    """Verifica procedencia e integridad de un snapshot antes de parsearlo.

    Args:
        snapshot_path: Archivo HTML raw versionado.
        metadata_path: Sidecar JSON; por defecto usa el mismo stem ``.metadata``.

    Returns:
        El DataFrame tipado producido por :func:`parse_elo_html`.

    Raises:
        EloManifestError: Si la metadata está incompleta o es incoherente.
        EloSnapshotIntegrityError: Si nombre, tamaño o SHA-256 no coinciden.
        EloSchemaError: Si el HTML guardado no conserva el esquema auditado.
    """

    resolved_snapshot = snapshot_path.resolve()
    resolved_metadata = (
        metadata_path.resolve()
        if metadata_path is not None
        else snapshot_path.with_suffix(".metadata.json").resolve()
    )
    metadata = _read_snapshot_metadata(resolved_metadata)
    if metadata.get("snapshot_file") != resolved_snapshot.name:
        raise EloSnapshotIntegrityError("La metadata Elo apunta a un nombre de snapshot distinto.")
    try:
        content = resolved_snapshot.read_bytes()
    except OSError as exc:
        raise EloSnapshotIntegrityError(
            f"No se puede leer el snapshot Elo: {resolved_snapshot}."
        ) from exc
    observed_sha256 = hashlib.sha256(content).hexdigest()
    expected_sha256 = metadata.get("sha256")
    if not isinstance(expected_sha256, str) or (observed_sha256 != expected_sha256):
        raise EloSnapshotIntegrityError(f"El SHA-256 no coincide para {resolved_snapshot}.")
    content_length = metadata.get("content_length")
    if not isinstance(content_length, int) or content_length != len(content):
        raise EloSnapshotIntegrityError(f"El tamaño no coincide para {resolved_snapshot}.")

    raw_gender = metadata.get("gender")
    if not isinstance(raw_gender, str):
        raise EloManifestError("La metadata Elo no contiene gender textual.")
    gender = _validate_gender(raw_gender)
    raw_url = metadata.get("url")
    if not isinstance(raw_url, str):
        raise EloManifestError("La metadata Elo no contiene una URL textual.")
    retrieved_at = _parse_utc_iso(
        metadata.get("retrieved_at_utc"),
        field_name="metadata.retrieved_at_utc",
    )
    return parse_elo_html(
        content,
        gender=gender,
        source_url=raw_url,
        retrieved_at_utc=retrieved_at,
    )


def load_latest_elo(
    gender: Gender,
    *,
    raw_dir: Path = TENNIS_ABSTRACT_ELO_RAW_DIR,
) -> pd.DataFrame:
    """Reparsea localmente el último snapshot íntegro de un género.

    Args:
        gender: Universo independiente, exactamente ``"M"`` o ``"F"``.
        raw_dir: Directorio que contiene ``manifest.json`` y los snapshots.

    Returns:
        El último ranking Elo local como DataFrame tipado.

    Raises:
        EloManifestError: Si no hay snapshot registrado o las rutas son inválidas.
        EloSnapshotIntegrityError: Si el contenido local fue alterado.
        EloSchemaError: Si el snapshot no coincide con el esquema auditado.
    """

    validated_gender = _validate_gender(gender)
    raw_root = raw_dir.resolve()
    manifest_path = raw_root / "manifest.json"
    manifest = _load_manifest(manifest_path, _utc_now())
    latest_paths = _latest_paths(manifest, raw_root, validated_gender)
    if latest_paths is None:
        raise EloManifestError(f"No existe un snapshot Elo registrado para {validated_gender}.")
    snapshot_path, metadata_path = latest_paths
    return load_elo_snapshot(
        snapshot_path,
        metadata_path=metadata_path,
    )
