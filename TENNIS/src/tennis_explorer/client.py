"""Cliente de bajo ruido y snapshots íntegros para Tennis Explorer.

La captura matinal realiza una petición de la página ``type=all`` solo cuando
la fecha no está cacheada. Los refrescos de resultados usan la misma URL,
pero se publican como observaciones append-only y nunca sustituyen la captura
matinal. Cada llamada de red emite un único GET sin reintentos; el cliente no
impone un límite artificial al número de ejecuciones que decida el orquestador.

La ruta diaria usa la excepción de ``robots.txt`` comunicada por el operador
al responsable del proyecto; la excepción no se extiende a ninguna otra URL.
Scrapling actúa únicamente como transporte HTTP estático, sin sigilo, proxies,
rotación, navegador ni resolución de desafíos. Los HTML se validan antes de
publicarse y un bloqueo abre un cortacircuitos persistente que impide nuevas
peticiones hasta una revisión manual.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Final, Protocol
from urllib.parse import parse_qs, urlsplit

import pandas as pd

from src.config import TENNIS_EXPLORER_RAW_DIR

from .parser import parse_daily_matches_html
from .transport import ScraplingHttpSession, ScraplingTransportError
from .types import (
    TennisExplorerBlockedError,
    TennisExplorerCacheError,
    TennisExplorerHttpError,
)


TENNIS_EXPLORER_BASE_URL: Final[str] = "https://www.tennisexplorer.com"
USER_AGENT: Final[str] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT: Final[tuple[float, float]] = (10.0, 30.0)
MIN_REQUEST_DELAY_SECONDS: Final[float] = 1.0
CACHE_SCHEMA_VERSION: Final[int] = 1

_DAILY_ARTIFACT: Final[str] = "tennis_explorer_daily_matches"
_RESULTS_ARTIFACT: Final[str] = "tennis_explorer_results_snapshot"
_ACCESS_BASIS: Final[str] = "user_reported_operator_authorization"
_ROBOTS_POLICY: Final[str] = "authorized_matches_endpoint_exception"
_TRANSPORT_NAME: Final[str] = "scrapling_fetcher_session"
_CIRCUIT_BREAKER_SCHEMA_VERSION: Final[int] = 1
_CHALLENGE_BODY_MARKERS: Final[tuple[bytes, ...]] = (
    b"/cdn-cgi/challenge-platform/",
    b"cf-chl-",
    b"<title>just a moment...</title>",
    b"attention required! | cloudflare",
    b"checking your browser before accessing",
    b"enable javascript and cookies to continue",
    b"challenges.cloudflare.com/turnstile/",
    b'class="cf-turnstile"',
    b"class='cf-turnstile'",
    b"cloudflare ray id",
)


class HttpSession(Protocol):
    """Define la parte mínima de una sesión HTTP que usa este módulo."""

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: tuple[float, float],
        allow_redirects: bool,
    ) -> Any:
        """Realiza un único ``GET`` sin redirecciones automáticas."""

    def close(self) -> None:
        """Libera los recursos de una sesión creada por el cliente."""


@dataclass(frozen=True)
class _CachePaths:
    """Agrupa las rutas de contenido y metadata de un artefacto cacheado."""

    content: Path
    metadata: Path


@dataclass(frozen=True, slots=True)
class TennisExplorerResultSnapshot:
    """Representa un refresco validado y append-only de una fecha.

    ``matches`` contiene el DataFrame tipado. Las rutas y la metadata de nivel
    snapshot conservan la procedencia incluso si la jornada no tiene filas.
    """

    match_date: date
    retrieved_at_utc: datetime
    source_url: str
    snapshot_sha256: str
    matches: pd.DataFrame
    html_path: Path
    metadata_path: Path


def build_daily_url(match_date: date) -> str:
    """Construye la URL canónica ``type=all`` para una fecha concreta."""

    if not isinstance(match_date, date) or isinstance(match_date, datetime):
        raise TypeError("match_date debe ser datetime.date, no datetime.")
    return (
        f"{TENNIS_EXPLORER_BASE_URL}/matches/"
        f"?day={match_date.day:02d}"
        f"&month={match_date.month:02d}"
        f"&type=all"
        f"&year={match_date.year:04d}"
    )


def _validate_authorized_matches_url(
    source_url: str,
    match_date: date,
) -> None:
    """Limita la excepción comunicada a la URL diaria canónica de partidos."""

    try:
        parsed = urlsplit(source_url)
        query = parse_qs(
            parsed.query,
            keep_blank_values=True,
            strict_parsing=True,
        )
        port = parsed.port
    except ValueError as exc:
        raise TennisExplorerHttpError(
            "La URL diaria no es válida; no se aplicará la excepción de acceso."
        ) from exc

    expected_query = {
        "day": [f"{match_date.day:02d}"],
        "month": [f"{match_date.month:02d}"],
        "type": ["all"],
        "year": [f"{match_date.year:04d}"],
    }
    is_authorized = (
        parsed.scheme == "https"
        and parsed.netloc == "www.tennisexplorer.com"
        and parsed.username is None
        and parsed.password is None
        and port is None
        and parsed.path == "/matches/"
        and not parsed.fragment
        and query == expected_query
    )
    if not is_authorized:
        raise TennisExplorerHttpError(
            "La excepción de acceso solo cubre la URL HTTPS canónica "
            "de /matches/ con day, month, year y type=all."
        )


def get_daily_matches(
    match_date: date | None = None,
    *,
    raw_dir: Path = TENNIS_EXPLORER_RAW_DIR,
    session: HttpSession | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], datetime] | None = None,
) -> pd.DataFrame:
    """Devuelve los partidos de una fecha usando caché o una descarga única.

    Args:
        match_date: Fecha solicitada. Si es ``None``, usa el día local del
            reloj inyectado o del sistema.
        raw_dir: Raíz de datos crudos reservada a Tennis Explorer.
        session: Sesión HTTP opcional para pruebas o reutilización controlada.
        sleeper: Función que aplica la pausa previa a la petición permitida.
        clock: Reloj que debe devolver un ``datetime`` con zona horaria.

    Returns:
        DataFrame tipado producido por el parser, con tres columnas de
        procedencia añadidas.

    Raises:
        TennisExplorerBlockedError: Si el servidor señala bloqueo o WAF.
        TennisExplorerCacheError: Si existe una caché parcial o corrupta.
        TennisExplorerHttpError: Si el transporte o el contenido fallan.
    """

    observed_at = _read_clock(clock)
    requested_date = match_date if match_date is not None else observed_at.date()
    if not isinstance(requested_date, date) or isinstance(
        requested_date,
        datetime,
    ):
        raise TypeError("match_date debe ser datetime.date o None.")

    raw_root = Path(raw_dir)
    source_url = build_daily_url(requested_date)
    daily_paths = _daily_cache_paths(raw_root, requested_date)
    cached = _load_cache_pair(
        daily_paths,
        artifact=_DAILY_ARTIFACT,
        date_field="match_date",
        expected_date=requested_date.isoformat(),
        expected_url=source_url,
    )
    if cached is not None:
        html, metadata = cached
        _raise_if_cached_challenge(html, source_url)
        frame = parse_daily_matches_html(html, requested_date)
        return _append_audit_columns(frame, metadata)

    _validate_authorized_matches_url(source_url, requested_date)
    _raise_if_circuit_open(raw_root)
    with _network_access_lock(raw_root):
        # Evita una carrera si otro proceso abrió el cortacircuitos mientras
        # esta ejecución esperaba el lock exclusivo.
        _raise_if_circuit_open(raw_root)
        owned_session = session is None
        http = session if session is not None else _build_http_session()
        try:
            sleeper(MIN_REQUEST_DELAY_SECONDS)
            response = _single_get(http, source_url)
            try:
                html, content_type = _validated_response_content(
                    response,
                    source_url,
                    allowed_content_types=("text/html",),
                )
            except TennisExplorerBlockedError:
                _open_circuit_breaker(
                    raw_root,
                    source_url=source_url,
                    opened_at_utc=_read_clock(clock).astimezone(UTC),
                    response=response,
                )
                raise

            # La estructura se valida antes de que el snapshot sea visible.
            frame = parse_daily_matches_html(html, requested_date)
            acquired_at_utc = _read_clock(clock).astimezone(UTC)
            metadata = _build_metadata(
                artifact=_DAILY_ARTIFACT,
                date_field="match_date",
                date_value=requested_date.isoformat(),
                source_url=source_url,
                retrieved_at_utc=acquired_at_utc,
                status_code=200,
                content_type=content_type,
                content=html,
            )
            metadata.update(
                {
                    "access_basis": _ACCESS_BASIS,
                    "robots_policy": _ROBOTS_POLICY,
                    "transport": _TRANSPORT_NAME,
                }
            )
            _write_cache_pair(daily_paths, html, metadata)
            return _append_audit_columns(frame, metadata)
        finally:
            if owned_session:
                http.close()


def refresh_daily_results(
    match_date: date,
    *,
    raw_dir: Path = TENNIS_EXPLORER_RAW_DIR,
    session: HttpSession | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], datetime] | None = None,
) -> TennisExplorerResultSnapshot:
    """Descarga y publica una observación append-only de resultados.

    Cada llamada realiza exactamente una tentativa a la misma URL canónica
    usada por la cartelera, tras el delay y el cortacircuitos compartidos. No
    consulta ni sustituye ``daily/YYYY/YYYY-MM-DD`` y tampoco impone un número
    máximo de ejecuciones: cada adquisición validada recibe una ruta UTC nueva.

    Args:
        match_date: Fecha civil cuyos estados se desean observar.
        raw_dir: Raíz local reservada a Tennis Explorer.
        session: Transporte inyectable; si falta se crea el adaptador estático.
        sleeper: Función inyectable que aplica el delay previo al GET.
        clock: Reloj consciente de zona para timestamp y pruebas.

    Returns:
        Snapshot estructurado con DataFrame, procedencia y rutas append-only.

    Raises:
        TennisExplorerBlockedError: Ante una señal inequívoca de WAF.
        TennisExplorerHttpError: Ante un único intento HTTP fallido.
        TennisExplorerSchemaError: Si el HTML no respeta el contrato real.
        TennisExplorerCacheError: Si la ruta append-only ya existe o falla.
    """

    if not isinstance(match_date, date) or isinstance(match_date, datetime):
        raise TypeError("match_date debe ser datetime.date, no datetime.")
    raw_root = Path(raw_dir)
    source_url = build_daily_url(match_date)
    _validate_authorized_matches_url(source_url, match_date)
    _raise_if_circuit_open(raw_root)
    with _network_access_lock(raw_root):
        _raise_if_circuit_open(raw_root)
        owned_session = session is None
        http = session if session is not None else _build_http_session()
        try:
            sleeper(MIN_REQUEST_DELAY_SECONDS)
            response = _single_get(http, source_url)
            try:
                html, content_type = _validated_response_content(
                    response,
                    source_url,
                    allowed_content_types=("text/html",),
                )
            except TennisExplorerBlockedError:
                _open_circuit_breaker(
                    raw_root,
                    source_url=source_url,
                    opened_at_utc=_read_clock(clock).astimezone(UTC),
                    response=response,
                )
                raise

            frame = parse_daily_matches_html(html, match_date)
            acquired_at_utc = _read_clock(clock).astimezone(UTC)
            metadata = _build_metadata(
                artifact=_RESULTS_ARTIFACT,
                date_field="match_date",
                date_value=match_date.isoformat(),
                source_url=source_url,
                retrieved_at_utc=acquired_at_utc,
                status_code=200,
                content_type=content_type,
                content=html,
            )
            metadata.update(
                {
                    "access_basis": _ACCESS_BASIS,
                    "robots_policy": _ROBOTS_POLICY,
                    "transport": _TRANSPORT_NAME,
                    "snapshot_purpose": "result_refresh",
                }
            )
            paths = _result_snapshot_paths(
                raw_root,
                match_date,
                acquired_at_utc,
                str(metadata["sha256"]),
            )
            _write_cache_pair(paths, html, metadata)
            return TennisExplorerResultSnapshot(
                match_date=match_date,
                retrieved_at_utc=acquired_at_utc,
                source_url=source_url,
                snapshot_sha256=str(metadata["sha256"]),
                matches=_append_audit_columns(frame, metadata),
                html_path=paths.content,
                metadata_path=paths.metadata,
            )
        finally:
            if owned_session:
                http.close()


def _read_clock(clock: Callable[[], datetime] | None) -> datetime:
    """Obtiene un instante con zona horaria y conserva su fecha local."""

    observed_at = (
        clock()
        if clock is not None
        else datetime.now().astimezone()
    )
    if not isinstance(observed_at, datetime):
        raise TypeError("clock debe devolver datetime.datetime.")
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("clock debe devolver un datetime con zona horaria.")
    return observed_at


def _daily_cache_paths(raw_dir: Path, match_date: date) -> _CachePaths:
    """Calcula las rutas estables de HTML y metadata para una fecha."""

    directory = raw_dir / "daily" / f"{match_date.year:04d}"
    stem = match_date.isoformat()
    return _CachePaths(
        content=directory / f"{stem}.html",
        metadata=directory / f"{stem}.metadata.json",
    )


def _result_snapshot_paths(
    raw_dir: Path,
    match_date: date,
    retrieved_at_utc: datetime,
    sha256: str,
) -> _CachePaths:
    """Calcula una ruta corta y append-only para una observación de resultados."""

    retrieved = retrieved_at_utc.astimezone(UTC)
    timestamp = retrieved.strftime("%Y%m%dT%H%M%S%fZ")
    stem = f"{timestamp}_{sha256[:12]}"
    directory = (
        raw_dir
        / "results"
        / f"{match_date.year:04d}"
        / match_date.isoformat()
    )
    return _CachePaths(
        content=directory / f"{stem}.html",
        metadata=directory / f"{stem}.metadata.json",
    )


def _circuit_breaker_path(raw_dir: Path) -> Path:
    """Devuelve la ruta estable del cortacircuitos de acceso a la web."""

    return raw_dir / "policy" / "circuit_breaker.json"


def _raise_if_circuit_open(raw_dir: Path) -> None:
    """Detiene todo tráfico nuevo mientras exista un bloqueo registrado."""

    path = _circuit_breaker_path(raw_dir)
    if not path.exists():
        return
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TennisExplorerCacheError(
            f"El cortacircuitos existe pero no puede validarse: {path}."
        ) from exc
    if not isinstance(value, dict):
        raise TennisExplorerCacheError(
            f"El cortacircuitos no contiene un objeto JSON válido: {path}."
        )

    expected = {
        "schema_version": _CIRCUIT_BREAKER_SCHEMA_VERSION,
        "host": "www.tennisexplorer.com",
        "access_basis": _ACCESS_BASIS,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise TennisExplorerCacheError(
            f"El cortacircuitos contiene metadata inesperada: {path}."
        )
    opened_at = value.get("opened_at_utc")
    source_url = value.get("source_url")
    reason = value.get("reason")
    if not all(isinstance(item, str) and item for item in (opened_at, source_url, reason)):
        raise TennisExplorerCacheError(
            f"El cortacircuitos está incompleto: {path}."
        )
    try:
        parsed_opened_at = datetime.fromisoformat(
            opened_at.replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise TennisExplorerCacheError(
            f"El cortacircuitos contiene una fecha inválida: {path}."
        ) from exc
    if (
        parsed_opened_at.tzinfo is None
        or parsed_opened_at.utcoffset() is None
    ):
        raise TennisExplorerCacheError(
            f"El cortacircuitos no contiene una fecha UTC: {path}."
        )

    raise TennisExplorerBlockedError(
        "El cortacircuitos de Tennis Explorer está abierto desde "
        f"{opened_at} ({reason}). Las fechas no cacheadas no emitirán tráfico "
        f"hasta que se revise y retire manualmente {path}."
    )


def _open_circuit_breaker(
    raw_dir: Path,
    *,
    source_url: str,
    opened_at_utc: datetime,
    response: Any,
) -> None:
    """Registra atómicamente un bloqueo sin guardar el cuerpo de respuesta."""

    path = _circuit_breaker_path(raw_dir)
    if path.exists():
        _raise_if_circuit_open(raw_dir)
    partial = _part_path(path)
    if partial.exists():
        raise TennisExplorerCacheError(
            f"Existe un cortacircuitos parcial que exige revisión: {partial}."
        )

    status_code = getattr(response, "status_code", None)
    reason = (
        f"http_{status_code}"
        if status_code in {403, 429}
        else "waf_challenge_signature"
    )
    opened = opened_at_utc.astimezone(UTC)
    payload = {
        "schema_version": _CIRCUIT_BREAKER_SCHEMA_VERSION,
        "host": "www.tennisexplorer.com",
        "opened_at_utc": opened.isoformat().replace("+00:00", "Z"),
        "source_url": source_url,
        "status_code": status_code if isinstance(status_code, int) else None,
        "reason": reason,
        "access_basis": _ACCESS_BASIS,
        "reopen_policy": "manual_review_only",
    }
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write_new_file(partial, encoded)
        os.replace(partial, path)
    except OSError as exc:
        _remove_own_partial(partial)
        raise TennisExplorerCacheError(
            f"No se pudo abrir el cortacircuitos de forma atómica: {exc}"
        ) from exc


@contextmanager
def _network_access_lock(raw_dir: Path) -> Iterator[None]:
    """Impide que dos procesos accedan simultáneamente al mismo host.

    Un lock abandonado tras una terminación abrupta se conserva para revisión
    manual. El cliente no lo elimina ni vuelve a intentar automáticamente,
    porque ante la duda es preferible emitir cero tráfico.
    """

    lock_path = raw_dir / "policy" / "network.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = lock_path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise TennisExplorerHttpError(
            "Ya existe una ejecución de red de Tennis Explorer o un lock "
            "abandonado. Se detiene sin emitir peticiones; revise "
            f"manualmente {lock_path}."
        ) from exc

    try:
        with handle:
            handle.write(
                f"pid={os.getpid()}\ncreated_at_utc="
                f"{datetime.now(UTC).isoformat()}\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _part_path(path: Path) -> Path:
    """Devuelve la ruta temporal adyacente usada en una escritura atómica."""

    return path.with_name(f"{path.name}.part")


def _load_cache_pair(
    paths: _CachePaths,
    *,
    artifact: str,
    date_field: str,
    expected_date: str,
    expected_url: str,
) -> tuple[bytes, dict[str, Any]] | None:
    """Carga y verifica un par de caché o informa de que aún no existe."""

    content_exists = paths.content.exists()
    metadata_exists = paths.metadata.exists()
    partial_paths = (
        _part_path(paths.content),
        _part_path(paths.metadata),
    )
    if any(path.exists() for path in partial_paths):
        raise TennisExplorerCacheError(
            f"Existe una escritura parcial en la caché de {expected_date}; "
            "se requiere revisión manual."
        )
    if not content_exists and not metadata_exists:
        return None
    if content_exists != metadata_exists:
        raise TennisExplorerCacheError(
            f"La caché de {expected_date} está incompleta; no se "
            "redescargará silenciosamente."
        )

    try:
        content = paths.content.read_bytes()
        metadata_value = json.loads(
            paths.metadata.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TennisExplorerCacheError(
            f"No se puede leer la caché de {expected_date}: {exc}"
        ) from exc
    if not isinstance(metadata_value, dict):
        raise TennisExplorerCacheError(
            f"La metadata de {expected_date} no es un objeto JSON."
        )
    metadata = dict(metadata_value)
    _validate_cache_metadata(
        metadata,
        content,
        artifact=artifact,
        date_field=date_field,
        expected_date=expected_date,
        expected_url=expected_url,
    )
    return content, metadata


def _validate_cache_metadata(
    metadata: Mapping[str, Any],
    content: bytes,
    *,
    artifact: str,
    date_field: str,
    expected_date: str,
    expected_url: str,
) -> None:
    """Verifica contrato, procedencia, tamaño y SHA-256 de una caché."""

    expected_values: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "artifact": artifact,
        date_field: expected_date,
        "source_url": expected_url,
        "status_code": 200,
    }
    for field, expected in expected_values.items():
        if metadata.get(field) != expected:
            raise TennisExplorerCacheError(
                f"Metadata inválida: {field!r} no coincide con {expected!r}."
            )

    size_bytes = metadata.get("size_bytes")
    if (
        not isinstance(size_bytes, int)
        or isinstance(size_bytes, bool)
        or size_bytes != len(content)
    ):
        raise TennisExplorerCacheError(
            "El tamaño del snapshot no coincide con su metadata."
        )

    expected_hash = metadata.get("sha256")
    actual_hash = hashlib.sha256(content).hexdigest()
    if not isinstance(expected_hash, str) or expected_hash != actual_hash:
        raise TennisExplorerCacheError(
            "El SHA-256 del snapshot no coincide con su metadata."
        )

    content_type = metadata.get("content_type")
    user_agent = metadata.get("user_agent")
    retrieved = metadata.get("retrieved_at_utc")
    if not isinstance(content_type, str) or not content_type:
        raise TennisExplorerCacheError(
            "La metadata no contiene un Content-Type válido."
        )
    if not isinstance(user_agent, str) or not user_agent:
        raise TennisExplorerCacheError(
            "La metadata no contiene el User-Agent usado."
        )
    if not isinstance(retrieved, str):
        raise TennisExplorerCacheError(
            "La metadata no contiene retrieved_at_utc."
        )
    try:
        parsed_retrieved = datetime.fromisoformat(
            retrieved.replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise TennisExplorerCacheError(
            "retrieved_at_utc no es un instante ISO-8601 válido."
        ) from exc
    if (
        parsed_retrieved.tzinfo is None
        or parsed_retrieved.utcoffset() is None
    ):
        raise TennisExplorerCacheError(
            "retrieved_at_utc debe incluir zona horaria."
        )


def _build_http_session() -> ScraplingHttpSession:
    """Crea el transporte Scrapling estático auditado para una sola tentativa."""

    return ScraplingHttpSession()


def _request_headers() -> dict[str, str]:
    """Devuelve cabeceras fijas y no rotatorias de un navegador normal."""

    return {
        "User-Agent": USER_AGENT,
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }


def _single_get(session: HttpSession, url: str) -> Any:
    """Ejecuta exactamente un GET y traduce fallos de transporte."""

    try:
        return session.get(
            url,
            headers=_request_headers(),
            timeout=REQUEST_TIMEOUT,
            allow_redirects=False,
        )
    except ScraplingTransportError as exc:
        raise TennisExplorerHttpError(
            f"Falló la única petición permitida a {url}: {exc}"
        ) from exc


def _validated_response_content(
    response: Any,
    url: str,
    *,
    allowed_content_types: tuple[str, ...],
) -> tuple[bytes, str]:
    """Valida estado, WAF, tipo MIME y bytes de una respuesta."""

    status_code = getattr(response, "status_code", None)
    headers_value = getattr(response, "headers", {})
    headers = (
        {str(key).lower(): str(value) for key, value in headers_value.items()}
        if isinstance(headers_value, Mapping)
        else {}
    )
    content_value = getattr(response, "content", b"")
    if not isinstance(content_value, (bytes, bytearray)):
        raise TennisExplorerHttpError(
            f"La respuesta de {url} no contiene bytes HTTP válidos."
        )
    content = bytes(content_value)

    if _looks_like_waf(status_code, headers, content):
        raise TennisExplorerBlockedError(
            "Tennis Explorer devolvió una señal de bloqueo, limitación o "
            "desafío WAF. Se detiene sin reintentar ni intentar evadirlo."
        )
    if not isinstance(status_code, int):
        raise TennisExplorerHttpError(
            f"La respuesta de {url} no contiene un estado HTTP válido."
        )
    if 300 <= status_code < 400:
        raise TennisExplorerHttpError(
            f"{url} respondió con redirect HTTP {status_code}; no se sigue."
        )
    if status_code != 200:
        raise TennisExplorerHttpError(
            f"{url} respondió con HTTP {status_code}; no se reintenta."
        )

    content_type = headers.get("content-type", "").strip()
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type not in allowed_content_types:
        raise TennisExplorerHttpError(
            f"{url} devolvió Content-Type {content_type!r}; se esperaba "
            f"uno de {allowed_content_types!r}."
        )
    return content, content_type


def _looks_like_waf(
    status_code: Any,
    headers: Mapping[str, str],
    content: bytes,
) -> bool:
    """Detecta únicamente indicadores fuertes de rate-limit o desafío WAF."""

    if status_code in {403, 429}:
        return True
    if headers.get("cf-mitigated", "").strip().lower() == "challenge":
        return True
    lowered = content.lower()
    return any(marker in lowered for marker in _CHALLENGE_BODY_MARKERS)


def _raise_if_cached_challenge(content: bytes, source_url: str) -> None:
    """Impide parsear una respuesta WAF que se haya incorporado manualmente."""

    if _looks_like_waf(200, {}, content):
        raise TennisExplorerBlockedError(
            f"La caché de {source_url} contiene un desafío WAF, no partidos."
        )


def _build_metadata(
    *,
    artifact: str,
    date_field: str,
    date_value: str,
    source_url: str,
    retrieved_at_utc: datetime,
    status_code: int,
    content_type: str,
    content: bytes,
) -> dict[str, Any]:
    """Construye metadata reproducible de procedencia e integridad."""

    retrieved = retrieved_at_utc.astimezone(UTC)
    metadata: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "artifact": artifact,
        date_field: date_value,
        "source_url": source_url,
        "retrieved_at_utc": retrieved.isoformat().replace("+00:00", "Z"),
        "status_code": status_code,
        "content_type": content_type,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "user_agent": USER_AGENT,
    }
    return metadata


def _write_cache_pair(
    paths: _CachePaths,
    content: bytes,
    metadata: Mapping[str, Any],
) -> None:
    """Publica contenido y metadata mediante archivos ``.part`` adyacentes."""

    paths.content.parent.mkdir(parents=True, exist_ok=True)
    content_part = _part_path(paths.content)
    metadata_part = _part_path(paths.metadata)
    if any(
        path.exists()
        for path in (
            paths.content,
            paths.metadata,
            content_part,
            metadata_part,
        )
    ):
        raise TennisExplorerCacheError(
            f"No se sobrescribe una caché existente o parcial: "
            f"{paths.content.parent}"
        )

    metadata_bytes = (
        json.dumps(
            dict(metadata),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    published_content = False
    try:
        _write_new_file(content_part, content)
        _write_new_file(metadata_part, metadata_bytes)
        os.replace(content_part, paths.content)
        published_content = True
        os.replace(metadata_part, paths.metadata)
    except OSError as exc:
        _remove_own_partial(content_part)
        _remove_own_partial(metadata_part)
        if published_content and not paths.metadata.exists():
            _remove_own_partial(paths.content)
        raise TennisExplorerCacheError(
            f"No se pudo publicar la caché de forma atómica: {exc}"
        ) from exc


def _write_new_file(path: Path, content: bytes) -> None:
    """Escribe, sincroniza y cierra un archivo nuevo sin sobrescribir."""

    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _remove_own_partial(path: Path) -> None:
    """Retira solo un artefacto parcial creado por esta transacción."""

    try:
        path.unlink(missing_ok=True)
    except OSError:
        # El error original de escritura sigue siendo el diagnóstico principal.
        return


def _append_audit_columns(
    frame: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> pd.DataFrame:
    """Añade procedencia con instante UTC y textos pandas reproducibles."""

    result = frame.copy()
    text_values = {
        "source_url": metadata["source_url"],
        "snapshot_sha256": metadata["sha256"],
    }
    for column in ("source_url", "snapshot_sha256"):
        result[column] = pd.Series(
            [text_values[column]] * len(result),
            index=result.index,
            dtype="string",
        )
    result["retrieved_at_utc"] = pd.Series(
        pd.to_datetime(
            [metadata["retrieved_at_utc"]] * len(result),
            utc=True,
        ),
        index=result.index,
        dtype="datetime64[ns, UTC]",
    )
    return result
