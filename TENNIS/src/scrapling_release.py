"""Comprueba la última release estable de Scrapling sin modificar el proyecto.

El módulo consulta una sola vez el endpoint oficial ``releases/latest`` de
GitHub, valida que la respuesta describa una release estable y compara su
versión con el pin aprobado. No sigue ``main``, no instala dependencias, no
actualiza archivos y no realiza reintentos.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any, Final, Protocol

import requests
from requests.adapters import HTTPAdapter


SCRAPLING_LATEST_RELEASE_URL: Final[str] = (
    "https://api.github.com/repos/D4Vinci/Scrapling/releases/latest"
)
PINNED_SCRAPLING_VERSION: Final[str] = "0.4.12"
GITHUB_API_VERSION: Final[str] = "2022-11-28"
REQUEST_TIMEOUT: Final[tuple[float, float]] = (10.0, 30.0)
USER_AGENT: Final[str] = "TENNIS-academic-release-check/1.0"

_STABLE_VERSION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[vV]?([0-9]+(?:\.[0-9]+)+)$"
)
_ALLOWED_MEDIA_TYPES: Final[frozenset[str]] = frozenset(
    {
        "application/json",
        "application/vnd.github+json",
    }
)


class HttpSession(Protocol):
    """Define la parte de una sesión HTTP necesaria para la comprobación."""

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: tuple[float, float],
        allow_redirects: bool,
    ) -> Any:
        """Realiza un único GET sin redirecciones automáticas."""

    def close(self) -> None:
        """Libera los recursos de una sesión creada por este módulo."""


class ScraplingReleaseCheckError(RuntimeError):
    """Indica que la release oficial no pudo comprobarse con seguridad."""


@dataclass(frozen=True)
class ScraplingReleaseCheck:
    """Resume la comparación entre el pin local y la última release estable."""

    pinned_version: str
    latest_version: str
    tag_name: str
    release_url: str

    @property
    def update_available(self) -> bool:
        """Indica si GitHub publica una versión estable posterior al pin."""

        return _version_key(self.latest_version) > _version_key(
            self.pinned_version
        )


def build_http_session() -> requests.Session:
    """Crea una sesión de Requests con reintentos desactivados explícitamente."""

    no_retry_adapter = HTTPAdapter(max_retries=0)
    session = requests.Session()
    session.mount("https://", no_retry_adapter)
    session.mount("http://", no_retry_adapter)
    return session


def check_latest_scrapling_release(
    *,
    session: HttpSession | None = None,
) -> ScraplingReleaseCheck:
    """Consulta y compara la última release estable publicada por Scrapling.

    Args:
        session: Sesión HTTP inyectable. Si se omite, se crea una sesión
            ``requests`` sin reintentos y se cierra al terminar.

    Returns:
        Resultado inmutable con el pin, la versión remota y la URL de release.

    Raises:
        ScraplingReleaseCheckError: Si falla HTTP, el JSON no cumple el
            contrato oficial, la release no es estable o GitHub informa una
            versión anterior al pin.
    """

    owned_session = session is None
    http = session if session is not None else build_http_session()
    try:
        try:
            response = http.get(
                SCRAPLING_LATEST_RELEASE_URL,
                headers=_request_headers(),
                timeout=REQUEST_TIMEOUT,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise ScraplingReleaseCheckError(
                "Falló la única petición a la API oficial de GitHub; "
                "no se reintentó."
            ) from exc

        payload = _validated_release_payload(response)
        tag_name = payload["tag_name"]
        latest_version = _normalise_stable_version(
            tag_name,
            field_name="tag_name",
        )
        pinned_version = _normalise_stable_version(
            PINNED_SCRAPLING_VERSION,
            field_name="pin local",
        )
        if _version_key(latest_version) < _version_key(pinned_version):
            raise ScraplingReleaseCheckError(
                "La API informa una release estable anterior al pin local: "
                f"GitHub={latest_version}, pin={pinned_version}."
            )

        release_url = payload["html_url"]
        return ScraplingReleaseCheck(
            pinned_version=pinned_version,
            latest_version=latest_version,
            tag_name=tag_name,
            release_url=release_url,
        )
    finally:
        if owned_session:
            http.close()


def _request_headers() -> dict[str, str]:
    """Construye las cabeceras recomendadas para la API versionada de GitHub."""

    return {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
        "User-Agent": USER_AGENT,
    }


def _validated_release_payload(response: Any) -> dict[str, Any]:
    """Valida HTTP, tipo MIME, JSON y campos de una release estable."""

    status_code = getattr(response, "status_code", None)
    if not isinstance(status_code, int) or isinstance(status_code, bool):
        raise ScraplingReleaseCheckError(
            "La API de GitHub no devolvió un estado HTTP válido."
        )
    if status_code != 200:
        raise ScraplingReleaseCheckError(
            "La API oficial de GitHub respondió con "
            f"HTTP {status_code}; no se reintentó."
        )

    headers_value = getattr(response, "headers", {})
    headers = (
        {str(key).lower(): str(value) for key, value in headers_value.items()}
        if isinstance(headers_value, Mapping)
        else {}
    )
    content_type = headers.get("content-type", "")
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type not in _ALLOWED_MEDIA_TYPES:
        raise ScraplingReleaseCheckError(
            "La API de GitHub devolvió un Content-Type inesperado: "
            f"{content_type!r}."
        )

    try:
        payload_value = response.json()
    except (TypeError, ValueError) as exc:
        raise ScraplingReleaseCheckError(
            "La respuesta de GitHub no contiene JSON válido."
        ) from exc
    if not isinstance(payload_value, dict):
        raise ScraplingReleaseCheckError(
            "La respuesta de GitHub no es un objeto JSON."
        )
    payload = dict(payload_value)

    tag_name = payload.get("tag_name")
    release_url = payload.get("html_url")
    if not isinstance(tag_name, str) or not tag_name.strip():
        raise ScraplingReleaseCheckError(
            "La release de GitHub no contiene un tag_name textual."
        )
    if not isinstance(release_url, str) or not release_url.strip():
        raise ScraplingReleaseCheckError(
            "La release de GitHub no contiene una html_url textual."
        )
    if payload.get("draft") is not False:
        raise ScraplingReleaseCheckError(
            "La respuesta marcada como latest corresponde a un borrador."
        )
    if payload.get("prerelease") is not False:
        raise ScraplingReleaseCheckError(
            "La respuesta marcada como latest corresponde a una prerelease."
        )

    return {
        "tag_name": tag_name.strip(),
        "html_url": release_url.strip(),
    }


def _normalise_stable_version(value: str, *, field_name: str) -> str:
    """Normaliza un tag numérico estable y rechaza sufijos prerelease."""

    match = _STABLE_VERSION_PATTERN.fullmatch(value.strip())
    if match is None:
        raise ScraplingReleaseCheckError(
            f"{field_name} no es una versión estable numérica: {value!r}."
        )
    return ".".join(
        str(int(component)) for component in match.group(1).split(".")
    )


def _version_key(value: str) -> tuple[int, ...]:
    """Convierte una versión normalizada en una tupla comparable."""

    components = tuple(int(component) for component in value.split("."))
    minimum_width = 3
    return components + (0,) * max(0, minimum_width - len(components))
