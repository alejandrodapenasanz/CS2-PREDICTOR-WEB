"""Transporte Scrapling mínimo y no evasivo para Tennis Explorer.

El módulo adapta ``FetcherSession`` de Scrapling 0.4.12 al pequeño contrato
HTTP usado por el cliente del proyecto. Por defecto conserva el perfil mínimo
de Tennis Explorer; TennisRatio activa identidad TLS/cabeceras de Chrome y
reintentos acotados. HTTP/3 permanece desactivado por fallo real de handshake.
No admite proxies, navegadores ni resolución de desafíos cuando el HTML
estático público ya contiene todos los datos necesarios.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Final, Protocol


SCRAPLING_REQUIRED_VERSION: Final[str] = "0.4.12"
SCRAPLING_TIMEOUT_SECONDS: Final[int] = 30
SCRAPLING_STATIC_BROWSER_IDENTITY: Final[str] = "chrome"
SCRAPLING_SESSION_OPTIONS: Final[dict[str, object]] = {
    "impersonate": None,
    "stealthy_headers": False,
    "http3": False,
    # En Scrapling 0.4.12 este valor es el total de tentativas, no las
    # repeticiones posteriores a una primera llamada.
    "retries": 1,
    "retry_delay": 1.0,
    "follow_redirects": False,
    "timeout": SCRAPLING_TIMEOUT_SECONDS,
    "verify": True,
}


class ScraplingTransportError(RuntimeError):
    """Representa un fallo controlado del transporte HTTP Scrapling."""


class _ScraplingBackend(Protocol):
    """Define el método GET del backend síncrono entregado por Scrapling."""

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
    ) -> Any:
        """Realiza la única tentativa GET configurada en la sesión."""


class _ScraplingSessionManager(Protocol):
    """Define el context manager síncrono creado por ``FetcherSession``."""

    def __enter__(self) -> _ScraplingBackend:
        """Abre la sesión y devuelve su backend síncrono."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> Any:
        """Cierra la sesión y sus conexiones."""


@dataclass(frozen=True)
class TransportResponse:
    """Respuesta inmutable compatible con el contrato usado por el cliente."""

    status_code: int
    content: bytes
    headers: Mapping[str, str]


class ScraplingHttpSession:
    """Adapta una ``FetcherSession`` estricta a una interfaz requests-like."""

    def __init__(
        self,
        *,
        browser_impersonation: bool = False,
        stealthy_headers: bool = False,
        retry_attempts: int = 1,
        retry_delay_seconds: float = 1.0,
        session_factory: Callable[..., _ScraplingSessionManager] | None = None,
        backend_error_types: tuple[type[BaseException], ...] | None = None,
    ) -> None:
        """Crea y abre una sesión Scrapling con configuración no evasiva.

        Args:
            browser_impersonation: Usa la identidad TLS fija ``chrome``.
            stealthy_headers: Genera cabeceras coherentes con esa identidad.
            retry_attempts: Total acotado de intentos ante errores de red.
            retry_delay_seconds: Espera determinista entre esos intentos.
            session_factory: Factoría inyectable para tests. Si se omite,
                carga ``FetcherSession`` de Scrapling 0.4.12 de forma perezosa.
            backend_error_types: Excepciones de transporte que deben traducirse
                a ``ScraplingTransportError``. Al usar la factoría real se
                configura automáticamente con ``curl_cffi.curl.CurlError``.

        Raises:
            ScraplingTransportError: Si falta la dependencia, su versión
                no coincide o el backend no puede abrirse.
        """

        if not isinstance(browser_impersonation, bool):
            raise TypeError("browser_impersonation debe ser booleano.")
        if not isinstance(stealthy_headers, bool):
            raise TypeError("stealthy_headers debe ser booleano.")
        if (
            isinstance(retry_attempts, bool)
            or not isinstance(retry_attempts, int)
            or retry_attempts <= 0
            or retry_attempts > 5
        ):
            raise ValueError("retry_attempts debe estar entre 1 y 5.")
        if (
            isinstance(retry_delay_seconds, bool)
            or not isinstance(retry_delay_seconds, (int, float))
            or retry_delay_seconds <= 0
            or retry_delay_seconds > 30
        ):
            raise ValueError("retry_delay_seconds debe estar entre 0 y 30.")
        if session_factory is None:
            factory, default_error_types = _load_scrapling_dependencies()
            session_factory = factory
            if backend_error_types is None:
                backend_error_types = default_error_types
        elif backend_error_types is None:
            backend_error_types = ()

        _validate_error_types(backend_error_types)
        self._backend_error_types = backend_error_types
        self._manager: _ScraplingSessionManager | None = None
        self._backend: _ScraplingBackend | None = None

        try:
            session_options = dict(SCRAPLING_SESSION_OPTIONS)
            if browser_impersonation:
                session_options["impersonate"] = SCRAPLING_STATIC_BROWSER_IDENTITY
            session_options["stealthy_headers"] = stealthy_headers
            session_options["retries"] = retry_attempts
            session_options["retry_delay"] = float(retry_delay_seconds)
            manager = session_factory(**session_options)
            backend = manager.__enter__()
        except self._backend_error_types as exc:
            raise ScraplingTransportError(f"No se pudo abrir la sesión Scrapling: {exc}") from exc
        except (TypeError, AttributeError) as exc:
            raise ScraplingTransportError(
                f"La factoría Scrapling no cumple el contrato esperado: {exc}"
            ) from exc

        self._manager = manager
        self._backend = backend

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: tuple[float, float],
        allow_redirects: bool,
    ) -> TransportResponse:
        """Realiza un GET y normaliza ``status/body/headers`` de Scrapling.

        ``timeout`` se conserva por compatibilidad con el protocolo existente.
        El backend usa siempre los 30 segundos fijados al crear la sesión.

        Args:
            url: URL HTTPS solicitada por el cliente.
            headers: Cabeceras fijas y explícitas de la petición.
            timeout: Timeout connect/read del protocolo anterior; no modifica
                la configuración Scrapling validada.
            allow_redirects: Debe ser ``False``; no se permite ampliarlo.

        Returns:
            Respuesta con ``status_code``, ``content`` y ``headers``.

        Raises:
            ScraplingTransportError: Ante sesión cerrada, redirect
                solicitado, error de curl o respuesta incompatible.
        """

        _validate_protocol_arguments(
            url=url,
            headers=headers,
            timeout=timeout,
            allow_redirects=allow_redirects,
        )
        if self._backend is None:
            raise ScraplingTransportError("La sesión Scrapling ya está cerrada.")

        try:
            response = self._backend.get(url, headers=dict(headers))
        except self._backend_error_types as exc:
            raise ScraplingTransportError(
                f"Falló la única tentativa Scrapling a {url}: {exc}"
            ) from exc
        return _adapt_response(response)

    def close(self) -> None:
        """Cierra una vez el context manager Scrapling; llamadas extra son no-op."""

        manager = self._manager
        if manager is None:
            return
        self._manager = None
        self._backend = None
        try:
            manager.__exit__(None, None, None)
        except self._backend_error_types as exc:
            raise ScraplingTransportError(f"Falló el cierre de la sesión Scrapling: {exc}") from exc

    def __enter__(self) -> ScraplingHttpSession:
        """Devuelve el transporte ya abierto para uso como context manager."""

        if self._backend is None:
            raise ScraplingTransportError("No se puede reabrir una sesión Scrapling cerrada.")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        """Cierra el transporte al abandonar un bloque ``with``."""

        del exc_type, exc_value, traceback
        self.close()


def _load_scrapling_dependencies() -> tuple[
    Callable[..., _ScraplingSessionManager],
    tuple[type[BaseException], ...],
]:
    """Carga la API oficial fijada y verifica la versión antes de usarla."""

    try:
        installed_version = version("scrapling")
    except PackageNotFoundError as exc:
        raise ScraplingTransportError(
            "Falta scrapling[fetchers]==0.4.12 en el entorno TENNIS."
        ) from exc
    if installed_version != SCRAPLING_REQUIRED_VERSION:
        raise ScraplingTransportError(
            "Versión Scrapling incompatible: "
            f"instalada={installed_version!r}, "
            f"requerida={SCRAPLING_REQUIRED_VERSION!r}."
        )

    try:
        from curl_cffi.curl import CurlError
        from scrapling.fetchers import FetcherSession
    except (ImportError, ModuleNotFoundError) as exc:
        raise ScraplingTransportError(
            "Scrapling está instalado sin las dependencias oficiales del extra [fetchers]."
        ) from exc
    return FetcherSession, (CurlError,)


def _validate_error_types(
    error_types: tuple[type[BaseException], ...],
) -> None:
    """Comprueba que las excepciones inyectadas forman una tupla segura."""

    if not isinstance(error_types, tuple) or any(
        not isinstance(error_type, type) or not issubclass(error_type, BaseException)
        for error_type in error_types
    ):
        raise TypeError("backend_error_types debe ser una tupla de excepciones.")


def _validate_protocol_arguments(
    *,
    url: str,
    headers: Mapping[str, str],
    timeout: tuple[float, float],
    allow_redirects: bool,
) -> None:
    """Valida el contrato requests-like sin relajar la política de red."""

    if not isinstance(url, str) or not url.startswith("https://"):
        raise ScraplingTransportError("ScraplingHttpSession solo admite URLs HTTPS explícitas.")
    if not isinstance(headers, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in headers.items()
    ):
        raise ScraplingTransportError("Las cabeceras deben ser un mapping de cadenas.")
    if (
        not isinstance(timeout, tuple)
        or len(timeout) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0
            for value in timeout
        )
    ):
        raise ScraplingTransportError("timeout debe ser una tupla positiva (connect, read).")
    if allow_redirects is not False:
        raise ScraplingTransportError("ScraplingHttpSession prohíbe seguir redirects.")


def _adapt_response(response: Any) -> TransportResponse:
    """Convierte una respuesta Scrapling en el contrato local estricto."""

    status = getattr(response, "status", None)
    body = getattr(response, "body", None)
    headers_value = getattr(response, "headers", None)
    if not isinstance(status, int) or isinstance(status, bool) or status < 100 or status > 599:
        raise ScraplingTransportError("Scrapling devolvió un estado HTTP inválido.")
    if not isinstance(body, (bytes, bytearray)):
        raise ScraplingTransportError("Scrapling devolvió un body que no son bytes.")
    if not isinstance(headers_value, Mapping):
        raise ScraplingTransportError("Scrapling devolvió headers incompatibles.")

    headers = {str(key): str(value) for key, value in headers_value.items()}
    return TransportResponse(
        status_code=status,
        content=bytes(body),
        headers=headers,
    )
