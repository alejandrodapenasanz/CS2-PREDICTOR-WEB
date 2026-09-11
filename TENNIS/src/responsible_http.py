"""Cliente HTTP común, cacheado y respetuoso con robots para TENNIS.

Este módulo es la única factoría de transporte de las fuentes remotas del
componente. Aplica ``robots.txt`` por defecto, serializa peticiones por host,
mantiene un intervalo mínimo configurable y conserva respuestas por contenido
SHA-256. No implementa navegadores, proxies, rotación de identidad ni bypass de
desafíos. Scrapling se usa únicamente como transporte HTTP estático.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Final, Literal, Protocol
from urllib.parse import parse_qs, urlsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import HTTP_CACHE_DIR, HTTP_CLIENT_CONFIG_PATH


DEFAULT_CONFIG_PATH: Final[Path] = HTTP_CLIENT_CONFIG_PATH
DEFAULT_CACHE_ROOT: Final[Path] = HTTP_CACHE_DIR
DEFAULT_USER_AGENT: Final[str] = "CS2-Predictor-TENNIS/1.0"
DEFAULT_MIN_INTERVAL_SECONDS: Final[float] = 1.0
DEFAULT_ROBOTS_TTL_SECONDS: Final[int] = 86_400
DEFAULT_RESPONSE_TTL_SECONDS: Final[int] = 300
DEFAULT_LOCK_TIMEOUT_SECONDS: Final[float] = 30.0
DEFAULT_LOCK_STALE_SECONDS: Final[float] = 300.0

CacheMode = Literal["default", "refresh", "no_store"]
TransportKind = Literal["requests", "scrapling"]
RobotsException = Callable[[str], bool]

_WAF_BODY_MARKERS: Final[tuple[bytes, ...]] = (
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
# JSD is also injected into ordinary HTML that already contains the requested
# data. Ignore only these exact script URLs, never the whole platform prefix.
_PASSIVE_JSD_SCRIPT: Final[re.Pattern[bytes]] = re.compile(
    rb"/cdn-cgi/challenge-platform/(?:h/[a-z0-9_-]+/)?scripts/jsd/"
    rb"(?:main|api)\.js(?=[?\s'\"<>]|$)"
)


class ResponsibleHttpError(requests.RequestException):
    """Error base de política, caché o transporte del cliente común."""


class RobotsDisallowedError(ResponsibleHttpError):
    """La política robots aplicable prohíbe solicitar la URL."""


class RobotsUnavailableError(ResponsibleHttpError):
    """No existe una política robots válida con la que autorizar el fetch."""


class HttpCacheError(ResponsibleHttpError):
    """La caché persistente no es íntegra o no puede publicarse."""


class RateLimitedError(ResponsibleHttpError):
    """Pausa persistente por host; no equivale a un desafío Cloudflare."""

    def __init__(self, url: str, retry_at: float, *, fresh_response: bool) -> None:
        """Expone el próximo instante permitido sin guardar el cuerpo del 429."""

        self.url = url
        self.retry_at = retry_at
        self.fresh_response = fresh_response
        try:
            deadline = datetime.fromtimestamp(retry_at, UTC).isoformat()
        except (OverflowError, OSError, ValueError):
            deadline = f"epoch {retry_at}"
        super().__init__(f"HTTP 429: {url}; pausa del host hasta {deadline}.")


class WafBlockedError(ResponsibleHttpError):
    """El servidor devolvió una señal inequívoca de WAF o rate-limit."""

    def __init__(self, message: str, response: HttpResponse) -> None:
        """Conserva la respuesta para que el cortacircuitos dueño la registre."""

        super().__init__(message)
        self.response = response


class _Transport(Protocol):
    """Superficie mínima compartida por Requests, Scrapling y fixtures."""

    def get(self, url: str, **kwargs: Any) -> Any:
        """Realiza un GET y devuelve una respuesta requests-like."""

    def close(self) -> None:
        """Libera los recursos del transporte."""


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """Respuesta materializada compatible con los descargadores existentes."""

    status_code: int
    content: bytes
    headers: Mapping[str, str]
    url: str
    from_cache: bool = False

    def raise_for_status(self) -> None:
        """Replica la comprobación HTTP usada por ``requests.Response``."""

        if self.status_code >= 400:
            raise requests.HTTPError(
                f"HTTP {self.status_code} for {self.url}",
                response=self,  # type: ignore[arg-type]
            )

    def json(self) -> Any:
        """Decodifica el cuerpo como JSON UTF-8."""

        return json.loads(self.content.decode("utf-8"))

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        """Expone el cuerpo materializado en bloques deterministas."""

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        for offset in range(0, len(self.content), chunk_size):
            yield self.content[offset : offset + chunk_size]

    def __enter__(self) -> HttpResponse:
        """Permite conservar el contrato ``with response``."""

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        """Finaliza el contexto; el cuerpo ya está materializado."""

        del exc_type, exc_value, traceback


@dataclass(frozen=True, slots=True)
class RobotsRule:
    """Una directiva Allow/Disallow perteneciente a un grupo robots."""

    allow: bool
    pattern: str


@dataclass(frozen=True, slots=True)
class RobotsGroup:
    """Grupo de agentes y reglas conservado en orden documental."""

    agents: tuple[str, ...]
    rules: tuple[RobotsRule, ...]


@dataclass(frozen=True, slots=True)
class RobotsPolicy:
    """Política robots parseada con precedencia por agente y regla más larga."""

    groups: tuple[RobotsGroup, ...]
    deny_all: bool = False

    def allows(self, url: str, *, user_agent: str) -> bool:
        """Decide una URL usando el grupo más específico y Allow en empates."""

        if self.deny_all:
            return False
        parsed = urlsplit(url)
        target = parsed.path or "/"
        if parsed.query:
            target = f"{target}?{parsed.query}"
        agent = user_agent.casefold()
        selected: list[RobotsGroup] = []
        best_specificity = -1
        for group in self.groups:
            matching = [
                token for token in group.agents if token == "*" or token.casefold() in agent
            ]
            if not matching:
                continue
            specificity = max(0 if token == "*" else len(token) for token in matching)
            if specificity > best_specificity:
                best_specificity = specificity
                selected = [group]
            elif specificity == best_specificity:
                selected.append(group)
        if not selected:
            return True

        winner: tuple[int, bool] | None = None
        for group in selected:
            for rule in group.rules:
                if not rule.pattern:
                    continue
                if _robots_pattern_matches(rule.pattern, target):
                    candidate = (len(rule.pattern), rule.allow)
                    if winner is None or candidate[0] > winner[0]:
                        winner = candidate
                    elif candidate[0] == winner[0] and candidate[1]:
                        winner = candidate
        return True if winner is None else winner[1]


@dataclass(frozen=True, slots=True)
class HttpClientSettings:
    """Configuración validada del cliente responsable."""

    minimum_interval_seconds: float = DEFAULT_MIN_INTERVAL_SECONDS
    robots_ttl_seconds: int = DEFAULT_ROBOTS_TTL_SECONDS
    response_ttl_seconds: int = DEFAULT_RESPONSE_TTL_SECONDS
    lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS
    lock_stale_seconds: float = DEFAULT_LOCK_STALE_SECONDS
    rate_limit_attempts: int = 1
    rate_limit_base_seconds: float = 30.0
    rate_limit_max_seconds: float = 300.0
    rate_limit_wait_budget_seconds: float = 90.0

    @classmethod
    def load(
        cls,
        path: Path = DEFAULT_CONFIG_PATH,
        *,
        host: str | None = None,
    ) -> HttpClientSettings:
        """Carga la configuración y un override explícito de host, si existe."""

        if not path.is_file():
            return cls()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ResponsibleHttpError(f"Configuración HTTP inválida: {path}") from exc
        if not isinstance(payload, Mapping):
            raise ResponsibleHttpError("La configuración HTTP debe ser un objeto JSON.")
        values: dict[str, Any] = {
            "minimum_interval_seconds": payload.get(
                "default_minimum_interval_seconds", DEFAULT_MIN_INTERVAL_SECONDS
            ),
            "robots_ttl_seconds": payload.get("robots_ttl_seconds", DEFAULT_ROBOTS_TTL_SECONDS),
            "response_ttl_seconds": payload.get(
                "response_cache_ttl_seconds", DEFAULT_RESPONSE_TTL_SECONDS
            ),
            "lock_timeout_seconds": payload.get(
                "lock_timeout_seconds", DEFAULT_LOCK_TIMEOUT_SECONDS
            ),
            "lock_stale_seconds": payload.get("lock_stale_seconds", DEFAULT_LOCK_STALE_SECONDS),
        }
        if float(values["minimum_interval_seconds"]) < DEFAULT_MIN_INTERVAL_SECONDS:
            raise ResponsibleHttpError(
                "El intervalo global no puede bajar de 1 s; use un override de host "
                "con access_basis de API/allowlist."
            )
        overrides = payload.get("host_overrides", {})
        if host is not None and isinstance(overrides, Mapping):
            override = overrides.get(host.casefold())
            if isinstance(override, Mapping):
                requested = override.get(
                    "minimum_interval_seconds", values["minimum_interval_seconds"]
                )
                if (
                    float(requested) < DEFAULT_MIN_INTERVAL_SECONDS
                    and not str(override.get("access_basis", "")).strip()
                ):
                    raise ResponsibleHttpError(
                        "Un ritmo inferior a 1 s exige access_basis de API/allowlist."
                    )
                values["minimum_interval_seconds"] = requested
                for key in (
                    "rate_limit_attempts",
                    "rate_limit_base_seconds",
                    "rate_limit_max_seconds",
                    "rate_limit_wait_budget_seconds",
                ):
                    if key in override:
                        values[key] = override[key]
        try:
            settings = cls(
                minimum_interval_seconds=float(values["minimum_interval_seconds"]),
                robots_ttl_seconds=int(values["robots_ttl_seconds"]),
                response_ttl_seconds=int(values["response_ttl_seconds"]),
                lock_timeout_seconds=float(values["lock_timeout_seconds"]),
                lock_stale_seconds=float(values["lock_stale_seconds"]),
                rate_limit_attempts=values.get("rate_limit_attempts", 1),
                rate_limit_base_seconds=float(values.get("rate_limit_base_seconds", 30.0)),
                rate_limit_max_seconds=float(values.get("rate_limit_max_seconds", 300.0)),
                rate_limit_wait_budget_seconds=float(
                    values.get("rate_limit_wait_budget_seconds", 90.0)
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ResponsibleHttpError("La configuración HTTP contiene números inválidos.") from exc
        settings.validate()
        return settings

    def validate(self) -> None:
        """Rechaza intervalos y TTL que harían inseguro el cliente."""

        if self.minimum_interval_seconds < 0:
            raise ResponsibleHttpError("minimum_interval_seconds no puede ser negativo.")
        if self.robots_ttl_seconds <= 0 or self.response_ttl_seconds < 0:
            raise ResponsibleHttpError("Los TTL HTTP no son válidos.")
        if self.lock_timeout_seconds <= 0 or self.lock_stale_seconds <= 0:
            raise ResponsibleHttpError("Los tiempos del lock HTTP deben ser positivos.")
        if type(self.rate_limit_attempts) is not int or not 1 <= self.rate_limit_attempts <= 5:
            raise ResponsibleHttpError("rate_limit_attempts debe ser un entero entre 1 y 5.")
        delays = (
            self.rate_limit_base_seconds,
            self.rate_limit_max_seconds,
            self.rate_limit_wait_budget_seconds,
        )
        if any(not math.isfinite(value) or value <= 0 for value in delays):
            raise ResponsibleHttpError(
                "Los tiempos de recuperación 429 deben ser finitos y positivos."
            )
        if self.rate_limit_max_seconds < self.rate_limit_base_seconds:
            raise ResponsibleHttpError("El máximo 429 no puede ser inferior a la espera base.")


def parse_robots_txt(content: bytes | str) -> RobotsPolicy:
    """Parsea grupos reales de robots, incluidos Allow, Disallow, ``*`` y ``$``."""

    if isinstance(content, bytes):
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise RobotsUnavailableError("robots.txt no es UTF-8 válido.") from exc
    elif isinstance(content, str):
        text = content
    else:
        raise TypeError("robots.txt debe ser bytes o texto.")

    groups: list[RobotsGroup] = []
    agents: list[str] = []
    rules: list[RobotsRule] = []
    rules_started = False

    def flush() -> None:
        nonlocal agents, rules, rules_started
        if agents:
            groups.append(RobotsGroup(tuple(agents), tuple(rules)))
        agents = []
        rules = []
        rules_started = False

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        field, value = (part.strip() for part in line.split(":", 1))
        field = field.casefold()
        if field == "user-agent":
            if rules_started:
                flush()
            if value:
                agents.append(value.casefold())
        elif field in {"allow", "disallow"} and agents:
            rules_started = True
            if field == "allow" or value:
                rules.append(RobotsRule(allow=field == "allow", pattern=value))
    flush()
    return RobotsPolicy(tuple(groups))


def authorized_matches_endpoint_exception(url: str) -> bool:
    """Autoriza solo la URL canónica diaria ``/matches/?...&type=all`` de TE."""

    try:
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "www.tennisexplorer.com"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port is not None
            or parsed.path != "/matches/"
            or parsed.fragment
        ):
            return False
        query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
        if set(query) != {"day", "month", "type", "year"}:
            return False
        day, month, kind, year = (
            query["day"],
            query["month"],
            query["type"],
            query["year"],
        )
        if (
            len(day) != 1
            or len(month) != 1
            or len(kind) != 1
            or len(year) != 1
            or re.fullmatch(r"[0-9]{2}", day[0]) is None
            or re.fullmatch(r"[0-9]{2}", month[0]) is None
            or re.fullmatch(r"[0-9]{4}", year[0]) is None
            or kind != ["all"]
        ):
            return False
        from datetime import date

        requested = date(int(year[0]), int(month[0]), int(day[0]))
        canonical = (
            "https://www.tennisexplorer.com/matches/"
            f"?day={requested.day:02d}&month={requested.month:02d}"
            f"&type=all&year={requested.year:04d}"
        )
        return url == canonical
    except (KeyError, TypeError, ValueError):
        return False


class ResponsibleHttpClient:
    """Cliente GET común con robots, rate-limit, lock y caché por SHA-256."""

    def __init__(
        self,
        *,
        transport: _Transport,
        transport_kind: TransportKind,
        cache_root: Path = DEFAULT_CACHE_ROOT,
        settings: HttpClientSettings | None = None,
        default_headers: Mapping[str, str] | None = None,
        robots_user_agent: str = "*",
        robots_exception: RobotsException | None = None,
        detect_waf: bool = False,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Configura el cliente sin emitir tráfico."""

        self._transport = transport
        self._transport_kind = transport_kind
        self._cache_root = Path(cache_root).resolve()
        self._explicit_settings = settings
        if settings is not None:
            settings.validate()
        self._default_headers = dict(default_headers or {})
        self._robots_user_agent = robots_user_agent
        self._robots_exception = robots_exception
        self._detect_waf = detect_waf
        self._sleeper = sleeper
        self._clock = clock
        self._closed = False
        self._expired_responses_pruned = False

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: tuple[float, float] = (10.0, 30.0),
        allow_redirects: bool = False,
        stream: bool = False,
        cache_mode: CacheMode = "default",
    ) -> HttpResponse:
        """Recupera 429 acotados fuera del lock; otros bloqueos se propagan."""

        canonical = _validate_url(url)
        settings = self._settings_for(_origin(canonical))
        waited = 0.0
        for attempt in range(settings.rate_limit_attempts):
            try:
                return self._get_once(
                    canonical,
                    headers=headers,
                    timeout=timeout,
                    allow_redirects=allow_redirects,
                    stream=stream,
                    cache_mode=cache_mode,
                )
            except RateLimitedError as exc:
                delay = max(0.0, exc.retry_at - self._clock())
                if (
                    not exc.fresh_response
                    or attempt + 1 >= settings.rate_limit_attempts
                    or waited + delay > settings.rate_limit_wait_budget_seconds
                ):
                    logging.getLogger(__name__).warning(
                        "%s Reintentos agotados o espera diferida; se conserva el último lote válido.",
                        exc,
                    )
                    raise
                logging.getLogger(__name__).warning(
                    "HTTP 429 %s: espera %.1fs; reintento %s/%s con la misma sesión Scrapling.",
                    exc.url,
                    delay,
                    attempt + 2,
                    settings.rate_limit_attempts,
                )
                waited += delay
                while delay > 0:
                    chunk = min(30.0, delay)
                    self._sleeper(chunk)
                    delay -= chunk
        raise AssertionError("Bucle HTTP sin respuesta ni excepción.")

    def _get_once(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None,
        timeout: tuple[float, float],
        allow_redirects: bool,
        stream: bool,
        cache_mode: CacheMode,
    ) -> HttpResponse:
        """Obtiene una URL permitida; una respuesta fresca evita todo fetch."""

        self._require_open()
        canonical = _validate_url(url)
        if allow_redirects:
            raise ResponsibleHttpError("El cliente común prohíbe seguir redirects.")
        if cache_mode not in {"default", "refresh", "no_store"}:
            raise ValueError(f"cache_mode inválido: {cache_mode!r}.")
        request_headers = dict(self._default_headers)
        request_headers.update(dict(headers or {}))
        cache_key = _request_cache_key(canonical, request_headers)
        cache_enabled = cache_mode != "no_store" and not stream
        if cache_mode == "default" and cache_enabled:
            cached = self._load_response(cache_key, canonical)
            if cached is not None:
                return cached

        if urlsplit(canonical).path != "/robots.txt":
            self._ensure_robots_allowed(canonical)
        origin = _origin(canonical)
        with self._host_lock(origin):
            if cache_mode == "default" and cache_enabled:
                cached = self._load_response(cache_key, canonical)
                if cached is not None:
                    return cached
            self._wait_for_host(origin)
            response = self._transport_get(
                canonical,
                headers=request_headers,
                timeout=timeout,
                stream=stream,
            )
            self._record_host_request(origin)
            self._inspect_rate_limit(origin, response)
            if self._detect_waf and _looks_like_waf(response):
                raise WafBlockedError(
                    f"{canonical} devolvió una señal WAF/rate-limit; no se cacheó.",
                    response,
                )
            if urlsplit(canonical).path == "/robots.txt":
                self._store_robots_response(origin, response)
            if cache_enabled and 200 <= response.status_code < 300:
                self._store_response(cache_key, response)
            return response

    def close(self) -> None:
        """Cierra una vez el transporte subyacente."""

        if self._closed:
            return
        self._closed = True
        self._transport.close()

    def __enter__(self) -> ResponsibleHttpClient:
        """Devuelve el cliente abierto."""

        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        """Cierra el transporte al abandonar el contexto."""

        del exc_type, exc_value, traceback
        self.close()

    def _require_open(self) -> None:
        """Evita reutilizar un transporte cerrado."""

        if self._closed:
            raise ResponsibleHttpError("El cliente HTTP común ya está cerrado.")

    def _transport_get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: tuple[float, float],
        stream: bool,
    ) -> HttpResponse:
        """Ejecuta un GET único y materializa una respuesta neutral."""

        try:
            kwargs: dict[str, Any] = {
                "headers": dict(headers),
                "timeout": timeout,
                "allow_redirects": False,
            }
            if self._transport_kind == "requests":
                kwargs["stream"] = stream
            raw = self._transport.get(url, **kwargs)
            status = getattr(raw, "status_code", None)
            content = getattr(raw, "content", None)
            headers_value = getattr(raw, "headers", {})
        except (requests.RequestException, OSError, RuntimeError) as exc:
            raise ResponsibleHttpError(f"Falló el GET responsable a {url}.") from exc
        if not isinstance(status, int) or isinstance(status, bool):
            raise ResponsibleHttpError(f"{url} no devolvió un estado HTTP válido.")
        if not isinstance(content, (bytes, bytearray)):
            raise ResponsibleHttpError(f"{url} no devolvió un cuerpo binario.")
        normalized_headers = (
            {str(key): str(value) for key, value in headers_value.items()}
            if isinstance(headers_value, Mapping)
            else {}
        )
        response_url = str(getattr(raw, "url", url))
        if response_url != url:
            raise ResponsibleHttpError(
                f"El transporte respondió desde {response_url!r}; redirects prohibidos."
            )
        return HttpResponse(status, bytes(content), normalized_headers, url)

    def _ensure_robots_allowed(self, url: str) -> None:
        """Aplica la excepción exacta o la política robots vigente."""

        if self._robots_exception is not None and self._robots_exception(url):
            return
        origin = _origin(url)
        policy = self._load_robots_policy(origin, allow_stale=False)
        if policy is None:
            policy = self._fetch_robots_policy(origin)
        if not policy.allows(url, user_agent=self._robots_user_agent):
            raise RobotsDisallowedError(f"robots.txt prohíbe solicitar {url}.")

    def _fetch_robots_policy(self, origin: str) -> RobotsPolicy:
        """Actualiza robots bajo el mismo lock/rate-limit; falla cerrado."""

        with self._host_lock(origin):
            cached = self._load_robots_policy(origin, allow_stale=False)
            if cached is not None:
                return cached
            robots_url = f"{origin}/robots.txt"
            self._wait_for_host(origin)
            try:
                response = self._transport_get(
                    robots_url,
                    headers={"User-Agent": DEFAULT_USER_AGENT, "Accept": "text/plain"},
                    timeout=(10.0, 30.0),
                    stream=False,
                )
            except ResponsibleHttpError:
                stale = self._load_robots_policy(origin, allow_stale=True)
                if stale is not None:
                    return stale
                raise RobotsUnavailableError(
                    f"No se pudo verificar robots.txt de {origin}; fetch bloqueado."
                )
            finally:
                self._record_host_request(origin)
            self._inspect_rate_limit(origin, response)
            if _looks_like_waf(response):
                stale = self._load_robots_policy(origin, allow_stale=True)
                if stale is not None:
                    return stale
                raise RobotsUnavailableError(
                    f"robots.txt de {origin} devolvió un bloqueo; fetch bloqueado."
                )
            self._store_robots_response(origin, response)
            policy = self._policy_from_robots_response(response)
            if policy is None:
                raise RobotsUnavailableError(
                    f"robots.txt de {origin} no tiene una respuesta utilizable."
                )
            return policy

    def _policy_from_robots_response(self, response: HttpResponse) -> RobotsPolicy | None:
        """Interpreta estados robots de forma conservadora."""

        if 200 <= response.status_code < 300:
            return parse_robots_txt(response.content)
        if response.status_code in {401, 403}:
            return RobotsPolicy((), deny_all=True)
        if response.status_code in {404, 410}:
            return RobotsPolicy(())
        return None

    def _robots_paths(self, origin: str) -> tuple[Path, Path]:
        """Devuelve metadata y objeto estable de robots para un origen."""

        key = hashlib.sha256(origin.encode("utf-8")).hexdigest()
        return (
            self._cache_root / "robots" / f"{key}.json",
            self._cache_root / "objects" / "robots" / key,
        )

    def _store_robots_response(self, origin: str, response: HttpResponse) -> None:
        """Publica robots y su hash; los estados no concluyentes también se auditan."""

        metadata_path, objects = self._robots_paths(origin)
        digest = hashlib.sha256(response.content).hexdigest()
        object_path = objects / f"{digest}.bin"
        _write_once_or_verify(object_path, response.content)
        payload = {
            "schema_version": 1,
            "origin": origin,
            "url": response.url,
            "status_code": response.status_code,
            "fetched_at_epoch": self._clock(),
            "sha256": digest,
            "object_path": str(object_path),
        }
        _atomic_write_json(metadata_path, payload)
        _prune_object_versions(objects, keep=object_path)

    def _load_robots_policy(
        self,
        origin: str,
        *,
        allow_stale: bool,
    ) -> RobotsPolicy | None:
        """Carga y verifica una política robots cacheada dentro de su TTL."""

        metadata_path, _ = self._robots_paths(origin)
        metadata = _read_json(metadata_path)
        if metadata is None:
            return None
        try:
            fetched = float(metadata["fetched_at_epoch"])
            status = int(metadata["status_code"])
            object_path = _validated_cached_object(
                metadata["object_path"],
                expected_parent=self._robots_paths(origin)[1],
            )
            expected = str(metadata["sha256"])
            if (
                not allow_stale
                and self._clock() - fetched > self._settings_for(origin).robots_ttl_seconds
            ):
                return None
            content = object_path.read_bytes()
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise HttpCacheError(f"Caché robots corrupta: {metadata_path}") from exc
        if hashlib.sha256(content).hexdigest() != expected:
            raise HttpCacheError(f"SHA-256 robots incorrecto: {metadata_path}")
        response = HttpResponse(status, content, {}, f"{origin}/robots.txt", True)
        return self._policy_from_robots_response(response)

    def _response_paths(self, cache_key: str) -> tuple[Path, Path]:
        """Devuelve metadata de petición y directorio de objetos compartido."""

        return (
            self._cache_root / "responses" / f"{cache_key}.json",
            self._cache_root / "objects" / "responses" / cache_key,
        )

    def _store_response(self, cache_key: str, response: HttpResponse) -> None:
        """Guarda una respuesta correcta por contenido SHA-256."""

        if not self._expired_responses_pruned:
            self._prune_expired_responses()
            self._expired_responses_pruned = True
        metadata_path, objects = self._response_paths(cache_key)
        digest = hashlib.sha256(response.content).hexdigest()
        object_path = objects / f"{digest}.bin"
        _write_once_or_verify(object_path, response.content)
        _atomic_write_json(
            metadata_path,
            {
                "schema_version": 1,
                "url": response.url,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "fetched_at_epoch": self._clock(),
                "sha256": digest,
                "object_path": str(object_path),
            },
        )
        _prune_object_versions(objects, keep=object_path)

    def _prune_expired_responses(self) -> None:
        """Retira una vez por proceso entradas vencidas para acotar la caché."""

        responses = self._cache_root / "responses"
        if not responses.is_dir():
            return
        for metadata_path in responses.glob("*.json"):
            try:
                metadata = _read_json(metadata_path)
                if metadata is None:
                    continue
                fetched = float(metadata["fetched_at_epoch"])
                url = str(metadata["url"])
                if self._clock() - fetched <= self._settings_for(_origin(url)).response_ttl_seconds:
                    continue
                cache_key = metadata_path.stem
                object_dir = self._response_paths(cache_key)[1]
                metadata_path.unlink()
                if object_dir.is_dir():
                    for candidate in object_dir.glob("*.bin"):
                        candidate.unlink()
                    object_dir.rmdir()
            except (HttpCacheError, KeyError, OSError, TypeError, ValueError):
                # Una entrada corrupta no amplía autoridad ni rompe una fuente
                # distinta. Si se solicita, la validación estricta sí fallará.
                continue

    def _load_response(self, cache_key: str, url: str) -> HttpResponse | None:
        """Sirve solo una entrada fresca cuya metadata, URL y hash coincidan."""

        metadata_path, _ = self._response_paths(cache_key)
        metadata = _read_json(metadata_path)
        if metadata is None:
            return None
        try:
            if metadata["url"] != url:
                raise HttpCacheError(f"La clave HTTP contradice su URL: {metadata_path}")
            fetched = float(metadata["fetched_at_epoch"])
            if self._clock() - fetched > self._settings_for(_origin(url)).response_ttl_seconds:
                return None
            object_path = _validated_cached_object(
                metadata["object_path"],
                expected_parent=self._response_paths(cache_key)[1],
            )
            content = object_path.read_bytes()
            expected = str(metadata["sha256"])
            status = int(metadata["status_code"])
            raw_headers = metadata.get("headers", {})
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise HttpCacheError(f"Caché HTTP corrupta: {metadata_path}") from exc
        if hashlib.sha256(content).hexdigest() != expected:
            raise HttpCacheError(f"SHA-256 HTTP incorrecto: {metadata_path}")
        headers = (
            {str(key): str(value) for key, value in raw_headers.items()}
            if isinstance(raw_headers, Mapping)
            else {}
        )
        return HttpResponse(status, content, headers, url, True)

    @contextmanager
    def _host_lock(self, origin: str) -> Iterator[None]:
        """Serializa procesos distintos mediante un lock exclusivo por host."""

        host_key = hashlib.sha256(origin.encode("utf-8")).hexdigest()
        lock_path = self._cache_root / "hosts" / host_key / "request.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        settings = self._settings_for(origin)
        deadline = self._clock() + settings.lock_timeout_seconds
        while True:
            try:
                descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    age = self._clock() - lock_path.stat().st_mtime
                except OSError:
                    age = 0.0
                if age > settings.lock_stale_seconds:
                    try:
                        lock_path.unlink()
                    except OSError:
                        pass
                    continue
                if self._clock() >= deadline:
                    raise ResponsibleHttpError(f"Timeout esperando lock HTTP: {origin}")
                self._sleeper(0.05)
                continue
            try:
                os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
            finally:
                os.close(descriptor)
            break
        try:
            yield
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass

    def _host_state_path(self, origin: str) -> Path:
        """Resuelve el timestamp compartido del último GET del origen."""

        host_key = hashlib.sha256(origin.encode("utf-8")).hexdigest()
        return self._cache_root / "hosts" / host_key / "last_request.json"

    def _wait_for_host(self, origin: str) -> None:
        """Espera solo el tiempo restante hasta el mínimo configurado."""

        rate_state = _read_json(self._host_state_path(origin).with_name("rate_limit.json"))
        if rate_state is not None:
            try:
                retry_at = float(rate_state["retry_at_epoch"])
                if not math.isfinite(retry_at):
                    raise ValueError("retry_at no finito")
            except (KeyError, TypeError, ValueError) as exc:
                raise HttpCacheError(f"Estado de pausa 429 corrupto para {origin}.") from exc
            if retry_at > self._clock():
                raise RateLimitedError(origin, retry_at, fresh_response=False)
        state = _read_json(self._host_state_path(origin))
        if state is None:
            return
        try:
            last_request = float(state["requested_at_epoch"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HttpCacheError(f"Estado de rate-limit corrupto para {origin}.") from exc
        remaining = self._settings_for(origin).minimum_interval_seconds - (
            self._clock() - last_request
        )
        if remaining > 0:
            self._sleeper(remaining)

    def _inspect_rate_limit(self, origin: str, response: HttpResponse) -> None:
        """Guarda una pausa compartida solo para los hosts con recuperación 429."""

        settings = self._settings_for(origin)
        if settings.rate_limit_attempts == 1:
            return
        state_path = self._host_state_path(origin).with_name("rate_limit.json")
        if response.status_code == 429 and not _has_waf_challenge(response):
            state = _read_json(state_path) or {}
            try:
                streak = max(0, int(state.get("streak", 0))) + 1
            except (TypeError, ValueError) as exc:
                raise HttpCacheError(f"Racha 429 corrupta para {origin}.") from exc
            delay = min(
                settings.rate_limit_max_seconds,
                settings.rate_limit_base_seconds * 2 ** min(streak - 1, 20),
            )
            headers = {key.casefold(): value for key, value in response.headers.items()}
            hinted = _retry_after_seconds(headers.get("retry-after"), now=self._clock())
            # Nunca acortar la espera indicada por el servidor al máximo local.
            retry_at = self._clock() + max(delay, hinted or 0.0)
            _atomic_write_json(
                state_path,
                {
                    "schema_version": 1,
                    "origin": origin,
                    "retry_at_epoch": retry_at,
                    "streak": streak,
                },
            )
            raise RateLimitedError(response.url, retry_at, fresh_response=True)
        if 200 <= response.status_code < 300 and not _looks_like_waf(response):
            if state_path.is_file():
                _atomic_write_json(
                    state_path,
                    {
                        "schema_version": 1,
                        "origin": origin,
                        "retry_at_epoch": 0.0,
                        "streak": 0,
                    },
                )

    def _record_host_request(self, origin: str) -> None:
        """Publica el instante del último intento de red del host."""

        _atomic_write_json(
            self._host_state_path(origin),
            {"schema_version": 1, "origin": origin, "requested_at_epoch": self._clock()},
        )

    def _settings_for(self, origin: str) -> HttpClientSettings:
        """Resuelve el override explícito del host sin relajar otros orígenes."""

        if self._explicit_settings is not None:
            return self._explicit_settings
        return HttpClientSettings.load(host=urlsplit(origin).hostname)


def build_http_client(
    *,
    transport_kind: TransportKind = "requests",
    cache_root: Path = DEFAULT_CACHE_ROOT,
    default_headers: Mapping[str, str] | None = None,
    robots_user_agent: str = "*",
    robots_exception: RobotsException | None = None,
    detect_waf: bool = False,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.time,
    request_retry_attempts: int = 1,
    request_retry_delay_seconds: float = 1.0,
    request_retry_statuses: tuple[int, ...] = (),
    browser_impersonation: bool = False,
    stealthy_headers: bool = False,
    settings: HttpClientSettings | None = None,
    transport: _Transport | None = None,
) -> ResponsibleHttpClient:
    """Construye el único cliente real, con estrategia de transporte elegible."""

    if transport is None:
        if transport_kind == "scrapling":
            # Import perezoso: ``tennis_explorer.__init__`` publica su cliente y
            # no debe crear un ciclo al importar esta infraestructura común.
            from .tennis_explorer.transport import ScraplingHttpSession

            transport = ScraplingHttpSession(
                browser_impersonation=browser_impersonation,
                stealthy_headers=stealthy_headers,
                retry_attempts=request_retry_attempts,
                retry_delay_seconds=request_retry_delay_seconds,
            )
        elif transport_kind == "requests":
            retry_total = max(0, request_retry_attempts - 1)
            retry = Retry(
                total=retry_total,
                connect=retry_total,
                read=retry_total,
                status=retry_total,
                backoff_factor=request_retry_delay_seconds,
                status_forcelist=request_retry_statuses,
                allowed_methods=frozenset({"GET"}),
                respect_retry_after_header=True,
            )
            adapter = HTTPAdapter(max_retries=retry)
            request_session = requests.Session()
            request_session.mount("https://", adapter)
            request_session.mount("http://", adapter)
            transport = request_session
        else:
            raise ValueError(f"transport_kind no admitido: {transport_kind!r}.")
    return ResponsibleHttpClient(
        transport=transport,
        transport_kind=transport_kind,
        cache_root=cache_root,
        settings=settings,
        default_headers=default_headers,
        robots_user_agent=robots_user_agent,
        robots_exception=robots_exception,
        detect_waf=detect_waf,
        sleeper=sleeper,
        clock=clock,
    )


def _validate_url(url: str) -> str:
    """Exige una URL HTTP(S) absoluta sin credenciales ni fragmento."""

    if not isinstance(url, str) or not url.strip():
        raise ResponsibleHttpError("La URL debe ser texto no vacío.")
    parsed = urlsplit(url.strip())
    try:
        port = parsed.port
    except ValueError as exc:
        raise ResponsibleHttpError(f"Puerto inválido en URL: {url!r}.") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 80, 443}
        or parsed.fragment
    ):
        raise ResponsibleHttpError(f"URL HTTP no autorizable: {url!r}.")
    return url.strip()


def _origin(url: str) -> str:
    """Normaliza el origen usado por robots, locks y rate-limit."""

    parsed = urlsplit(url)
    default_port = 443 if parsed.scheme == "https" else 80
    port = parsed.port or default_port
    suffix = "" if port == default_port else f":{port}"
    return f"{parsed.scheme}://{(parsed.hostname or '').casefold()}{suffix}"


def _request_cache_key(url: str, headers: Mapping[str, str]) -> str:
    """Deriva una clave estable sin almacenar secretos fuera de la metadata."""

    safe_headers = {
        str(key).casefold(): str(value)
        for key, value in headers.items()
        if str(key).casefold() not in {"authorization", "cookie", "proxy-authorization"}
    }
    canonical = json.dumps(
        {"method": "GET", "url": url, "headers": safe_headers},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _robots_pattern_matches(pattern: str, target: str) -> bool:
    """Evalúa wildcards robots con anclaje inicial y ``$`` final opcional."""

    anchored = pattern.endswith("$")
    body = pattern[:-1] if anchored else pattern
    expression = re.escape(body).replace(r"\*", ".*")
    suffix = "$" if anchored else ".*"
    return re.fullmatch(f"{expression}{suffix}", target) is not None


def _looks_like_waf(response: HttpResponse) -> bool:
    """Detecta bloqueo real sin confundir un script JSD pasivo con un reto.

    Only successful HTML qualifies for the narrow JSD path distinction. Other
    challenge markers, HTTP 403/429 and CF-Mitigated always take precedence.
    Response bytes are never changed, and source parsers still validate them.
    """

    if response.status_code in {403, 429}:
        return True
    return _has_waf_challenge(response)


def _has_waf_challenge(response: HttpResponse) -> bool:
    """Distingue firmas de desafío de un 429 sin challenge recuperable."""

    headers = {str(key).casefold(): str(value) for key, value in response.headers.items()}
    if headers.get("cf-mitigated", "").strip().casefold() == "challenge":
        return True
    lowered = response.content[:524_288].lower()
    content_type = headers.get("content-type", "").split(";", 1)[0].strip().casefold()
    if response.status_code == 200 and content_type in {"text/html", "application/xhtml+xml"}:
        lowered = _PASSIVE_JSD_SCRIPT.sub(b"/passive-jsd-script", lowered)
    return any(marker in lowered for marker in _WAF_BODY_MARKERS)


def _retry_after_seconds(value: str | None, *, now: float) -> float | None:
    """Interpreta segundos o fecha HTTP, rechazando valores no finitos."""

    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            seconds = parsed.timestamp() - now
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None


def _read_json(path: Path) -> Mapping[str, Any] | None:
    """Lee un objeto JSON o devuelve ``None`` cuando aún no existe."""

    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HttpCacheError(f"JSON HTTP corrupto: {path}") from exc
    if not isinstance(payload, Mapping):
        raise HttpCacheError(f"JSON HTTP no es un objeto: {path}")
    return payload


def _validated_cached_object(value: object, *, expected_parent: Path) -> Path:
    """Impide que metadata manipulada haga leer fuera de la caché asignada."""

    candidate = Path(str(value)).resolve()
    parent = expected_parent.resolve()
    if candidate.parent != parent or candidate.suffix != ".bin":
        raise HttpCacheError(f"Ruta de objeto HTTP fuera de caché: {candidate}")
    return candidate


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publica metadata JSON de forma atómica."""

    content = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.{os.getpid()}.part")
    try:
        partial.write_bytes(content)
        os.replace(partial, path)
    except OSError as exc:
        raise HttpCacheError(f"No se pudo publicar metadata HTTP: {path}") from exc
    finally:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass


def _write_once_or_verify(path: Path, content: bytes) -> None:
    """Crea un objeto por hash o verifica que el objeto existente sea idéntico."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_bytes() != content:
            raise HttpCacheError(f"Colisión o corrupción en objeto HTTP: {path}")
        return
    partial = path.with_name(f"{path.name}.{os.getpid()}.part")
    try:
        partial.write_bytes(content)
        try:
            os.link(partial, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise HttpCacheError(f"Colisión o corrupción en objeto HTTP: {path}")
        except OSError:
            if not path.exists():
                os.replace(partial, path)
                return
    finally:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass


def _prune_object_versions(directory: Path, *, keep: Path) -> None:
    """Conserva una sola versión corporal por clave para evitar acumulación."""

    if not directory.is_dir():
        return
    for candidate in directory.glob("*.bin"):
        if candidate.resolve() == keep.resolve():
            continue
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass
