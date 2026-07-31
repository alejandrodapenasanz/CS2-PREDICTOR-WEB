"""Parser puro y estricto de la agenda diaria de Tennis Explorer.

La implementación refleja el HTML real capturado el 30 de julio de 2026. La
tabla diaria se identifica por su pestaña de fecha; los partidos son parejas
de filas cuyo identificador sigue el patrón ``id``/``idb``. Solo se devuelven
individuales ATP, WTA, Challenger e ITF. Dobles y competiciones UTR quedan
fuera del alcance aprobado.

La fuente diaria no siempre distingue cancelaciones, walkovers y partidos en
juego. Por ello el estado solo se asigna cuando existe evidencia estructural o
texto explícito; cualquier caso restante se conserva como ``unknown`` junto
con ``status_evidence``. El ganador solo se expone cuando un final
convencional tiene contadores terminales, scores visibles y slug estable; no
se deduce de la posición de la fila, la cuota ni una clase CSS.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import math
import re
from typing import Final
from urllib.parse import urlsplit

from bs4 import BeautifulSoup, NavigableString, Tag
import pandas as pd

from .types import (
    OUTPUT_COLUMNS,
    OUTPUT_DTYPES,
    Gender,
    MatchStatus,
    Surface,
    TennisExplorerSchemaError,
    TourLevel,
    WinnerSide,
)


_DATE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(\d{1,2})\.\s*(\d{1,2})\.\s*(\d{4})\b"
)
_TIME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<!\d)([01]\d|2[0-3]):[0-5]\d(?!\d)"
)
_PLAYER_HREF_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^/player/([^/?#]+)/$"
)
_TOURNAMENT_HREF_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^/[^/?#]+/(\d{4})/(atp-men|wta-women)/"
    r"(?:\?type=double)?$"
)
_MATCH_DETAIL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^/match-detail/\?id=(\d+)$"
)
_ODDS_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^\d+(?:\.\d+)?$"
)
_GENERIC_FUTURES_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^Futures\s+\d{4}$",
    flags=re.IGNORECASE,
)
_SEED_SUFFIX_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\s*\(\d+\)\s*$"
)
_WALKOVER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?:\bwalkover\b|\bw\s*[./-]?\s*o\b\.?)",
    flags=re.IGNORECASE,
)
_CANCELLED_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\bcancel(?:l)?ed\b",
    flags=re.IGNORECASE,
)
_LIVE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?:\blive\b|\binterrupted\b)",
    flags=re.IGNORECASE,
)
_FINISHED_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\bfinished\b",
    flags=re.IGNORECASE,
)
_SCHEDULED_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\bscheduled\b",
    flags=re.IGNORECASE,
)
_SUPPORTED_SURFACES: Final[frozenset[str]] = frozenset(
    {"Hard", "Clay", "Grass", "Carpet"}
)
_TERMINAL_SET_RESULTS: Final[frozenset[tuple[int, int]]] = frozenset(
    {
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
        (3, 2),
    }
)


@dataclass(frozen=True)
class _Tournament:
    """Contiene el contexto heredado por las filas de un torneo."""

    include: bool
    name: str
    href: str | None
    gender: Gender
    tour_level: TourLevel | None
    surface: Surface | None


@dataclass(frozen=True)
class _Player:
    """Representa un jugador visible y su identificador web estable opcional."""

    name: str
    href: str | None
    slug: str | None


@dataclass(frozen=True)
class _ParsedResult:
    """Conserva sets observados y un ganador solo cuando es demostrable."""

    player_1_sets_won: int | None
    player_2_sets_won: int | None
    sets_score: str | None
    winner_side: WinnerSide | None
    winner_slug: str | None
    evidence: str


def parse_daily_matches_html(
    html: bytes | str,
    match_date: date,
) -> pd.DataFrame:
    """Convierte un HTML diario real en un DataFrame tipado de individuales.

    Args:
        html: Documento HTML completo, como bytes originales o texto.
        match_date: Fecha solicitada, usada para escoger la tabla exacta.

    Returns:
        Un DataFrame con las columnas y dtypes definidos en ``OUTPUT_DTYPES``.
        Si la tabla válida no contiene partidos dentro del alcance, devuelve
        el mismo esquema sin filas.

    Raises:
        TypeError: Si los argumentos no respetan el contrato de tipos.
        TennisExplorerSchemaError: Si cambia la estructura inspeccionada o un
            valor no puede interpretarse sin inventar información.
    """

    if not isinstance(html, (bytes, str)):
        raise TypeError("html debe ser bytes o str.")
    if not isinstance(match_date, date) or isinstance(match_date, datetime):
        raise TypeError("match_date debe ser datetime.date, no datetime.")
    if not html:
        raise TennisExplorerSchemaError("El HTML diario está vacío.")

    soup = BeautifulSoup(html, "lxml")
    surfaces_by_tournament = _parse_surface_catalog(soup, match_date)
    table = _select_date_table(soup, match_date)
    rows = table.find_all("tr")
    if not rows:
        return empty_matches_dataframe()

    records: list[dict[str, object]] = []
    tournament: _Tournament | None = None
    index = 0
    saw_tournament_header = False

    while index < len(rows):
        row = rows[index]
        classes = set(row.get("class", []))

        if "head" in classes:
            if "flags" not in classes:
                raise TennisExplorerSchemaError(
                    "Se encontró una cabecera de tabla sin la clase 'flags'."
                )
            tournament = _parse_tournament_header(
                row,
                match_date,
                surfaces_by_tournament=surfaces_by_tournament,
            )
            saw_tournament_header = True
            index += 1
            continue

        row_id = row.get("id")
        if not row_id:
            if _normalise_text(row.get_text(" ", strip=True)):
                raise TennisExplorerSchemaError(
                    "La tabla diaria contiene una fila no reconocida sin id."
                )
            index += 1
            continue
        if tournament is None:
            raise TennisExplorerSchemaError(
                f"La fila {row_id!r} aparece antes de una cabecera de torneo."
            )
        if row_id.endswith("b"):
            raise TennisExplorerSchemaError(
                f"La segunda fila {row_id!r} no tiene una primera fila previa."
            )
        if index + 1 >= len(rows):
            raise TennisExplorerSchemaError(
                f"Falta la segunda fila del partido {row_id!r}."
            )

        second_row = rows[index + 1]
        expected_second_id = f"{row_id}b"
        if second_row.get("id") != expected_second_id:
            raise TennisExplorerSchemaError(
                f"El partido {row_id!r} debería continuar en "
                f"{expected_second_id!r}, no en {second_row.get('id')!r}."
            )

        if tournament.include:
            records.append(
                _parse_match_pair(
                    row,
                    second_row,
                    tournament=tournament,
                    match_date=match_date,
                )
            )
        index += 2

    if not saw_tournament_header:
        return empty_matches_dataframe()
    return _to_typed_dataframe(records)


def empty_matches_dataframe() -> pd.DataFrame:
    """Crea un DataFrame vacío que conserva exactamente el contrato público."""

    return pd.DataFrame(
        {
            column: pd.Series(dtype=dtype)
            for column, dtype in OUTPUT_DTYPES.items()
        },
        columns=OUTPUT_COLUMNS,
    )


def _select_date_table(soup: BeautifulSoup, match_date: date) -> Tag:
    """Selecciona la única tabla ``result`` asociada a la fecha solicitada."""

    candidates: list[Tag] = []
    for table in soup.find_all("table"):
        if set(table.get("class", [])) != {"result"}:
            continue
        date_navigation = table.find_previous_sibling()
        if not isinstance(date_navigation, Tag):
            continue
        if date_navigation.name != "ul" or "tabs" not in date_navigation.get(
            "class", []
        ):
            continue
        dates = _dates_in_text(date_navigation.get_text(" ", strip=True))
        if match_date in dates:
            candidates.append(table)

    if len(candidates) != 1:
        raise TennisExplorerSchemaError(
            "Se esperaba una única tabla diaria 'result' para "
            f"{match_date.isoformat()}, pero se encontraron {len(candidates)}."
        )
    return candidates[0]


def _dates_in_text(text: str) -> set[date]:
    """Extrae fechas válidas ``DD. MM. YYYY`` de un texto de navegación."""

    parsed: set[date] = set()
    for day_text, month_text, year_text in _DATE_PATTERN.findall(text):
        try:
            parsed.add(
                date(
                    int(year_text),
                    int(month_text),
                    int(day_text),
                )
            )
        except ValueError as error:
            raise TennisExplorerSchemaError(
                f"La navegación contiene una fecha inválida: "
                f"{day_text}.{month_text}.{year_text}."
            ) from error
    return parsed


def _parse_surface_catalog(
    soup: BeautifulSoup,
    match_date: date,
) -> dict[str, Surface]:
    """Extrae superficies explícitas del catálogo incluido en el mismo HTML.

    La agenda diaria no repite la superficie en cada cabecera, pero el bloque
    ``This week's tournaments`` enlaza el mismo ``tournament_href`` y publica
    un ``td.s-color``. Una ausencia permanece ausente; un título nuevo o dos
    superficies contradictorias para el mismo enlace detienen el parser.
    """

    surfaces: dict[str, Surface] = {}
    for surface_cell in soup.select("td.s-color"):
        row = surface_cell.find_parent("tr")
        if not isinstance(row, Tag):
            raise TennisExplorerSchemaError(
                "Una celda de superficie no pertenece a una fila de torneo."
            )
        name_cells = row.select("td.t-name")
        if len(name_cells) != 1:
            raise TennisExplorerSchemaError(
                "Una fila del catálogo con superficie no tiene un td.t-name."
            )
        direct_links = name_cells[0].find_all("a", recursive=False)
        if len(direct_links) != 1:
            raise TennisExplorerSchemaError(
                "Una fila del catálogo con superficie no tiene un enlace único."
            )
        raw_href = direct_links[0].get("href")
        if not isinstance(raw_href, str):
            raise TennisExplorerSchemaError(
                "El enlace de una fila del catálogo no es texto."
            )
        href_match = _TOURNAMENT_HREF_PATTERN.fullmatch(raw_href)
        if href_match is None:
            continue
        href_year, _ = href_match.groups()
        if int(href_year) != match_date.year or "?type=double" in raw_href:
            continue

        titled_spans = surface_cell.find_all("span", title=True)
        if not titled_spans:
            continue
        if len(titled_spans) != 1:
            raise TennisExplorerSchemaError(
                f"El catálogo contiene varias superficies para {raw_href!r}."
            )
        raw_surface = titled_spans[0].get("title")
        if not isinstance(raw_surface, str):
            raise TennisExplorerSchemaError(
                f"La superficie de {raw_href!r} no es texto."
            )
        surface_text = _normalise_text(raw_surface)
        if surface_text not in _SUPPORTED_SURFACES:
            raise TennisExplorerSchemaError(
                f"Superficie no reconocida en el catálogo: {surface_text!r}."
            )
        surface: Surface = surface_text  # type: ignore[assignment]
        previous = surfaces.get(raw_href)
        if previous is not None and previous != surface:
            raise TennisExplorerSchemaError(
                f"Superficies contradictorias para {raw_href!r}: "
                f"{previous!r} y {surface!r}."
            )
        surfaces[raw_href] = surface
    return surfaces


def _parse_tournament_header(
    row: Tag,
    match_date: date,
    *,
    surfaces_by_tournament: dict[str, Surface],
) -> _Tournament:
    """Interpreta género, modalidad y nivel desde una cabecera observada."""

    name_cells = row.select("td.t-name")
    if len(name_cells) != 1:
        raise TennisExplorerSchemaError(
            "La cabecera de torneo no contiene exactamente un td.t-name."
        )
    name_cell = name_cells[0]
    name = _normalise_text(name_cell.get_text(" ", strip=True))
    if not name:
        raise TennisExplorerSchemaError(
            "La cabecera de torneo no contiene un nombre."
        )

    type_classes = [
        class_name
        for span in name_cell.find_all("span")
        for class_name in span.get("class", [])
        if class_name.startswith("type-")
    ]
    if len(type_classes) != 1:
        raise TennisExplorerSchemaError(
            f"El torneo {name!r} no tiene un único marcador de modalidad."
        )
    type_class = type_classes[0]
    supported_types = {
        "type-men2": ("M", True),
        "type-women2": ("F", True),
        "type-men4": ("M", False),
        "type-women4": ("F", False),
    }
    if type_class not in supported_types:
        raise TennisExplorerSchemaError(
            f"Marcador de modalidad no reconocido: {type_class!r}."
        )
    gender, is_singles = supported_types[type_class]

    direct_links = name_cell.find_all("a", recursive=False)
    if len(direct_links) > 1:
        raise TennisExplorerSchemaError(
            f"El torneo {name!r} contiene varios enlaces principales."
        )
    href = direct_links[0].get("href") if direct_links else None
    if href is not None:
        if not isinstance(href, str):
            raise TennisExplorerSchemaError(
                f"El enlace del torneo {name!r} no es texto."
            )
        href_match = _TOURNAMENT_HREF_PATTERN.fullmatch(href)
        if href_match is None:
            raise TennisExplorerSchemaError(
                f"Formato de enlace de torneo no reconocido: {href!r}."
            )
        href_year, href_tour = href_match.groups()
        expected_tour = "atp-men" if gender == "M" else "wta-women"
        if int(href_year) != match_date.year or href_tour != expected_tour:
            raise TennisExplorerSchemaError(
                f"El enlace {href!r} contradice la fecha o el género."
            )

    is_generic_futures = bool(_GENERIC_FUTURES_PATTERN.fullmatch(name))
    if href is None and not (gender == "M" and is_generic_futures):
        raise TennisExplorerSchemaError(
            f"Falta el enlace del torneo {name!r}; solo se observó esta "
            "excepción en la cabecera agregada de Futures masculinos."
        )

    is_utr = name.casefold().startswith("utr ") or (
        href is not None
        and urlsplit(href).path.casefold().startswith("/utr-")
    )
    if not is_singles or is_utr:
        return _Tournament(
            include=False,
            name=name,
            href=href,
            gender=gender,
            tour_level=None,
            surface=None,
        )

    tour_level = _classify_tour_level(
        name,
        href=href,
        gender=gender,
        is_generic_futures=is_generic_futures,
    )
    surface = (
        None
        if href is None
        else surfaces_by_tournament.get(href)
    )
    return _Tournament(
        include=True,
        name=name,
        href=href,
        gender=gender,
        tour_level=tour_level,
        surface=surface,
    )


def _classify_tour_level(
    name: str,
    *,
    href: str | None,
    gender: Gender,
    is_generic_futures: bool,
) -> TourLevel:
    """Aplica únicamente los marcadores de nivel observados y aprobados."""

    normalised_name = name.casefold()
    first_path_component = ""
    if href:
        path_parts = urlsplit(href).path.strip("/").split("/")
        first_path_component = path_parts[0].casefold() if path_parts else ""

    has_challenger_marker = (
        normalised_name.endswith(" challenger")
        or first_path_component.endswith("-challenger")
    )
    has_itf_marker = (
        normalised_name.endswith(" itf")
        or first_path_component.endswith("-itf")
    )

    if gender == "M":
        if has_challenger_marker:
            return "Challenger"
        if is_generic_futures or has_itf_marker:
            return "ITF"
        return "ATP"
    if has_itf_marker:
        return "ITF"
    return "WTA"


def _parse_match_pair(
    first_row: Tag,
    second_row: Tag,
    *,
    tournament: _Tournament,
    match_date: date,
) -> dict[str, object]:
    """Convierte las dos filas HTML de un individual en un registro."""

    first_cells = first_row.find_all("td", recursive=False)
    second_cells = second_row.find_all("td", recursive=False)
    if len(first_cells) != 13 or len(second_cells) != 8:
        raise TennisExplorerSchemaError(
            f"El partido {first_row.get('id')!r} tiene "
            f"{len(first_cells)}/{len(second_cells)} celdas; se esperaban 13/8."
        )

    _validate_match_cell_signature(first_cells, second_cells, first_row)
    player_1 = _parse_player(first_cells[1])
    player_2 = _parse_player(second_cells[0])

    time_and_status_text = _visible_status_text(first_cells[0])
    scheduled_time_match = _TIME_PATTERN.search(time_and_status_text)
    scheduled_time: str | None = (
        scheduled_time_match.group(0) if scheduled_time_match else None
    )

    outcome_1 = _normalise_text(first_cells[2].get_text(" ", strip=True))
    outcome_2 = _normalise_text(second_cells[1].get_text(" ", strip=True))
    scores_1 = [
        _normalise_text(cell.get_text(" ", strip=True))
        for cell in first_cells[3:8]
    ]
    scores_2 = [
        _normalise_text(cell.get_text(" ", strip=True))
        for cell in second_cells[2:7]
    ]
    status, evidence = _parse_status(
        time_and_status_text=time_and_status_text,
        outcome_1=outcome_1,
        outcome_2=outcome_2,
        scores_1=scores_1,
        scores_2=scores_2,
        scheduled_time=scheduled_time,
    )
    parsed_result = _parse_result(
        status=status,
        outcome_1=outcome_1,
        outcome_2=outcome_2,
        scores_1=scores_1,
        scores_2=scores_2,
        player_1_slug=player_1.slug,
        player_2_slug=player_2.slug,
    )
    match_detail_href, source_match_id = _parse_match_detail(first_cells[12])

    if tournament.tour_level is None:
        raise TennisExplorerSchemaError(
            "Un torneo incluido no tiene tour_level."
        )

    return {
        "match_date": pd.Timestamp(match_date),
        "tournament": tournament.name,
        "tournament_href": tournament.href,
        "tour_level": tournament.tour_level,
        "gender": tournament.gender,
        "surface": tournament.surface,
        "scheduled_time": scheduled_time,
        "player_1_name": player_1.name,
        "player_2_name": player_2.name,
        "player_1_href": player_1.href,
        "player_2_href": player_2.href,
        "player_1_slug": player_1.slug,
        "player_2_slug": player_2.slug,
        "player_1_has_link": player_1.href is not None,
        "player_2_has_link": player_2.href is not None,
        "player_1_odds": _parse_odds(first_cells[9]),
        "player_2_odds": _parse_odds(first_cells[10]),
        "status": status,
        "status_evidence": evidence,
        "player_1_sets_won": parsed_result.player_1_sets_won,
        "player_2_sets_won": parsed_result.player_2_sets_won,
        "sets_score": parsed_result.sets_score,
        "winner_side": parsed_result.winner_side,
        "winner_slug": parsed_result.winner_slug,
        "result_evidence": parsed_result.evidence,
        "source_match_id": source_match_id,
        "match_detail_href": match_detail_href,
    }


def _validate_match_cell_signature(
    first_cells: list[Tag],
    second_cells: list[Tag],
    first_row: Tag,
) -> None:
    """Valida las clases y posiciones que delimitan una pareja de filas."""

    match_id = first_row.get("id")
    if not {"first", "time"}.issubset(first_cells[0].get("class", [])):
        raise TennisExplorerSchemaError(
            f"El partido {match_id!r} no empieza por td.first.time."
        )
    if first_cells[0].get("rowspan") != "2":
        raise TennisExplorerSchemaError(
            f"La hora del partido {match_id!r} no abarca dos filas."
        )
    if "t-name" not in first_cells[1].get("class", []):
        raise TennisExplorerSchemaError(
            f"Falta td.t-name para el primer jugador de {match_id!r}."
        )
    if "t-name" not in second_cells[0].get("class", []):
        raise TennisExplorerSchemaError(
            f"Falta td.t-name para el segundo jugador de {match_id!r}."
        )

    for outcome_cell in (first_cells[2], second_cells[1]):
        classes = set(outcome_cell.get("class", []))
        if not ({"result", "nbr"} & classes):
            raise TennisExplorerSchemaError(
                f"Resultado estructural no reconocido en {match_id!r}."
            )
    for score_cell in (*first_cells[3:8], *second_cells[2:7]):
        if "score" not in score_cell.get("class", []):
            raise TennisExplorerSchemaError(
                f"Celda de set no reconocida en {match_id!r}."
            )

    odds_cells = first_cells[9:11]
    if any(
        not ({"course", "coursew"} & set(cell.get("class", [])))
        for cell in odds_cells
    ):
        raise TennisExplorerSchemaError(
            f"Las dos cuotas de {match_id!r} no ocupan las posiciones H/A."
        )


def _parse_player(cell: Tag) -> _Player:
    """Extrae el texto visible y, si existe, el href/slug estable del jugador."""

    direct_links = cell.find_all("a", recursive=False)
    if len(direct_links) > 1:
        raise TennisExplorerSchemaError(
            "Una celda de jugador contiene más de un enlace principal."
        )
    if not direct_links:
        visible_name = _SEED_SUFFIX_PATTERN.sub(
            "",
            _normalise_text(cell.get_text(" ", strip=True)),
        )
        if not visible_name:
            raise TennisExplorerSchemaError(
                "Una celda de jugador sin enlace tampoco contiene nombre."
            )
        return _Player(name=visible_name, href=None, slug=None)

    link = direct_links[0]
    href = link.get("href")
    if not isinstance(href, str):
        raise TennisExplorerSchemaError(
            "El enlace de un jugador no contiene href textual."
        )
    href_match = _PLAYER_HREF_PATTERN.fullmatch(href)
    if href_match is None:
        raise TennisExplorerSchemaError(
            f"Formato de enlace de jugador no reconocido: {href!r}."
        )
    visible_name = _normalise_text(link.get_text(" ", strip=True))
    if not visible_name:
        raise TennisExplorerSchemaError(
            f"El enlace de jugador {href!r} no contiene texto visible."
        )
    return _Player(
        name=visible_name,
        href=href,
        slug=href_match.group(1),
    )


def _parse_odds(cell: Tag) -> float | None:
    """Convierte una cuota decimal; conserva una celda vacía como nula."""

    text = _normalise_text(cell.get_text(" ", strip=True))
    if not text:
        return None
    if _ODDS_PATTERN.fullmatch(text) is None:
        raise TennisExplorerSchemaError(
            f"Cuota decimal no reconocida: {text!r}."
        )
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise TennisExplorerSchemaError(
            f"Cuota decimal fuera de rango: {text!r}."
        )
    return value


def _parse_status(
    *,
    time_and_status_text: str,
    outcome_1: str,
    outcome_2: str,
    scores_1: list[str],
    scores_2: list[str],
    scheduled_time: str | None,
) -> tuple[MatchStatus, str]:
    """Determina un estado solo cuando las celdas aportan evidencia suficiente."""

    status_text = _normalise_text(
        " ".join((time_and_status_text, outcome_1, outcome_2))
    )
    if _WALKOVER_PATTERN.search(status_text):
        return "walkover", "explicit_walkover"
    if _CANCELLED_PATTERN.search(status_text):
        return "cancelled", "explicit_cancelled"
    if _LIVE_PATTERN.search(status_text):
        return "in_progress", "explicit_live_or_interrupted"
    if _FINISHED_PATTERN.search(status_text):
        return "finished", "explicit_finished"
    if _SCHEDULED_PATTERN.search(status_text):
        return "scheduled", "explicit_scheduled"

    has_outcome_1 = bool(outcome_1)
    has_outcome_2 = bool(outcome_2)
    any_scores = any((*scores_1, *scores_2))

    if has_outcome_1 and has_outcome_2:
        if not outcome_1.isdigit() or not outcome_2.isdigit():
            return "unknown", "unrecognised_status_text"
        result_pair = (int(outcome_1), int(outcome_2))
        winner_sets = max(result_pair)
        loser_sets = min(result_pair)
        if (
            winner_sets != loser_sets
            and (winner_sets, loser_sets) in _TERMINAL_SET_RESULTS
            and any_scores
        ):
            return "finished", "terminal_result_with_set_scores"
        return "unknown", "nonterminal_or_scoreless_result"
    if has_outcome_1 != has_outcome_2:
        return "unknown", "partial_terminal_result"
    if any_scores:
        return "unknown", "set_scores_without_explicit_live_state"
    if scheduled_time is not None:
        return "scheduled", "scheduled_time_with_empty_result_and_scores"
    if "--:--" in time_and_status_text:
        return "unknown", "undetermined_no_start_time"
    return "unknown", "unrecognised_status_text"


def _optional_sets_won(value: str) -> int | None:
    """Convierte un contador de sets decimal y conserva otros textos como nulos."""

    if not value or not value.isdigit():
        return None
    parsed = int(value)
    if parsed > 3:
        raise TennisExplorerSchemaError(
            f"Contador de sets fuera del rango observado: {value!r}."
        )
    return parsed


def _parse_result(
    *,
    status: MatchStatus,
    outcome_1: str,
    outcome_2: str,
    scores_1: list[str],
    scores_2: list[str],
    player_1_slug: str | None,
    player_2_slug: str | None,
) -> _ParsedResult:
    """Infiere un ganador solo desde sets terminales y un slug estable.

    Un walkover, una cancelación, una retirada no observada explícitamente o
    un marcador parcial nunca se convierten en ganador por orden de fila,
    clase CSS ni cuota. Los contadores numéricos se conservan aunque el
    encuentro aún no sea liquidable.
    """

    sets_1 = _optional_sets_won(outcome_1)
    sets_2 = _optional_sets_won(outcome_2)
    sets_score = (
        None
        if sets_1 is None or sets_2 is None
        else f"{sets_1}-{sets_2}"
    )
    has_scores = any((*scores_1, *scores_2))
    terminal_pair = (
        sets_1 is not None
        and sets_2 is not None
        and sets_1 != sets_2
        and (max(sets_1, sets_2), min(sets_1, sets_2))
        in _TERMINAL_SET_RESULTS
        and has_scores
    )
    if status != "finished":
        return _ParsedResult(
            player_1_sets_won=sets_1,
            player_2_sets_won=sets_2,
            sets_score=sets_score,
            winner_side=None,
            winner_slug=None,
            evidence="winner_not_inferred_nonterminal_or_special_status",
        )
    if not terminal_pair:
        return _ParsedResult(
            player_1_sets_won=sets_1,
            player_2_sets_won=sets_2,
            sets_score=sets_score,
            winner_side=None,
            winner_slug=None,
            evidence="winner_not_inferred_without_terminal_sets",
        )

    assert sets_1 is not None and sets_2 is not None
    if sets_1 > sets_2:
        winner_side: WinnerSide = "player_1"
        winner_slug = player_1_slug
    else:
        winner_side = "player_2"
        winner_slug = player_2_slug
    if winner_slug is None:
        return _ParsedResult(
            player_1_sets_won=sets_1,
            player_2_sets_won=sets_2,
            sets_score=sets_score,
            winner_side=None,
            winner_slug=None,
            evidence="winner_not_inferred_missing_winner_slug",
        )
    return _ParsedResult(
        player_1_sets_won=sets_1,
        player_2_sets_won=sets_2,
        sets_score=sets_score,
        winner_side=winner_side,
        winner_slug=winner_slug,
        evidence="winner_from_terminal_sets_and_slug",
    )


def _parse_match_detail(cell: Tag) -> tuple[str, str]:
    """Extrae el href y el ID fuente opaco que identifican el partido."""

    links = [
        link.get("href")
        for link in cell.find_all("a")
        if isinstance(link.get("href"), str)
        and str(link.get("href")).startswith("/match-detail/")
    ]
    match = (
        None
        if len(links) != 1
        else _MATCH_DETAIL_PATTERN.fullmatch(links[0])
    )
    if match is None:
        raise TennisExplorerSchemaError(
            "El partido no contiene un único enlace /match-detail/?id=NUMERO."
        )
    return links[0], match.group(1)


def _visible_status_text(cell: Tag) -> str:
    """Obtiene texto de hora/estado excluyendo el bloque publicitario streams."""

    pieces: list[str] = []
    for descendant in cell.descendants:
        if not isinstance(descendant, NavigableString):
            continue
        if any(
            isinstance(parent, Tag)
            and "streams" in parent.get("class", [])
            for parent in descendant.parents
            if parent is not cell
        ):
            continue
        text = _normalise_text(str(descendant))
        if text:
            pieces.append(text)
    return _normalise_text(" ".join(pieces))


def _normalise_text(value: str) -> str:
    """Colapsa espacios HTML sin alterar el contenido significativo."""

    return " ".join(value.replace("\xa0", " ").split())


def _to_typed_dataframe(records: list[dict[str, object]]) -> pd.DataFrame:
    """Materializa registros en el orden y dtypes estables del contrato."""

    if not records:
        return empty_matches_dataframe()
    frame = pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)
    for column, dtype in OUTPUT_DTYPES.items():
        frame[column] = frame[column].astype(dtype)
    duplicated_ids = frame["source_match_id"].duplicated(keep=False)
    if bool(duplicated_ids.any()):
        repeated = sorted(
            set(frame.loc[duplicated_ids, "source_match_id"].astype(str))
        )
        raise TennisExplorerSchemaError(
            "La tabla diaria repite source_match_id: "
            f"{repeated[:5]}."
        )
    return frame
