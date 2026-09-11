"""Pure, strict parsers for the inspected public TennisRatio documents.

The schedule pages currently expose two server-rendered representations:
``.match-card`` for featured matches and ``.compact-row`` for the remainder.
Profile facts are embedded as JavaScript assignments to ``window.playerData``
and ``window.matchesData``.  No browser execution or private endpoint is used.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
import hashlib
import json
import math
import re
from typing import Final, Literal, cast
from urllib.parse import urljoin, urlsplit
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup, Tag
import pandas as pd

from .types import (
    AGENDA_AUDIT_COLUMNS,
    AGENDA_DTYPES,
    AGENDA_OUTPUT_COLUMNS,
    TENNISRATIO_BASE_URL,
    Gender,
    ParsedProfile,
    ProfileMatch,
    ProfileMatchStats,
    SitemapPlayer,
    TennisRatioSchemaError,
)


_PROFILE_PATH_RE: Final[re.Pattern[str]] = re.compile(r"^/players/(?P<slug>[A-Za-z0-9_-]+)\.html$")
_PROFILE_IMAGE_RE: Final[re.Pattern[str]] = re.compile(
    r"^/static/pyBreak/img/profile_pics/(?P<slug>[A-Za-z0-9_-]+)\.jpg$"
)
_MATCH_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9]+$")
_DOB_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9]{8}$")
_SCORE_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<left>[0-9]+)-(?P<right>[0-9]+)(?P<tiebreak>\([^)]*\))?$"
)
_ALLOWED_LEVELS: Final[dict[str, str]] = {
    "ATP": "ATP",
    "WTA": "WTA",
    "Challengers": "Challenger",
    "Futures": "ITF",
    "Grand Slams": "grand_slam",
}
_ALLOWED_SURFACES: Final[frozenset[str]] = frozenset({"Hard", "Clay", "Grass", "Carpet"})
_CANONICAL_SURFACES: Final[dict[str, str]] = {
    surface.casefold(): surface for surface in _ALLOWED_SURFACES
}
_SITEMAP_NAMESPACE: Final[str] = "http://www.sitemaps.org/schemas/sitemap/0.9"
SCHEDULE_CONTEXT_COLUMNS: Final[tuple[str, ...]] = (
    "tournament_level_source",
    "round_source",
    "round_evidence",
)
_SCHEDULE_DTYPES: Final[dict[str, str]] = {
    **AGENDA_DTYPES,
    **dict.fromkeys(SCHEDULE_CONTEXT_COLUMNS, "string"),
}


def empty_agenda_dataframe() -> pd.DataFrame:
    """Return an empty frame with the shared agenda and audit dtypes."""

    columns = AGENDA_OUTPUT_COLUMNS + AGENDA_AUDIT_COLUMNS + SCHEDULE_CONTEXT_COLUMNS
    return pd.DataFrame(
        {column: pd.Series(dtype=_SCHEDULE_DTYPES[column]) for column in columns},
        columns=columns,
    )


def parse_agenda_html(
    html: bytes | str,
    *,
    gender: Gender,
    source_url: str,
    retrieved_at_utc: datetime,
    snapshot_sha256: str | None = None,
) -> pd.DataFrame:
    """Parse all visible schedule dates into the shared daily agenda schema.

    Args:
        html: Complete public ATP or WTA schedule document.
        gender: Domain gender represented by that schedule page.
        source_url: Exact public URL from which the bytes were obtained.
        retrieved_at_utc: Time at which the response became locally available.
        snapshot_sha256: Digest of the original bytes.  It is recomputed when
            omitted, which is convenient for fixture-level tests.

    Returns:
        A typed DataFrame compatible with the Tennis Explorer output columns,
        plus ``source_url``, ``retrieved_at_utc`` and ``snapshot_sha256``.

    Raises:
        TennisRatioSchemaError: If structural evidence is missing, duplicated,
            contradictory or outside the inspected enum values.
    """

    if gender not in {"M", "F"}:
        raise TennisRatioSchemaError(f"Unsupported agenda gender: {gender!r}.")
    raw = _require_document(html)
    retrieved = _require_utc_datetime(retrieved_at_utc)
    sha256 = snapshot_sha256 or hashlib.sha256(raw).hexdigest()
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise TennisRatioSchemaError("snapshot_sha256 must be lowercase SHA-256.")
    _require_schedule_url(source_url, gender)

    soup = BeautifulSoup(raw, "lxml")
    date_groups = soup.select(".matches-date-group[data-date]")
    if not date_groups:
        raise TennisRatioSchemaError("Schedule has no .matches-date-group[data-date].")

    records: list[dict[str, object]] = []
    seen_dates: set[date] = set()
    seen_match_ids: set[str] = set()
    for date_group in date_groups:
        match_date = _parse_iso_date(_required_attr(date_group, "data-date"), "agenda date")
        if match_date in seen_dates:
            raise TennisRatioSchemaError(f"Duplicate schedule date {match_date}.")
        seen_dates.add(match_date)
        tournament_groups = date_group.select(".tournament-group")
        for tournament_group in tournament_groups:
            records.extend(
                _parse_tournament_group(
                    tournament_group,
                    match_date=match_date,
                    gender=gender,
                    source_url=source_url,
                    retrieved_at_utc=retrieved,
                    snapshot_sha256=sha256,
                    seen_match_ids=seen_match_ids,
                )
            )

        orphan_matches = [
            node
            for node in date_group.select(".match-card, .compact-row")
            if node.find_parent(class_="tournament-group") is None
        ]
        if orphan_matches:
            raise TennisRatioSchemaError("Schedule contains matches without tournament context.")

    return _agenda_frame(records)


def parse_player_sitemap(xml: bytes | str) -> tuple[SitemapPlayer, ...]:
    """Parse the official player inventory without accepting foreign URLs."""

    raw = _require_document(xml)
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise TennisRatioSchemaError("Player sitemap is not valid XML.") from exc
    expected_root = f"{{{_SITEMAP_NAMESPACE}}}urlset"
    if root.tag != expected_root:
        raise TennisRatioSchemaError(f"Unexpected sitemap root: {root.tag!r}.")

    result: list[SitemapPlayer] = []
    seen_urls: set[str] = set()
    for url_node in root.findall(f"{{{_SITEMAP_NAMESPACE}}}url"):
        loc_node = url_node.find(f"{{{_SITEMAP_NAMESPACE}}}loc")
        if loc_node is None or not loc_node.text:
            raise TennisRatioSchemaError("Sitemap entry has no loc.")
        source_url = loc_node.text.strip()
        slug = profile_slug_from_url(source_url)
        if source_url in seen_urls:
            raise TennisRatioSchemaError(f"Duplicate sitemap URL: {source_url}.")
        seen_urls.add(source_url)
        lastmod_node = url_node.find(f"{{{_SITEMAP_NAMESPACE}}}lastmod")
        lastmod = None
        if lastmod_node is not None and lastmod_node.text:
            lastmod = _parse_iso_date(lastmod_node.text.strip(), "sitemap lastmod")
        result.append(
            SitemapPlayer(
                source_url=source_url,
                source_player_key=source_player_key(slug),
                profile_slug=slug,
                lastmod=lastmod,
            )
        )
    if not result:
        raise TennisRatioSchemaError("Player sitemap is empty.")
    return tuple(result)


def parse_profile_html(
    html: bytes | str,
    *,
    source_url: str,
) -> ParsedProfile:
    """Parse the strict causal subset of a public player profile.

    TennisRatio serializes missing numerical statistics as JavaScript ``NaN``.
    The parser explicitly maps only that token to ``None`` before retaining the
    original raw snapshot hash; infinities and other constants are rejected.
    """

    raw = _require_document(html)
    url_slug = profile_slug_from_url(source_url)
    soup = BeautifulSoup(raw, "lxml")
    scripts = [script.get_text() for script in soup.find_all("script")]
    script_text = "\n".join(scripts)
    player_data = _extract_window_object(script_text, "playerData")
    matches_data = _extract_window_object(script_text, "matchesData")

    profile_slug = _required_text(player_data, "id", "playerData")
    if profile_slug != url_slug:
        raise TennisRatioSchemaError(
            f"Profile id {profile_slug!r} contradicts URL slug {url_slug!r}."
        )
    player_name = _required_text(player_data, "name", "playerData")
    category = _required_text(player_data, "category", "playerData")
    if category == "ATP":
        gender: Gender = "M"
    elif category == "WTA":
        gender = "F"
    else:
        raise TennisRatioSchemaError(f"Unsupported player category {category!r}.")

    rank = _optional_positive_int(player_data.get("rank"), "playerData.rank")
    dob = _optional_dob(player_data.get("birthdate"))
    country = _optional_text(player_data.get("country"), "playerData.country")
    hand = _optional_text(player_data.get("hand"), "playerData.hand")
    raw_matches = matches_data.get("matches")
    if not isinstance(raw_matches, list):
        raise TennisRatioSchemaError("matchesData.matches must be a list.")
    matches = tuple(_parse_profile_match(item) for item in raw_matches)

    return ParsedProfile(
        source_url=source_url,
        source_player_key=source_player_key(profile_slug),
        profile_slug=profile_slug,
        gender=gender,
        player_name=player_name,
        dob=dob,
        rank=rank,
        country=country,
        hand=hand,
        matches=matches,
        player_payload=player_data,
    )


def parse_robots_txt(content: bytes | str) -> frozenset[str]:
    """Return ``Disallow`` paths for the wildcard user-agent group."""

    raw = _require_document(content)
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise TennisRatioSchemaError("robots.txt is not UTF-8 text.") from exc
    group_agents: set[str] = set()
    group_has_rules = False
    disallowed: set[str] = set()
    saw_agent = False
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        field, value = (part.strip() for part in line.split(":", 1))
        lowered = field.casefold()
        if lowered == "user-agent":
            saw_agent = True
            if group_has_rules:
                group_agents = set()
                group_has_rules = False
            group_agents.add(value.casefold())
        elif lowered in {"allow", "disallow"}:
            group_has_rules = True
            if lowered == "disallow" and "*" in group_agents and value:
                disallowed.add(value)
    if not saw_agent:
        raise TennisRatioSchemaError("robots.txt has no User-agent directive.")
    return frozenset(disallowed)


def source_player_key(profile_slug: str) -> str:
    """Namespace a TennisRatio profile slug for cross-source safety."""

    if not re.fullmatch(r"[A-Za-z0-9_-]+", profile_slug):
        raise TennisRatioSchemaError(f"Invalid profile slug {profile_slug!r}.")
    return f"tennisratio:{profile_slug}"


def profile_slug_from_url(source_url: str) -> str:
    """Validate and extract the slug of an authorized public profile URL."""

    parsed = urlsplit(source_url)
    if (
        parsed.scheme != "https"
        or parsed.netloc.casefold() != "www.tennisratio.com"
        or parsed.query
        or parsed.fragment
    ):
        raise TennisRatioSchemaError(f"Unauthorized profile URL: {source_url!r}.")
    match = _PROFILE_PATH_RE.fullmatch(parsed.path)
    if match is None:
        raise TennisRatioSchemaError(f"Unexpected profile path: {parsed.path!r}.")
    return match.group("slug")


def _parse_tournament_group(
    group: Tag,
    *,
    match_date: date,
    gender: Gender,
    source_url: str,
    retrieved_at_utc: datetime,
    snapshot_sha256: str,
    seen_match_ids: set[str],
) -> list[dict[str, object]]:
    """Parse one tournament container and both inspected match layouts."""

    level_label = _required_attr(group, "data-level")
    surface = _required_attr(group, "data-surface")
    mapped_level = _ALLOWED_LEVELS.get(level_label)
    if mapped_level is None:
        raise TennisRatioSchemaError(f"Unsupported tournament level {level_label!r}.")
    if mapped_level == "grand_slam":
        tour_level = "ATP" if gender == "M" else "WTA"
    else:
        tour_level = mapped_level
    if surface not in _ALLOWED_SURFACES:
        raise TennisRatioSchemaError(f"Unsupported surface {surface!r}.")
    header = group.select_one(":scope > .tournament-header")
    if header is None:
        raise TennisRatioSchemaError("Tournament group has no direct header.")
    title_nodes = header.select(".tournament-title")
    if len(title_nodes) != 1:
        raise TennisRatioSchemaError("Tournament header needs exactly one title.")
    tournament = title_nodes[0].get_text(" ", strip=True)
    if not tournament:
        raise TennisRatioSchemaError("Tournament title is empty.")

    match_nodes = group.select(
        ":scope > .matches-container .match-card, :scope > .matches-compact .compact-row"
    )
    records: list[dict[str, object]] = []
    for node in match_nodes:
        match_id = _required_attr(node, "data-match-id")
        if _MATCH_ID_RE.fullmatch(match_id) is None:
            raise TennisRatioSchemaError(f"Invalid source match id {match_id!r}.")
        namespaced_id = f"tennisratio:{gender}:{match_id}"
        if namespaced_id in seen_match_ids:
            raise TennisRatioSchemaError(f"Duplicate source match id {namespaced_id}.")
        seen_match_ids.add(namespaced_id)
        if _required_attr(node, "data-level") != level_label:
            raise TennisRatioSchemaError(f"Match {match_id} contradicts tournament level.")
        if _required_attr(node, "data-surface") != surface:
            raise TennisRatioSchemaError(f"Match {match_id} contradicts surface.")
        if "match-card" in set(node.get("class", [])):
            players = _featured_players(node)
            utc_text = _unique_attr(node, ".match-time-display[data-utc]", "data-utc")
            detail = _unique_href(node, "a.compare-btn[href]")
        elif "compact-row" in set(node.get("class", [])):
            players = _compact_players(node)
            utc_text = _required_attr(node.select_one(".compact-time[data-utc]"), "data-utc")
            detail = _absolute_public_href(_required_attr(node, "href"))
        else:
            raise TennisRatioSchemaError("Unrecognized match representation.")
        if utc_text == "None":
            scheduled_time: object = pd.NA
            scheduled_start_utc: object = pd.NaT
            status_evidence = "tennisratio:agenda_time_tbd"
        else:
            scheduled = _parse_utc_instant(utc_text, "scheduled time")
            if scheduled.date() != match_date:
                raise TennisRatioSchemaError(
                    f"Match {match_id} UTC date contradicts group date {match_date}."
                )
            scheduled_time = scheduled.strftime("%H:%M")
            scheduled_start_utc = scheduled
            status_evidence = "tennisratio:agenda_scheduled_utc"
        round_raw, round_evidence = _schedule_round(node, header)
        records.append(
            {
                "match_date": pd.Timestamp(match_date),
                "tournament": tournament,
                "tournament_href": pd.NA,
                "tour_level": tour_level,
                "gender": gender,
                "surface": surface,
                "scheduled_time": scheduled_time,
                "scheduled_start_utc": scheduled_start_utc,
                "player_1_name": players[0][0],
                "player_2_name": players[1][0],
                "player_1_href": players[0][1],
                "player_2_href": players[1][1],
                "player_1_slug": source_player_key(players[0][2]),
                "player_2_slug": source_player_key(players[1][2]),
                "player_1_has_link": True,
                "player_2_has_link": True,
                "player_1_odds": pd.NA,
                "player_2_odds": pd.NA,
                "status": "scheduled",
                "status_evidence": status_evidence,
                "player_1_sets_won": pd.NA,
                "player_2_sets_won": pd.NA,
                "sets_score": pd.NA,
                "winner_side": pd.NA,
                "winner_slug": pd.NA,
                "result_evidence": "tennisratio:no_result_on_schedule",
                "source_match_id": namespaced_id,
                "match_detail_href": detail,
                "source_url": source_url,
                "retrieved_at_utc": retrieved_at_utc,
                "snapshot_sha256": snapshot_sha256,
                "tournament_level_source": level_label,
                "round_source": round_raw,
                "round_evidence": round_evidence,
            }
        )
    return records


def _schedule_round(node: Tag, header: Tag) -> tuple[object, str]:
    """Read the inspected per-match label or its exact enclosing draw group.

    The group header describes this group, not the tournament in general.
    Conflicting labels never propagate to compact rows or become a guessed code.
    """

    direct = {
        item.get_text(" ", strip=True)
        for item in node.select(".match-time > span:not([class])")
        if item.get_text(" ", strip=True)
    }
    grouped: set[str] = set()
    for item in header.select(".tournament-meta"):
        match = re.fullmatch(r"(.+?)\s*·\s*\d+\s+matches?", item.get_text(" ", strip=True))
        if match:
            grouped.add(match[1].strip())
    labels = direct | grouped
    sibling_labels = (
        {
            item.get_text(" ", strip=True)
            for item in header.parent.select(".match-time > span:not([class])")
            if item.get_text(" ", strip=True)
        }
        if isinstance(header.parent, Tag)
        else set()
    )
    if grouped and sibling_labels and not sibling_labels.issubset(grouped):
        return pd.NA, "conflict"
    if not labels:
        return pd.NA, "missing"
    if len(labels) != 1:
        return pd.NA, "conflict"
    evidence = (
        "match_and_group" if direct and grouped else ("match" if direct else "tournament_group")
    )
    return next(iter(labels)), evidence


def _featured_players(node: Tag) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
    """Extract two linked players from a featured match card."""

    player_nodes = node.select(".players > .player")
    if len(player_nodes) != 2:
        raise TennisRatioSchemaError("Featured match does not have exactly two players.")
    result: list[tuple[str, str, str]] = []
    for player in player_nodes:
        anchors = player.select("a.player-name[href]")
        if len(anchors) != 1:
            raise TennisRatioSchemaError("Featured player needs exactly one profile link.")
        name = anchors[0].get_text(" ", strip=True)
        href = _absolute_public_href(_required_attr(anchors[0], "href"))
        slug = profile_slug_from_url(href)
        if not name:
            raise TennisRatioSchemaError("Featured player name is empty.")
        result.append((name, href, slug))
    return cast(tuple[tuple[str, str, str], tuple[str, str, str]], tuple(result))


def _compact_players(node: Tag) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
    """Extract compact players, deriving the profile slug from inspected avatars."""

    result: list[tuple[str, str, str]] = []
    for selector in (".compact-player.compact-p1", ".compact-player.compact-p2"):
        players = node.select(selector)
        if len(players) != 1:
            raise TennisRatioSchemaError(f"Compact match needs one node {selector}.")
        name_nodes = players[0].select(".compact-name")
        image_nodes = players[0].select("img.compact-avatar[src]")
        if len(name_nodes) != 1 or len(image_nodes) != 1:
            raise TennisRatioSchemaError("Compact player lacks a unique name/avatar.")
        name = name_nodes[0].get_text(" ", strip=True)
        image_path = urlsplit(_required_attr(image_nodes[0], "src")).path
        match = _PROFILE_IMAGE_RE.fullmatch(image_path)
        if match is None or match.group("slug") == "default":
            raise TennisRatioSchemaError(f"Cannot derive profile slug from {image_path!r}.")
        slug = match.group("slug")
        href = f"{TENNISRATIO_BASE_URL}/players/{slug}.html"
        if not name:
            raise TennisRatioSchemaError("Compact player name is empty.")
        result.append((name, href, slug))
    return cast(tuple[tuple[str, str, str], tuple[str, str, str]], tuple(result))


def _parse_profile_match(value: object) -> ProfileMatch:
    """Validate one completed match embedded in ``matchesData``."""

    if not isinstance(value, dict):
        raise TennisRatioSchemaError("Each matchesData.matches item must be an object.")
    raw = cast(dict[str, object], value)
    effective_date = _parse_iso_date(_required_text(raw, "date", "match"), "match date")
    tournament = _required_text(raw, "tournament", "match")
    tour_level = _required_text(raw, "tournament_level", "match")
    raw_surface = _optional_text(raw.get("surface"), "match.surface")
    if raw_surface is None:
        # An empty source surface remains empty.  Downstream normalisation
        # converts it to None, so it cannot affect a surface-specific Elo.
        surface = ""
    else:
        surface = _CANONICAL_SURFACES.get(raw_surface.casefold(), "")
        if not surface:
            raise TennisRatioSchemaError(f"Unsupported profile match surface {raw_surface!r}.")
    round_name = _required_text(raw, "round", "match")
    rival_name = _required_text(raw, "rival_name", "match")
    result = _required_text(raw, "result", "match")
    if result not in {"Win", "Lose"}:
        raise TennisRatioSchemaError(f"Unsupported match result {result!r}.")
    rival_exists = raw.get("rival_exists")
    rival_slug_raw = raw.get("rival_slugname")
    rival_key: str | None
    if rival_exists is True:
        if not isinstance(rival_slug_raw, str) or not rival_slug_raw:
            raise TennisRatioSchemaError("Existing rival has no rival_slugname.")
        rival_key = source_player_key(rival_slug_raw)
    elif rival_exists in {False, None}:
        rival_key = None
    else:
        raise TennisRatioSchemaError("rival_exists must be boolean or null.")
    score = _optional_text(raw.get("score"), "match.score")
    score_winner, winner_sets, loser_sets = _winner_score(score, result)
    player_odd = _optional_finite_float(raw.get("player_odd"), "match.player_odd")
    rival_odd = _optional_finite_float(raw.get("rival_odd"), "match.rival_odd")
    stats = _parse_profile_match_stats(raw)
    return ProfileMatch(
        effective_date=effective_date,
        tournament=tournament,
        tour_level=tour_level,
        surface=surface,
        round=round_name,
        rival_name=rival_name,
        rival_source_key=rival_key,
        result=cast(Literal["Win", "Lose"], result),
        score=score,
        score_winner_perspective=score_winner,
        winner_sets_won=winner_sets,
        loser_sets_won=loser_sets,
        player_odd=player_odd,
        rival_odd=rival_odd,
        raw_payload=raw,
        stats=stats,
    )


def _parse_profile_match_stats(raw: dict[str, object]) -> ProfileMatchStats | None:
    """Validate the public service/return fields without inventing missing data."""

    breakpoints_saved, breakpoints_faced = _optional_fraction(
        raw.get("breakpoints_saved_count"),
        "match.breakpoints_saved_count",
    )
    service_games_won, service_games_played = _optional_fraction(
        raw.get("service_games_won"),
        "match.service_games_won",
    )
    breakpoints_converted, breakpoint_opportunities = _optional_fraction(
        raw.get("breakpoints_converted_count"),
        "match.breakpoints_converted_count",
    )
    return_games_won, return_games_played = _optional_fraction(
        raw.get("return_games_won"),
        "match.return_games_won",
    )
    stats = ProfileMatchStats(
        first_serve_accuracy_pct=_optional_percentage(
            raw.get("first_serve_accuracy"), "match.first_serve_accuracy"
        ),
        first_serve_points_won_pct=_optional_percentage(
            raw.get("first_serve_points"), "match.first_serve_points"
        ),
        second_serve_points_won_pct=_optional_percentage(
            raw.get("second_serve_points"), "match.second_serve_points"
        ),
        aces=_optional_nonnegative_int(raw.get("aces"), "match.aces"),
        double_faults=_optional_nonnegative_int(raw.get("double_faults"), "match.double_faults"),
        breakpoints_saved=breakpoints_saved,
        breakpoints_faced=breakpoints_faced,
        service_games_won=service_games_won,
        service_games_played=service_games_played,
        return_first_serve_points_won_pct=_optional_percentage(
            raw.get("return_1st_serve_points"), "match.return_1st_serve_points"
        ),
        return_second_serve_points_won_pct=_optional_percentage(
            raw.get("return_2nd_serve_points"), "match.return_2nd_serve_points"
        ),
        breakpoints_converted=breakpoints_converted,
        breakpoint_opportunities=breakpoint_opportunities,
        return_games_won=return_games_won,
        return_games_played=return_games_played,
        serve_pressure_points=_optional_nonnegative_int(
            raw.get("serve_pressure_all"), "match.serve_pressure_all"
        ),
        serve_pressure_points_won=_optional_nonnegative_int(
            raw.get("serve_pressure_won"), "match.serve_pressure_won"
        ),
        return_pressure_points=_optional_nonnegative_int(
            raw.get("return_pressure_all"), "match.return_pressure_all"
        ),
        return_pressure_points_won=_optional_nonnegative_int(
            raw.get("return_pressure_won"), "match.return_pressure_won"
        ),
    )
    if (
        stats.serve_pressure_points is not None
        and stats.serve_pressure_points_won is not None
        and stats.serve_pressure_points_won > stats.serve_pressure_points
    ):
        raise TennisRatioSchemaError("match serve pressure wins exceed opportunities.")
    if (
        stats.return_pressure_points is not None
        and stats.return_pressure_points_won is not None
        and stats.return_pressure_points_won > stats.return_pressure_points
    ):
        raise TennisRatioSchemaError("match return pressure wins exceed opportunities.")
    return stats if stats.available_fields else None


def _winner_score(
    score: str | None,
    result: str,
) -> tuple[str | None, int | None, int | None]:
    """Normalize score tokens to winner/loser orientation without guessing sets.

    Textual suffixes are preserved.  Set totals are returned only when every
    numeric token is a terminal conventional set; retirement and partial-set
    strings therefore remain auditable but have nullable totals.
    """

    if score is None:
        return None, None, None
    normalized_tokens: list[str] = []
    winner_sets = 0
    loser_sets = 0
    all_terminal = True
    saw_numeric = False
    for token in score.split():
        match = _SCORE_TOKEN_RE.fullmatch(token)
        if match is None:
            normalized_tokens.append(token)
            all_terminal = False
            continue
        saw_numeric = True
        left = int(match.group("left"))
        right = int(match.group("right"))
        suffix = match.group("tiebreak") or ""
        if result == "Lose":
            left, right = right, left
        normalized_tokens.append(f"{left}-{right}{suffix}")
        if not _is_terminal_set(left, right):
            all_terminal = False
        elif left > right:
            winner_sets += 1
        else:
            loser_sets += 1
    normalized = " ".join(normalized_tokens)
    if not saw_numeric or not all_terminal or winner_sets <= loser_sets:
        return normalized, None, None
    return normalized, winner_sets, loser_sets


def _is_terminal_set(winner_games: int, loser_games: int) -> bool:
    """Return whether a numeric score is unambiguously a completed set."""

    high = max(winner_games, loser_games)
    low = min(winner_games, loser_games)
    if high == 6 and high - low >= 2:
        return True
    if high == 7 and low in {5, 6}:
        return True
    return high >= 10 and high - low >= 2


def _extract_window_object(script_text: str, variable: str) -> dict[str, object]:
    """Decode one and only one JSON object assigned to a named window field."""

    marker = f"window.{variable}"
    positions = [match.start() for match in re.finditer(re.escape(marker), script_text)]
    if len(positions) != 1:
        raise TennisRatioSchemaError(
            f"Expected exactly one {marker} assignment, found {len(positions)}."
        )
    equals = script_text.find("=", positions[0] + len(marker))
    if equals < 0:
        raise TennisRatioSchemaError(f"{marker} has no assignment operator.")
    decoder = json.JSONDecoder(parse_constant=_parse_json_constant)
    try:
        value, _ = decoder.raw_decode(script_text[equals + 1 :].lstrip())
    except json.JSONDecodeError as exc:
        raise TennisRatioSchemaError(f"{marker} is not decodable JSON.") from exc
    if not isinstance(value, dict):
        raise TennisRatioSchemaError(f"{marker} must contain an object.")
    return cast(dict[str, object], value)


def _parse_json_constant(value: str) -> None:
    """Normalize non-finite source numbers to missing audit-only values."""

    if value in {"NaN", "Infinity", "-Infinity"}:
        return None
    raise TennisRatioSchemaError(f"Unsupported JavaScript constant {value!r}.")


def _agenda_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    """Apply deterministic nullable pandas dtypes to parsed agenda rows."""

    if not records:
        return empty_agenda_dataframe()
    columns = AGENDA_OUTPUT_COLUMNS + AGENDA_AUDIT_COLUMNS + SCHEDULE_CONTEXT_COLUMNS
    frame = pd.DataFrame.from_records(records, columns=columns)
    for column, dtype in _SCHEDULE_DTYPES.items():
        if column in {"retrieved_at_utc", "scheduled_start_utc"}:
            frame[column] = pd.to_datetime(frame[column], utc=True).astype(dtype)
        elif column == "match_date":
            frame[column] = pd.to_datetime(frame[column], errors="raise").astype(dtype)
        else:
            frame[column] = frame[column].astype(dtype)
    return frame


def _require_document(value: bytes | str) -> bytes:
    """Return non-empty source bytes without lossy decoding."""

    if isinstance(value, bytes):
        raw = value
    elif isinstance(value, str):
        raw = value.encode("utf-8")
    else:
        raise TypeError("Document must be bytes or str.")
    if not raw.strip():
        raise TennisRatioSchemaError("Source document is empty.")
    return raw


def _require_schedule_url(source_url: str, gender: Gender) -> None:
    """Reject schedule provenance that does not match its declared gender."""

    expected = f"{TENNISRATIO_BASE_URL}/{'atp' if gender == 'M' else 'wta'}-matches.html"
    if source_url != expected:
        raise TennisRatioSchemaError(f"Schedule URL {source_url!r} does not match gender {gender}.")


def _absolute_public_href(value: str) -> str:
    """Normalize a public site href while rejecting another host or scheme."""

    result = urljoin(f"{TENNISRATIO_BASE_URL}/", value)
    parsed = urlsplit(result)
    if (
        parsed.scheme != "https"
        or parsed.netloc.casefold() != "www.tennisratio.com"
        or parsed.fragment
    ):
        raise TennisRatioSchemaError(f"Unexpected external href {value!r}.")
    return result


def _required_attr(node: Tag | None, name: str) -> str:
    """Read one non-empty scalar HTML attribute."""

    if node is None:
        raise TennisRatioSchemaError(f"Missing node required for attribute {name}.")
    value = node.get(name)
    if not isinstance(value, str) or not value.strip():
        raise TennisRatioSchemaError(f"Missing required attribute {name}.")
    return value.strip()


def _unique_attr(node: Tag, selector: str, attribute: str) -> str:
    """Read an attribute from exactly one descendant."""

    selected = node.select(selector)
    if len(selected) != 1:
        raise TennisRatioSchemaError(f"Expected exactly one node {selector!r}.")
    return _required_attr(selected[0], attribute)


def _unique_href(node: Tag, selector: str) -> str:
    """Return one absolute public href selected under a node."""

    selected = node.select(selector)
    if len(selected) != 1:
        raise TennisRatioSchemaError(f"Expected exactly one link {selector!r}.")
    return _absolute_public_href(_required_attr(selected[0], "href"))


def _required_text(mapping: dict[str, object], key: str, context: str) -> str:
    """Read one required non-empty string from a JSON object."""

    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TennisRatioSchemaError(f"{context}.{key} must be non-empty text.")
    return value.strip()


def _optional_text(value: object, context: str) -> str | None:
    """Validate optional source text without coercion."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise TennisRatioSchemaError(f"{context} must be text or null.")
    stripped = value.strip()
    return stripped or None


def _optional_positive_int(value: object, context: str) -> int | None:
    """Validate an optional positive integral value."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TennisRatioSchemaError(f"{context} must be a positive integer or null.")
    return value


def _optional_finite_float(value: object, context: str) -> float | None:
    """Validate optional audit-only odds without accepting booleans/nonfinite values."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TennisRatioSchemaError(f"{context} must be numeric or null.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise TennisRatioSchemaError(f"{context} must be positive and finite.")
    return result


def _optional_percentage(value: object, context: str) -> float | None:
    """Validate a nullable finite percentage on TennisRatio's 0..100 scale."""

    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TennisRatioSchemaError(f"{context} must be numeric or null.")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 100.0:
        raise TennisRatioSchemaError(f"{context} must be finite and between 0 and 100.")
    return result


def _optional_nonnegative_int(value: object, context: str) -> int | None:
    """Validate a nullable non-negative integral count."""

    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TennisRatioSchemaError(f"{context} must be an integer or null.")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise TennisRatioSchemaError(f"{context} must be a non-negative integer or null.")
    return int(numeric)


def _optional_fraction(value: object, context: str) -> tuple[int | None, int | None]:
    """Split a nullable ``won/opportunities`` source field after validation."""

    if value is None or value == "":
        return None, None
    if not isinstance(value, str) or re.fullmatch(r"[0-9]+/[0-9]+", value) is None:
        raise TennisRatioSchemaError(f"{context} must use non-negative N/D text or null.")
    numerator, denominator = (int(part) for part in value.split("/", 1))
    if numerator > denominator:
        raise TennisRatioSchemaError(f"{context} numerator cannot exceed denominator.")
    return numerator, denominator


def _optional_dob(value: object) -> date | None:
    """Parse optional YYYYMMDD profile birth dates."""

    if value is None or value == "":
        return None
    if not isinstance(value, str) or _DOB_RE.fullmatch(value) is None:
        raise TennisRatioSchemaError("playerData.birthdate must be YYYYMMDD or null.")
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError as exc:
        raise TennisRatioSchemaError("playerData.birthdate is not a calendar date.") from exc


def _parse_iso_date(value: str, context: str) -> date:
    """Parse an exact ISO calendar date."""

    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise TennisRatioSchemaError(f"Invalid {context}: {value!r}.") from exc
    if parsed.isoformat() != value:
        raise TennisRatioSchemaError(f"Non-canonical {context}: {value!r}.")
    return parsed


def _parse_utc_instant(value: str, context: str) -> datetime:
    """Parse an ISO instant and normalize it to aware UTC."""

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TennisRatioSchemaError(f"Invalid {context}: {value!r}.") from exc
    if parsed.tzinfo is None:
        raise TennisRatioSchemaError(f"{context} must include a timezone.")
    return parsed.astimezone(UTC)


def _require_utc_datetime(value: datetime) -> datetime:
    """Normalize an aware timestamp to UTC and reject naive clocks."""

    if not isinstance(value, datetime) or value.tzinfo is None:
        raise TennisRatioSchemaError("retrieved_at_utc must be timezone-aware.")
    return value.astimezone(UTC)
