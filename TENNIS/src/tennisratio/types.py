"""Public data contracts for the causal TennisRatio sidecar.

The source uses ``ATP``/``WTA`` labels, while the TENNIS domain represents
gender as ``M``/``F``.  Dates in the mapped APIs are deliberately split into
an effective date and an availability date: both must be strictly before a
prediction date before an observation is admissible.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Final, Literal


Gender = Literal["M", "F"]
SourceCategory = Literal["ATP", "WTA"]
RefreshStatus = Literal[
    "published",
    "published_with_profile_failures",
    "already_refreshed",
]
IdentityStatus = Literal["mapped", "unmapped", "ambiguous"]

AGENDA_OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "match_date",
    "tournament",
    "tournament_href",
    "tour_level",
    "gender",
    "surface",
    "scheduled_time",
    "scheduled_start_utc",
    "player_1_name",
    "player_2_name",
    "player_1_href",
    "player_2_href",
    "player_1_slug",
    "player_2_slug",
    "player_1_has_link",
    "player_2_has_link",
    "player_1_odds",
    "player_2_odds",
    "status",
    "status_evidence",
    "player_1_sets_won",
    "player_2_sets_won",
    "sets_score",
    "winner_side",
    "winner_slug",
    "result_evidence",
    "source_match_id",
    "match_detail_href",
)

AGENDA_AUDIT_COLUMNS: Final[tuple[str, ...]] = (
    "source_url",
    "retrieved_at_utc",
    "snapshot_sha256",
)

AGENDA_DTYPES: Final[dict[str, str]] = {
    "match_date": "datetime64[ns]",
    "tournament": "string",
    "tournament_href": "string",
    "tour_level": "string",
    "gender": "string",
    "surface": "string",
    "scheduled_time": "string",
    "scheduled_start_utc": "datetime64[ns, UTC]",
    "player_1_name": "string",
    "player_2_name": "string",
    "player_1_href": "string",
    "player_2_href": "string",
    "player_1_slug": "string",
    "player_2_slug": "string",
    "player_1_has_link": "boolean",
    "player_2_has_link": "boolean",
    "player_1_odds": "Float64",
    "player_2_odds": "Float64",
    "status": "string",
    "status_evidence": "string",
    "player_1_sets_won": "Int64",
    "player_2_sets_won": "Int64",
    "sets_score": "string",
    "winner_side": "string",
    "winner_slug": "string",
    "result_evidence": "string",
    "source_match_id": "string",
    "match_detail_href": "string",
    "source_url": "string",
    "retrieved_at_utc": "datetime64[ns, UTC]",
    "snapshot_sha256": "string",
}

TENNISRATIO_BASE_URL: Final[str] = "https://www.tennisratio.com"
LICENSE_URL: Final[str] = "https://creativecommons.org/licenses/by-nc/4.0/"
ATTRIBUTION: Final[str] = "TennisRatio.com, CC BY-NC 4.0"


class TennisRatioError(RuntimeError):
    """Base class for controlled TennisRatio acquisition failures."""


class TennisRatioSchemaError(TennisRatioError):
    """Raised when a response does not satisfy the inspected source schema."""


class TennisRatioHttpError(TennisRatioError):
    """Raised for a rejected URL or an invalid HTTP response."""


class TennisRatioBlockedError(TennisRatioHttpError):
    """Stop the batch after exhausted rate-limit retries or an actual WAF."""


class TennisRatioLockError(TennisRatioError):
    """Raised when another cross-platform refresh lock is already held."""


class TennisRatioStoreError(TennisRatioError):
    """Raised when the append-only sidecar cannot preserve its contract."""


@dataclass(frozen=True, slots=True)
class HttpPayload:
    """Validated bytes returned by one authorized HTTP GET."""

    source_url: str
    content: bytes
    content_type: str
    retrieved_at_utc: datetime
    sha256: str
    etag: str | None = None
    last_modified: str | None = None


@dataclass(frozen=True, slots=True)
class SitemapPlayer:
    """One allowed player profile advertised by the official sitemap."""

    source_url: str
    source_player_key: str
    profile_slug: str
    lastmod: date | None


@dataclass(frozen=True, slots=True)
class ProfileMatchStats:
    """Validated per-player statistics for one completed match.

    Percentages are stored on the source's 0..100 scale.  Fraction fields are
    split into numerator/denominator pairs so downstream code never needs to
    interpret strings such as ``"5/7"``.
    """

    first_serve_accuracy_pct: float | None
    first_serve_points_won_pct: float | None
    second_serve_points_won_pct: float | None
    aces: int | None
    double_faults: int | None
    breakpoints_saved: int | None
    breakpoints_faced: int | None
    service_games_won: int | None
    service_games_played: int | None
    return_first_serve_points_won_pct: float | None
    return_second_serve_points_won_pct: float | None
    breakpoints_converted: int | None
    breakpoint_opportunities: int | None
    return_games_won: int | None
    return_games_played: int | None
    serve_pressure_points: int | None
    serve_pressure_points_won: int | None
    return_pressure_points: int | None
    return_pressure_points_won: int | None

    @property
    def available_fields(self) -> int:
        """Count non-null scalar fields for coverage diagnostics."""

        return sum(
            value is not None
            for value in (
                self.first_serve_accuracy_pct,
                self.first_serve_points_won_pct,
                self.second_serve_points_won_pct,
                self.aces,
                self.double_faults,
                self.breakpoints_saved,
                self.breakpoints_faced,
                self.service_games_won,
                self.service_games_played,
                self.return_first_serve_points_won_pct,
                self.return_second_serve_points_won_pct,
                self.breakpoints_converted,
                self.breakpoint_opportunities,
                self.return_games_won,
                self.return_games_played,
                self.serve_pressure_points,
                self.serve_pressure_points_won,
                self.return_pressure_points,
                self.return_pressure_points_won,
            )
        )


@dataclass(frozen=True, slots=True)
class ProfileMatch:
    """A completed match embedded in a player profile.

    Odds are retained here only so the store can preserve the raw audit
    evidence.  Mapped result APIs intentionally never expose them as model
    inputs.
    """

    effective_date: date
    tournament: str
    tour_level: str
    surface: str
    round: str
    rival_name: str
    rival_source_key: str | None
    result: Literal["Win", "Lose"]
    score: str | None
    score_winner_perspective: str | None
    winner_sets_won: int | None
    loser_sets_won: int | None
    player_odd: float | None
    rival_odd: float | None
    raw_payload: dict[str, object]
    stats: ProfileMatchStats | None = None


@dataclass(frozen=True, slots=True)
class ParsedProfile:
    """Strict subset of ``window.playerData`` and ``window.matchesData``."""

    source_url: str
    source_player_key: str
    profile_slug: str
    gender: Gender
    player_name: str
    dob: date | None
    rank: int | None
    country: str | None
    hand: str | None
    matches: tuple[ProfileMatch, ...]
    player_payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class MappedRanking:
    """A uniquely mapped ranking admissible only after its availability date.

    ``first_seen_at_utc`` is the mapping-complete instant: it is no earlier
    than source capture, batch publication, or the identity resolution.
    """

    gender: Gender
    sackmann_player_id: int
    source_player_key: str
    player_name: str
    rank: int
    effective_date: date
    available_date: date
    first_seen_at_utc: datetime
    source_url: str
    source_sha256: str


@dataclass(frozen=True, slots=True)
class MappedIdentity:
    """Latest unique Sackmann identity for a namespaced schedule participant.

    Its first-seen instant includes source observation and batch publication,
    not merely the time at which the mapping row was staged.
    """

    gender: Gender
    source_player_key: str
    sackmann_player_id: int
    first_seen_at_utc: datetime
    source_url: str
    source_sha256: str


@dataclass(frozen=True, slots=True)
class IdentityDecision:
    """Exact, append-only outcome of mapping one profile to Sackmann."""

    status: IdentityStatus
    sackmann_player_id: int | None
    method: str
    candidate_count: int


@dataclass(frozen=True, slots=True)
class IdentityRemapCandidate:
    """Latest published profile fact and its currently published resolution."""

    player_observation_id: str
    gender: Gender
    source_player_key: str
    player_name: str
    dob: date | None
    source_url: str
    source_sha256: str
    current_status: IdentityStatus
    current_sackmann_player_id: int | None
    current_method: str
    current_candidate_count: int


@dataclass(frozen=True, slots=True)
class GoodProfileSnapshot:
    """Newest published, successfully parsed raw profile object."""

    source_player_key: str
    source_sha256: str
    compressed_path: Path


@dataclass(frozen=True, slots=True)
class MappedResult:
    """A canonical completed result with unique Sackmann identities.

    ``first_seen_at_utc`` is the mapping-complete available instant.  ``score``
    and the optional set totals are always oriented winner-to-loser.
    """

    canonical_match_id: str
    gender: Gender
    effective_date: date
    available_date: date
    first_seen_at_utc: datetime
    tournament: str
    tour_level: str
    surface: str
    round: str
    winner_sackmann_id: int
    loser_sackmann_id: int
    winner_source_key: str
    loser_source_key: str
    winner_name: str
    loser_name: str
    score: str | None
    winner_sets_won: int | None
    loser_sets_won: int | None
    source_url: str
    source_sha256: str


@dataclass(frozen=True, slots=True)
class MappedMatchStats:
    """One mapped player-match stat observation admissible before a cutoff."""

    gender: Gender
    sackmann_player_id: int
    source_player_key: str
    effective_date: date
    available_date: date
    first_seen_at_utc: datetime
    surface: str
    tour_level: str
    stats: ProfileMatchStats
    source_url: str
    source_sha256: str


@dataclass(frozen=True, slots=True)
class RefreshReport:
    """Machine-readable outcome of one UTC-daily refresh attempt."""

    status: RefreshStatus
    utc_date: date
    retrieved_at_utc: datetime
    already_refreshed: bool
    agenda_rows: int
    profiles_selected: int
    profiles_attempted: int
    profiles_succeeded: int
    profiles_failed: int
    profiles_unchanged: int
    inventory_entries: int
    canonical_fingerprint: str
    manifest_path: Path
    database_path: Path
    identity_remaps: int = 0

    @property
    def changed(self) -> bool:
        """Return whether this invocation published a new last-good batch."""

        return not self.already_refreshed or self.identity_remaps > 0


@dataclass(frozen=True, slots=True)
class IdentityRemapReport:
    """Outcome of a local Sackmann identity reevaluation with no HTTP GET."""

    batch_id: str
    observed_at_utc: datetime
    published_at_utc: datetime
    candidates: int
    remaps: int
    master_sha256_by_gender: dict[Gender, str]
    canonical_fingerprint: str
    published: bool

    @property
    def changed(self) -> bool:
        """Return whether at least one new resolution became visible."""

        return self.remaps > 0
