"""Select source profiles for a schedule date, not a canonical identity/feature mapping.

Only unique full-name + gender matches against inspected Tennis Abstract links
are eligible for acquisition. No guessed URLs, initials or fuzzy matching.
Same-day schedule evidence may choose a download, never a same-day model input.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Any

import pandas as pd

from .player_mapping.normalization import normalize_name_text
from .tennis_abstract_daily import InventoryPlayer, load_inventory, utc_now
from .tennisratio import load_active_agenda


@dataclass(frozen=True)
class AgendaSelection:
    """Selected inspected profiles and visible gaps in a dated acquisition scope."""

    players: tuple[InventoryPlayer, ...]
    report: dict[str, Any]


def select_agenda_players(
    agenda: pd.DataFrame,
    inventory: list[InventoryPlayer],
    match_date: date,
    *,
    observed_at: datetime,
) -> AgendaSelection:
    """Deduplicate today's participants and leave missing/ambiguous names unresolved."""

    observed = pd.Timestamp(observed_at)
    if observed.tzinfo is None:
        raise ValueError("Agenda selection requires an aware observation time.")
    required = {"match_date", "gender", "retrieved_at_utc", "source_url", "snapshot_sha256"}
    required.update(f"player_{side}_{field}" for side in (1, 2) for field in ("name", "href"))
    if not required.issubset(agenda.columns):
        raise ValueError("Agenda does not satisfy the published TennisRatio selection contract.")
    indices: dict[tuple[str, str], list[InventoryPlayer]] = defaultdict(list)
    for player in inventory:
        indices[(player.gender, normalize_name_text(player.name))].append(player)
    dated = agenda.loc[pd.to_datetime(agenda["match_date"]).dt.date.eq(match_date)]
    participants: dict[tuple[str, str], dict[str, Any]] = {}
    source_identities: dict[tuple[str, str], set[str]] = defaultdict(set)
    unusable_rows = 0
    for row in dated.to_dict("records"):
        captured = pd.to_datetime(row["retrieved_at_utc"], errors="coerce", utc=True)
        if pd.isna(captured) or captured > observed or row["gender"] not in {"M", "F"}:
            unusable_rows += 1
            continue
        for side in (1, 2):
            name = row[f"player_{side}_name"]
            href = row[f"player_{side}_href"]
            if not isinstance(name, str) or not name.strip():
                name = ""
            normalized = normalize_name_text(name) if name else ""
            gender = str(row["gender"])
            source_identity = str(href) if isinstance(href, str) and href else normalized
            key = (gender, source_identity)
            participant = participants.setdefault(
                key,
                {
                    "gender": gender,
                    "name": name,
                    "normalized_names": set(),
                    "source_identity": source_identity,
                    "evidence": [],
                },
            )
            participant["normalized_names"].add(normalized)
            source_identities[(gender, normalized)].add(source_identity)
            participant["evidence"].append(
                {
                    "match_date": match_date.isoformat(),
                    "agenda_player_name": name,
                    "agenda_player_href": source_identity,
                    "source_url": str(row["source_url"]),
                    "snapshot_sha256": str(row["snapshot_sha256"]),
                    "retrieved_at_utc": captured.isoformat(),
                }
            )
    selected: dict[tuple[str, str], InventoryPlayer] = {}
    unresolved: list[dict[str, Any]] = []
    for participant in participants.values():
        names = participant["normalized_names"]
        normalized = next(iter(names)) if len(names) == 1 else ""
        key = (participant["gender"], normalized)
        candidates = indices.get(key, [])
        reason = None
        if (
            not normalized
            or len(normalized.split()) < 2
            or any(len(p) == 1 for p in normalized.split())
        ):
            reason = "full_name_required_no_initial_expansion"
        elif len(names) != 1 or len(source_identities[key]) != 1 or len(candidates) > 1:
            reason = "ambiguous_name_or_source_identity"
        elif not candidates:
            reason = "not_in_inspected_tennis_abstract_inventory"
        if reason:
            unresolved.append(
                {
                    "gender": participant["gender"],
                    "name": participant["name"],
                    "source_identity": participant["source_identity"],
                    "reason": reason,
                }
            )
            continue
        player = candidates[0]
        selected[(player.gender, player.key)] = replace(
            player, selection_evidence=tuple(participant["evidence"])
        )
    players = tuple(sorted(selected.values(), key=lambda p: (p.rank, p.gender, p.key)))
    report = {
        "scope": "daily_agenda",
        "match_date": match_date.isoformat(),
        "agenda_source": "tennisratio_published_local",
        "agenda_rows": len(dated),
        "unusable_agenda_rows": unusable_rows,
        "participant_count": len(participants),
        "selected_players": [{"gender": p.gender, "key": p.key, "name": p.name} for p in players],
        "selected_count": len(players),
        "unresolved_players": unresolved,
        "identity_policy": "unique_full_name_gender_acquisition_only_not_canonical_mapping",
        "model_ready": False,
    }
    return AgendaSelection(players, report)


def load_daily_selection(match_date: date) -> AgendaSelection:
    """Use the existing source-owned local agenda API, without downloading another inventory."""

    return select_agenda_players(
        load_active_agenda(match_date), load_inventory(), match_date, observed_at=utc_now()
    )
