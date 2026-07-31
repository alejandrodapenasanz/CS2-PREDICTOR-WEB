from __future__ import annotations

import re
from typing import Any


def _clean(value: str | None) -> str:
    return " ".join((value or "").strip().split())


def _notes(meta_text: str) -> list[str]:
    notes = []
    for raw in re.split(r"[\r\n]+", meta_text or ""):
        line = _clean(raw)
        if not line:
            continue
        line = re.sub(r"^\*+\s*", "", line).strip()
        if line and not re.match(r"^best\s+of\s+\d+", line, flags=re.I):
            notes.append(line)
    return notes


def parse_match_context_meta(meta_text: str | None) -> dict[str, Any]:
    """Parsea el bloque Maps de HLTV: formato, LAN/Online, stage e incentivo.

    Ejemplos reales:
      Best of 3 (LAN)
      * Swiss round 4 (teams with a 2-1 record). Winner advances to playoffs.
      ** rikon substitutes maQuein.
    """

    raw = meta_text or ""
    text = _clean(raw).lower()
    notes = _notes(raw)
    notes_text = " ".join(notes).lower()
    combined = f"{text} {notes_text}".strip()

    best_of = None
    match = re.search(r"best\s+of\s+(\d+)", combined)
    if match:
        best_of = int(match.group(1))

    environment = "unknown"
    if re.search(r"\(\s*lan\s*\)", combined) or " lan " in f" {combined} ":
        environment = "lan"
    elif re.search(r"\(\s*online\s*\)", combined) or " online " in f" {combined} ":
        environment = "online"

    stage = "unknown"
    if "swiss" in combined:
        stage = "swiss"
    elif "round of 16" in combined or "ro16" in combined:
        stage = "ro16"
    elif "quarter-final" in combined or "quarter final" in combined:
        stage = "quarter"
    elif "semi-final" in combined or "semi final" in combined:
        stage = "semi"
    elif "grand final" in combined or re.search(r"\bfinal\b", combined):
        stage = "final"
    elif "playoff" in combined:
        stage = "playoff"
    elif "group" in combined:
        stage = "group"
    elif "qualifier" in combined:
        stage = "qualifier"
    elif "league" in combined:
        stage = "league"
    elif "upper bracket" in combined:
        stage = "upper_bracket"
    elif "lower bracket" in combined:
        stage = "lower_bracket"

    swiss_round = None
    match = re.search(r"swiss\s+round\s+(\d+)", combined)
    if match:
        swiss_round = int(match.group(1))
    swiss_record = None
    match = re.search(r"teams?\s+with\s+a\s+([0-9]+-[0-9]+)\s+record", combined)
    if match:
        swiss_record = match.group(1)

    bracket = None
    if "upper bracket" in combined:
        bracket = "upper"
    elif "lower bracket" in combined:
        bracket = "lower"

    substitution_notes = [
        note for note in notes
        if re.search(r"\b(substitute|substitutes|stand-?in|replaces|replaced)\b", note, flags=re.I)
    ]
    has_substitution_note = bool(substitution_notes)

    winner_advances = bool(re.search(r"winner\s+(advances|qualifies|proceeds)", combined))
    loser_eliminated = "elimination match" in combined or bool(
        re.search(r"(loser|losing\s+team)\s+(is\s+)?eliminated", combined)
    )
    winners_match = "winners' match" in combined or "winners match" in combined
    opening_match = "opening match" in combined
    placement_match = any(token in combined for token in ("3rd place", "third place", "placement", "consolation"))

    high_stakes = (
        winner_advances
        or loser_eliminated
        or stage in {"quarter", "semi", "final", "playoff", "lower_bracket"}
    )

    if placement_match:
        incentive_label = "placement_match"
        incentive_uncertainty = "high"
    elif loser_eliminated:
        incentive_label = "elimination_match"
        incentive_uncertainty = "low"
    elif winner_advances:
        incentive_label = "winner_advances"
        incentive_uncertainty = "low"
    elif winners_match:
        incentive_label = "winners_match"
        incentive_uncertainty = "medium"
    elif opening_match:
        incentive_label = "opening_match"
        incentive_uncertainty = "medium"
    elif stage in {"quarter", "semi", "final", "playoff", "lower_bracket"}:
        incentive_label = "playoff_or_bracket"
        incentive_uncertainty = "low"
    elif stage in {"group", "swiss", "league", "unknown"}:
        incentive_label = "group_or_swiss"
        incentive_uncertainty = "medium"
    else:
        incentive_label = "unknown"
        incentive_uncertainty = "medium"

    return {
        "raw_meta": raw,
        "best_of": best_of,
        "environment": environment,
        "is_lan": environment == "lan",
        "stage": stage,
        "stage_detail": notes[0] if notes else None,
        "notes": notes,
        "swiss_round": swiss_round,
        "swiss_record": swiss_record,
        "bracket": bracket,
        "winner_advances": winner_advances,
        "loser_eliminated": loser_eliminated,
        "winners_match": winners_match,
        "opening_match": opening_match,
        "placement_match": placement_match,
        "high_stakes": high_stakes,
        "incentive_label": incentive_label,
        "incentive_uncertainty": incentive_uncertainty,
        "has_substitution_note": has_substitution_note,
        "substitution_notes": substitution_notes,
    }


def schema_stage(stage: str | None) -> str:
    mapping = {
        "group": "group",
        "swiss": "swiss",
        "ro16": "ro16",
        "quarter": "quarter",
        "semi": "semi",
        "final": "final",
        "playoff": "other",
        "upper_bracket": "other",
        "lower_bracket": "other",
        "qualifier": "other",
        "league": "other",
    }
    return mapping.get(stage or "", "other")
