"""Adapta resultados laterales de TennisRatio al contrato operativo.

La tabla lateral identifica jugadores mediante IDs Sackmann y claves propias
de TennisRatio. La base operativa, en cambio, liquida exclusivamente contra la
identidad y los slugs congelados por la primera predicción oficial. Este módulo
solo construye una observación cuando ambas identidades pueden unirse de forma
completa e inequívoca; en cualquier otro caso devuelve ``None`` para que el
orquestador use su fuente de resultados de respaldo.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
from typing import Protocol

import pandas as pd

from ..config import PLAYER_MAPPING_DATABASE_PATH
from ..player_mapping import PlayerMappingStore
from ..tennis_explorer import TennisExplorerResultSnapshot
from ..tennisratio import MappedResult, load_mapped_results
from .store import OperationsStore


class MappedResultLoader(Protocol):
    """Define la consulta causal mínima al SQLite lateral de TennisRatio."""

    def __call__(
        self,
        as_of_date: date,
        *,
        base_cutoff_by_gender: Mapping[str, date],
    ) -> Sequence[MappedResult]:
        """Devuelve resultados efectivos y disponibles antes de ``as_of``."""


@dataclass(frozen=True, slots=True)
class TennisRatioResultSnapshot:
    """Agrupa observaciones laterales inequívocas para una jornada pendiente."""

    match_date: date
    retrieved_at_utc: datetime
    source_url: str
    snapshot_sha256: str
    matches: pd.DataFrame
    canonical_match_ids: tuple[str, ...]
    official_pending_count: int
    matched_count: int
    unmatched_count: int


@dataclass(frozen=True, slots=True)
class TennisExplorerMappedResultSnapshot(TennisExplorerResultSnapshot):
    """Snapshot de respaldo reducido a resultados enlazados con seguridad.

    ``raw_snapshot_sha256`` identifica el HTML adquirido. ``snapshot_sha256``
    identifica tambien la traduccion exacta aplicada, por lo que un cambio del
    mapping no puede reutilizar silenciosamente el mismo run. Las filas no
    enlazadas permanecen en el snapshot HTML append-only, pero no se insertan
    como observaciones de partidos desconocidos.
    """

    raw_snapshot_sha256: str
    official_pending_count: int
    terminal_result_count: int
    exact_id_count: int
    identity_mapped_count: int
    paired_inferred_count: int
    unmatched_official_count: int
    unmapped_player_rows: int
    ambiguous_pair_rows: int


@dataclass(frozen=True, slots=True)
class _OfficialMatch:
    """Identidad mínima ya congelada en la predicción oficial."""

    source_match_id: str
    gender: str
    player_a_id: int
    player_b_id: int
    player_a_slug: str
    player_b_slug: str

    @property
    def player_pair(self) -> tuple[int, int]:
        """Devuelve el par no ordenado de IDs Sackmann."""

        return (
            min(self.player_a_id, self.player_b_id),
            max(self.player_a_id, self.player_b_id),
        )


@dataclass(frozen=True, slots=True)
class _ExplorerCandidate:
    """Resultado terminal de Explorer con ambas identidades resueltas."""

    row: dict[str, object]
    player_1_id: int
    player_2_id: int
    player_1_mapping_method: str
    player_2_mapping_method: str
    player_1_mapping_updated_at_utc: datetime
    player_2_mapping_updated_at_utc: datetime
    identity_join: str

    @property
    def player_pair(self) -> tuple[int, int]:
        """Devuelve el par canÃ³nico independiente del orden de la web."""

        return (
            min(self.player_1_id, self.player_2_id),
            max(self.player_1_id, self.player_2_id),
        )


@dataclass(frozen=True, slots=True)
class _PartialExplorerCandidate:
    """Resultado con una identidad exacta que puede anclar la pareja."""

    row: dict[str, object]
    gender: str
    known_player_id: int
    known_side: int
    mapping_method: str
    mapping_updated_at_utc: datetime


def _strict_integer(value: object) -> int | None:
    """Acepta solo IDs enteros reales, sin truncar floats ni booleanos."""

    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, Integral):
        return int(value)
    return None


def _set_integer(value: object) -> int | None:
    """Acepta enteros de Pandas/JSON sin permitir redondeos implÃ­citos."""

    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    numeric = float(value)
    if not numeric.is_integer():
        return None
    return int(numeric)


def _optional_text(value: object) -> str | None:
    """Normaliza texto escalar de una fila Pandas sin inventar valores."""

    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        return None
    text = str(value).strip()
    return text or None


def _terminal_explorer_row(row: dict[str, object]) -> bool:
    """Exige evidencia terminal internamente coherente antes de mapearla."""

    if _optional_text(row.get("status")) != "finished":
        return False
    first_sets = _set_integer(row.get("player_1_sets_won"))
    second_sets = _set_integer(row.get("player_2_sets_won"))
    winner_side = _optional_text(row.get("winner_side"))
    winner_slug = _optional_text(row.get("winner_slug"))
    if (
        first_sets is None
        or second_sets is None
        or first_sets < 0
        or second_sets < 0
        or first_sets == second_sets
        or winner_slug is None
        or _optional_text(row.get("sets_score")) is None
        or _optional_text(row.get("result_evidence")) is None
    ):
        return False
    if winner_side == "player_1":
        return first_sets > second_sets
    if winner_side == "player_2":
        return second_sets > first_sets
    return False


def _load_pending_official_matches(
    store: OperationsStore,
    pending_date: date,
) -> tuple[_OfficialMatch, ...] | None:
    """Carga partidos oficiales sin liquidar o marca el lote incompleto."""

    rows = store.connection.execute(
        """
        SELECT
            m.source_match_id,
            m.gender,
            p.player_a_id,
            p.player_b_id,
            p.player_a_slug,
            p.player_b_slug
        FROM matches AS m
        JOIN official_predictions AS op
            ON op.source_match_id = m.source_match_id
        JOIN predictions AS p
            ON p.prediction_id = op.prediction_id
        LEFT JOIN settlements AS s
            ON s.source_match_id = m.source_match_id
        WHERE m.match_date = ?
          AND s.source_match_id IS NULL
        ORDER BY m.source_match_id
        """,
        (pending_date.isoformat(),),
    ).fetchall()
    if not rows:
        return None

    matches: list[_OfficialMatch] = []
    for row in rows:
        player_a_id = _strict_integer(row["player_a_id"])
        player_b_id = _strict_integer(row["player_b_id"])
        source_match_id = row["source_match_id"]
        gender = row["gender"]
        player_a_slug = row["player_a_slug"]
        player_b_slug = row["player_b_slug"]
        if (
            player_a_id is None
            or player_b_id is None
            or player_a_id == player_b_id
            or not isinstance(source_match_id, str)
            or not source_match_id
            or gender not in {"M", "F"}
            or not isinstance(player_a_slug, str)
            or not player_a_slug
            or not isinstance(player_b_slug, str)
            or not player_b_slug
            or player_a_slug == player_b_slug
        ):
            return None
        matches.append(
            _OfficialMatch(
                source_match_id=source_match_id,
                gender=gender,
                player_a_id=player_a_id,
                player_b_id=player_b_id,
                player_a_slug=player_a_slug,
                player_b_slug=player_b_slug,
            )
        )
    return tuple(matches)


def _tennisratio_slug(source_player_key: str) -> str:
    """Normaliza una clave lateral al namespace público de TennisRatio."""

    key = source_player_key.strip()
    if key.startswith("tennisratio:"):
        return key
    return f"tennisratio:{key}"


def _valid_terminal_sets(result: MappedResult) -> tuple[int, int] | None:
    """Devuelve sets ganador/perdedor solo si la fuente los cerró sin duda."""

    winner_sets = _strict_integer(result.winner_sets_won)
    loser_sets = _strict_integer(result.loser_sets_won)
    if (
        winner_sets is None
        or loser_sets is None
        or winner_sets <= loser_sets
        or loser_sets < 0
        or not isinstance(result.score, str)
        or not result.score.strip()
    ):
        return None
    return winner_sets, loser_sets


def _result_pair(result: MappedResult) -> tuple[int, int] | None:
    """Valida y devuelve el par no ordenado de IDs de un resultado."""

    winner_id = _strict_integer(result.winner_sackmann_id)
    loser_id = _strict_integer(result.loser_sackmann_id)
    if winner_id is None or loser_id is None or winner_id == loser_id:
        return None
    return min(winner_id, loser_id), max(winner_id, loser_id)


def _snapshot_fingerprint(rows: Sequence[dict[str, object]]) -> str:
    """Calcula una huella estable del conjunto exacto de evidencia lateral."""

    payload = [
        {
            "canonical_match_id": row["tennisratio_canonical_match_id"],
            "source_match_id": row["source_match_id"],
            "source_snapshot_sha256": row["source_snapshot_sha256"],
            "winner_slug": row["winner_slug"],
            "winner_side": row["winner_side"],
            "player_1_sets_won": row["player_1_sets_won"],
            "player_2_sets_won": row["player_2_sets_won"],
            "sets_score": row["sets_score"],
            "observed_at_utc": str(row["observed_at_utc"]),
        }
        for row in rows
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _explorer_mapping_fingerprint(
    raw_snapshot_sha256: str,
    rows: Sequence[dict[str, object]],
) -> str:
    """Sella la evidencia web y la traducciÃ³n oficial exacta aplicada."""

    payload = {
        "raw_snapshot_sha256": raw_snapshot_sha256,
        "mapped_rows": [
            {
                "source_match_id": row["source_match_id"],
                "tennis_explorer_source_match_id": row["tennis_explorer_source_match_id"],
                "identity_join": row["identity_join"],
                "winner_slug": row["winner_slug"],
                "winner_side": row["winner_side"],
                "player_1_sets_won": row["player_1_sets_won"],
                "player_2_sets_won": row["player_2_sets_won"],
            }
            for row in rows
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _exact_explorer_observation(
    row: dict[str, object],
    official: _OfficialMatch,
) -> dict[str, object] | None:
    """Conserva un ID nativo solo si su ganador pertenece al partido oficial."""

    winner_slug = _optional_text(row.get("winner_slug"))
    if not _terminal_explorer_row(row) or winner_slug not in {
        official.player_a_slug,
        official.player_b_slug,
    }:
        return None
    translated = dict(row)
    translated.update(
        {
            "source_match_id": official.source_match_id,
            "tennis_explorer_source_match_id": official.source_match_id,
            "identity_join": "stable_source_match_id_exact",
            "official_player_a_id": official.player_a_id,
            "official_player_b_id": official.player_b_id,
        }
    )
    return translated


def _mapped_explorer_observation(
    candidate: _ExplorerCandidate,
    official: _OfficialMatch,
) -> dict[str, object] | None:
    """Orienta un resultado Explorer a los slugs congelados por la predicciÃ³n."""

    row = candidate.row
    first_slug = _optional_text(row.get("player_1_slug"))
    second_slug = _optional_text(row.get("player_2_slug"))
    winner_slug = _optional_text(row.get("winner_slug"))
    winner_side = _optional_text(row.get("winner_side"))
    if (
        first_slug is None
        or second_slug is None
        or winner_slug is None
        or winner_side not in {"player_1", "player_2"}
    ):
        return None
    expected_winner_slug = first_slug if winner_side == "player_1" else second_slug
    if winner_slug != expected_winner_slug:
        return None

    winner_id = candidate.player_1_id if winner_side == "player_1" else candidate.player_2_id
    if winner_id == official.player_a_id:
        official_winner_side = "player_1"
        official_winner_slug = official.player_a_slug
    elif winner_id == official.player_b_id:
        official_winner_side = "player_2"
        official_winner_slug = official.player_b_slug
    else:
        return None

    first_sets = _set_integer(row.get("player_1_sets_won"))
    second_sets = _set_integer(row.get("player_2_sets_won"))
    if first_sets is None or second_sets is None:
        return None
    player_a_is_source_first = candidate.player_1_id == official.player_a_id
    official_first_sets = first_sets if player_a_is_source_first else second_sets
    official_second_sets = second_sets if player_a_is_source_first else first_sets
    source_match_id = _optional_text(row.get("source_match_id"))
    if source_match_id is None:
        return None

    translated = dict(row)
    translated.update(
        {
            "source_match_id": official.source_match_id,
            "player_1_slug": official.player_a_slug,
            "player_2_slug": official.player_b_slug,
            "player_1_sets_won": official_first_sets,
            "player_2_sets_won": official_second_sets,
            "winner_side": official_winner_side,
            "winner_slug": official_winner_slug,
            "result_evidence": "tennis_explorer_identity_mapped_terminal_result",
            "tennis_explorer_source_match_id": source_match_id,
            "tennis_explorer_player_1_slug": first_slug,
            "tennis_explorer_player_2_slug": second_slug,
            "tennis_explorer_winner_slug": winner_slug,
            "identity_join": candidate.identity_join,
            "mapped_player_1_id": candidate.player_1_id,
            "mapped_player_2_id": candidate.player_2_id,
            "player_1_mapping_method": candidate.player_1_mapping_method,
            "player_2_mapping_method": candidate.player_2_mapping_method,
            "player_1_mapping_updated_at_utc": (candidate.player_1_mapping_updated_at_utc),
            "player_2_mapping_updated_at_utc": (candidate.player_2_mapping_updated_at_utc),
        }
    )
    return translated


def build_tennis_explorer_mapped_result_snapshot(
    store: OperationsStore,
    snapshot: TennisExplorerResultSnapshot,
    *,
    mapping_database_path: Path = PLAYER_MAPPING_DATABASE_PATH,
) -> TennisExplorerMappedResultSnapshot:
    """Cruza el fallback con partidos oficiales sin usar nombres aproximados.

    Primero conserva coincidencias por ``source_match_id`` exacto. Para agendas
    procedentes de otra fuente resuelve ambos slugs de Tennis Explorer mediante
    el mapping auditable a Sackmann y exige una relaciÃ³n uno-a-uno por fecha,
    gÃ©nero y par de IDs. Una pareja duplicada en cualquiera de los lados se
    considera ambigua y no produce observaciÃ³n ni settlement.
    """

    official_matches = _load_pending_official_matches(store, snapshot.match_date) or ()
    raw_rows = [dict(row) for row in snapshot.matches.to_dict(orient="records")]
    terminal_rows = [row for row in raw_rows if _terminal_explorer_row(row)]
    raw_by_id: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in terminal_rows:
        source_match_id = _optional_text(row.get("source_match_id"))
        if source_match_id is not None:
            raw_by_id[source_match_id].append(row)

    observations: list[dict[str, object]] = []
    matched_official_ids: set[str] = set()
    used_explorer_ids: set[str] = set()
    for official in official_matches:
        exact_rows = raw_by_id.get(official.source_match_id, [])
        if len(exact_rows) != 1:
            continue
        translated = _exact_explorer_observation(exact_rows[0], official)
        if translated is None:
            continue
        observations.append(translated)
        matched_official_ids.add(official.source_match_id)
        used_explorer_ids.add(official.source_match_id)

    cross_rows = [
        row
        for row in terminal_rows
        if _optional_text(row.get("source_match_id")) not in used_explorer_ids
    ]
    mapping_keys: set[tuple[str, str]] = set()
    for row in cross_rows:
        gender = _optional_text(row.get("gender"))
        first_slug = _optional_text(row.get("player_1_slug"))
        second_slug = _optional_text(row.get("player_2_slug"))
        if gender in {"M", "F"} and first_slug is not None and second_slug is not None:
            mapping_keys.add((gender, first_slug))
            mapping_keys.add((gender, second_slug))

    mappings = {}
    if mapping_keys and Path(mapping_database_path).is_file():
        with PlayerMappingStore(Path(mapping_database_path)) as mapping_store:
            mappings = mapping_store.get_many(mapping_keys)

    candidates_by_key: dict[
        tuple[str, tuple[int, int]],
        list[_ExplorerCandidate],
    ] = defaultdict(list)
    partial_candidates_by_anchor: dict[
        tuple[str, int],
        list[_PartialExplorerCandidate],
    ] = defaultdict(list)
    unmapped_player_rows = 0
    for row in cross_rows:
        gender = _optional_text(row.get("gender"))
        first_slug = _optional_text(row.get("player_1_slug"))
        second_slug = _optional_text(row.get("player_2_slug"))
        if gender not in {"M", "F"} or first_slug is None or second_slug is None:
            unmapped_player_rows += 1
            continue
        first_mapping = mappings.get((gender, first_slug))
        second_mapping = mappings.get((gender, second_slug))
        if first_mapping is None and second_mapping is None:
            unmapped_player_rows += 1
            continue
        if first_mapping is None or second_mapping is None:
            known = first_mapping or second_mapping
            assert known is not None
            partial_candidates_by_anchor[(gender, known.player_id)].append(
                _PartialExplorerCandidate(
                    row=row,
                    gender=gender,
                    known_player_id=known.player_id,
                    known_side=1 if first_mapping is not None else 2,
                    mapping_method=known.resolution_method,
                    mapping_updated_at_utc=known.updated_at_utc,
                )
            )
            continue
        if first_mapping.player_id == second_mapping.player_id:
            unmapped_player_rows += 1
            continue
        candidate = _ExplorerCandidate(
            row=row,
            player_1_id=first_mapping.player_id,
            player_2_id=second_mapping.player_id,
            player_1_mapping_method=first_mapping.resolution_method,
            player_2_mapping_method=second_mapping.resolution_method,
            player_1_mapping_updated_at_utc=first_mapping.updated_at_utc,
            player_2_mapping_updated_at_utc=second_mapping.updated_at_utc,
            identity_join="match_date_gender_unordered_sackmann_ids",
        )
        candidates_by_key[(gender, candidate.player_pair)].append(candidate)

    official_by_key: dict[
        tuple[str, tuple[int, int]],
        list[_OfficialMatch],
    ] = defaultdict(list)
    for official in official_matches:
        if official.source_match_id not in matched_official_ids:
            official_by_key[(official.gender, official.player_pair)].append(official)

    ambiguous_pair_rows = 0
    identity_mapped_count = 0
    for key, candidates in candidates_by_key.items():
        same_pair_officials = official_by_key.get(key, [])
        if len(candidates) != 1 or len(same_pair_officials) != 1:
            if same_pair_officials:
                ambiguous_pair_rows += len(candidates)
            continue
        candidate = candidates[0]
        official = same_pair_officials[0]
        translated = _mapped_explorer_observation(candidate, official)
        if translated is None:
            ambiguous_pair_rows += 1
            continue
        explorer_id = str(translated["tennis_explorer_source_match_id"])
        if explorer_id in used_explorer_ids:
            ambiguous_pair_rows += 1
            continue
        observations.append(translated)
        matched_official_ids.add(official.source_match_id)
        used_explorer_ids.add(explorer_id)
        identity_mapped_count += 1

    official_by_anchor: dict[tuple[str, int], list[_OfficialMatch]] = defaultdict(list)
    for official in official_matches:
        if official.source_match_id in matched_official_ids:
            continue
        official_by_anchor[(official.gender, official.player_a_id)].append(official)
        official_by_anchor[(official.gender, official.player_b_id)].append(official)

    paired_inferred_count = 0
    for anchor, partial_candidates in partial_candidates_by_anchor.items():
        same_anchor_officials = official_by_anchor.get(anchor, [])
        if len(partial_candidates) != 1 or len(same_anchor_officials) != 1:
            if same_anchor_officials:
                ambiguous_pair_rows += len(partial_candidates)
            continue
        partial = partial_candidates[0]
        official = same_anchor_officials[0]
        remaining_player_id = (
            official.player_b_id
            if partial.known_player_id == official.player_a_id
            else official.player_a_id
        )
        if partial.known_side == 1:
            first_id, second_id = partial.known_player_id, remaining_player_id
            first_method, second_method = (
                partial.mapping_method,
                "paired_official_complement",
            )
            first_updated = partial.mapping_updated_at_utc
            second_updated = snapshot.retrieved_at_utc
        else:
            first_id, second_id = remaining_player_id, partial.known_player_id
            first_method, second_method = (
                "paired_official_complement",
                partial.mapping_method,
            )
            first_updated = snapshot.retrieved_at_utc
            second_updated = partial.mapping_updated_at_utc
        candidate = _ExplorerCandidate(
            row=partial.row,
            player_1_id=first_id,
            player_2_id=second_id,
            player_1_mapping_method=first_method,
            player_2_mapping_method=second_method,
            player_1_mapping_updated_at_utc=first_updated,
            player_2_mapping_updated_at_utc=second_updated,
            identity_join=("match_date_gender_single_sackmann_id_official_pair_complement"),
        )
        translated = _mapped_explorer_observation(candidate, official)
        if translated is None:
            ambiguous_pair_rows += 1
            continue
        explorer_id = str(translated["tennis_explorer_source_match_id"])
        if explorer_id in used_explorer_ids or official.source_match_id in matched_official_ids:
            ambiguous_pair_rows += 1
            continue
        observations.append(translated)
        matched_official_ids.add(official.source_match_id)
        used_explorer_ids.add(explorer_id)
        identity_mapped_count += 1
        paired_inferred_count += 1

    observations.sort(key=lambda row: str(row["source_match_id"]))
    mapped_frame = pd.DataFrame(observations)
    if mapped_frame.empty:
        mapped_frame = pd.DataFrame(
            columns=[
                "source_match_id",
                "status",
                "player_1_sets_won",
                "player_2_sets_won",
                "sets_score",
                "winner_side",
                "winner_slug",
                "result_evidence",
            ]
        )
    mapped_sha256 = _explorer_mapping_fingerprint(
        snapshot.snapshot_sha256,
        observations,
    )
    return TennisExplorerMappedResultSnapshot(
        match_date=snapshot.match_date,
        retrieved_at_utc=snapshot.retrieved_at_utc,
        source_url=snapshot.source_url,
        snapshot_sha256=mapped_sha256,
        matches=mapped_frame,
        html_path=snapshot.html_path,
        metadata_path=snapshot.metadata_path,
        raw_snapshot_sha256=snapshot.snapshot_sha256,
        official_pending_count=len(official_matches),
        terminal_result_count=len(terminal_rows),
        exact_id_count=len(observations) - identity_mapped_count,
        identity_mapped_count=identity_mapped_count,
        paired_inferred_count=paired_inferred_count,
        unmatched_official_count=len(official_matches) - len(matched_official_ids),
        unmapped_player_rows=unmapped_player_rows,
        ambiguous_pair_rows=ambiguous_pair_rows,
    )


def build_tennisratio_result_snapshot(
    store: OperationsStore,
    *,
    pending_date: date,
    as_of_date: date,
    loader: MappedResultLoader = load_mapped_results,
) -> TennisRatioResultSnapshot | None:
    """Construye un lote causal e inequívoco para conciliación.

    ``pending_date`` es la fecha efectiva del resultado y ``as_of_date`` es la
    fecha de la ejecución diaria. Ambas condiciones se vuelven a comprobar
    aquí aunque el loader lateral ya las aplique. Para agendas cuyo ID oficial
    pertenece a TennisRatio se aceptan las coincidencias parciales seguras: un
    resultado ausente o ambiguo no invalida los demás y queda pendiente para
    otro run. Para IDs de otra fuente se conserva el contrato completo, de modo
    que su reconciliador nativo pueda seguir actuando como fallback.
    """

    if not pending_date < as_of_date:
        raise ValueError("pending_date debe ser estrictamente anterior a as_of_date.")
    if pending_date == date.min:
        return None
    official_matches = _load_pending_official_matches(store, pending_date)
    if official_matches is None:
        return None

    cutoff = pending_date - timedelta(days=1)
    loaded = loader(
        as_of_date,
        base_cutoff_by_gender={"M": cutoff, "F": cutoff},
    )
    eligible = tuple(
        result
        for result in loaded
        if result.effective_date == pending_date
        and result.effective_date < as_of_date
        and result.available_date < as_of_date
    )
    candidates: dict[tuple[str, tuple[int, int]], list[MappedResult]] = defaultdict(list)
    for result in eligible:
        player_pair = _result_pair(result)
        if result.gender not in {"M", "F"} or player_pair is None:
            continue
        candidates[(result.gender, player_pair)].append(result)

    observations: list[dict[str, object]] = []
    used_canonical_ids: set[str] = set()
    official_by_key: dict[
        tuple[str, tuple[int, int]],
        list[_OfficialMatch],
    ] = defaultdict(list)
    for official in official_matches:
        official_by_key[(official.gender, official.player_pair)].append(official)

    for key, same_pair_officials in official_by_key.items():
        # Dos partidos del mismo par en la misma fecha no se pueden enlazar
        # causalmente con una única evidencia lateral sin inventar identidad.
        if len(same_pair_officials) != 1:
            continue
        official = same_pair_officials[0]
        matches = candidates.get(key, [])
        if len(matches) != 1:
            continue
        result = matches[0]
        if result.canonical_match_id in used_canonical_ids:
            continue
        terminal_sets = _valid_terminal_sets(result)
        if terminal_sets is None:
            continue
        winner_sets, loser_sets = terminal_sets
        if result.winner_sackmann_id == official.player_a_id:
            winner_side = "player_1"
            winner_slug = official.player_a_slug
            player_1_sets, player_2_sets = winner_sets, loser_sets
        elif result.winner_sackmann_id == official.player_b_id:
            winner_side = "player_2"
            winner_slug = official.player_b_slug
            player_1_sets, player_2_sets = loser_sets, winner_sets
        else:
            continue

        source_winner_slug = _tennisratio_slug(result.winner_source_key)
        source_loser_slug = _tennisratio_slug(result.loser_source_key)
        observations.append(
            {
                # La FK operativa debe conservar el ID congelado por la
                # predicción; la identidad canónica lateral queda auditada.
                "source_match_id": official.source_match_id,
                "match_date": pending_date,
                "status": "finished",
                "player_1_sets_won": player_1_sets,
                "player_2_sets_won": player_2_sets,
                "sets_score": result.score.strip(),
                "winner_side": winner_side,
                "winner_slug": winner_slug,
                "result_evidence": "tennisratio_mapped_terminal_result",
                "observed_at_utc": result.first_seen_at_utc,
                "retrieved_at_utc": result.first_seen_at_utc,
                "source_url": result.source_url,
                "source_snapshot_sha256": result.source_sha256,
                "tennisratio_canonical_match_id": result.canonical_match_id,
                "tennisratio_winner_slug": source_winner_slug,
                "tennisratio_loser_slug": source_loser_slug,
            }
        )
        used_canonical_ids.add(result.canonical_match_id)

    if not observations:
        return None
    complete = len(observations) == len(official_matches)
    tennisratio_owned_agenda = all(
        official.source_match_id.startswith("tennisratio:") for official in official_matches
    )
    if not complete and not tennisratio_owned_agenda:
        return None
    observations.sort(key=lambda row: str(row["source_match_id"]))
    retrieved_candidates = [
        row["observed_at_utc"]
        for row in observations
        if isinstance(row["observed_at_utc"], datetime)
    ]
    if len(retrieved_candidates) != len(observations):
        return None
    retrieved_at = max(retrieved_candidates)
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        return None
    source_urls = sorted({str(row["source_url"]) for row in observations})
    return TennisRatioResultSnapshot(
        match_date=pending_date,
        retrieved_at_utc=retrieved_at.astimezone(UTC),
        source_url=source_urls[0],
        snapshot_sha256=_snapshot_fingerprint(observations),
        matches=pd.DataFrame(observations),
        canonical_match_ids=tuple(sorted(used_canonical_ids)),
        official_pending_count=len(official_matches),
        matched_count=len(observations),
        unmatched_count=len(official_matches) - len(observations),
    )
