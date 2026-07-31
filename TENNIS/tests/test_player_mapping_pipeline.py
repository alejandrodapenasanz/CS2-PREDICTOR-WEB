"""Tests offline del resolvedor diario Tennis Explorer -> Sackmann.

Las pruebas inyectan maestros e índices pequeños con identidades reales. No
leen los datos crudos del proyecto, no acceden a la red y crean todos sus
artefactos temporales dentro de ``TENNIS/tests``.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.player_mapping import (  # noqa: E402
    ActiveCandidate,
    PlayerMappingStore,
    build_sackmann_name_key,
    load_unresolved_players,
    resolve_scraped_matches,
    summarize_mapping_coverage,
)


TESTS_ROOT = Path(__file__).resolve().parent
BATCH_DATE = date(2026, 7, 30)
FIXED_CLOCK = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)


def _temporary_test_directory() -> TemporaryDirectory[str]:
    """Crea un directorio temporal aislado bajo la carpeta de tests."""

    return TemporaryDirectory(prefix="player-mapping-pipeline-", dir=TESTS_ROOT)


def _candidate(
    player_id: int,
    first: str,
    last: str,
    *,
    gender: str = "M",
    ioc: str = "ESP",
) -> ActiveCandidate:
    """Construye un candidato activo realista y correctamente normalizado."""

    return ActiveCandidate(
        gender=gender,  # type: ignore[arg-type]
        player_id=player_id,
        player_name=f"{first} {last}",
        name_first=first,
        name_last=last,
        ioc=ioc,
        last_match_date=date(2026, 6, 1),
        name_key=build_sackmann_name_key(first, last),
    )


def _candidate_index(
    *candidates: ActiveCandidate,
) -> dict[object, tuple[ActiveCandidate, ...]]:
    """Agrupa candidatos conservando explícitamente las colisiones."""

    grouped: dict[object, list[ActiveCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.name_key, []).append(candidate)
    return {
        key: tuple(sorted(values, key=lambda value: value.player_id))
        for key, values in grouped.items()
    }


def _players_frame(*candidates: ActiveCandidate) -> pd.DataFrame:
    """Crea el subconjunto de maestro requerido por el pipeline."""

    records = [
        {
            "gender": candidate.gender,
            "player_id": candidate.player_id,
            "player_name": candidate.player_name,
            "ioc": candidate.ioc,
        }
        for candidate in candidates
    ]
    return pd.DataFrame.from_records(records)


def _daily_frame(
    rows: list[dict[str, object]],
) -> pd.DataFrame:
    """Completa registros de partido con fecha y metadatos diarios."""

    completed: list[dict[str, object]] = []
    for row_number, row in enumerate(rows, start=1):
        record = {
            "match_date": pd.Timestamp(BATCH_DATE),
            "tournament": "Madrid Test",
            "tour_level": "ATP",
            "gender": "M",
            "player_1_name": f"Playerone{row_number} P.",
            "player_1_slug": f"playerone-{row_number}",
            "player_2_name": f"Playertwo{row_number} P.",
            "player_2_slug": f"playertwo-{row_number}",
        }
        record.update(row)
        completed.append(record)
    return pd.DataFrame.from_records(completed)


def _write_empty_overrides(path: Path) -> None:
    """Escribe la cabecera exacta de una tabla de overrides vacía."""

    path.write_text(
        "gender,slug,player_id,reason\n",
        encoding="utf-8",
        newline="",
    )


def _resolve_in_directory(
    directory: Path,
    matches: pd.DataFrame,
    *,
    candidates: tuple[ActiveCandidate, ...],
    overrides_text: str = "gender,slug,player_id,reason\n",
) -> pd.DataFrame:
    """Ejecuta el pipeline con rutas e identidades totalmente inyectadas."""

    overrides_path = directory / "overrides.csv"
    overrides_path.write_text(
        overrides_text,
        encoding="utf-8",
        newline="",
    )
    genders = {str(value) for value in matches["gender"]}
    indexes = {
        gender: _candidate_index(
            *(
                candidate
                for candidate in candidates
                if candidate.gender == gender
            )
        )
        for gender in genders
    }
    return resolve_scraped_matches(
        matches,
        as_of_date=BATCH_DATE,
        database_path=directory / "mapping.sqlite3",
        overrides_path=overrides_path,
        unresolved_path=directory / "unresolved.csv",
        players=_players_frame(*candidates),
        candidate_indexes=indexes,
        clock=lambda: FIXED_CLOCK,
    )


class PlayerMappingPipelineRealNamesTest(unittest.TestCase):
    """Comprueba los nombres compuestos, acentos y el aislamiento solicitado."""

    def test_easy_compound_and_accented_real_names_map_exactly(self) -> None:
        """Resuelve cuatro casos reales sin fuzzy ni recortar apellidos."""

        bautista = _candidate(105138, "Roberto", "Bautista Agut", ioc="ESP")
        auger = _candidate(200000, "Felix", "Auger Aliassime", ioc="CAN")
        zandschulp = _candidate(
            122298,
            "Botic",
            "Van De Zandschulp",
            ioc="NED",
        )
        cerundolo = _candidate(
            202103,
            "Francisco",
            "Cerundolo",
            ioc="ARG",
        )
        matches = _daily_frame(
            [
                {
                    "player_1_name": "Bautista Agut R.",
                    "player_1_slug": "bautista-agut-roberto",
                    "player_2_name": "Auger-Aliassime F.",
                    "player_2_slug": "auger-aliassime-felix",
                },
                {
                    "player_1_name": "Van De Zandschulp B.",
                    "player_1_slug": "van-de-zandschulp-botic",
                    "player_2_name": "Cerúndolo F.",
                    "player_2_slug": "cerundolo-francisco",
                },
            ]
        )

        with _temporary_test_directory() as temporary:
            mapped = _resolve_in_directory(
                Path(temporary),
                matches,
                candidates=(bautista, auger, zandschulp, cerundolo),
            )

        self.assertEqual(
            mapped["player_1_id"].tolist(),
            [105138, 122298],
        )
        self.assertEqual(
            mapped["player_2_id"].tolist(),
            [200000, 202103],
        )
        self.assertEqual(
            set(mapped["mapping_status"]),
            {"mapped"},
        )
        self.assertEqual(
            set(mapped["player_1_mapping_method"]),
            {"automatic_name"},
        )

    def test_same_name_key_never_crosses_gender(self) -> None:
        """Usa el universo femenino aunque exista el mismo nombre masculino."""

        woman = _candidate(
            220416,
            "Moyuka",
            "Uchijima",
            gender="F",
            ioc="JPN",
        )
        man = _candidate(
            999001,
            "Moyuka",
            "Uchijima",
            gender="M",
            ioc="JPN",
        )
        opponent = _candidate(
            216347,
            "Iga",
            "Swiatek",
            gender="F",
            ioc="POL",
        )
        matches = _daily_frame(
            [
                {
                    "gender": "F",
                    "tour_level": "WTA",
                    "player_1_name": "Uchijima M.",
                    "player_1_slug": "uchijima-moyuka",
                    "player_2_name": "Swiatek I.",
                    "player_2_slug": "swiatek-iga",
                }
            ]
        )

        with _temporary_test_directory() as temporary:
            mapped = _resolve_in_directory(
                Path(temporary),
                matches,
                candidates=(woman, man, opponent),
            )

        self.assertEqual(mapped.loc[0, "player_1_id"], 220416)
        self.assertEqual(mapped.loc[0, "player_2_id"], 216347)


class PlayerMappingPipelineDegradationTest(unittest.TestCase):
    """Comprueba overrides, colisiones y continuidad del lote diario."""

    def test_collision_is_resolved_by_manual_override_and_audited(self) -> None:
        """Usa un override motivado para dos Boyer activos del mismo país."""

        tristan = _candidate(207729, "Tristan", "Boyer", ioc="USA")
        toby = _candidate(208142, "Toby", "Boyer", ioc="USA")
        michelsen = _candidate(210506, "Alex", "Michelsen", ioc="USA")
        matches = _daily_frame(
            [
                {
                    "player_1_name": "Boyer T.",
                    "player_1_slug": "boyer-tristan",
                    "player_2_name": "Michelsen A.",
                    "player_2_slug": "michelsen-alex",
                }
            ]
        )
        overrides = (
            "gender,slug,player_id,reason\n"
            "M,boyer-tristan,207729,Perfil verificado manualmente\n"
        )

        with _temporary_test_directory() as temporary:
            directory = Path(temporary)
            mapped = _resolve_in_directory(
                directory,
                matches,
                candidates=(tristan, toby, michelsen),
                overrides_text=overrides,
            )
            with PlayerMappingStore(
                directory / "mapping.sqlite3"
            ) as store:
                stored = store.get("M", "boyer-tristan")

        self.assertEqual(mapped.loc[0, "player_1_id"], 207729)
        self.assertEqual(
            mapped.loc[0, "player_1_mapping_method"],
            "manual_override",
        )
        self.assertIsNotNone(stored)
        self.assertEqual(stored.player_id, 207729)  # type: ignore[union-attr]

    def test_unresolved_collision_is_logged_without_stopping_other_matches(
        self,
    ) -> None:
        """Conserva un partido ambiguo y permite mapear el resto del lote."""

        brandon = _candidate(206909, "Brandon", "Nakashima", ioc="USA")
        bryce = _candidate(210416, "Bryce", "Nakashima", ioc="USA")
        michelsen = _candidate(210506, "Alex", "Michelsen", ioc="USA")
        bautista = _candidate(105138, "Roberto", "Bautista Agut", ioc="ESP")
        matches = _daily_frame(
            [
                {
                    "player_1_name": "Michelsen A.",
                    "player_1_slug": "michelsen-alex",
                    "player_2_name": "Bautista Agut R.",
                    "player_2_slug": "bautista-agut-roberto",
                },
                {
                    "tour_level": "ITF",
                    "player_1_name": "Nakashima B.",
                    "player_1_slug": "nakashima-baff8",
                    "player_2_name": "Michelsen A.",
                    "player_2_slug": "michelsen-alex",
                },
            ]
        )

        with _temporary_test_directory() as temporary:
            directory = Path(temporary)
            mapped = _resolve_in_directory(
                directory,
                matches,
                candidates=(brandon, bryce, michelsen, bautista),
            )
            unresolved = load_unresolved_players(
                directory / "unresolved.csv"
            )

        self.assertEqual(mapped["mapping_status"].tolist(), ["mapped", "unmapped"])
        self.assertTrue(pd.isna(mapped.loc[1, "player_1_id"]))
        self.assertEqual(mapped.loc[1, "player_2_id"], 210506)
        self.assertEqual(len(unresolved), 1)
        self.assertEqual(
            unresolved.loc[0, "reason"],
            "ambiguous_active_candidates",
        )
        self.assertEqual(unresolved.loc[0, "candidate_count"], 2)
        self.assertEqual(
            unresolved.loc[0, "candidate_player_ids"],
            "206909|210416",
        )

    def test_missing_slug_is_unresolved_and_batch_continues(self) -> None:
        """No asigna por nombre cuando falta la clave estable de la ficha."""

        michelsen = _candidate(210506, "Alex", "Michelsen", ioc="USA")
        bautista = _candidate(105138, "Roberto", "Bautista Agut", ioc="ESP")
        matches = _daily_frame(
            [
                {
                    "player_1_name": "Michelsen A.",
                    "player_1_slug": pd.NA,
                    "player_2_name": "Bautista Agut R.",
                    "player_2_slug": "bautista-agut-roberto",
                }
            ]
        )

        with _temporary_test_directory() as temporary:
            directory = Path(temporary)
            mapped = _resolve_in_directory(
                directory,
                matches,
                candidates=(michelsen, bautista),
            )
            unresolved = load_unresolved_players(
                directory / "unresolved.csv"
            )

        self.assertEqual(mapped.loc[0, "mapping_status"], "unmapped")
        self.assertTrue(pd.isna(mapped.loc[0, "player_1_id"]))
        self.assertEqual(unresolved.loc[0, "reason"], "missing_player_slug")
        self.assertTrue(pd.isna(unresolved.loc[0, "slug"]))

    def test_unparseable_visible_name_is_logged_without_inventing_a_key(
        self,
    ) -> None:
        """Deja vacía la normalización de ``TBD`` y conserva el otro jugador."""

        bautista = _candidate(105138, "Roberto", "Bautista Agut", ioc="ESP")
        matches = _daily_frame(
            [
                {
                    "player_1_name": "TBD",
                    "player_1_slug": "pending-player",
                    "player_2_name": "Bautista Agut R.",
                    "player_2_slug": "bautista-agut-roberto",
                }
            ]
        )

        with _temporary_test_directory() as temporary:
            directory = Path(temporary)
            mapped = _resolve_in_directory(
                directory,
                matches,
                candidates=(bautista,),
            )
            unresolved = load_unresolved_players(
                directory / "unresolved.csv"
            )

        self.assertEqual(mapped.loc[0, "mapping_status"], "unmapped")
        self.assertEqual(mapped.loc[0, "player_2_id"], 105138)
        self.assertEqual(unresolved.loc[0, "reason"], "invalid_visible_name")
        self.assertTrue(pd.isna(unresolved.loc[0, "normalized_last_name"]))
        self.assertTrue(pd.isna(unresolved.loc[0, "first_initial"]))


class PlayerMappingPipelineCacheAndSummaryTest(unittest.TestCase):
    """Comprueba inmutabilidad de caché, idempotencia y métricas."""

    def test_empty_daily_frame_returns_typed_mapping_columns(self) -> None:
        """Mantiene el contrato en un día válido sin partidos ni E/S auxiliar."""

        empty = pd.DataFrame(
            {
                column: pd.Series(dtype="string")
                for column in (
                    "match_date",
                    "tournament",
                    "tour_level",
                    "gender",
                    "player_1_name",
                    "player_1_slug",
                    "player_2_name",
                    "player_2_slug",
                )
            }
        )

        with _temporary_test_directory() as temporary:
            directory = Path(temporary)
            mapped = resolve_scraped_matches(
                empty,
                as_of_date=BATCH_DATE,
                database_path=directory / "mapping.sqlite3",
                overrides_path=directory / "missing-overrides.csv",
                unresolved_path=directory / "unresolved.csv",
            )

        self.assertTrue(mapped.empty)
        self.assertEqual(str(mapped["player_1_id"].dtype), "Int64")
        self.assertEqual(
            str(mapped["player_1_mapping_method"].dtype),
            "string",
        )
        self.assertEqual(str(mapped["mapping_status"].dtype), "string")

    def test_cached_slug_is_never_recalculated(self) -> None:
        """Conserva el primer ID aunque un índice posterior proponga otro."""

        alex = _candidate(210506, "Alex", "Michelsen", ioc="USA")
        other = _candidate(999506, "Adam", "Michelsen", ioc="USA")
        bautista = _candidate(105138, "Roberto", "Bautista Agut", ioc="ESP")
        matches = _daily_frame(
            [
                {
                    "player_1_name": "Michelsen A.",
                    "player_1_slug": "michelsen-stable",
                    "player_2_name": "Bautista Agut R.",
                    "player_2_slug": "bautista-agut-roberto",
                }
            ]
        )

        with _temporary_test_directory() as temporary:
            directory = Path(temporary)
            overrides_path = directory / "overrides.csv"
            _write_empty_overrides(overrides_path)
            common = {
                "as_of_date": BATCH_DATE,
                "database_path": directory / "mapping.sqlite3",
                "overrides_path": overrides_path,
                "unresolved_path": directory / "unresolved.csv",
                "players": _players_frame(alex, other, bautista),
                "clock": lambda: FIXED_CLOCK,
            }
            first = resolve_scraped_matches(
                matches,
                candidate_indexes={
                    "M": _candidate_index(alex, bautista)
                },
                **common,
            )
            second = resolve_scraped_matches(
                matches,
                candidate_indexes={
                    "M": _candidate_index(other, bautista)
                },
                **common,
            )

        self.assertEqual(first.loc[0, "player_1_id"], 210506)
        self.assertEqual(second.loc[0, "player_1_id"], 210506)
        self.assertEqual(
            second.loc[0, "player_1_mapping_method"],
            "automatic_name",
        )

    def test_summary_reports_slots_and_complete_matches_by_level(self) -> None:
        """Calcula porcentajes explícitos sin ocultar mapeos parciales."""

        mapped = pd.DataFrame(
            {
                "gender": pd.Series(["M", "M"], dtype="string"),
                "tour_level": pd.Series(["ATP", "ATP"], dtype="string"),
                "player_1_id": pd.Series([1, pd.NA], dtype="Int64"),
                "player_2_id": pd.Series([2, 2], dtype="Int64"),
                "player_1_mapping_method": pd.Series(
                    ["automatic_name", pd.NA],
                    dtype="string",
                ),
                "player_2_mapping_method": pd.Series(
                    ["automatic_name", "manual_override"],
                    dtype="string",
                ),
                "mapping_status": pd.Series(
                    ["mapped", "unmapped"],
                    dtype="string",
                ),
            }
        )

        summary = summarize_mapping_coverage(mapped)

        self.assertEqual(summary.loc[0, "matches"], 2)
        self.assertEqual(summary.loc[0, "player_slots"], 4)
        self.assertEqual(summary.loc[0, "automatic_player_slots"], 2)
        self.assertEqual(summary.loc[0, "automatic_player_slot_pct"], 50.0)
        self.assertEqual(summary.loc[0, "mapped_matches"], 1)
        self.assertEqual(summary.loc[0, "mapped_match_pct"], 50.0)
        self.assertEqual(summary.loc[0, "automatic_match_pct"], 50.0)


if __name__ == "__main__":
    unittest.main()
