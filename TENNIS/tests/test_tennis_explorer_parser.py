"""Tests offline del parser diario de Tennis Explorer.

La suite usa exclusivamente el snapshot HTML real de 2026-07-30 guardado en
``tests/fixtures``. Las variantes de error se obtienen modificando una copia
en memoria, por lo que ningún test realiza peticiones de red.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import sys
import unittest

from bs4 import BeautifulSoup
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tennis_explorer import (  # noqa: E402
    OUTPUT_COLUMNS,
    OUTPUT_DTYPES,
    TennisExplorerSchemaError,
    parse_daily_matches_html,
)


FIXTURE_PATH = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "tennis_explorer"
    / "matches_2026-07-30_all.html"
)
FIXTURE_DATE = date(2026, 7, 30)


class TennisExplorerParserTests(unittest.TestCase):
    """Valida cobertura, tipos y fallos explícitos contra el HTML observado."""

    @classmethod
    def setUpClass(cls) -> None:
        """Carga una sola vez el snapshot real y su resultado de referencia."""

        cls.html = FIXTURE_PATH.read_bytes()
        cls.matches = parse_daily_matches_html(cls.html, FIXTURE_DATE)

    def test_real_fixture_has_expected_scope_and_level_counts(self) -> None:
        """Incluye 313 individuales y reproduce los cinco segmentos reales."""

        self.assertEqual(len(self.matches), 313)
        actual_counts = (
            self.matches.groupby(
                ["gender", "tour_level"],
                observed=True,
            )
            .size()
            .to_dict()
        )
        self.assertEqual(
            actual_counts,
            {
                ("M", "ATP"): 13,
                ("M", "Challenger"): 24,
                ("M", "ITF"): 133,
                ("F", "WTA"): 21,
                ("F", "ITF"): 122,
            },
        )
        self.assertFalse(
            self.matches["tournament"]
            .str.casefold()
            .str.startswith("utr ")
            .any()
        )
        self.assertFalse(
            self.matches["player_1_href"]
            .str.startswith("/doubles-team/", na=False)
            .any()
        )

    def test_real_fixture_preserves_columns_and_nullable_dtypes(self) -> None:
        """Mantiene el orden y los dtypes pandas definidos por el contrato."""

        self.assertEqual(tuple(self.matches.columns), OUTPUT_COLUMNS)
        for column, expected_dtype in OUTPUT_DTYPES.items():
            with self.subTest(column=column):
                self.assertEqual(str(self.matches[column].dtype), expected_dtype)
        self.assertTrue(
            (self.matches["match_date"] == pd.Timestamp(FIXTURE_DATE)).all()
        )

    def test_real_fixture_extracts_names_slugs_odds_and_futures_context(
        self,
    ) -> None:
        """Extrae valores reales y conserva la agregación Futures sin href."""

        row = self.matches.loc[
            (self.matches["player_1_name"] == "Kuzuhara B.")
            & (self.matches["player_2_name"] == "Roddick J.")
        ].iloc[0]

        self.assertEqual(row["tournament"], "Futures 2026")
        self.assertEqual(row["gender"], "M")
        self.assertEqual(row["tour_level"], "ITF")
        self.assertTrue(pd.isna(row["tournament_href"]))
        self.assertTrue(pd.isna(row["surface"]))
        self.assertEqual(row["player_1_href"], "/player/kuzuhara/")
        self.assertEqual(row["player_2_href"], "/player/roddick-30f1b/")
        self.assertEqual(row["player_1_slug"], "kuzuhara")
        self.assertEqual(row["player_2_slug"], "roddick-30f1b")
        self.assertAlmostEqual(float(row["player_1_odds"]), 1.41)
        self.assertAlmostEqual(float(row["player_2_odds"]), 2.71)

    def test_real_fixture_extracts_stable_match_ids_and_conservative_winners(
        self,
    ) -> None:
        """Liquida solo los 76 finales convencionales observados."""

        self.assertEqual(self.matches["source_match_id"].nunique(), 313)
        expected_hrefs = (
            "/match-detail/?id=" + self.matches["source_match_id"]
        )
        pd.testing.assert_series_equal(
            self.matches["match_detail_href"],
            expected_hrefs.rename("match_detail_href"),
        )

        finished = self.matches.loc[self.matches["status"].eq("finished")]
        self.assertEqual(len(finished), 76)
        self.assertTrue(finished["winner_slug"].notna().all())
        self.assertTrue(finished["winner_side"].eq("player_1").all())
        self.assertTrue(
            finished["result_evidence"]
            .eq("winner_from_terminal_sets_and_slug")
            .all()
        )
        self.assertEqual(
            finished["sets_score"].value_counts().to_dict(),
            {"2-0": 64, "2-1": 12},
        )

        partial = self.matches.loc[
            self.matches["player_1_name"].eq("Ibraimi Y.")
        ].iloc[0]
        self.assertEqual(int(partial["player_1_sets_won"]), 1)
        self.assertEqual(int(partial["player_2_sets_won"]), 0)
        self.assertEqual(partial["sets_score"], "1-0")
        self.assertTrue(pd.isna(partial["winner_slug"]))
        self.assertTrue(pd.isna(partial["winner_side"]))

    def test_terminal_sets_can_identify_player_2_without_row_order_assumption(
        self,
    ) -> None:
        """Deriva el ganador del contador, no de la primera fila ni coursew."""

        soup = self._fixture_soup()
        first_row = soup.find("tr", id="r10")
        second_row = soup.find("tr", id="r10b")
        self.assertIsNotNone(first_row)
        self.assertIsNotNone(second_row)
        first_row.select_one("td.result").string = "0"
        second_row.select_one("td.result").string = "2"

        frame = parse_daily_matches_html(str(soup), FIXTURE_DATE)
        row = frame.loc[
            frame["source_match_id"].eq("3281237")
        ].iloc[0]
        self.assertEqual(row["sets_score"], "0-2")
        self.assertEqual(row["winner_side"], "player_2")
        self.assertEqual(row["winner_slug"], "mannarino-a7108")

    def test_real_fixture_joins_explicit_weekly_catalog_surfaces(self) -> None:
        """Une superficies por href sin inferir los Futures agregados."""

        self.assertEqual(int(self.matches["surface"].notna().sum()), 180)
        examples = {
            ("Washington", "M"): "Hard",
            ("San Marino challenger", "M"): "Clay",
            ("Maspalomas ITF", "F"): "Clay",
        }
        for (tournament, gender), expected in examples.items():
            with self.subTest(tournament=tournament, gender=gender):
                observed = set(
                    self.matches.loc[
                        self.matches["tournament"].eq(tournament)
                        & self.matches["gender"].eq(gender),
                        "surface",
                    ].dropna()
                )
                self.assertEqual(observed, {expected})

    def test_indoors_is_missing_material_not_assumed_hard(self) -> None:
        """Conserva un recinto indoor sin inventar su superficie material."""

        soup = self._fixture_soup()
        tournament_link = next(
            (
                link
                for link in soup.select(
                    'a[href="/washington/2026/atp-men/"]'
                )
                if link.find_parent("tr").select_one(
                    "td.s-color span[title]"
                )
                is not None
            ),
            None,
        )
        self.assertIsNotNone(tournament_link)
        catalog_row = tournament_link.find_parent("tr")
        surface_span = catalog_row.select_one("td.s-color span[title]")
        self.assertIsNotNone(surface_span)
        surface_span["title"] = "Indoors"

        frame = parse_daily_matches_html(str(soup), FIXTURE_DATE)
        washington = frame.loc[
            frame["tournament"].eq("Washington")
            & frame["gender"].eq("M")
        ]
        self.assertGreater(len(washington), 0)
        self.assertTrue(washington["surface"].isna().all())

    def test_unmarked_wta_tournament_uses_approved_wta_convention(self) -> None:
        """Clasifica Targu Mures como WTA al no tener marcador ITF explícito."""

        rows = self.matches.loc[
            self.matches["tournament"] == "Targu Mures"
        ]
        self.assertEqual(len(rows), 4)
        self.assertEqual(set(rows["gender"]), {"F"})
        self.assertEqual(set(rows["tour_level"]), {"WTA"})

    def test_status_is_conservative_and_ignores_live_stream_adverts(self) -> None:
        """No confunde publicidad Live streams ni resuelve un 1-0 ambiguo."""

        scheduled = self.matches.loc[
            self.matches["player_1_name"] == "Nakashima B."
        ].iloc[0]
        self.assertEqual(scheduled["status"], "scheduled")
        self.assertEqual(
            scheduled["status_evidence"],
            "scheduled_time_with_empty_result_and_scores",
        )

        no_set_scores = self.matches.loc[
            self.matches["player_1_name"] == "Ibraimi Y."
        ].iloc[0]
        self.assertEqual(no_set_scores["status"], "unknown")
        self.assertEqual(
            no_set_scores["status_evidence"],
            "nonterminal_or_scoreless_result",
        )

        unknown = self.matches.loc[
            self.matches["player_1_name"] == "Price G."
        ].iloc[0]
        self.assertTrue(pd.isna(unknown["scheduled_time"]))
        self.assertEqual(
            unknown["status_evidence"],
            "undetermined_no_start_time",
        )

    def test_player_without_link_and_blank_odds_are_nullable(self) -> None:
        """Conserva nombre visible y nulos cuando faltan enlace y cuotas."""

        soup = self._fixture_soup()
        player_link = soup.find(
            "a",
            href="/player/michelsen-a98bb/",
        )
        self.assertIsNotNone(player_link)
        player_link.replace_with("Michelsen A.")
        first_row = soup.find("tr", id="r10")
        self.assertIsNotNone(first_row)
        for odds_cell in first_row.select("td.course, td.coursew"):
            odds_cell.clear()

        frame = parse_daily_matches_html(str(soup), FIXTURE_DATE)
        row = frame.loc[
            frame["match_detail_href"] == "/match-detail/?id=3281237"
        ].iloc[0]
        self.assertEqual(row["player_1_name"], "Michelsen A.")
        self.assertFalse(bool(row["player_1_has_link"]))
        self.assertTrue(pd.isna(row["player_1_href"]))
        self.assertTrue(pd.isna(row["player_1_slug"]))
        self.assertTrue(pd.isna(row["player_1_odds"]))
        self.assertTrue(pd.isna(row["player_2_odds"]))
        self.assertTrue(pd.isna(row["winner_side"]))
        self.assertTrue(pd.isna(row["winner_slug"]))
        self.assertEqual(
            row["result_evidence"],
            "winner_not_inferred_missing_winner_slug",
        )

    def test_malformed_player_link_raises_clear_schema_error(self) -> None:
        """Rechaza un anchor presente cuyo href ya no sigue /player/slug/."""

        soup = self._fixture_soup()
        player_link = soup.find(
            "a",
            href="/player/michelsen-a98bb/",
        )
        self.assertIsNotNone(player_link)
        player_link["href"] = "/profiles/michelsen"

        with self.assertRaisesRegex(
            TennisExplorerSchemaError,
            "enlace de jugador",
        ):
            parse_daily_matches_html(str(soup), FIXTURE_DATE)

    def test_non_numeric_non_empty_odds_raise_schema_error(self) -> None:
        """Rechaza una cuota no vacía si no es decimal."""

        soup = self._fixture_soup()
        first_row = soup.find("tr", id="r10")
        self.assertIsNotNone(first_row)
        first_row.select("td.course, td.coursew")[0].string = "evens"

        with self.assertRaisesRegex(
            TennisExplorerSchemaError,
            "Cuota decimal no reconocida",
        ):
            parse_daily_matches_html(str(soup), FIXTURE_DATE)

    def test_broken_row_pair_raises_schema_error(self) -> None:
        """Detecta si la segunda fila deja de llamarse exactamente id+b."""

        soup = self._fixture_soup()
        second_row = soup.find("tr", id="r10b")
        self.assertIsNotNone(second_row)
        second_row["id"] = "row-with-new-format"

        with self.assertRaisesRegex(
            TennisExplorerSchemaError,
            "debería continuar",
        ):
            parse_daily_matches_html(str(soup), FIXTURE_DATE)

    def test_duplicate_source_match_id_raises_clear_schema_error(self) -> None:
        """Impide que dos filas compartan la clave estable de la fuente."""

        soup = self._fixture_soup()
        second_detail = soup.find(
            "a",
            href="/match-detail/?id=3281239",
        )
        self.assertIsNotNone(second_detail)
        second_detail["href"] = "/match-detail/?id=3281237"

        with self.assertRaisesRegex(
            TennisExplorerSchemaError,
            "repite source_match_id",
        ):
            parse_daily_matches_html(str(soup), FIXTURE_DATE)

    def test_explicit_walkover_cancelled_and_live_tokens_take_priority(
        self,
    ) -> None:
        """Reconoce estados explícitos sin consultar el widget externo."""

        cases = {
            "17:00 W.O.": ("walkover", "explicit_walkover"),
            "17:00 Cancelled": ("cancelled", "explicit_cancelled"),
            "1st Set - Live": (
                "in_progress",
                "explicit_live_or_interrupted",
            ),
        }
        for time_text, expected in cases.items():
            with self.subTest(time_text=time_text):
                soup = self._fixture_soup()
                time_cell = soup.find("tr", id="r12").select_one("td.time")
                self.assertIsNotNone(time_cell)
                direct_text = next(
                    child
                    for child in time_cell.contents
                    if isinstance(child, str) and child.strip()
                )
                direct_text.replace_with(time_text)

                frame = parse_daily_matches_html(str(soup), FIXTURE_DATE)
                row = frame.loc[
                    frame["match_detail_href"]
                    == "/match-detail/?id=3281285"
                ].iloc[0]
                self.assertEqual(
                    (row["status"], row["status_evidence"]),
                    expected,
                )
                self.assertTrue(pd.isna(row["winner_side"]))
                self.assertTrue(pd.isna(row["winner_slug"]))
                self.assertEqual(
                    row["result_evidence"],
                    "winner_not_inferred_nonterminal_or_special_status",
                )

    def test_set_scores_without_explicit_live_state_remain_unknown(self) -> None:
        """No convierte sets parciales en directo sin un marcador Live explícito."""

        soup = self._fixture_soup()
        for row_id in ("r10", "r10b"):
            outcome = soup.find("tr", id=row_id).select_one("td.result")
            self.assertIsNotNone(outcome)
            outcome["class"] = ["nbr"]
            outcome.clear()

        frame = parse_daily_matches_html(str(soup), FIXTURE_DATE)
        row = frame.loc[
            frame["match_detail_href"] == "/match-detail/?id=3281237"
        ].iloc[0]
        self.assertEqual(row["status"], "unknown")
        self.assertEqual(
            row["status_evidence"],
            "set_scores_without_explicit_live_state",
        )

    def test_valid_table_with_no_in_scope_singles_returns_typed_empty(
        self,
    ) -> None:
        """Devuelve esquema vacío si la tabla válida solo ofrece excluidos."""

        soup = self._fixture_soup()
        for span in soup.find_all("span"):
            classes = span.get("class", [])
            span["class"] = [
                {
                    "type-men2": "type-men4",
                    "type-women2": "type-women4",
                }.get(class_name, class_name)
                for class_name in classes
            ]

        frame = parse_daily_matches_html(str(soup), FIXTURE_DATE)
        self.assertTrue(frame.empty)
        self.assertEqual(tuple(frame.columns), OUTPUT_COLUMNS)
        for column, expected_dtype in OUTPUT_DTYPES.items():
            with self.subTest(column=column):
                self.assertEqual(str(frame[column].dtype), expected_dtype)

    def test_missing_requested_date_table_is_not_silently_empty(self) -> None:
        """Trata una fecha ausente como cambio/error, no como jornada vacía."""

        with self.assertRaisesRegex(
            TennisExplorerSchemaError,
            "una única tabla diaria",
        ):
            parse_daily_matches_html(self.html, date(2025, 1, 1))

    def test_datetime_is_rejected_instead_of_losing_time_silently(self) -> None:
        """Exige una fecha pura igual que la API de descarga pública."""

        with self.assertRaisesRegex(TypeError, "no datetime"):
            parse_daily_matches_html(
                self.html,
                datetime(2026, 7, 30, 12, 0),
            )

    def _fixture_soup(self) -> BeautifulSoup:
        """Crea una copia DOM independiente del snapshot para cada mutación."""

        return BeautifulSoup(self.html, "lxml")


if __name__ == "__main__":
    unittest.main()
