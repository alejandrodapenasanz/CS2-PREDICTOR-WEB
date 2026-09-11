"""Offline selection/CLI regressions: agenda-only, provenance, no guessed identities."""

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
import json
from unittest.mock import Mock

import pandas as pd
import pytest

from scripts import update_tennis_abstract as cli
from src.tennis_abstract_daily import update_daily
from src.tennis_abstract_selection import AgendaSelection, select_agenda_players
from src.tennis_abstract_store import TennisAbstractStore
from tests.test_tennis_abstract_daily import client_for, page, player

DAY = date(2026, 9, 9)
NOW = datetime(2026, 9, 9, 8, tzinfo=UTC)


def native(key="JannikSinner", name="Jannik Sinner", gender="M", rank=1):
    """An already inspected source-owned profile, not a slug derived by the selector."""

    return replace(player(key, gender, rank), name=name)


def agenda_row(a="Jannik Sinner", b="Carlos Alcaraz", **overrides):
    """Create one published agenda row with full names and exact source provenance."""

    return {
        "match_date": pd.Timestamp(DAY),
        "gender": "M",
        "retrieved_at_utc": pd.Timestamp(NOW - timedelta(minutes=5)),
        "source_url": "https://www.tennisratio.com/atp-matches.html",
        "snapshot_sha256": "a" * 64,
        "player_1_name": a,
        "player_2_name": b,
        "player_1_href": "https://www.tennisratio.com/players/" + a.replace(" ", "-") + ".html",
        "player_2_href": "https://www.tennisratio.com/players/" + b.replace(" ", "-") + ".html",
        **overrides,
    }


def test_only_day_participants_are_selected_once_and_future_agenda_cannot_change_them():
    """Repeated players are one fetch; adding tomorrow's matches leaves today's targets stable."""

    inventory = [
        native(),
        native("CarlosAlcaraz", "Carlos Alcaraz", rank=2),
        native("DaniilMedvedev", "Daniil Medvedev", rank=3),
    ]
    rows = [agenda_row(), agenda_row()]
    before = select_agenda_players(pd.DataFrame(rows), inventory, DAY, observed_at=NOW)
    after = select_agenda_players(
        pd.DataFrame(
            rows
            + [
                agenda_row(
                    "Daniil Medvedev",
                    "Carlos Alcaraz",
                    match_date=pd.Timestamp(DAY + timedelta(days=1)),
                )
            ]
        ),
        inventory,
        DAY,
        observed_at=NOW,
    )
    assert [p.key for p in before.players] == ["JannikSinner", "CarlosAlcaraz"]
    assert before == after
    assert before.report["participant_count"] == 2
    assert before.players[0].selection_evidence[0]["snapshot_sha256"] == "a" * 64
    assert before.report["model_ready"] is False


def test_full_name_normalization_does_not_expand_initials_or_cross_genders():
    """Typography can match; an initial or wrong source gender cannot select a profile."""

    inventory = [native("FabianMarozsan", "Fabian Marozsan"), native()]
    result = select_agenda_players(
        pd.DataFrame(
            [
                agenda_row("Fábián Marozsán", "Sinner J."),
                agenda_row("Jannik Sinner", "Unknown Player", gender="F"),
            ]
        ),
        inventory,
        DAY,
        observed_at=NOW,
    )
    assert [p.key for p in result.players] == ["FabianMarozsan"]
    assert len(result.report["unresolved_players"]) == 3


@pytest.mark.parametrize("collision", ["inventory", "agenda"])
def test_name_collisions_remain_unresolved(collision):
    """Two source identities or two inspected links never get silently merged."""

    inventory = [native()]
    rows = [agenda_row()]
    if collision == "inventory":
        inventory.append(native("DifferentNativeKey", "Jannik Sinner"))
    else:
        rows.append(agenda_row(player_1_href="https://www.tennisratio.com/players/other-id.html"))
    result = select_agenda_players(pd.DataFrame(rows), inventory, DAY, observed_at=NOW)
    assert result.players == ()
    assert any(
        r["reason"] == "ambiguous_name_or_source_identity"
        for r in result.report["unresolved_players"]
    )


def test_future_capture_is_rejected_but_today_capture_only_selects_acquisition():
    """Selecting a download today does not authorize a feature dated today."""

    inventory = [native(), native("CarlosAlcaraz", "Carlos Alcaraz")]
    future = select_agenda_players(
        pd.DataFrame([agenda_row(retrieved_at_utc=pd.Timestamp(NOW + timedelta(minutes=1)))]),
        inventory,
        DAY,
        observed_at=NOW,
    )
    assert not future.players and future.report["unusable_agenda_rows"] == 1
    current = select_agenda_players(pd.DataFrame([agenda_row()]), inventory, DAY, observed_at=NOW)
    assert len(current.players) == 2 and not current.report["model_ready"]


def test_selected_acquisition_preserves_provenance_and_reports_missing_participants(tmp_path):
    """A successful subset is partial selection, not full coverage or a model input."""

    selection = select_agenda_players(
        pd.DataFrame([agenda_row()]), [native()], DAY, observed_at=NOW
    )
    path = tmp_path / "tennis_abstract.sqlite3"
    wire = client_for(page())
    result = update_daily(
        store_path=path,
        players=list(selection.players),
        selection_context=selection.report,
        client=wire,
        clock=lambda: NOW,
    )
    assert result["status"] == "partial_selection"
    assert result["selection_scope"] == "daily_agenda" and result["inventory_players"] == 1
    assert wire.get.call_count == 1 and not result["production_changed"]
    store = TennisAbstractStore(path, read_only=True)
    try:
        assert store.state("last_agenda_selection") == selection.report
        assert store.snapshot_before("M", "JannikSinner", DAY) is None
        saved = store.snapshot_before("M", "JannikSinner", DAY + timedelta(days=1))
        assert saved["inventory"]["selection_evidence"][0]["match_date"] == DAY.isoformat()
        assert saved["model_ready"] is False
    finally:
        store.close()


def test_default_cli_uses_agenda_and_full_inventory_requires_explicit_flag(monkeypatch, capsys):
    """Exercise the real entrypoint without browser/network or model side effects."""

    selection = AgendaSelection((native(),), {"scope": "daily_agenda"})
    loader = Mock(return_value=selection)
    update = Mock(return_value={"status": "completed"})
    monkeypatch.setattr(cli, "load_daily_selection", loader)
    monkeypatch.setattr(cli, "update_daily", update)
    monkeypatch.setattr("sys.argv", ["update_tennis_abstract.py", "--date", DAY.isoformat()])
    assert cli.main() == 0
    loader.assert_called_once_with(DAY)
    assert update.call_args.kwargs["players"] == list(selection.players)
    assert update.call_args.kwargs["selection_context"] == selection.report
    loader.reset_mock()
    monkeypatch.setattr("sys.argv", ["update_tennis_abstract.py", "--full-inventory"])
    assert cli.main() == 0
    loader.assert_not_called()
    assert update.call_args.kwargs["players"] is None


def test_missing_agenda_never_falls_back_to_the_full_catalogue(monkeypatch, capsys):
    """A missing schedule is visible and triggers zero profile requests."""

    monkeypatch.setattr(
        cli, "load_daily_selection", lambda day: AgendaSelection((), {"agenda_rows": 0})
    )
    update = Mock(side_effect=AssertionError("No catalogue fallback"))
    monkeypatch.setattr(cli, "update_daily", update)
    monkeypatch.setattr("sys.argv", ["update_tennis_abstract.py"])
    assert cli.main() == 2
    assert json.loads(capsys.readouterr().out)["status"] == "agenda_unavailable"
    update.assert_not_called()


def test_status_does_not_load_agenda_or_open_a_download(monkeypatch):
    """The status command stays read-only even after agenda selection becomes the default."""

    loader = Mock(side_effect=AssertionError("Status must not initialize the other source store"))
    monkeypatch.setattr(cli, "load_daily_selection", loader)
    monkeypatch.setattr(cli, "acquisition_status", lambda: {"status": "pending"})
    monkeypatch.setattr("sys.argv", ["update_tennis_abstract.py", "--status"])
    assert cli.main() == 0
    loader.assert_not_called()
