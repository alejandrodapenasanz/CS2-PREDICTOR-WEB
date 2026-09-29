"""Offline regressions for agenda-first acquisition plus a resumable daily extra quota."""

from contextlib import closing
from datetime import timedelta
from unittest.mock import Mock

import pytest

from scripts import update_tennis_abstract as cli
from src import tennis_abstract_daily as daily
from src.tennis_abstract_selection import AgendaSelection
from src.tennis_abstract_store import TennisAbstractStore
from tests.test_tennis_abstract_daily import START, client_for, page, player


def test_all_agenda_players_precede_fifty_extras_and_same_day_rerun_does_not_expand(
    tmp_path, monkeypatch
):
    """Even an agenda larger than fifty is uncapped; extras cannot outrank it."""

    path = tmp_path / "tennis_abstract.sqlite3"
    agenda = [player(f"Agenda{index}", rank=100 + index) for index in range(60)]
    extras = [player(f"Extra{index}", rank=index + 1) for index in range(70)]
    monkeypatch.setattr(daily, "load_inventory", lambda: agenda + extras)
    wire = client_for(*(page() for _ in range(110)))
    arguments = dict(
        store_path=path,
        players=agenda,
        client=wire,
        clock=lambda: START,
        selection_context={"scope": "daily_agenda"},
        extra_profiles=50,
    )
    first = daily.update_daily(**arguments)
    assert first["refreshed_agenda"] == 60
    assert first["refreshed_extra"] == first["extra_players_selected"] == 50
    assert first["pending"] == 0
    assert first["extra_backlog_deferred"] == 20
    keys = [call.args[0].split("p=")[1] for call in wire.get.call_args_list]
    assert keys[:60] == [p.key for p in agenda]
    assert keys[60:] == [p.key for p in extras[:50]]
    repeated = daily.update_daily(**arguments)
    assert repeated["status"] == "already_refreshed"
    assert repeated["already_refreshed"] == 110
    assert wire.get.call_count == 110
    with closing(TennisAbstractStore(path)) as store:
        saved = store.snapshot_before("M", agenda[0].key, START.date() + timedelta(days=1))
        assert store.snapshot_before("M", agenda[0].key, START.date()) is None
    tomorrow_wire = client_for(*(page("65") for _ in range(110)))
    tomorrow = daily.update_daily(
        **{**arguments, "client": tomorrow_wire, "clock": lambda: START + timedelta(days=1)}
    )
    assert tomorrow["refreshed_extra"] == 50
    tomorrow_keys = [call.args[0].split("p=")[1] for call in tomorrow_wire.get.call_args_list]
    assert tomorrow_keys[60:80] == [p.key for p in extras[50:]]
    with closing(TennisAbstractStore(path)) as store:
        assert store.snapshot_before("M", agenda[0].key, START.date() + timedelta(days=1)) == saved


def test_interrupted_quota_resumes_same_extras_after_today_players(tmp_path, monkeypatch):
    """A partial run cannot consume another fifty identities on restart."""

    path = tmp_path / "tennis_abstract.sqlite3"
    agenda = [player("Today", rank=100)]
    extras = [player(f"Extra{index}", rank=index + 1) for index in range(6)]
    monkeypatch.setattr(daily, "load_inventory", lambda: agenda + extras)
    wire = client_for(*(page() for _ in range(4)))
    arguments = dict(
        store_path=path, players=agenda, client=wire, clock=lambda: START, extra_profiles=3
    )
    first = daily.update_daily(**arguments, max_profiles=2)
    assert first["pending_agenda"] == 0
    assert first["pending_extra"] == 2
    again = daily.update_daily(**arguments)
    assert again["refreshed_extra"] == 2
    assert again["refreshed_agenda"] == 0
    assert [call.args[0].split("p=")[1] for call in wire.get.call_args_list] == [
        "Today",
        "Extra0",
        "Extra1",
        "Extra2",
    ]
    daily.update_daily(**arguments)
    assert wire.get.call_count == 4


def test_already_refreshed_primary_does_not_prevent_acquiring_new_extras(tmp_path, monkeypatch):
    """Migration on a day already processed by the old agenda-only policy still works."""

    path = tmp_path / "tennis_abstract.sqlite3"
    agenda = [player("Today")]
    wire = client_for(page())
    daily.update_daily(store_path=path, players=agenda, client=wire, clock=lambda: START)
    monkeypatch.setattr(daily, "load_inventory", lambda: agenda + [player("Pending")])
    extra_wire = client_for(page())
    result = daily.update_daily(
        store_path=path, players=agenda, client=extra_wire, clock=lambda: START, extra_profiles=50
    )
    assert result["already_refreshed"] == 1
    assert result["refreshed_agenda"] == 0 and result["refreshed_extra"] == 1
    assert extra_wire.get.call_args.args[0].endswith("p=Pending")


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_invalid_quota_is_rejected_before_io(tmp_path, value):
    """Do not convert a malformed budget into unlimited acquisition."""

    with pytest.raises(ValueError, match="extra_profiles"):
        daily.update_daily(store_path=tmp_path / "tennis_abstract.sqlite3", extra_profiles=value)


def test_cli_can_explicitly_disable_extras_without_dropping_agenda(monkeypatch):
    """The sole shared entrypoint owns the setting used by both launchers."""

    selection = AgendaSelection((player(),), {"scope": "daily_agenda"})
    update = Mock(return_value={"status": "completed"})
    monkeypatch.setattr(cli, "load_daily_selection", lambda day: selection)
    monkeypatch.setattr(cli, "update_daily", update)
    monkeypatch.setattr("sys.argv", ["update_tennis_abstract.py", "--extra-profiles", "0"])
    assert cli.main() == 0
    assert update.call_args.kwargs["extra_profiles"] == 0
    assert update.call_args.kwargs["players"] == list(selection.players)
