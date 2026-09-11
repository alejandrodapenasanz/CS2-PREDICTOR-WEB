"""Causal pistol strength beyond general Elo, fitted on earlier available rounds.

One observation per pistol (CT orientation); whole-day prequential replay.
Past map/side context is used ONLY after that source becomes available. The
prematch output marginalizes both sides and never consumes the target map/veto.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
import hashlib
from itertools import groupby
import json
import math
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from .pistols import MIN_PISTOLS, OPPONENT_DIFF_COLUMNS, WINDOW_DAYS, civil_date

VERSION = 1
MIN_FIT_PISTOLS = 100
MIN_FIT_MATCHES = 20
PRIOR_INFORMATION = 5.0  # 20 neutral Bernoulli trials x variance .25.
MAX_OFFSET = 1.5
MAPS = ("ancient", "anubis", "dust2", "inferno", "mirage", "nuke", "overpass", "train", "vertigo")
OPPONENT_SYM_COLUMNS = ["pistol_opponent_available", "pistol_opponent_sample_min", "pistol_opponent_model_ready"]
OPPONENT_COLUMNS = OPPONENT_DIFF_COLUMNS + OPPONENT_SYM_COLUMNS
CHALLENGER_COLUMNS = ["pistol_opponent_edge_diff", "pistol_opponent_available", "pistol_opponent_sample_min"]


def sigmoid(value: float) -> float:
    """Bounded logistic without overflow, shared by diagnostic and serving paths."""
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def residual_offset(actual: list[float], expected: list[float]) -> float:
    """Shrunk one-step logistic offset; never trust a few surprising wins fully."""
    score = sum(y - p for y, p in zip(actual, expected, strict=True))
    information = sum(p * (1.0 - p) for p in expected)
    return max(-MAX_OFFSET, min(MAX_OFFSET, score / (information + PRIOR_INFORMATION)))


def state_before_day(rows: list[dict[str, Any]], as_of: str | date) -> Any:
    """Shared strict serving state for the new challenger; legacy artifacts stay unchanged."""
    from .features import ChronologicalState
    from .config import get_runtime_config

    state = ChronologicalState(form_half_life=get_runtime_config().training.form_half_life_days)
    day = civil_date(as_of)
    for row in sorted(rows, key=lambda r: (r["date_obj"], str(r["id"]))):
        if civil_date(row["date"]) >= day:
            break
        state.observe(row)
    return state


class SymmetricPistolScorer:
    """Evaluate the artifact with the same A/B averaging used by production."""

    def __init__(self, artifact: Any) -> None:
        self.artifact = artifact
        self.metadata = artifact.metadata

    def predict_proba_team1(self, rows: list[dict[str, float]]) -> np.ndarray:
        from .features import DIFF_COLUMNS, EXTENDED_DIFF_COLUMNS

        directional = set(DIFF_COLUMNS + EXTENDED_DIFF_COLUMNS) | {
            "opening_odds_prob_centered",
            "roster_glicko_prob_centered",
        }
        reverse = [{k: -v if k in directional else v for k, v in row.items()} for row in rows]
        probability, _ = self.artifact.predict_symmetric_proba_team1_with_uncertainty(rows, reverse)
        return probability


class CivilEloHistory:
    """Replay the existing rating update; expose only end-of-previous-day states."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        # Lazy import avoids a cycle when features imports the candidate columns.
        from .features import ChronologicalState

        state = ChronologicalState()
        self.by_match: dict[str, dict[str, Any]] = {}
        self.by_team: dict[str, list[tuple[date, float]]] = defaultdict(list)
        ordered = sorted(rows, key=lambda r: (r["date_obj"], str(r["id"])))
        for day_text, iterator in groupby(ordered, key=lambda r: str(r["date"])[:10]):
            day = date.fromisoformat(day_text)
            matches = list(iterator)
            for row in matches:
                self.by_match[str(row["id"])] = {
                    "day": day,
                    "team1": str(row.get("team1_id")),
                    "team2": str(row.get("team2_id")),
                    "key1": row["team1_key"],
                    "key2": row["team2_key"],
                    "elo1": state.elos[row["team1_key"]],
                    "elo2": state.elos[row["team2_key"]],
                }
            touched: set[str] = set()
            for row in matches:
                state.observe(row)
                touched.update((row["team1_key"], row["team2_key"]))
            for key in sorted(touched):
                self.by_team[key].append((day, state.elos[key]))

    def rating(self, key: str, day: date) -> float:
        """No same-day outcomes, even when every match has an exact kickoff time."""
        history = self.by_team.get(key, [])
        index = bisect_left(history, (day, -math.inf))
        return history[index - 1][1] if index else 1500.0


@dataclass
class PistolSnapshot:
    """Read-only-in-use state frozen before any event of its civil day."""

    day: date
    beta: float = 0.0
    ct_intercept: float = 0.0
    map_bias: dict[str, float] = field(default_factory=dict)
    fit_n: int = 0
    fit_matches: int = 0
    teams: dict[str, dict[str, float]] = field(default_factory=dict)
    raw: dict[str, tuple[float, int]] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.fit_n >= MIN_FIT_PISTOLS and self.fit_matches >= MIN_FIT_MATCHES

    def contextual(self, elo_ct_minus_t: float, map_name: str) -> float:
        """Retrospective benchmark for a PAST round, not a prematch map prediction."""
        return sigmoid(self.ct_intercept + self.map_bias.get(map_name.lower(), 0.0) + self.beta * elo_ct_minus_t / 400)

    def neutral(self, elo_diff: float, offset: float = 0.0) -> float:
        """Unknown target map/starting side: average CT and T, preserve swap symmetry."""
        strength = self.beta * elo_diff / 400 + offset
        return 0.5 * (sigmoid(self.ct_intercept + strength) + sigmoid(-self.ct_intercept + strength))

    def features(self, a: str, b: str, elo_diff: float) -> dict[str, float]:
        """Only incremental information beyond Elo is in the challenger matrix."""
        sa, sb = self.teams.get(a, {}), self.teams.get(b, {})
        n = min(sa.get("n", 0.0), sb.get("n", 0.0))
        available = self.ready and n >= MIN_PISTOLS
        difference = sa.get("offset", 0.0) - sb.get("offset", 0.0) if available else 0.0
        baseline = self.neutral(elo_diff)
        adjusted = self.neutral(elo_diff, difference)
        return {
            "pistol_opponent_edge_diff": adjusted - baseline,
            "pistol_opponent_probability_centered": adjusted - 0.5,
            "pistol_opponent_baseline_centered": baseline - 0.5,
            "pistol_opponent_residual_diff": difference,
            "pistol_opponent_available": float(available),
            "pistol_opponent_sample_min": float(n),
            "pistol_opponent_model_ready": float(self.ready),
        }


class PistolOpponentHistory:
    """Daily replay of original captures, past Elo, and frozen out-of-sample residuals."""

    def __init__(self, rows: list[dict[str, Any]], store: dict[str, Any]) -> None:
        self.elo = CivilEloHistory(rows)
        self.events: list[dict[str, Any]] = []
        self.exclusions: Counter[str] = Counter()
        self.snapshots: dict[date, PistolSnapshot] = {}
        self.predictions: list[dict[str, Any]] = []
        self._played: dict[date, list[dict[str, Any]]] = defaultdict(list)
        self._received: dict[date, list[dict[str, Any]]] = defaultdict(list)
        self._eligible: list[dict[str, Any]] = []
        seen: set[tuple[Any, Any]] = set()
        for team, records in store.get("teams", {}).items():
            for record in records:
                if record["side"] != "ct":
                    continue  # Never count the reversed row as a second observation.
                key = (record["map_id"], record.get("half"))
                if key in seen:
                    continue
                seen.add(key)
                match = self.elo.by_match.get(str(record.get("match_id")))
                if not match or {str(team), record["opponent"]} != {match["team1"], match["team2"]}:
                    self.exclusions["missing_exact_match_or_team_identity"] += 1
                    continue
                played, captured = civil_date(record["played_at"]), civil_date(record["captured_at"])
                if played != match["day"] or captured < played:
                    self.exclusions["date_or_availability_conflict"] += 1
                    continue
                ct_first = str(team) == match["team1"]
                event = {
                    "match_id": str(record["match_id"]),
                    "map_id": record["map_id"],
                    "half": record["half"],
                    "played": played,
                    "captured": captured,
                    "ct": str(team),
                    "t": record["opponent"],
                    "map": record["map_name"].lower(),
                    "y": float(record["win"]),
                    "elo_ct": match["elo1"] if ct_first else match["elo2"],
                    "elo_t": match["elo2"] if ct_first else match["elo1"],
                }
                self.events.append(event)
                self._played[played].append(event)
                self._received[max(played, captured) + timedelta(days=1)].append(event)
        self.events.sort(key=lambda e: (e["played"], e["match_id"], e["map_id"], e["half"]))
        for events in (*self._played.values(), *self._received.values()):
            events.sort(key=lambda e: (e["match_id"], e["map_id"], e["half"]))
        self.first_day = min(self._played, default=date.max)
        self._last_day = self.first_day - timedelta(days=1)
        self.fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "version": VERSION,
                    "source": store.get("fingerprint"),
                    "events": self.events,
                    "window": WINDOW_DAYS,
                    "min_fit": MIN_FIT_PISTOLS,
                    "prior": PRIOR_INFORMATION,
                },
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()

    def _fit_day(self, day: date) -> PistolSnapshot:
        self._eligible.extend(self._received.get(day, []))
        floor = day - timedelta(days=WINDOW_DAYS)
        self._eligible = [e for e in self._eligible if e["played"] >= floor]
        snapshot = PistolSnapshot(
            day, fit_n=len(self._eligible), fit_matches=len({e["match_id"] for e in self._eligible})
        )
        if snapshot.ready and len({e["y"] for e in self._eligible}) == 2:
            matrix = np.asarray(
                [[(e["elo_ct"] - e["elo_t"]) / 400, *[float(e["map"] == m) for m in MAPS]] for e in self._eligible]
            )
            model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=300, random_state=42)
            model.fit(matrix, np.asarray([e["y"] for e in self._eligible]))
            snapshot.beta = float(model.coef_[0, 0])
            snapshot.ct_intercept = float(model.intercept_[0])
            snapshot.map_bias = {m: float(v) for m, v in zip(MAPS, model.coef_[0, 1:], strict=True)}
        else:
            # Counts alone must never advertise a fitted model for a single class.
            snapshot.fit_n = 0
        contributions: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        raw: dict[str, list[float]] = defaultdict(list)
        for e in self._eligible:
            for team, actual, expected, rival_elo in (
                (e["ct"], e["y"], e.get("expected_ct"), e["elo_t"]),
                (e["t"], 1 - e["y"], 1 - e["expected_ct"] if e.get("expected_ct") is not None else None, e["elo_ct"]),
            ):
                raw[team].append(actual)
                if expected is not None:
                    contributions[team].append((actual, expected, rival_elo))
        snapshot.raw = {t: ((sum(v) + 10) / (len(v) + 20), len(v)) for t, v in raw.items()}
        for team, values in contributions.items():
            actual, expected, rivals = zip(*values, strict=True)
            snapshot.teams[team] = {
                "n": float(len(values)),
                "offset": residual_offset(list(actual), list(expected)),
                "observed": sum(actual) / len(values),
                "expected": sum(expected) / len(values),
                "mean_opponent_elo": sum(rivals) / len(values),
            }
        return snapshot

    def snapshot(self, as_of: str | date) -> PistolSnapshot:
        """Replaying a later capture can never amend an earlier frozen snapshot."""
        day = civil_date(as_of)
        if day < self.first_day:
            return PistolSnapshot(day)
        while self._last_day < day:
            current = self._last_day + timedelta(days=1)
            snapshot = self._fit_day(current)
            self.snapshots[current] = snapshot
            # All pistols of all matches of D use the same morning state.
            for event in self._played.get(current, []):
                gap = event["elo_ct"] - event["elo_t"]
                values = snapshot.features(event["ct"], event["t"], gap)
                if snapshot.ready:
                    event["expected_ct"] = snapshot.contextual(gap, event["map"])
                    raw_a = snapshot.raw.get(event["ct"], (0.5, 0))[0]
                    raw_b = snapshot.raw.get(event["t"], (0.5, 0))[0]
                    raw_p = sigmoid(math.log(raw_a / (1 - raw_a)) - math.log(raw_b / (1 - raw_b)))
                    # Canonical ID orientation is independent of who won/played CT.
                    reverse = event["ct"] > event["t"]
                    orient = lambda p: 1.0 - p if reverse else p
                    self.predictions.append(
                        {
                            "match_id": event["match_id"],
                            "map_id": event["map_id"],
                            "half": event["half"],
                            "date": current.isoformat(),
                            "actual": int(orient(event["y"])),
                            "elo_only": orient(values["pistol_opponent_baseline_centered"] + 0.5),
                            "adjusted": orient(values["pistol_opponent_probability_centered"] + 0.5),
                            "raw_rate": orient(raw_p),
                            "available": values["pistol_opponent_available"],
                            "fit_n": snapshot.fit_n,
                        }
                    )
            self._last_day = current
        return self.snapshots[day]

    def features(self, a: str, b: str, key_a: str, key_b: str, as_of: str | date) -> dict[str, float]:
        day = civil_date(as_of)
        gap = self.elo.rating(key_a, day) - self.elo.rating(key_b, day)
        return self.snapshot(day).features(str(a), str(b), gap)

    def match_features(self, row: dict[str, Any]) -> dict[str, float]:
        return self.features(
            str(row.get("team1_id")),
            str(row.get("team2_id")),
            row["team1_key"],
            row["team2_key"],
            str(row["date"])[:10],
        )
