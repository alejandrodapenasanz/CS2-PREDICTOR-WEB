"""Read-only model probe for a specified code+VAULT root and isolated environment.

Run with the owning component's Python: --component CS2|TENNIS --root PATH.
Uses fixed synthetic inputs to test identical inference after relocation, not
to publish picks, estimate accuracy, retrain or alter sacred data.
"""

import argparse
from datetime import date, timedelta
import json
from pathlib import Path
import sys


def probe(root: Path, component: str) -> dict:
    """Load exact live artifacts and emit comparable path-independent evidence."""
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / component))
    if component == "CS2":
        sys.path.insert(0, str(root / "CS2/MODEL"))
        from cs2model.artifacts import load_artifact
        from cs2model.promotion import read_model_pointer

        model = load_artifact()
        if model is None:
            raise RuntimeError("Missing live CS2 model")
        registry = root / "VAULT/CS2/MODEL/artifacts/registry"
        live = read_model_pointer(registry, "latest")
        previous = read_model_pointer(registry, "last_good")
        row = {column: 0.0 for column in model.feature_columns}
        row.update(glicko_diff=120.0, elo_diff=120.0)
        probabilities = model.predict_proba_team1([row]).tolist()
        return {"live": live.sha256, "previous": previous.sha256, "probabilities": probabilities}

    import pandas as pd
    from src.config import ELO_HANDOFF_CONFIG_PATH, STATE_ROOT
    from src.elo.operational import load_elo_handoff_config
    from src.modeling.service import load_active_deployment_model

    handoff = load_elo_handoff_config(ELO_HANDOFF_CONFIG_PATH, project_root=STATE_ROOT)
    result = {"base_commit": handoff.base_source_commit, "models": {}}
    for gender in ("M", "F"):
        model = load_active_deployment_model(gender)
        row = {column: 0.0 for column in model.estimator.feature_columns}
        row.update(gender=gender, surface="Hard", tour_level="A", round="R32", best_of=3)
        row.update(elo_general_diff=125.0, elo_surface_diff=80.0, rank_diff=-12.0)
        cutoff = date.fromisoformat(model.training_available_max_date) + timedelta(days=1)
        values = model.predict(pd.DataFrame([row]), as_of_date=cutoff)
        result["models"][gender] = {
            "fingerprint": model.run_fingerprint,
            "probabilities": values["model_probability_a"].tolist(),
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--component", choices=("CS2", "TENNIS"), required=True)
    args = parser.parse_args()
    print(json.dumps(probe(args.root.resolve(), args.component), sort_keys=True))
