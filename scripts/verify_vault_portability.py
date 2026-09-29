"""Copy code + live/previous models to a new path and compare real inference.

No training, SQL writes, network, Telegram sends or production pointer changes.
The disposable copy contains models and the frozen source manifest, not another
copy of the 10GB acquisition database. Full DB integrity is checked separately
by vault_audit.py. Run with CPython 3.13 after migration.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

from vault_migrate import ROOT, save_json


def model_probe(root: Path, component: str) -> dict:
    """Run a fresh owning-domain process; imports must come from the given copy."""
    python = ROOT / "VAULT" / component / ".venv/Scripts/python.exe"
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run(
        [str(python), str(ROOT / "scripts/probe_vault_models.py"), "--root", str(root), "--component", component],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"{component} probe failed: {completed.stderr}")
    return json.loads(completed.stdout)


def main() -> None:
    """Validate independent copied model bytes and remove only the owned fixture."""
    verification = ROOT / "VAULT/verification"
    verification.mkdir(parents=True, exist_ok=True)
    before = {component: model_probe(ROOT, component) for component in ("CS2", "TENNIS")}
    with TemporaryDirectory(prefix="portable-models-", dir=verification) as temporary:
        copied = Path(temporary) / "checkout with spaces"
        copied.mkdir()
        assert copied.resolve().is_relative_to(verification.resolve())
        for relative in ("CS2/MODEL/cs2model", "TENNIS/src", "TENNIS/config"):
            shutil.copytree(ROOT / relative, copied / relative, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(ROOT / "CS2/MODEL/config.yaml", copied / "CS2/MODEL/config.yaml")
        target = copied / "VAULT/CS2/MODEL/artifacts/registry"
        target.mkdir(parents=True)
        original = ROOT / "VAULT/CS2/MODEL/artifacts/registry"
        versions = set()
        for pointer in ("latest.json", "last_good.json"):
            payload = json.loads((original / pointer).read_text(encoding="utf-8"))
            versions.add(payload.get("version") or payload.get("latest"))
            shutil.copy2(original / pointer, target / pointer)
        for version in versions:
            shutil.copytree(original / version, target / version)
        shutil.copy2(original.parent / "model.pkl", target.parent / "model.pkl")
        source = ROOT / "VAULT/TENNIS/models/phase7"
        destination = copied / "VAULT/TENNIS/models/phase7"
        destination.mkdir(parents=True)
        active = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        shutil.copy2(source / "manifest.json", destination / "manifest.json")
        shutil.copytree(source / active["active_run"], destination / active["active_run"])
        handoff = json.loads((ROOT / "TENNIS/config/elo_handoff.json").read_text(encoding="utf-8"))
        relative = Path("TENNIS") / handoff["base"]["manifest_path"]
        (copied / "VAULT" / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "VAULT" / relative, copied / "VAULT" / relative)
        shutil.copy2(ROOT / "VAULT/migration.json", copied / "VAULT/migration.json")
        after = {component: model_probe(copied, component) for component in ("CS2", "TENNIS")}
        if before != after:
            raise RuntimeError("Model fingerprint/prediction changed after relocation")
        report = {"status": "ok", "different_path_with_spaces": True, "exact_match": True, "models": after}
        save_json(verification / "portability.json", report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
