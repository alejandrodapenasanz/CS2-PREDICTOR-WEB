"""Domain-owned maintenance entrypoint used by scripts/soluciona_errores.py."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
STATE = REPO / "VAULT/CS2"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from BBDD.fetch_recovery import persist_recovery, plan_recovery, readonly, utcnow
from PIPELINE.operation_lock import operation_lock


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def execute(*, diagnose: bool, limit: int) -> int:
    db = STATE / "BBDD/cs2.db"
    master_path = STATE / "PIPELINE/master/matches.json"
    master = json.loads(master_path.read_text(encoding="utf-8")) if master_path.exists() else {}
    with closing(readonly(db)) as conn:
        plan = plan_recovery(conn, master, now=utcnow(), limit=limit)
    print(
        f"SQLite: OK. Incidencias de descarga: {plan['issue_count']}; sin muestra (no errores): {plan['no_sample_count']}.",
        flush=True,
    )
    print(f"Por entidad: {plan['by_entity']}", flush=True)
    print(
        f"Reintentables: {len(plan['selected'])}; aplazados: {len(plan['deferred'])}; falsos OK: {len(plan['corrections'])}.",
        flush=True,
    )
    reasons = Counter(item["deferred_reason"] for item in plan["deferred"])
    print(f"Motivos de aplazamiento: {dict(reasons)}", flush=True)
    if diagnose:
        for item in plan["selected"]:
            print(f"  {item['entity_type']} {item['entity_key']}: {item['url']}")
        return 2 if plan["issue_count"] else 0
    python = (
        STATE / "SCRAPER/hltv-scraper-api/.venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    )
    if plan["selected"] and not python.is_file():
        raise RuntimeError("Falta el entorno del scraper. Ejecuta start.ps1 para prepararlo.")
    token = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_%fZ")
    run_dir = STATE / "PIPELINE/maintenance" / token
    write_report(run_dir / "plan.json", plan)
    # This is deliberately NOT a daily publication and never updates master/latest.
    write_report(run_dir / "manifest.json", {"started_at": utcnow(), "kind": "fetch_recovery"})
    child_code = 0
    if plan["selected"]:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["HLTV_MAX_HTTP_REQUESTS_PER_RUN"] = str(
            min(150, max(1, int(env.get("HLTV_MAX_HTTP_REQUESTS_PER_RUN", "150"))))
        )
        env.setdefault("HLTV_FETCH_MIN_INTERVAL", "2.5")
        env.setdefault("HLTV_SOLVE_CLOUDFLARE", "0")
        try:
            child_code = subprocess.run(
                [str(python), str(ROOT / "PIPELINE/fetch_recovery_worker.py"), "--plan", str(run_dir / "plan.json")],
                cwd=ROOT,
                env=env,
                check=False,
            ).returncode
        except KeyboardInterrupt:
            print("\nInterrumpido. Las capturas terminadas quedan guardadas; no se repite la descarga.")
            child_code = 130
    conn = sqlite3.connect(db.resolve().as_uri() + "?mode=rw", uri=True, timeout=30)
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        counts = persist_recovery(conn, plan, run_dir)
    finally:
        conn.close()
    report_path = run_dir / "results.json"
    results = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {"results": []}
    with closing(readonly(db)) as conn:
        pending = [
            dict(row)
            for row in conn.execute(
                "SELECT entity_type,entity_key,last_status,note,next_eligible_at_utc FROM fetch_state "
                "WHERE last_status IN ('blocked','error','partial') ORDER BY entity_type,entity_key"
            )
        ]
    remaining = len(pending)
    if child_code and not report_path.exists():
        results.update(
            stopped_reason="worker_failed_before_report",
            unattempted=[
                {
                    "entity_type": item["entity_type"],
                    "entity_key": item["entity_key"],
                    "reason": "worker_failed_before_report",
                }
                for item in plan["selected"]
            ],
        )
    summary = {
        "run_dir": str(run_dir),
        "finished_at": utcnow(),
        "before": plan["issue_count"],
        "remaining": remaining,
        "attempted": sum(bool(r.get("attempted")) for r in results["results"]),
        "statuses": dict(Counter(r["status"] for r in results["results"] if r.get("attempted"))),
        "deferred": [
            {k: item.get(k) for k in ("entity_type", "entity_key", "deferred_reason", "next_eligible_at_utc")}
            for item in plan["deferred"]
        ],
        "unattempted": results.get("unattempted", []),
        "stopped_reason": results.get("stopped_reason"),
        "rows_inserted": counts,
        "worker_exit_code": child_code,
        "predictions_untouched": True,
        "pending": pending,
    }
    # Update only presentation through its real entrypoint. No training, Telegram or full ingest.
    web_code = subprocess.run([sys.executable, str(REPO / "WEB/build_web.py")], cwd=REPO, check=False).returncode
    summary["web_exit_code"] = web_code
    write_report(run_dir / "summary.json", summary)
    write_report(STATE / "PIPELINE/maintenance/latest.json", summary)
    print(f"Recuperación: {summary['statuses']}. Siguen pendientes: {remaining}.", flush=True)
    if summary["stopped_reason"]:
        print(
            f"Parada de seguridad: {summary['stopped_reason']}; sin intentar: {len(summary['unattempted'])}.",
            flush=True,
        )
    print(f"Informe: {run_dir / 'summary.json'}", flush=True)
    print(
        "No se han reentrenado modelos ni modificado predicciones. start.ps1 usará las capturas nuevas cuando sean causales."
    )
    if child_code not in (0, 2) or web_code:
        return 130 if child_code == 130 else 1
    return 2 if remaining else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Recuperación acotada de fetch_state de CS2.")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 13):
        parser.error("Se requiere CPython 3.13")
    if args.limit < 0:
        parser.error("--limit debe ser 0 (todas) o un entero positivo")
    try:
        with operation_lock(STATE / "PIPELINE/operation.lock"):
            return execute(diagnose=args.diagnose, limit=args.limit)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError, sqlite3.DatabaseError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
