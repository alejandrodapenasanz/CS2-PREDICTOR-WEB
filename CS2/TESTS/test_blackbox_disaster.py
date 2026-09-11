"""Simulacro de DESASTRE para la caja negra BLACKBOX.

Verifica el ciclo completo sobre una BBDD sintetica (sin red, sin datos reales):

  1. export -> escribe BLACKBOX y auto-verifica.
  2. verify -> los checksums del manifest cuadran.
  3. desastre -> se mueve cs2.db a un lado.
  4. restore -> reconstruye cs2.db desde BLACKBOX.
  5. checksums -> el contenido de CADA tabla fuente es identico al original.
  6. lo derivado (ratings_history) NO se guarda y queda vacio tras restaurar.
  7. guardian -> restore sin --force sobre una BBDD sana se rechaza.
  8. autoheal -> sobre una BBDD vacia, restaura sola.
  9. dry-run -> el lector de entrenamiento (dataio) carga las mismas filas.
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # .../CS2
sys.path.insert(0, str(ROOT / "BBDD"))
sys.path.insert(0, str(ROOT / "MODEL"))

import blackbox as bb  # noqa: E402

SCHEMA_SQL = ROOT / "BBDD" / "cs2_prediction_schema.sql"


def _make_synthetic_db(db_path: Path) -> None:
    """BBDD minima pero valida: filas en tablas fuente + una derivada."""
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL.read_text(encoding="utf-8"))
        conn.execute("INSERT INTO teams(team_id,name,country,hltv_id) VALUES (1,'Alpha','SE',101),(2,'Beta','DK',102)")
        conn.execute("INSERT INTO players(player_id,nick,hltv_id) VALUES (1,'p1',201),(2,'p2',202)")
        conn.execute("INSERT INTO events(event_id,name,hltv_event_id,is_lan,tier) VALUES (1,'Major 2025','E1',1,'S')")
        conn.execute(
            "INSERT INTO team_rosters(roster_id,team_id,player_id,valid_from) VALUES (1,1,1,'2025-01-01'),(2,2,2,'2025-01-01')"
        )
        conn.execute(
            "INSERT INTO matches(match_id,hltv_match_id,event_id,datetime_utc,team1_id,team2_id,"
            "best_of,status,data_tier,winner_team_id,score_t1,score_t2) "
            "VALUES (1,'1000001',1,'2025-06-01T10:00:00Z',1,2,3,'completed','completed',1,2,1)"
        )
        conn.execute(
            "INSERT INTO maps(map_id,match_id,map_number,map_name,winner_team_id,rounds_t1,rounds_t2) "
            "VALUES (1,1,1,'Mirage',1,13,7),(2,1,2,'Inferno',2,10,13),(3,1,3,'Nuke',1,13,11)"
        )
        conn.execute(
            "INSERT INTO veto(veto_id,match_id,step_order,team_id,action,map_name) VALUES (1,1,1,1,'ban','Anubis')"
        )
        conn.execute(
            "INSERT INTO map_round_sources(round_source_id,map_id,source_file,source_url,sha256,"
            "captured_at_utc,played_at_utc,round_format,map_name,team_left_hltv_id,team_right_hltv_id,parser_version) "
            "VALUES(1,1,'fixture.html.gz','https://www.hltv.org/stats/matches/mapstatsid/1/a-b',"
            "'fixture-sha','2025-06-01T12:00:00Z','2025-06-01T10:00:00Z','mr12','Mirage','101','102',1)"
        )
        conn.execute("INSERT INTO map_rounds VALUES(1,1,1,1,0,1,'101','ct','ct_win.svg')")
        conn.execute("INSERT INTO match_lineups(match_id,team_id,player_id,is_standin) VALUES (1,1,1,0),(1,2,2,0)")
        conn.execute(
            "INSERT INTO map_player_stats(map_id,player_id,team_id,kills,deaths,adr,kast,rating) "
            "VALUES (1,1,1,20,10,85.5,75.0,1.30)"
        )
        conn.execute(
            "INSERT INTO odds(odds_id,match_id,bookmaker,captured_at_utc,market_type,odds_t1,odds_t2,prob_t1,prob_t2) "
            "VALUES (1,1,'bookie','2025-05-31T00:00:00Z','opening',1.5,2.5,0.62,0.38)"
        )
        conn.execute(
            "INSERT INTO predictions(prediction_id,match_id,hltv_match_id,model_version,predicted_at_utc,prob_team1) "
            "VALUES (1,1,'1000001','m@2025',' 2025-05-31T00:00:00Z',0.6)"
        )
        conn.execute(
            "INSERT INTO prediction_ledger(ledger_id,match_id,hltv_match_id,team1_id,team2_id,"
            "kickoff_utc,predicted_at_utc,model_version,prob_team1,prediction_regime,"
            "prediction_json,ledger_status,result_filled_at_utc,actual_team1_win,"
            "prediction_correct,realized_log_loss,realized_brier,created_at_utc,updated_at_utc) "
            "VALUES (1,1,'1000001',1,2,'2025-06-01T10:00:00Z','2025-05-31T00:00:00Z',"
            "'m@2025',0.6,'odds','{}','evaluated','2025-06-01T12:00:00Z',1,1,"
            "0.5108256237659907,0.16,'2025-05-31T00:00:00Z','2025-06-01T12:00:00Z')"
        )
        conn.execute(
            "INSERT INTO raw_snapshots(raw_snapshot_id,kind,run_id,source_file,payload_json) "
            "VALUES (1,'match_snapshot','R1','runs/R1/x.json','{\"a\":1}')"
        )
        # DERIVADA: debe quedar FUERA de la caja negra.
        conn.execute(
            "INSERT INTO ratings_history(rating_row_id,entity_type,entity_id,before_match_id,as_of_date,rating,rd) "
            "VALUES (1,'team',1,1,'2025-06-01T10:00:00Z',1500.0,60.0)"
        )
        conn.commit()
    finally:
        conn.close()


def _table_hash(db_path: Path, table: str) -> str:
    conn = sqlite3.connect(db_path)
    try:
        return bb.table_content_hash(conn, table)
    finally:
        conn.close()


def _count(db_path: Path, table: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return bb._table_row_count(conn, table)
    finally:
        conn.close()


class BlackboxDisasterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.work = Path(tempfile.mkdtemp(prefix="bb_disaster_"))
        self.db = self.work / "cs2.db"
        self.blackbox = self.work / "BLACKBOX"
        self.backups = self.work / "backups"
        _make_synthetic_db(self.db)

    def tearDown(self) -> None:
        shutil.rmtree(self.work, ignore_errors=True)

    def test_disaster_restore_roundtrip(self) -> None:
        # 1) export + auto-verify.
        self.assertEqual(bb.export(self.db, self.blackbox), 0, "export deberia devolver 0")
        # 2) verify profundo.
        self.assertEqual(bb.verify(self.blackbox, deep=True), 0, "verify deberia devolver 0")

        # Huella de las tablas fuente ANTES del desastre.
        before = {t: _table_hash(self.db, t) for t in bb.SOURCE_OF_TRUTH_TABLES}

        # Filas de entrenamiento que ve el pipeline ANTES (dry-run de referencia).
        rows_before = self._training_rows(self.db)

        # 3) DESASTRE: mover cs2.db a un lado.
        aside = self.work / "cs2.db.disaster"
        shutil.move(str(self.db), str(aside))
        self.assertFalse(self.db.exists())

        # 4) restore desde la caja negra (sin build_db para no depender de deps ML).
        rc = bb.restore(self.db, self.blackbox, run_build_db=False, backup_dir=self.backups)
        self.assertEqual(rc, 0, "restore deberia devolver 0")
        self.assertTrue(self.db.exists(), "cs2.db deberia existir tras restore")

        # 5) checksums: contenido IDENTICO tabla a tabla.
        after = {t: _table_hash(self.db, t) for t in bb.SOURCE_OF_TRUTH_TABLES}
        for t in bb.SOURCE_OF_TRUTH_TABLES:
            self.assertEqual(before[t], after[t], f"la tabla fuente '{t}' difiere tras restaurar")
        self.assertEqual(_count(self.db, "prediction_ledger"), 1)

        # 6) la tabla DERIVADA no se guardo y queda vacia.
        self.assertGreater(_count(aside, "ratings_history"), 0, "el original tenia una fila derivada")
        self.assertEqual(_count(self.db, "ratings_history"), 0, "lo derivado NO debe restaurarse")

        # 7) guardian: restaurar sobre una BBDD SANA sin --force se rechaza.
        rc_guard = bb.restore(self.db, self.blackbox, force=False, run_build_db=False, backup_dir=self.backups)
        self.assertEqual(rc_guard, 4, "el guardian deberia rechazar pisar una BBDD sana")

        # 8) autoheal: sobre una BBDD vacia, restaura sola.
        self.db.unlink()
        sqlite3.connect(self.db).close()  # fichero vacio, sin tablas -> 'empty'
        self.assertEqual(bb._db_health(self.db), "empty")
        rc_heal = bb.autoheal(self.db, self.blackbox, backup_dir=self.backups, run_build_db=False)
        self.assertEqual(rc_heal, 0)
        self.assertEqual(_count(self.db, "matches"), 1, "autoheal deberia haber restaurado los partidos")

        # 9) dry-run del pipeline: el lector de entrenamiento carga lo mismo.
        rows_after = self._training_rows(self.db)
        self.assertEqual(rows_before, rows_after, "el pipeline debe leer las mismas filas tras restaurar")
        self.assertGreaterEqual(rows_after, 1, "deberia haber al menos una serie entrenable")

    def _training_rows(self, db_path: Path) -> int:
        """Pasada dry-run: numero de series entrenables que ve dataio.

        Si el stack ML (numpy) no esta disponible, cae a una lectura SQL directa
        equivalente para no depender de dependencias pesadas.
        """
        try:
            from cs2model import dataio  # noqa: E402

            return len(dataio.load_training_rows_from_db(db_path))
        except ImportError:
            conn = sqlite3.connect(db_path)
            try:
                return int(
                    conn.execute(
                        "SELECT COUNT(*) FROM matches WHERE status='completed' "
                        "AND score_t1 IS NOT NULL AND score_t2 IS NOT NULL AND score_t1 <> score_t2"
                    ).fetchone()[0]
                )
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
