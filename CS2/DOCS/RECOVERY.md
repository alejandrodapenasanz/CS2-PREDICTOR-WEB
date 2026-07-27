# RECOVERY — cómo guardar y recomponer el proyecto

> Guía de continuidad. Responde a: *"quiero poder guardarme la carpeta y, si me llega sin datos,
> recomponerla"*. Escrita el 2026-07-27 tras la auditoría (`docs/AUDIT.md`).

---

## 1. Idea en una frase

**El código es reproducible; los datos NO están en git.** Un clon limpio trae *solo* el código.
Para recomponer necesitas **el código (git) + al menos un artefacto de datos** (ver §3). Con eso,
todo lo derivado (ratings, features, predicciones, artefacto del modelo, web) se **recalcula**.

---

## 2. Qué es código (versionado) y qué es dato (fuera de git)

| Tipo | Ejemplos | ¿En git? | Cómo se recupera |
|---|---|---|---|
| **Código / config** | `MODEL/`, `PIPELINE/*.py`, `BBDD/*.py`, `*.sql`, `config.yaml`, `start.ps1`, docs | ✅ Sí | `git clone` |
| **Fuente de verdad** | `BBDD/cs2.db` | ❌ No (`.gitignore`) | Backup manual (§3) |
| **Seed histórico** | `.../data/raw/history_10000_2026-06-28/results_all.json` | ❌ No | Backup manual |
| **Master compat** | `PIPELINE/master/matches.json`, `manifest.json`, `roster_history.json` | ❌ No | Backup / re-export |
| **Runs crudos** | `PIPELINE/runs/<RUN_ID>/` (snapshots, odds, analytics, assets) | ❌ No | Backup / re-scrape |
| **Artefacto modelo** | `MODEL/artifacts/model.pkl`, `registry/` | ❌ No | Reentrenar (`train.py`) |
| **Informes generados** | `MODEL/results/*.json/*.csv`, `WEB/data.js` | ❌ No | Se regeneran solos |
| **Secretos** | `cf_session.json`, `corp_ca_bundle.pem` | ❌ No (correcto) | Se regeneran en runtime |

**Backups existentes** (según PROJECT.md §8): cada `ingest` deja copia en `BBDD/backups/` y espejo
en `../CS2-Predictor-Backups/` (variable `CS2_BACKUP_MIRROR_DIR`). **Esos backups son tu red de
seguridad real** — cópialos a un sitio duradero (nube/disco externo).

---

## 2bis. La forma recomendada: BLACKBOX (implementado)

Existe una herramienta dedicada que empaqueta **solo la fuente de verdad** en una
carpeta portátil `CS2/BBDD/BLACKBOX/` lista para USB/nube, con checksums y
restauración manual documentada. Es el mecanismo preferido sobre copiar `cs2.db`
a mano.

```powershell
cd CS2
python BBDD\blackbox.py export          # crea/actualiza BBDD\BLACKBOX
python BBDD\blackbox.py verify           # comprueba integridad (SHA-256)
python BBDD\blackbox.py restore          # reconstruye cs2.db (con guardián)
# integrado en el pipeline:
.\start.ps1 -BackupBlackbox              # exporta al terminar
.\start.ps1 -RestoreBlackbox             # restaura al empezar
# y auto-heal automático al inicio si la BBDD falta/está vacía/corrupta.
```

Qué guarda BLACKBOX (no recomputable) y qué excluye (derivado/operacional) está
en `CS2/BBDD/BLACKBOX/README.md` y en el `manifest.json`. Copia esa carpeta al
USB/nube tras cada `-BackupBlackbox`.

## 3. Qué respaldar (mínimo → ideal)

Elige según cuánto quieras poder reconstruir:

- **Mínimo imprescindible (recuperación total del histórico):** **`BBDD/cs2.db`**.
  Es la fuente de verdad; con ella se regenera todo lo demás. Un único fichero.
- **Alternativa equivalente:** el **seed** `results_all.json` **+** `PIPELINE/master/matches.json`
  **+** (si quieres odds/analytics/box scores) la carpeta `PIPELINE/runs/`. Con esto `build_db.py`
  reconstruye `cs2.db` desde cero.
- **Ideal (arranque instantáneo sin recalcular):** `cs2.db` **+** `MODEL/artifacts/model.pkl`.
  Así ni siquiera hay que reentrenar para tener predicciones/web.

Comandos para sacar una copia SQL legible de la BBDD (de PROJECT.md §8):

```powershell
sqlite3 BBDD\cs2.db ".backup 'BBDD\backups\cs2_manual_backup.db'"
sqlite3 BBDD\cs2.db ".dump" > BBDD\cs2_dump.sql
```

---

## 4. Escenarios de recuperación

### 4.1. Tengo `cs2.db` (el caso feliz)

```powershell
cd CS2
# 1) coloca cs2.db en CS2/BBDD/cs2.db
python BBDD\build_db.py          # asegura/migra el esquema vivo (no borra datos)
python MODEL\train.py            # regenera MODEL/artifacts/model.pkl desde la BBDD
python ..\WEB\build_web.py --sport-root .   # regenera WEB/data.js
```
Todo lo derivado queda reconstruido. Las features/ratings/predicciones son vistas point-in-time
recalculables, así que no importa que no tuvieras los .json de resultados.

### 4.2. Tengo el seed `results_all.json` (o master) pero no la `.db`

```powershell
cd CS2
python BBDD\build_db.py --raw <ruta\results_all.json> --master PIPELINE\master\matches.json
python MODEL\train.py
python ..\WEB\build_web.py --sport-root .
```
`build_db.py` siembra la BBDD desde el JSON **solo si `matches` está vacía**
([build_db.py:18-20](../BBDD/build_db.py#L18-L20)).

### 4.3. No tengo NINGÚN dato (solo el clon de git)

Realidad: **no se puede reconstruir el histórico completo** — el seed de ~10k series no está en
git. Lo que sí puedes hacer es **arrancar una base viva desde hoy**:

```powershell
cd CS2
.\start.ps1              # scrapea HLTV en vivo y empieza a poblar cs2.db (requiere red + Python 3.10-3.13)
```
A partir de ahí, cada `.\start.ps1` va acumulando partidos nuevos, odds, analytics y contexto. El
histórico profundo solo vuelve si recuperas un backup (§3) o repites el backfill masivo original.

### 4.4. Tengo `cs2.db` pero está corrupta / dudo de su integridad

```powershell
cd CS2
sqlite3 BBDD\cs2.db "PRAGMA foreign_key_check;"      # debe devolver 0 filas
python BBDD\deduplicate_matches.py                    # debe dar duplicate_pairs=0
```

---

## 5. Regla de oro para no perder nada

1. **Versiona el código** (ya está en git, rama `pre-dev`).
2. **Respalda `cs2.db`** (o el seed) a un sitio duradero, con fecha. Es un fichero; hazlo tras
   cada `.\start.ps1 -Retrain` importante. Los backups automáticos de `BBDD/backups/` ayudan, pero
   viven en la misma carpeta: cópialos fuera.
3. **No versiones datos ni secretos** en git (el `.gitignore` ya lo impide; respétalo).
4. Todo lo demás (ratings, features, `model.pkl`, `data.js`, informes) es **derivado y
   recalculable** desde 1+2.

---

## 6. Estado de automatización (hecho)

La comprobación/reconstrucción automática ya está implementada como **BLACKBOX**
(ver §2bis): `BBDD/blackbox.py` (`export`/`verify`/`restore`/`autoheal`) + su
integración en `start.ps1` (`-BackupBlackbox`, `-RestoreBlackbox`, auto-heal al
inicio) + el test de desastre `TESTS/test_blackbox_disaster.py`. Entregado en la
rama `feature/blackbox`.

Flujo de continuidad recomendado:
1. `.\start.ps1 -Retrain -BackupBlackbox` cuando incorpores datos importantes.
2. Copia `CS2/BBDD/BLACKBOX/` al USB y a la nube.
3. Si algún día abres el proyecto sin datos, `.\start.ps1` detecta la BBDD
   ausente y la restaura sola desde BLACKBOX antes de seguir.
