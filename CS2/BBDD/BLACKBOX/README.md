# BLACKBOX — caja negra de la base de datos CS2

Esta carpeta es un **respaldo portátil y autocontenido** de la **fuente de verdad**
de la base de datos de CS2-Predictor. Guárdala en un USB y en la nube.

**Con solo esta carpeta puedes reconstruir la base de datos completa**, aunque
pierdas el resto del proyecto. No necesitas internet ni volver a scrapear HLTV.

---

## 1. Qué contiene (y qué NO)

Aquí se guarda **solo lo que NO se puede recomputar**: los hechos crudos y
canónicos (partidos, mapas, box scores, odds, rankings, snapshots raw, etc.) y
el `prediction_ledger` prospectivo completo. El ledger se copia y restaura sin
reescribir sus filas.

**NO** se guarda lo *derivado*, porque se regenera solo a partir de los hechos:
- `ratings_history` y `match_features` → los recalcula el modelo (Glicko-2 y
  features point-in-time).
- `fetch_state` (caché del scraper) e `ingest_runs` (log) → se rehacen solos.

La lista exacta de tablas incluidas/excluidas está en `manifest.json`
(`source_of_truth_tables` / `derived_excluded_tables`).

```
BLACKBOX/
├── README.md                     <- este fichero
├── restore_standalone.py         <- restaurador sin dependencias del proyecto
├── manifest.json                 <- versión, fecha, SHA-256 de cada fichero y tabla
├── schema.sql                    <- esquema canónico de la BBDD
└── data/
    ├── cs2_blackbox.sqlite.gz    <- CANÓNICO: SQLite con las tablas fuente (gzip)
    └── cs2_blackbox.sql.gz       <- fallback: volcado SQL de texto (gzip)
```

---

## 2. Cómo restaurar

### Opción A — con el restaurador incluido (recomendada, solo Python 3)

```bash
# desde dentro de esta carpeta:
python restore_standalone.py --target ./cs2.db          # verifica y restaura
python restore_standalone.py --verify-only              # solo comprobar integridad
```
Coloca la `cs2.db` resultante en `CS2/BBDD/cs2.db` del proyecto.

### Opción B — con el proyecto (regenera además lo derivado)

```powershell
cd CS2
python BBDD\blackbox.py restore --blackbox BBDD\BLACKBOX --db BBDD\cs2.db
```
Esto restaura y luego ejecuta `build_db.py` (migraciones + índices). Después,
`python MODEL\train.py` reconstruye ratings/features/artefacto.

### Opción C — 100% a mano, sin ningún script (solo `sqlite3` y `gzip`)

```bash
# a partir del volcado de texto (lo más portable que existe):
gzip -dk data/cs2_blackbox.sql.gz          # -> data/cs2_blackbox.sql
sqlite3 cs2.db < data/cs2_blackbox.sql

# o a partir del SQLite canónico directamente:
gzip -dk data/cs2_blackbox.sqlite.gz       # -> data/cs2_blackbox.sqlite
mv data/cs2_blackbox.sqlite cs2.db
```
En Windows sin `gzip`: cualquier descompresor (7-Zip) abre los `.gz`.

---

## 3. Cómo verificar la integridad a mano

Cada fichero tiene su SHA-256 en `manifest.json`. Para comprobarlo:

```bash
# Linux/macOS
sha256sum data/cs2_blackbox.sqlite.gz
# Windows PowerShell
Get-FileHash data\cs2_blackbox.sqlite.gz -Algorithm SHA256
```
Compara el resultado con el valor de ese fichero en `manifest.json > files`.
`python restore_standalone.py --verify-only` hace esta comprobación por ti.

---

## 4. Qué hago después de restaurar

La `cs2.db` restaurada ya contiene todos los hechos. Para tener el sistema
completo (predicciones, web), desde el proyecto:

```powershell
cd CS2
python MODEL\train.py                      # regenera el modelo y las features
python ..\WEB\build_web.py --sport-root .  # regenera el dashboard
```

---

## 5. Cómo se genera / actualiza esta carpeta

Desde el proyecto, tras un pipeline con datos nuevos:

```powershell
cd CS2
python BBDD\blackbox.py export             # reescribe esta carpeta desde cs2.db
# o dentro del pipeline:
.\start.ps1 -BackupBlackbox
```
El export es atómico y se auto-verifica antes de publicar; conserva la versión
anterior en `.prev/`. **Vuelve a copiar esta carpeta al USB/nube tras cada
export importante.**

---

*Formato: SQLite + volcado SQL, ambos abiertos y de archivo. Sin dependencias
propietarias. Versión de formato y fecha de creación en `manifest.json`.*
