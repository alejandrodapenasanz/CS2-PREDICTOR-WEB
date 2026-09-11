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
| **Artefacto modelo** | copia runtime `MODEL/artifacts/model.pkl`; versiones y punteros `registry/latest.json`, `registry/last_good.json` | ❌ No | Rollback desde registry o reentrenar por la puerta |
| **Informes generados** | `MODEL/results/*.json/*.csv`, `WEB/data.js` | ❌ No | Se regeneran solos |
| **Secretos** | `cf_session.json`, `corp_ca_bundle.pem` | ❌ No (correcto) | Se regeneran en runtime |

**Backups existentes** (según PROJECT.md §8): cada `ingest` deja una copia en
`BBDD/backups/`. El espejo está desactivado por defecto para no guardar dos
copias en el mismo disco. Si quieres redundancia real, define
`CS2_BACKUP_MIRROR_DIR` con una ruta de otro disco o almacenamiento
sincronizado. **Esos backups son tu red de seguridad real**: conserva al menos
una copia fuera del equipo.

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
- **Ideal (arranque instantáneo y rollback):** `cs2.db` **+** toda la carpeta
  `MODEL/artifacts/` (`model.pkl`, registry, `latest.json` y `last_good.json`).
  Copiar solo `model.pkl` permite inferencia, pero pierde la versión lógica, sus
  hashes y la vía de rollback.

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
python MODEL\train.py            # genera challenger; la puerta decide la copia runtime
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
.\start.ps1              # scrapea HLTV en vivo y empieza a poblar cs2.db (requiere red + CPython 3.13)
```
A partir de ahí, cada `.\start.ps1` va acumulando partidos nuevos, odds, analytics y contexto. El
histórico profundo solo vuelve si recuperas un backup (§3) o repites el backfill masivo original.

### 4.4. Tengo `cs2.db` pero está corrupta / dudo de su integridad

```powershell
cd CS2
sqlite3 BBDD\cs2.db "PRAGMA foreign_key_check;"      # debe devolver 0 filas
python BBDD\deduplicate_matches.py                    # debe dar duplicate_pairs=0
```

### 4.5. El modelo nuevo degrada producción

No copies manualmente un `.pkl` antiguo encima del vivo. El registro valida
versiones, rutas y SHA-256 y restaura la copia runtime de forma atómica:

```powershell
cd CS2
.\start.ps1 -RollbackModel
# equivalente directo, sin ejecutar el resto del pipeline:
python MODEL\manage_models.py rollback
```

`registry/last_good.json` pasa a ser `latest`, su artefacto se copia a
`MODEL/artifacts/model.pkl` y el vivo saliente rota a `last_good`. Así el rollback
es reversible. Si falta un puntero, una versión sale del registry o cualquier hash
no coincide, la operación falla cerrada y no debe repararse editando JSON a mano.

Para corregir un puntero legado antes del rollback, usa la vía sancionada y el
SHA-256 previamente inspeccionado:

```powershell
python MODEL\manage_models.py set-last-good --version <YYYYMMDD_HHMMSSZ> --expected-sha256 <SHA256>
```

El comando carga y puntúa el candidato con semilla 42 y después revalida bajo
lock/CAS `latest`, el `last_good` anterior, el runtime y el SHA objetivo. Solo
cambia `last_good.json`; no publica el modelo elegido.

Un `-Retrain` posterior tampoco pisa el rollback a ciegas: manual y automático
usan la misma comparación emparejada, delimitada por
`live_cutoff = metadata.date_max` del vivo. El automático se programa aparte:
`attempt_cutoff` es el máximo entre `live_cutoff` y el último intento terminal
válido registrado, y exige al menos 100 etiquetas estrictamente posteriores a
ese corte. Así un rechazo o aplazamiento no se reintenta con idéntica evidencia.
Con menos de 100 filas comunes posteriores a `live_cutoff` se aplaza; si el
challenger no supera log loss por `0.001`, o dentro de la banda de empate no
mejora Brier por `0.0005`, se rechaza y producción permanece byte a byte intacta.

La selección limita `recipe_mask` a `<= live_cutoff`, congela features,
`best_name`, Optuna y pesos, y ajusta el shadow una sola vez con ese prefijo. El
shadow predice todo el sufijo sin refits ni acceso a sus etiquetas; después se
compara con el incumbente. La misma receta se puede refitear sobre todo el
histórico antes de cerrar la decisión porque no influye en el shadow, pero solo
se publica si gana. Esta distinción importa al investigar una regresión: las
métricas de promoción certifican el shadow fijo as-of, no constituyen un test
independiente del pickle full-history. El health gate final valida directamente
`registry/<version>/model.pkl`; la `candidate_reference` comprueba su SHA-256 y
ese mismo hash es la precondición CAS obligatoria.

El bundle núcleo (`model.pkl`, metadata, SHAP, manifest y config) se construye en
staging y se mueve una sola vez a una versión nueva del registry. Desde entonces
es inmutable: no repares sus archivos a mano. Los únicos añadidos posteriores
sancionados son sidecars auditables como `promotion_decision.json` y
`deployment.json`; no cambian el bundle ni el SHA del modelo. Su metadata puede
conservar `promotion_approved` y
`deployment_state=pending_pointer_commit` aunque el commit posterior haya
terminado; esos campos describen la decisión previa, no el estado vivo. Para
saber qué se publicó comprueba `latest.json`, el hash de `model.pkl` y el recibo
`deployment.json` cuando exista.

Para auditar la decisión, `promotion_decision.json` aporta `holdout_sha256` sobre
IDs/fechas/etiquetas y `prediction_sha256` sobre esa identidad más ambas
probabilidades (incumbente y shadow). Una huella de predicciones vacía indica que
no hubo muestra suficiente para calcular las métricas, no una publicación.

Una promoción se serializa mediante `registry/.deployment.lock` y, bajo el lock,
revalida por CAS la versión+SHA del incumbente y el SHA del challenger observados
durante la evaluación. Ambos argumentos son obligatorios; bootstrap exige al
menos el SHA esperado del candidato. Si alguno cambió, aborta sin tocar
producción. `deployment.json` contiene referencias, resultado y ambas huellas de
decisión (`decision_holdout_sha256` y `decision_prediction_sha256`). Úsalo como
recibo de auditoría, pero para recuperar manda el estado
comprobado de `latest.json` y del hash runtime; un fallo posterior al escribir el
recibo se informa sin deshacer un despliegue ya confirmado.

### 4.6. Liberar espacio sin perder recuperación

La política conserva exactamente `latest` y `last_good` (`registry_keep=0`).
Para `PIPELINE/runs/` conserva los 2 últimos no referenciados, el run publicado por
`master/manifest.json`, todos los runs todavía referenciados y cualquier carpeta
de nombre desconocido/no canónico. Estos conjuntos protegidos prevalecen sobre N.

Antes de la primera poda revisa la keep-list y delete-list completas. El preview
genera un token ligado al contenido actual; aplicar requiere confirmar
explícitamente ese mismo token. Si aparece, desaparece o cambia una carpeta, el
plan queda obsoleto y hay que generar otro. Esa primera confirmación registra la
política aprobada; la poda automática posterior se bloquea si cambian la versión,
el tipo (`registry`/`runs`) o N. Conserva además un backup externo: la poda es
intencionadamente irreversible una vez confirmada.

La aplicación de una poda del registry mantiene `registry/.deployment.lock`
desde la revalidación final del plan hasta terminar los borrados y escribir el
marker. Así una promoción, rollback o cambio de `last_good` concurrente no puede
invalidar la keep-list durante la eliminación.

Antes del primer borrado se enumeran y se inspeccionan con `lstat` todos los
árboles candidatos. Después se renombran atómicamente **todos** bajo
`.retention-quarantine/<TOKEN>/`, todavía bajo el lock. Si falla cualquier
rename, se revierten todos los anteriores y no se borra ningún byte. Solo cuando
todos están en quarantine comienza `rmtree`, por lo que un fallo de Windows no
puede dejar un bundle parcial bajo un nombre canónico.

Un fallo durante `rmtree` conserva el manifest, los bundles pendientes y el
detalle estructurado (`deleted`, `pending`, `failed`, `quarantine`). Tras reparar
la ACL se reanuda exactamente esa operación, sin editar JSON ni reutilizar el
preview como si el filesystem no hubiese cambiado:

```powershell
python MODEL\manage_models.py prune --scope registry --resume <TOKEN>
```

`--resume` interpreta el estado sellado del manifest. Para `setup_failed`,
`staging`, `rollback_failed` o `rolled_back` exige que cada bundle exista en
exactamente uno de los dos lados (canónico XOR quarantine), comprueba su
fingerprint y restaura todos los nombres canónicos sin borrar. Para `staged`,
`deleting`, `deletion_failed`, `marker_failed` o `complete` valida primero
`latest`, `last_good`, sus SHA y el hash del runtime, y continúa únicamente desde
quarantine. Si existen ambas copias, no existe ninguna o cambia el fingerprint,
falla cerrado.

Mientras haya una quarantine pendiente, cualquier preview la marca como
`recovery_required`, no ofrece token de confirmación y `--auto` falla antes de
consultar su marker. La recuperación no carga la configuración actual: usa la
política, scope, marker y N sellados en su propio manifest.

La política de staging es la versión 3 y el schema de quarantine es la versión
2: cualquier token, marker o manifest de una versión anterior queda invalidado.
Los atributos Windows `ReadOnly` se limpian únicamente después de mover todos
los bundles a quarantine; nunca se cambia ese atributo en un nombre canónico ni
en `latest`/`last_good`.

Desde `CS2/`, genera el preview sin mutaciones y repite con los tokens que muestre:

```powershell
python MODEL\manage_models.py prune --scope all
python MODEL\manage_models.py prune --scope all --confirm "registry=<TOKEN_REGISTRY>,runs=<TOKEN_RUNS>"
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
