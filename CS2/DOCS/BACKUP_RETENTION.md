# Retención sancionada de backups SQLite

La única vía de poda de copias de la BBDD es `BBDD/manage_backups.py`. El
comando no abre `cs2.db`: inventaría metadatos del filesystem y valida el
backup que se conservará con SQLite `mode=ro&immutable=1`, `PRAGMA quick_check`,
una tabla `matches` no vacía y la presencia de `prediction_ledger`. Además lee
el conteo del ledger de cada candidato: si cualquiera contiene más evidencia
que el retenido, o no permite demostrar su conteo, bloquea toda la poda. Un WAL
o rollback journal adyacente no vacío o inseguro también bloquea.

## Contrato de seguridad

- Sin `--confirm` solo se emite el preview JSON; no se modifica nada.
- `--auto` tampoco borra en su primera ejecución: sin un marker compatible
  devuelve `mode=preview`, `applied=false` y `reason=approval_required`. Sin
  `--required-keep` devuelve `reason=current_snapshot_required`, aunque el
  marker exista.
- El token SHA-256 contiene política, `--keep`, configuración y estado exacto
  de los ficheros. Cualquier cambio exige otro preview.
- Una confirmación manual de `prune` crea
  `BBDD/backups/.backup-retention-approved.json`. La aprobación queda ligada a
  versión de política, hash completo de configuración y `N`; cambiar cualquiera
  de los tres invalida el modo automático hasta confirmar otro preview.
- `prune` solo puede hacer `unlink` de ficheros regulares con nombre fechado
  sancionado dentro del hijo exacto `BBDD/backups/`. No usa glob, recursión ni
  rutas aportadas por el preview.
- Siempre queda como mínimo un backup (`--keep >= 1`). Los nombres desconocidos
  se conservan como `unmanaged_name`.
- En automático, `required_keep` identifica por ruta e identidad filesystem el
  snapshot que acaba de crear `backup_database`; cuenta dentro de `N`. Con
  `N=1` es el único `.db` gestionado que se conserva, aunque otro nombre parezca
  posterior por reloj. Se revalida antes de cada `unlink`.
- Los snapshots nuevos usan `YYYYMMDD_HHMMSS_ffffffZ` para que dos backups del
  mismo segundo no colisionen. El parser mantiene los nombres históricos
  `YYYYMMDD_HHMMSSZ`.
- `cs2.db`, `cs2.db-wal`, `cs2.db-shm`, `cs2.db-journal` y `BLACKBOX/` son
  externos al scope y aparecen como protegidos. Un live ausente, symlink,
  reparse point, fichero no regular o hardlink desde un candidato bloquea toda
  la aplicación.
- Un lock exclusivo en `BBDD/backups/.backup-retention.lock` se adquiere antes
  de crear el nombre/temporal del snapshot y cubre su creación, publicación,
  espejo, refresh autoritativo, prevalidación y todos los `unlink`. La poda
  manual usa el mismo lock, por lo que no puede intercalarse con un backup. Un
  lock previo bloquea el comando. Si queda uno tras un crash, se inspecciona el
  PID/token antes de una decisión manual; nunca se borra a ciegas. La identidad
  del lock se revalida antes y después de cada `unlink`; si desaparece o cambia,
  el lote se detiene y conserva la evidencia de cualquier borrado ya realizado.
- Una parada de sistema a mitad de lote no puede ser transaccional en un
  filesystem. El CLI falla con exit `3` y JSON estructurado (`partial`,
  `deleted`, `deleted_bytes`) para saber exactamente qué llegó a eliminarse.
  Si además no puede retirar el lock, conserva ese error parcial como principal
  y añade el lock residual en `warnings`.

## Poda por lotes

Desde `CS2/`, primero congela el hash de la base viva:

```powershell
Get-FileHash BBDD\cs2.db -Algorithm SHA256
python BBDD\manage_backups.py prune --keep 100
python BBDD\manage_backups.py prune --keep 100 --confirm <TOKEN_DEL_PREVIEW>
Get-FileHash BBDD\cs2.db -Algorithm SHA256
```

Repite preview, confirmación y hash con un token nuevo en cada escalón. Para el
inventario de 172 copias, una secuencia conservadora es `--keep 150`, `100`,
`50`, `25`, `10`, `5` y finalmente `1`. Se puede usar una secuencia más corta,
pero cada salto aumenta el tamaño del lote no transaccional. Tras cada paso se
comprueba que el hash de `cs2.db` no cambió y que el preview siguiente mantiene
`retained_backup_validation.ok=true`, además de revisar `ledger_counts`.

El valor por defecto vive en `BBDD/backup_retention.json` y es `1`; `--keep`
solo lo sustituye para el plan/confirmación actuales.

## Retención automática tras cada backup

`BBDD/build_db.py::backup_database` es la única integración operativa. Crea el
snapshot con la API de backup de SQLite leyendo la base viva en `mode=ro`, lo
valida como copia autocontenida y solo entonces lo publica en `BBDD/backups/`.
Tras copiar opcionalmente el espejo desde ese snapshot ya verificado, ejecuta
la decisión automática. `BBDD/ingest.py`, build, repair y deduplicación ya pasan
por este helper; no hay una segunda implementación en PowerShell.

Hasta que exista aprobación, cada backup se conserva y el resultado incluye un
preview/no-op. Para habilitar la política por defecto, incluso si el inventario
ya fue reducido y el plan no tiene nada que borrar, se confirma una vez:

```powershell
python BBDD\manage_backups.py prune
python BBDD\manage_backups.py prune --confirm <TOKEN_DEL_PREVIEW>
python BBDD\manage_backups.py prune --required-keep <SNAPSHOT_ACTUAL>.db --auto
```

`--required-keep` acepta el basename respecto de `BBDD/backups/` o una ruta
absoluta; no acepta una ruta relativa prefijada de nuevo con `BBDD/backups/`.

Un fallo de creación, espejo o validación ocurre antes de la decisión automática
y por tanto nunca dispara una poda. Un marker ausente o una política cambiada
es un no-op normal; un marker corrupto o inseguro falla cerrado.

El contrato físico con la política `N=1` es exactamente **base viva + un archivo
de snapshot operativo verificado**. Ese archivo corresponde a la llamada más
reciente a `backup_database`; no promete ser el estado anterior a la mutación
que motivó la ejecución, porque cada caller invoca el helper en su punto ya
existente. No se crea un segundo backup pre/post ni se modifica el orden de
ingesta bajo este contrato.

`start.ps1`, `start.ps1 -Retrain` y el reentreno automático convergen en el
mismo ingest final con backup habilitado. Las etapas de persistencia que puedan
crear un snapshot usan también `backup_database`, de modo que cada snapshot
nuevo reemplaza al anterior mediante la política aprobada `N=1`. Al terminar
correctamente cualquiera de esos flujos, `BBDD/backups/` contiene un solo `.db`
gestionado. `-NoDb` es la excepción explícita: no toca la base ni crea snapshot.
Si una validación de integridad o del `prediction_ledger` bloquea la poda, se
conserva la evidencia adicional y el flujo falla cerrado antes de arriesgar
información sagrada.

## Extras heredados

Los dos extras de la raíz BBDD no forman parte de `prune`. Requieren el
subcomando separado `cleanup` y selección nominal explícita:

```powershell
python BBDD\manage_backups.py cleanup --target cs2_integrity_test.db
python BBDD\manage_backups.py cleanup --target cs2_integrity_test.db --confirm <TOKEN>

python BBDD\manage_backups.py cleanup --target cs2_dump.sql
python BBDD\manage_backups.py cleanup --target cs2_dump.sql --confirm <TOKEN>
```

Para seleccionar ambos en un único preview se repite `--target`. Ningún otro
nombre, ruta relativa o directorio puede entrar en `cleanup`.
