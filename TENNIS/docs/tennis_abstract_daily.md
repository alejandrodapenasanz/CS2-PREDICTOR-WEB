# Tennis Abstract: migración diaria por fases (06/09/2026)

## Decisión y límites

El usuario aprobó conservar Sackmann como **archivo de entrenamiento**, y
preparar Tennis Abstract para las actualizaciones diarias. No se borra el
archivo ni se mueve su commit/corte. La retirada de fuentes operativas requiere
demostrar cobertura e identidades/fechas compatibles, no solo recibir HTTP 200.

Esta entrega implementa adquisición y persistencia, no cambia la arquitectura
predictiva: el Elo/overlay y las features vigentes siguen usando sus contratos
anteriores. TA continúa aportando sus informes Elo por el adaptador existente;
las nuevas matrices de jugador todavía **no alimentan Elo ni entrenamiento**.
TennisRatio y Tennis Explorer siguen proporcionando cartelera/resultados.

## Ruta automática

`start.ps1` raíz → `TENNIS/run_tennis.ps1` → `update_sources.py --skip-sackmann`
→ actualización TennisRatio → `update_tennis_abstract.py` → pipeline existente.

La ruta `run_tennis.ps1 -UpdateOnly` también actualiza los dos informes Elo
(omite Sackmann y Match Charting), después TennisRatio y las fichas TA de la cartelera, sin
entrenar ni escribir predicciones/settlements ni publicar WEB. La tarea Windows
ya existente `CS2-Predictor-TennisRatio-Daily` usa esa ruta: no se ha creado otra
tarea ni duplicado el pipeline. Su nombre legacy se conserva. Necesita sesión
del usuario y red; no puede descargar con el PC apagado. El horario y estado se
consultan con `scripts/manage_tennisratio_daily_task.ps1 -Mode Status`.

Los enlaces exactos del último informe Elo validado proporcionan género, clave
nativa y procedencia del inventario. No hay matching por nombre, IDs Sackmann
inventados ni reinterpretación automática de rutas modernas numéricas. El
inventario actual tiene **551 hombres + 544 mujeres = 1.095 jugadores**: es la
cobertura del ranking Elo TA, no de todos los tenistas existentes.

Desde el 09/09 el modo diario selecciona únicamente los participantes de la
cartelera TennisRatio publicada para hoy UTC. `--date YYYY-MM-DD` cambia la fecha
de cartelera, nunca la fecha real de captura. Se deduplican por identidad de
fuente y se selecciona un enlace ya inspeccionado de TA solo cuando el nombre
completo normalizado y género son únicos. No se usa fuzzy matching, iniciales,
slugs adivinados ni se considera esto un mapping canónico para Elo/features.
Las coincidencias ambiguas o ausentes se reportan; una selección incompleta
devuelve `partial_selection`, no `completed`. Sin cartelera devuelve
`agenda_unavailable` y no descarga perfiles. No se recurre al catálogo entero.

El inventario completo requiere **`--full-inventory` explícito**. Se conserva el
histórico ya adquirido y el progreso; cambiar el alcance no elimina perfiles.
Las peticiones siguen siendo secuenciales, con mínimo de 1 s por host y control
diario por ficha. `--max-profiles N` permite una prueba parcial y **no marca la
selección completa**. El launcher y la tarea diaria nunca añaden `--full-inventory`.
Cada éxito se confirma en SQLite antes del siguiente jugador; la siguiente
ejecución omite los ya capturados ese día y atiende primero los menos intentados.
El lock SQLite independiente se libera incluso si muere el proceso.

La selección puede usar una cartelera capturada durante el día porque solo
decide qué documentos adquirir. **No autoriza esas capturas como features de
ese día.** Para tener un snapshot anterior a D hay que adquirirlo previamente
(por ejemplo seleccionando la cartelera de mañana ya publicada). La evidencia
de selección acompaña las nuevas fichas y el resumen; las matrices siguen
marcadas `model_ready=False`. No se cambian producción, ratings ni fechas.

`--status` conserva contadores globales del almacén. `last_agenda_selection`
identifica la última fecha/selección diaria; los pendientes de todo el catálogo
no significan que el modo diario vaya a descargarlos todos. No consulta ni
inicializa TennisRatio ni realiza red al leer ese estado.

## Transporte y fallos

Se utiliza `TennisAbstractAcquisitionClient` → cliente HTTP común → Scrapling
HTTP, con recuperación opcional mediante navegador autorizada el 06/09 en
`AGENTS.md` y `config/tennis_abstract_access.json`. Se preservan robots,
validación de URL, caché SHA-256 y lock/cadencia por host. La excepción de datos
TA permanece limitada a las rutas inspeccionadas y la declaración de permiso
del operador. El navegador puede ejecutar los scripts necesarios para el
challenge; el parser sigue tratando los históricos JS exclusivamente como datos.

`config/http_client.json` aplica a TA la misma recuperación 429 que TennisRatio:
3 intentos, backoff 30/60 s, presupuesto de espera acumulada 90 s. `Retry-After`
no se acorta: si pide más tiempo, se persiste y se difiere, sin esperar horas.
Un WAF no se trata como un 429 ordinario: se permite una recuperación por
sesión mediante `AsyncStealthySession`, Chrome visible, perfil dedicado y
`solve_cloudflare=True`. No se instala software desde el runtime. Un timeout
global de 90 s cancela incluso el bucle del solver; las operaciones individuales
tienen 30 s y un solo intento. Si falla, se conserva el progreso y se pausa
24 h ante WAF, esquema o fallo sin clasificar; nunca se sustituye la última
ficha válida por un fallo. La excepción para timeouts de navegación aislados
se detalla debajo (corrección del 08/09).

El navegador captura los **bytes de red y cabeceras de la navegación final**.
No usa el DOM modificado ni las cabeceras del primer challenge (Scrapling puede
conservar estas últimas en su respuesta envolvente). Se verifica la URL final
exacta, y el cliente común sigue comprobando WAF y esquema antes de aceptar el
contenido. Las dependencias del navegador tienen una allow-list de assets y
rutas de challenge, cadencia por host y bloqueo de navegaciones ajenas. Los
históricos secundarios se solicitan explícitamente desde el cliente común,
no por descargas implícitas del navegador.

Las cookies emitidas por el servidor se conservan en un único perfil privado
`data/raw/_browser_profiles/tennis_abstract/`, ignorado por Git. No se usa el
perfil personal ni se imprimen tokens. La caché de disco Chrome se limita a
10 MiB; esto no limita el tamaño total del perfil. La falta de Chrome o un
fallo de arranque se informa, sin instalaciones ni fallback silencioso.

La nueva versión puede reintentar una vez la antigua pausa **local** WAF al
activar la recuperación autorizada. `--browser-recovery` solicita otro intento
explícito ante errores locales. Ninguna de estas vías elimina una pausa del
servidor (`RateLimitedError`/`Retry-After`). Un fallo persistente sigue siendo
un fallo: que el navegador esté autorizado no garantiza obtener todos los datos.

El resumen JSON indica inventario, éxitos previos/nuevos, pendientes, causa y
`retry_at_utc`, además de `mode=shadow_acquisition_not_model_input` y
`model_ready_rows=0`. Los estados incompletos devuelven código 2 desde Python.
El arranque completo avisa y continúa usando el modelo/fuentes vigentes;
`-UpdateOnly` comunica el estado incompleto al programador de tareas.

## Persistencia, peso y anti-fugas

`data/processed/tennis_abstract.sqlite3` es un almacén de fuente independiente:

- `ta_rows`: arrays originales deduplicados por SHA-256, sin adivinar columnas.
- `ta_snapshots` y `ta_sightings`: manifiestos y cambios observados, con claves
  nativas, procedencia, cabeceras, hashes y timestamps de captura.
- `ta_checks`/`ta_state`: progreso mutable y último resumen/pausa, no features.
- `ta_raw_documents`: cuerpos comprimidos, **solo dos versiones por URL**.

No se generan copias completas diarias de la BBDD. Las filas de fuente y los
manifiestos necesarios para reconstruir el pasado no se podan; solo se rotan
cuerpos HTTP de este almacén nuevo. Los arrays preservan también las celdas
finales desconocidas; no se mapean a estadísticas con nombres supuestos.

La captura se fecha **después** de terminar la descarga, incluidos reintentos
que crucen medianoche. Se puede consultar evidencia capturada estrictamente
antes de D con `snapshot_before`, pero esto **no la hace una feature válida**:
`date` sigue siendo fecha del torneo, `exact_match_date=None`,
`canonical_player_id=None`, `model_ready=False`. Las tasas numéricas y las
fechas exactas necesitan su validación de dominio posterior. No se aplica un
snapshot recién capturado a partidos históricos. Una vuelta A→B→A se registra
como nueva observación, sin reescribir el pasado ni duplicar la fila A.

No se abre ni se muta `BBDD/tennis.sqlite3` desde este actualizador. Tampoco se
tocan champion, last_good, ratings, features, splits, calibradores ni semilla.
No hay dependencias nuevas.

## Evidencia real y punto pendiente

La prueba real del 06/09 a las 14:05 UTC obtuvo HTTP 200 con Scrapling para
Sabalenka, Sinner, Gauff y Alcaraz: **1.285 filas de jugador**, **1.162 con el
campo de puntos de servicio presente**. No equivalen necesariamente a 1.285
partidos únicos ni a estadísticas ya aprobadas para entrenamiento. Se guardan
4/1.095 jugadores; no se presenta esa muestra como cobertura completa.

La continuación a las 14:06 UTC llegó a Rybakina (también HTTP 200). Su matriz
contiene **138 filas con anchuras 27 y 44**; tres filas de BJK Cup tienen solo
27 celdas. SHA-256 del documento observado:
`6ce6d4a4f9e48f0eb59de79208a4c004bc4fa98437f29f40bafaab74e3ad0f93`.
El contrato inspeccionado exige al menos 44. Se detuvo sin rellenar columnas
ni aceptar parcialmente esa ficha, conservando las cuatro fichas anteriores.
El usuario aprobó después **cuarentena para las filas incompletas y continuar**.
El contrato v2 conserva índice original, anchura, motivo, hash y celdas crudas
de las filas cortas, separadas de las válidas; no hace padding ni mapea campos
faltantes. La recaptura incorporó 135 filas válidas de Rybakina y 3 en cuarentena.

La siguiente ampliación llegó a **8 jugadores, 3.986 filas completas y 3 en
cuarentena** antes de un HTTP 429 de Pegula clasificado como WAF. Después de la
autorización de navegador, la prueba real del 06/09 a las **20:36 UTC** obtuvo
HTTP 200 para su ficha y `jsmatches/JessicaPegula.js`, con **134 filas completas
y 3 en cuarentena**. Total almacenado entonces: **9 jugadores, 4.120 filas
completas, 6 en cuarentena y 3.749 con puntos de servicio presentes**.
Faltan 1.086 jugadores del inventario completo: no se ha completado el barrido.

En esa prueba Chrome no encontró un challenge activo. Demuestra descarga,
parseo y guardado por la ruta del navegador, pero **no demuestra que el solver
haya superado un challenge real** ni que el 429 anterior desapareciera por usar
Chrome. La recuperación challenge→200, límite total, cabeceras finales y
respeto de Retry-After se verifican con fixtures. La biblioteca puede escribir
`No Cloudflare challenge found` a nivel ERROR incluso ante un HTTP 200 válido;
la aceptación depende del contenido y de nuestros validadores, no de ese mensaje.

Pruebas añadidas: idempotencia diaria, reanudación, retención, invariancia
append-future, corte civil, cambios que revierten a un snapshot previo, pausa
persistente WAF/429, recuperación 429 por la ruta real con fixtures, exclusión
de rutas/IDs no inspeccionados, conservación ante HTML inválido y lock de
adquisición. El smoke incluye el entrypoint nuevo.

Para continuar la adquisición después de resolver el esquema/pausa:

```powershell
.\TENNIS\.venv\Scripts\python.exe .\TENNIS\scripts\update_tennis_abstract.py
```

El reintento explícito de un fallo local utiliza `--browser-recovery`. No hay
flag que ignore Retry-After del servidor, invente fechas o publique estas filas
al modelo. El parser/esquema siguen siendo independientes del transporte.

### Estado durante la adquisición (07/09/2026)

`update_tennis_abstract.py --status` lee el progreso confirmado sin acceder a la
red, sin crear archivos y sin tomar el lock escritor de adquisición. Abre SQLite
con `mode=ro` y lee contadores/fechas en una única transacción, para que no se
mezclen dos fichas a medio confirmar. Funciona mientras continúa la descarga.

El estado distingue jugadores del inventario alguna vez adquiridos, nunca
adquiridos, refrescados hoy y pendientes hoy. La cobertura de filas incluye todo
el almacén y se etiqueta por separado. El último resumen de ejecución se muestra
con fecha y tamaño de selección: una prueba antigua de un jugador no significa
que todo el inventario esté completo. La consulta no detecta procesos activos,
no reintenta nada ni modifica una pausa WAF/429. Su código 0 significa que la
consulta funcionó, no que haya terminado la descarga.

### Timeout aislado frente a bloqueo del host (08/09/2026)

El barrido del 07/09 guardó 710 fichas nuevas y llegó a 719 jugadores adquiridos,
204.538 filas completas y 293 en cuarentena. Se detuvo a las 06:56 UTC cuando
`Page.goto` agotó 60 s en Nicolas Kicker. El cliente compartido envolvía el
timeout como `ResponsibleHttpError`, y el actualizador aplicaba su pausa genérica
local de 24 h. Eso no acreditaba un WAF ni una orden de espera del servidor.

Ahora el adaptador de navegador emite `TennisAbstractPlayerTimeout` únicamente
para el timeout `Page.goto` tipado de la biblioteca inspeccionada. No basta que
un mensaje genérico contenga la palabra timeout. Si se ha observado WAF, un
`Retry-After` positivo o un cuerpo de respuesta que no se ha podido verificar,
no se aplica esta clasificación. El deadline global del solver/arranque tampoco
se reclasifica. El cliente HTTP común permanece intacto: se recupera la señal
tipada de su cadena de excepciones, sin alterar robots, caché o rate-limit.

- Un timeout aislado se registra en `ta_checks` y en el estado operativo de
  adquisición `player_timeouts`; conserva la ficha anterior y continúa el lote.
- Se aplaza esa identidad 30 minutos (`PLAYER_TIMEOUT_RETRY_SECONDS=1800`).
  La siguiente ejecución respeta el plazo, incluso con `--browser-recovery`.
- Tres timeouts consecutivos (`MAX_CONSECUTIVE_PLAYER_TIMEOUTS=3`) detienen el
  lote durante 15 minutos (`TIMEOUT_STREAK_PAUSE_SECONDS=900`). Una ficha válida
  reinicia la racha. No hay bucle ilimitado de reintentos.
- El resumen registra intentos, fichas aplazadas y pendientes; nunca anuncia
  `completed` si queda una ficha pendiente. `--status` expone las pausas por
  jugador sin escribir. Al recuperar una ficha se limpia solo su pausa mutable.
- WAF, robots, 429, `Retry-After`, errores de esquema y fallos ambiguos siguen
  deteniendo el lote. No se reinterpretan automáticamente antiguas pausas
  genéricas: la de Kicker ya había vencido al reanudar el 08/09.

Es exclusivamente una corrección de adquisición. No cambia fechas disponibles,
matrices originales, identidades canónicas, features, modelo ni datos sagrados.

## Verificación de cierre

El 06/09/2026 las puertas completas de TENNIS terminaron en verde: Ruff lint,
ratchet de formato (94 archivos protegidos; 94 legacy pendientes, sin aumento),
mypy sobre 22 módulos, **599 tests + 264 subtests**, cobertura de imports y
smoke determinista de 16 imports y 9 entrypoints. `pip check` no encontró
dependencias incompatibles. No se añadieron dependencias.

La auditoría operativa `scripts/audit_match_format.py --date 2026-09-06`
ejecutó inferencia real a las 20:39 UTC: estado `ok`, cartelera de 40 partidos,
2 predicciones, `published=false`. Es una comprobación de arranque e inferencia,
no una publicación de WEB ni una prueba de cobertura completa de la cartelera.
Los hashes SHA-256 del triplete en `BBDD/tennis.sqlite3`, del manifest activo,
de `last_good.json` y de `config/elo_handoff.json` permanecieron idénticos antes
y después de la adquisición y las pruebas. Los tres checklists de arquitectura,
anti-fugas y dependencias se revisaron sin cambiar el contrato del modelo.

Continuación del 07/09/2026: el barrido completo se reanudó por la ruta de
navegador persistente. A las 05:14:56 UTC había **49 jugadores almacenados**,
40 capturados ese día, **16.639 filas completas y 24 en cuarentena**; 1.046
jugadores nunca adquiridos y 1.055 pendientes de refresco diario. Son contadores
de un barrido todavía activo, no una declaración de cobertura completa.

Tras añadir el estado de solo lectura pasaron las puertas completas: **604 tests
+ 264 subtests**, mypy (22 módulos), lint, ratchet de formato, cobertura de
imports y smoke (16 imports / 9 entrypoints), además de `pip check`. Se probó
lectura durante el lock de adquisición, invariancia del archivo SQLite,
rechazo de escrituras por `mode=ro`, ausencia de creación de archivos, separación
de inventario/hoy y conservación de pausas del servidor. No cambian dependencias
ni el contrato causal; permanecen intactos los hashes del modelo activo,
`last_good` y `BBDD/tennis.sqlite3`.

La inferencia real para el 07/09 terminó `ok` a las 05:15:30 UTC: 82 filas de
cartelera y **53 predicciones**, `published=false`. Estas predicciones usan el
modelo y fuentes previos, no las nuevas matrices TA. La primera ejecución en
sandbox no pudo escribir un temporal del catálogo de superficies; la repetición
con permisos de ejecución suficientes finalizó correctamente, sin cambiar código
del catálogo ni relajar su contrato. No se publicó WEB.

El 08/09 se verificó la nueva clasificación con **615 tests + 264 subtests**:
timeout de navegación con/sin respuesta, respuesta WAF por cabeceras o cuerpo,
cuerpo desconocido, deadline del solver, conservación as-of del snapshot,
continuación con otros jugadores, pausa persistente por ficha, límite de tres
fallos y reinicio de racha tras un éxito. No se añadieron dependencias;
`pip check` pasó. Los hashes del modelo, `last_good` y la BBDD sagrada fueron
idénticos antes y después de los cambios y la auditoría (la BBDD había avanzado
por operaciones ajenas a esta sesión desde la verificación anterior).

La reanudación real por navegador empezó el 08/09 a las 18:19 UTC. A las
18:25:09 UTC: **754/1.095 jugadores adquiridos**, 213.885 filas completas y
298 en cuarentena; 341 jugadores nunca adquiridos. Sin pausa activa ni timeout
local observado en ese tramo: la corrección del timeout se verifica con
fixtures, no se presenta como una recuperación real de Kicker ya demostrada.
El barrido sigue separado del modelo y todavía no está completo.

El arranque diario del 08/09 terminó `ok`, `published=false`, con 54 filas y
cero predicciones nuevas: todas tenían el flag
`prediction_not_strictly_before_scheduled_start` a la hora de la auditoría.
También había 18 flags de conflicto de género/nivel y una identidad sin mapping;
no se tocaron esos contratos ni se forzaron predicciones retrospectivas.

## Cambio a selección por cartelera: verificación del 09/09/2026

Por petición del usuario, el barrido de catálogo deja de ser la ruta diaria.
La última cartelera publicada disponible al verificar era la del **08/09**:
54 partidos, 108 participantes, **106 perfiles TA seleccionables**, frente a
los 1.095 del catálogo completo. Tiago Torres y Sean Cuenin no tenían enlace en
el inventario inspeccionado y quedaron explícitamente sin resolver. Estas
cifras son de ayer, no se presentan como cobertura de hoy. La adquisición previa
había conservado 1.094 jugadores, 285.317 filas completas y 461 en cuarentena;
no se ha borrado nada para cambiar de alcance.

Pasaron **624 tests + 264 subtests**, mypy (23 módulos), lint, formato incremental
(96 protegidos, 94 legacy pendientes sin crecimiento), cobertura de imports,
smoke (16 imports / 9 entrypoints) y `pip check`. Las regresiones verifican el
orden TennisRatio → TA en el launcher, modo completo solo con flag, deduplicación,
colisiones de nombre/género, ausencia de fallback al catálogo, evidencia de
selección y aislamiento temporal: añadir agenda futura no cambia los perfiles
seleccionados para hoy; una captura de hoy continúa fuera de las features de hoy.

El entrypoint real sin flags devolvió `agenda_unavailable` para el 09/09 y no
abrió ninguna descarga TA: una actualización TennisRatio iniciada a las 06:04 UTC
seguía activa y aún no había publicado esa fecha. Su lock y proceso se respetaron;
no se lanzó una segunda actualización ni se esperó al barrido completo para
validar el código. La adquisición real de los perfiles de hoy queda pendiente
de esa publicación. Los hashes de producción, `last_good` y la BBDD sagrada
permanecieron iguales durante estas verificaciones. No se publicó WEB ni se
reentrenó/promovió un modelo desde esta tarea.
