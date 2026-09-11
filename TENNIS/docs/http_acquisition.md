# Adquisición HTTP responsable y consolidada

## Auditoría previa

La revisión realizada antes de modificar código encontró una integración
parcial, no un cliente común:

- TennisRatio tenía allow-list estricta, snapshots SHA-256 y un parser local
  limitado al `Disallow` del grupo `*`. La política se validaba al comenzar el
  lote, pero el rate-limit estaba en el servicio (`0,5 s` por defecto) y no
  cubría de forma común todos los GET.
- Tennis Explorer tenía caché/snapshots propios, lock, espera de `1 s`, la
  excepción `authorized_matches_endpoint_exception` y un cortacircuitos WAF.
- Sackmann y Match Charting creaban dos sesiones Requests independientes con
  reintentos conservadores y manifiestos incrementales.
- Tennis Abstract limitaba el adaptador a las dos páginas Elo exactas y
  rechazaba rutas de jugador/JS, pero no consultaba un parser común de robots.
- No existía una infraestructura reutilizable que pudiera consumir el catálogo
  de superficies A2.

## Contrato vigente

`src.responsible_http.ResponsibleHttpClient` es el único cliente real construido
por las fuentes TENNIS. Centraliza:

- `robots.txt` antes de cada fetch de red no cacheado, con TTL de 24 horas;
- grupos por user-agent, `Allow`, `Disallow`, wildcard `*`, anclaje `$` y
  precedencia de la regla coincidente más larga;
- fallo cerrado cuando robots no puede verificarse y no existe una copia
  anterior íntegra; `401/403` deniega y `404/410` permite;
- lock interproceso y timestamp compartidos por host;
- separación mínima de `1 s` entre intentos de red al mismo host;
- caché de respuestas con cuerpo direccionado por SHA-256 y TTL;
- detección de señales inequívocas de WAF antes de cachear una respuesta.

Cada clave de caché conserva solo su cuerpo vigente. Los snapshots crudos y
manifiestos propios de cada fuente siguen siendo la evidencia de dominio; no se
reescriben ni sustituyen por la caché HTTP. Las descargas masivas en streaming
de CSV fijados a commit no duplican sus bytes en esta caché: su archivo destino
y su hash Git ya cumplen esa función.

La excepción original a robots es
`authorized_matches_endpoint_exception`: compara la cadena HTTPS canónica
exacta de Tennis Explorer para `/matches/` con `day`, `month`, `type=all` y
`year`. Reordenar parámetros, añadir parámetros, usar otra ruta o cualquier URL
de jugador deja de ser excepción. El cortacircuitos de Tennis Explorer se
mantiene como dueño del registro persistente del bloqueo; el cliente común le
entrega la respuesta WAF y nunca intenta resolverla.

## Transporte por fuente

| Fuente | Transporte | Estrategia conservada |
|---|---|---|
| TennisRatio | Scrapling estático | identidad TLS Chrome fija, `stealthy_headers=True`, recuperación 429 acotada con pausa persistente; sin navegador/proxy |
| Tennis Explorer | Scrapling estático | sin impersonación, sin stealth, un intento; excepción exacta y cortacircuitos WAF |
| Sackmann mirror | Requests | GET GitHub conservadores e integridad Git por commit/blob |
| Match Charting | Requests | GET GitHub conservadores e integridad Git por commit/blob |
| Tennis Abstract Elo | Scrapling estático | identidad Chrome/cabeceras coherentes; recuperación 429 acotada del host; solo las dos páginas Elo públicas exactas |
| Tennis Abstract adquisición autorizada | Scrapling HTTP + navegador de recuperación | conector para fichas/datos inspeccionados; Chrome persistente visible, timeout total y excepción de datos acotada |

El descargador **Elo** de Tennis Abstract sigue limitado a sus dos informes:
no sigue enlaces a jugadores ni scripts. Desde la autorización expresa del
05/09/2026 existe además `tennis_abstract_access.py`, con una allow-list propia
y separada para fichas/históricos. La declaración del operador queda en
`config/tennis_abstract_access.json`; no es una verificación independiente de
permiso. Solo esas rutas de datos pueden usar la excepción de robots. Las
reglas de todas las demás fuentes permanecen intactas.

## Recuperación HTTP 429 de TennisRatio (06/09/2026)

Se reutiliza el patrón de HLTV (Retry-After, backoff y pausa por dominio),
implementado dentro del cliente TENNIS, sin importar código de CS2. Antes,
TennisRatio convertía el 429 en un error genérico y el servicio podía seguir
intentando el resto de perfiles del mismo host.

La política se activó para `www.tennisratio.com` y desde la migración diaria
del 06/09 también para **`www.tennisabstract.com`**, en
`config/http_client.json → host_overrides`:

| Parámetro | Valor predeterminado |
|---|---:|
| `rate_limit_attempts` (incluye intento inicial) | 3 |
| `rate_limit_base_seconds` | 30 |
| `rate_limit_max_seconds` (máximo del backoff, no del servidor) | 300 |
| `rate_limit_wait_budget_seconds` (espera acumulada por llamada) | 90 |

Sin indicación del servidor las dos esperas son 30 y 60 segundos. Una nueva
respuesta 429 aumenta la racha del host; una respuesta válida la reinicia.
`Retry-After` admite segundos y fecha HTTP y nunca se recorta al máximo local:
si el servidor pide una hora, se registra la hora completa y se difiere el
lote, sin dormir una hora ni reintentar antes. La espera sale en el log y se
hace fuera del lock, en tramos de hasta 30 segundos, sin jitter aleatorio.

La pausa se guarda en un único `rate_limit.json` bajo la caché común del host.
Sobrevive al reinicio del cliente/proceso y bloquea nuevas peticiones de red,
incluido `--force`, hasta que expire. La caché válida se puede seguir leyendo;
el cuerpo 429 nunca se almacena como dato. Robots pasa por la misma política,
sin saltarse sus reglas. Los tres intentos internos de Scrapling por errores
de transporte se conservan: no reintentan estados HTTP ni sustituyen esta
política. Se mantiene la misma sesión, identidad TLS y cabeceras durante la
recuperación.

Un 429 con firmas reales de challenge, un 403 o un WAF no entra en estos
reintentos HTTP ordinarios. TennisRatio se detiene. Solo Tennis Abstract tiene
la autorización posterior para [recuperación por navegador](tennis_abstract_daily.md),
manteniendo detección WAF y los límites/Retry-After del servidor.
Agotar los reintentos también detiene el recorrido de perfiles, cierra la
sesión y **no publica el lote truncado ni mueve `last_good`**. Los datos
previamente publicados siguen disponibles. La CLI devuelve error explícito;
el launcher normal registra fallo de actualización y continúa con su lógica
existente de último dato válido/frescura. `-UpdateOnly` propaga el error.

El timestamp de disponibilidad se toma al recibir los bytes validados, no
antes de las esperas: cruzar medianoche nunca convierte un dato de D en
información disponible en D-1. No se cambian fórmulas de features ni tablas
sagradas. Ambos launchers usan este comportamiento automáticamente, sin
flags nuevos. La adquisición experimental de Tennis Abstract conserva su
contrato independiente; este cambio no sustituye las fuentes de producción.

Tests offline: `tests/test_tennisratio_rate_limit.py` y
`test_exhausted_429_stops_profiles_and_preserves_published_batch`, además de
las regresiones de robots/WAF/JSD existentes.

Validación local del cambio: **565 tests y 262 subtests**, Ruff lint,
ratchet de formato (94 pendientes heredados sin crecimiento), Mypy,
cobertura de imports (13 raíces externas / 17 dependencias declaradas),
smoke (16 imports y 8 entrypoints deterministas) y `pip check` en verde.
El auditor manual de Tennis Abstract tiene un smoke separado de la lista de
scripts que realmente ejecuta el launcher: no se ha activado en producción.
Los reintentos 429 se verifican con respuestas grabadas y reloj simulado;
no se provoca un rate-limit contra el proveedor para probarlos.

`service.py` y `types.py` pertenecen al inventario sellado del Elo. Su cambio
exige reconstruir Elo/features por los entrypoints normales antes de servir;
no se elimina ni se elude esa comprobación. Una identidad nueva de features
se valida con la puerta de modelos existente, nunca editando manifiestos a
mano ni dando por buena una equivalencia sin comprobarla.

### Cierre operativo verificado (06/09/2026)

La reconstrucción publicó el Elo actualizado y las features
`68a6c12722b66a6089ab67aff4fbb054f6079377d11ae88b070b2ea9faa49a95`.
El launcher registrado en `logs/run_tennis_20260906_093428139_33296.log`
terminó con `status=ok`, `exit_code=0` a las 09:56:06 UTC, incluida la
generación de `WEB/data.js`.

La puerta evaluó el challenger `6ec499e9…` frente a `d080b15c…` sobre
55.798 partidos: ambos obtuvieron accuracy **69,1889 %**, log-loss
**0,574737800603** y Brier **0,196565504322**. La diferencia máxima de
probabilidades y métricas fue **0** (tolerancia `1e-12`). Decisión:
`reject / brier_tiebreak_not_improved`; no se forzó una promoción. La
evidencia de equivalencia permite servir el modelo vivo con las features
reconstruidas. Los hashes de `models/phase7/manifest.json` y `last_good.json`
permanecen iguales a los registrados antes de reconstruir.

A las 13:44:29 UTC se repitió `scripts/audit_match_format.py --date 2026-09-06`
con datos reales, sin publicación: `pipeline=ok`, **40 filas de cartelera y
12 predicciones** (10 MEDIUM y 2 LOW). No implica que toda fila sea elegible
para pronóstico ni que se hayan rellenado datos ausentes: en esas 12 faltaba
evidencia anterior a D para best_of/round y se conservaron sus flags. No se
ejecutaron escrituras operativas ni se regeneró la web en este diagnóstico.

## Configuración

`config/http_client.json` define el intervalo, TTL y tiempos del lock. Un
override de host puede reducir el segundo mínimo únicamente si declara un
`access_basis` explícito de API/allowlist. Así A2 y futuras adquisiciones pueden
reutilizar el mismo cliente sin crear otra política paralela.

La consolidación por sí sola no autoriza Selenium/headless, proxies, rotación,
endpoints privados ni resolución de Cloudflare/WAF. Tennis Abstract cuenta
desde el 06/09 con una autorización adicional de navegador en `AGENTS.md`,
descrita en su informe de adquisición diaria; no se extiende a otras fuentes.
La excepción a robots para datos de
Tennis Abstract necesita `operator_authorization_confirmed=true` con
procedencia; no es una desactivación global del parser.

## Auditoría del histórico de Tennis Abstract (05/09/2026)

`scripts/audit_tennis_abstract.py` usa el conector autorizado. Adquiere páginas
clásicas y, si corresponde, el `jsmatches/<clave>.js` referenciado exactamente.
No descarga scripts de UI ni ejecuta JavaScript. Extrae arrays literales con
un parser de delimitadores y `ast.literal_eval`, rechazando expresiones de
código, profundidad inesperada y matrices inválidas. Compara las 44 primeras
cabeceras conocidas y **no mapea las columnas restantes al modelo**.

| Fuente inspeccionada | Filas | Anchura | Última fecha de fuente | Filas con puntos de servicio |
|---|---:|---:|---|---:|
| `player-classic.cgi?p=JannikSinner` | 565 | 48 | 2026-06-29 (Wimbledon) | 506 |
| `jsmatches/ArynaSabalenka.js` | 129 | 44 | 2026-08-31 (US Open) | 118 |

No son cifras de cobertura global. El WTA inspeccionado es una partición
reciente; la página anuncia además un histórico `Career.js` que todavía no se
ha auditado. La cabecera mostrada por las páginas tiene 47 campos, mientras
las filas ATP/WTA tienen anchuras diferentes. Tampoco `date` es una fecha
fiable de partido: varias rondas comparten la fecha de inicio del torneo.
`chartlink` está presente en 326/565 y 70/129 filas respectivamente; no se ha
convertido esa referencia en fecha efectiva sin verificar su contrato.

El 05/09 a las 20:50:29 (hora del log de Scrapling), la ampliación de la muestra a
`wplayer-classic.cgi?p=CocoGauff` devolvió **429**. Se detuvo el lote sin
reintentos automáticos. El cliente no vuelve a solicitar nada tras un WAF/429
en la misma instancia; la CLI conserva los resultados parciales, informa del
fallo y devuelve código 1. La cuota o cadencia aprobada por el proveedor no
está documentada todavía; no se ha configurado una descarga masiva diaria.

**Estado: adquisición autorizada y auditoría, no sustitución de producción.**
La base Sackmann, precedencia TennisRatio/Explorer, Elo operativo, features y
modelo activo siguen intactos. No hay ingesta de estas matrices a la BBDD
operativa ni entrenamiento con ellas. Antes de esa migración se requieren
contratos inequívocos de fechas/columnas/IDs y comparación de cobertura; un
HTTP 200 no es evidencia de datos recientes ni de compatibilidad de features.

Para repetir la auditoría cuando el proveedor permita nuevas peticiones:

```powershell
.\TENNIS\.venv\Scripts\python.exe .\TENNIS\scripts\audit_tennis_abstract.py
```

El auditor manual anterior se mantiene como diagnóstico. Desde el 06/09 se
reutiliza su parser en la [adquisición diaria persistente](tennis_abstract_daily.md),
invocada por `run_tennis.ps1` y su tarea diaria. Es una fase de adquisición
paralela: no conecta fechas ambiguas al Elo ni reemplaza features/modelos.
Los números y limitaciones del apartado anterior corresponden a la auditoría
inicial; el informe enlazado describe la nueva ingesta y su cobertura real.

## Corrección del falso positivo JSD (2026-09-05)

El cliente rechazaba carteleras y perfiles válidos HTTP 200 porque trataba
cualquier aparición de `/cdn-cgi/challenge-platform/` como un bloqueo. La
respuesta inspeccionada contenía los partidos completos y un script inyectado
`/cdn-cgi/challenge-platform/scripts/jsd/main.js`.

[Cloudflare distingue JavaScript Detections de una página de desafío](https://developers.cloudflare.com/cloudflare-challenges/concepts/how-challenges-work/):
el primero puede estar presente en HTML ordinario sin interrumpir su entrega.
El detector ahora distingue únicamente las rutas JSD `main.js` y `api.js`
(incluido el prefijo `h/<variante>/`) en respuestas **200 HTML**. No ignora el
prefijo de plataforma completo. La distinción solo afecta a la clasificación:
los bytes originales, su SHA-256 y el timestamp de captura se conservan.

`403`, `429`, `CF-Mitigated: challenge`, títulos/textos de challenge,
`cf-chl-`, Turnstile y rutas desconocidas/de orquestación siguen deteniendo la
respuesta antes de cachearla. La presencia de datos con aspecto de cartelera
no anula estas señales. Los parsers de fuente siguen exigiendo su esquema:
un HTML con solo un script JSD no se acepta como partidos.

La ruta continúa siendo `TennisRatioClient → ResponsibleHttpClient →
ScraplingHttpSession`: GET estático, robots, lock por host y mínimo de 1 s.
No se descarga ni ejecuta el script JSD, no se resuelve un challenge y no se
abre ningún navegador. El cortacircuitos persistente de Tennis Explorer no se
ha cambiado.

Regresión: `tests/test_tennisratio_passive_jsd.py` cubre la ruta real del
cliente con transporte Scrapling simulado, lectura de cartelera, integridad de
bytes/hash, caché, rate-limit y coexistencia de JSD con bloqueos reales.
Puertas locales: **522 tests y 262 subtests**, Ruff lint/formato, mypy,
cobertura de imports, smoke y `pip check`.

### Validación en la máquina

`run_tennis.ps1 -UpdateOnly` terminó con código **0** el 05/09/2026 a las
14:05:56 UTC. Lote `20260905T134711230835Z-82e274c561ff`: **35 partidos**
(22 de hoy y 13 del día siguiente), **418 perfiles validados de 469 intentos**.
Los 51 restantes fallaron controles de calidad del contenido, no transporte:
41 proporciones con numerador mayor que denominador, 7 porcentajes fuera de
0–100 y 3 representaciones inválidas de juegos de servicio. Se conservan sus
últimos snapshots buenos; no se relajan los validadores para aceptarlos.

Una segunda ejecución del actualizador devolvió `already_refreshed=true` y
`changed=false`: el control diario evita repetir la descarga. La inferencia
real, ejecutada sin publicación, produjo 16 predicciones de hoy y 12 para
mañana. No se lanzó un reentreno ni se regeneró la web dentro de esta prueba.
Los hashes completos de predictions (6.022 filas), observations (7.480),
settlements (1.563), sus triggers y el manifiesto de modelo activo permanecieron
idénticos durante la actualización.

No cambió ningún contrato de features, semilla, split, calibración ni
dependencia. La corrección pertenece al cliente de adquisición; el arranque
normal de ambos launchers la utiliza sin flags especiales.
