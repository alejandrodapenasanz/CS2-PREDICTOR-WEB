# Reglas permanentes del proyecto CS2

Estas reglas se aplican a todo `CS2/` y complementan el `AGENTS.md` de la raíz.
Si una tarea no puede cumplir ambos contratos, se detiene y se consulta al
usuario antes de editar.

## Arquitectura del componente

- `SCRAPER/hltv-scraper-api/` adquiere y normaliza datos de HLTV en un entorno
  aislado; no contiene features ni lógica de entrenamiento.
- `PIPELINE/` orquesta la adquisición y el enriquecimiento y entrega contratos
  explícitos a persistencia y predicción. No implementa modelos paralelos.
- `BBDD/` es la única capa que define, migra y escribe el esquema SQLite. Toda
  mutación pasa por sus APIs sancionadas y conserva procedencia temporal.
- `MODEL/cs2model/` contiene features, ratings, entrenamiento, calibración y
  evaluación; los scripts de `MODEL/` son adaptadores/entrypoints finos sobre
  esa lógica reusable.
- `TESTS/` contiene las regresiones del dominio. Los tests propios del scraper
  permanecen en `SCRAPER/hltv-scraper-api/tests/`.
- `start.ps1` es el entrypoint operativo real de CS2 y `PIPELINE/start.py` el
  entrypoint Python del pipeline. Un cambio debe ejercitar esas rutas, no solo
  importar una utilidad aislada.
- El modelo no importa internals del scraper ni obtiene datos de `WEB/`,
  `TELEGRAM/` o `TENNIS/`. Las capas compartidas consumen las salidas publicadas
  por CS2, nunca su estado mutable interno.

## Anti-fugas de CS2

- Para predecir un partido en fecha `D`, cada feature usa únicamente información
  con fecha efectiva y disponibilidad estrictamente anteriores a `D`.
- Elo, Glicko y cualquier estado cronológico se consultan/emiten antes de llamar
  a la operación que observa el resultado. Nunca se rellena un partido pasado
  con el rating actual de un equipo.
- Con partidos sin hora fiable, todos los partidos de `D` se puntúan desde el
  estado congelado al inicio de `D`; sus resultados se aplican en bloque después.
- Rankings, rosters, stand-ins, estadísticas de jugador, mapas, contexto y odds
  deben apuntar a un snapshot prepartido demostrable. Resultados, veto real
  conocido después, cuotas live/cierre o snapshots capturados tarde no pueden
  alimentar una predicción histórica.
- Como única excepción al corte civil `< D`, las **opening odds** pueden haberse
  capturado durante `D` cuando la evidencia conserva timestamps UTC fiables y
  cumple estrictamente `captured_at_utc < kickoff_utc`. Se usa la primera
  captura válida con dos cuotas decimales coherentes y de-vigada a suma 1. Si
  falta cualquiera de esos requisitos se enruta a `no_odds`; nunca se sustituye
  por una cuota live, de cierre, proxy o capturada al inicio/después del partido.
- Añadir filas posteriores a `D` no puede alterar las features ni el rating
  as-of del partido. Se mantiene una prueba explícita de invariancia por prefijo.
- Entrenamiento, selección, tuning, calibración y evaluación usan exclusivamente
  splits temporales walk-forward (con embargo cuando corresponda). No se permite
  `shuffle=True`, `train_test_split` aleatorio ni K-fold aleatorio.
- Todo preprocesamiento aprendido se ajusta dentro del fold de entrenamiento.
  La semilla fija de producción es `42` y se pasa explícitamente a cada camino
  estocástico.

## Integridad de `prediction_ledger`

- `prediction_ledger` es evidencia prospectiva sagrada. Está prohibido editarla
  con SQL manual, cambiar resultados ya evaluados, borrar filas, recrearla desde
  predicciones actuales o hacer backfills que simulen predicciones históricas.
- Las únicas mutaciones permitidas son las vías operativas existentes y
  sancionadas: `BBDD.ingest.upsert_prediction_ledger` registra/actualiza la última
  predicción válida **antes** del kickoff mientras el registro está abierto, y
  `BBDD.ingest.finalize_prediction_ledger` congela, invalida o evalúa su ciclo de
  vida con el resultado real. No se crean atajos alternativos.
- Un nuevo dato sobre una predicción ya congelada/evaluada se representa de forma
  aditiva o en una tabla lateral trazable; nunca sobrescribe el registro original.
- Las migraciones del ledger solo pueden añadir estructuras compatibles o tablas
  laterales. Cualquier cambio destructivo o necesidad de reparación se detiene y
  requiere autorización explícita del usuario.

## Runtime, dependencias y puertas

- CS2 usa **CPython 3.13**. Las dependencias directas de modelo/pipeline/tests se
  declaran en `CS2/requirements.txt`; las exclusivas del scraper, en
  `CS2/SCRAPER/hltv-scraper-api/requirements.txt`.
- Cada cambio de requirements sincroniza su `requirements.lock.txt`. No se
  compensa un import ausente instalándolo desde el código o apoyándose en el
  entorno del otro componente.
- Antes de cerrar: `ruff check`, `ruff format --check`, `mypy`, `pytest TESTS/`,
  el smoke determinista de `MODEL/smoke_pipeline.py`, el check de cobertura de
  imports y el boot de los entrypoints reales deben pasar bajo Python 3.13. El
  scraper ejecuta sus mismas puertas dentro de su propio entorno.

## Roles obligatorios de Codex

### Rol: arquitecto

- [ ] El cambio está en la capa dueña (`SCRAPER`, `PIPELINE`, `BBDD` o `MODEL`)
      y no introduce imports inversos o cruzados.
- [ ] Los contratos de entrada/salida y las rutas reales de arranque siguen
      siendo explícitos y están probados.
- [ ] La documentación y las pruebas reflejan cualquier cambio observable.
- [ ] `prediction_ledger` solo se tocó mediante su API operativa sancionada y
      toda extensión de persistencia es aditiva o lateral.

### Rol: revisor anti-fugas

- [ ] Audité fecha efectiva y disponibilidad de cada input y ambas son `< D`.
- [ ] Ratings/features se emiten antes de observar resultados y el test de
      append-future conserva exactamente el valor as-of.
- [ ] Splits, calibración, selección, tuning y preprocessing son temporales y
      fold-locales, sin shuffle.
- [ ] La semilla de producción es `42` en todos los caminos estocásticos.
- [ ] Ningún dato postpartido se presenta como snapshot prepartido.
- [ ] Toda opening odd de `D` acredita primera captura válida y
      `captured_at_utc < kickoff_utc`; la ausencia de evidencia activa el régimen
      sin odds y no un fallback a mercado tardío.

### Rol: revisor de dependencias

- [ ] Cada import de terceros está declarado directamente en el
      `requirements.txt` del componente que lo ejecuta.
- [ ] Requirements y lock están sincronizados y `pip check` pasa en Python 3.13.
- [ ] Pasan ruff lint/formato, mypy, pytest, smoke determinista, cobertura de
      imports y boot real tanto donde aplique a CS2 como al scraper aislado.
