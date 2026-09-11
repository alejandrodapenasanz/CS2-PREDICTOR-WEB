# Reglas permanentes del proyecto TENNIS

Estas reglas se aplican en todas las fases y sesiones del proyecto.

## Alcance del repositorio

- Todo el proyecto vive dentro de la carpeta `TENNIS/` situada en la raíz del repositorio.
- Nunca se debe crear, editar ni borrar nada fuera de `TENNIS/`, salvo las
  excepciones explícitas documentadas debajo.
- No se debe modificar ningún código de CS:GO. El `start.ps1` de la raíz solo
  puede cambiar dentro de la excepción de orquestación autorizada debajo.

### Excepciones explícitas y limitadas autorizadas por el usuario

- El usuario autorizó expresamente que la fase 9 adapte de forma mínima
  `WEB/build_web.py`, `WEB/index.html` y el `WEB/data.js` generado para mostrar
  los partidos de tenis, su ganador previsto, probabilidad y fiabilidad.
- Esta excepción no autoriza ningún rediseño general de la web ni cambios en
  `CS2/` o cualquier otro archivo situado fuera de `TENNIS/`.
- Tras la fase 9, el usuario autorizó expresamente adaptar el `start.ps1` de la
  raíz únicamente para ejecutar primero `CS2/start.ps1`, después
  `TENNIS/run_tennis.ps1` y propagar `-Retrain` a ambos. Esta autorización no
  permite modificar `CS2/start.ps1` ni ningún otro archivo de CS2.

## Dudas y ambigüedades

- El 05/09/2026 el operador confirmó autorización expresa de Tennis Abstract
  para adquisición automatizada de fichas/históricos y datos de `jsfrags`,
  `jsmatches` y `jsplayers`. Se registra como declaración del operador en
  `config/tennis_abstract_access.json`. La excepción debe estar aislada al host
  exacto y rutas de datos inspeccionadas; no se generaliza a otras fuentes ni
  elimina la detección WAF, los límites por host o la validación de URLs.
  La recuperación de challenges de Tennis Abstract se rige por la autorización
  posterior de navegador descrita debajo. Los archivos JS de históricos se
  parsean como datos, sin ejecutarlos en el parser. La autorización no permite inventar
  fechas de partido, mapear columnas ambiguas ni sustituir producción sin validar.

- El 06/09/2026 el usuario aprobó una migración por fases: conservar Sackmann
  como archivo de entrenamiento y adquirir diariamente históricos/estadísticas
  de Tennis Abstract en un almacén de fuente separado. La adquisición no activa
  por sí sola nuevas features ni reemplaza el overlay/modelo. Fechas de torneo,
  IDs sin mapping y filas con esquema ambiguo no se convierten en datos de
  partido; requieren validación explícita antes de su uso predictivo.

- Ante cualquier duda o ambigüedad —incluidos formatos de datos, decisiones de diseño o detalles no especificados— hay que detenerse y preguntar al usuario.
- No se deben inventar nombres de columnas, rutas ni supuestos. Es preferible formular una pregunta que adoptar un supuesto silencioso.

## Prioridad de adquisición: Tennis Abstract

- El 09/09/2026 el usuario pidió limitar la adquisición diaria a quienes juegan
  en la cartelera seleccionada. Primero se actualiza la cartelera TennisRatio;
  después se adquieren solo sus participantes identificables en los enlaces
  inspeccionados de TA. El catálogo completo requiere `--full-inventory`
  explícito. No se borra el histórico ya adquirido ni se adivinan URLs/identidades
  para cubrir ausencias. La selección por nombre completo único + género sirve
  solo para adquirir perfiles; no constituye mapping canónico para el modelo.
  Una captura de hoy sigue sin ser una feature válida para predecir hoy.
- El 06/09/2026 el usuario confirmó expresamente actualizar estas reglas para
  habilitar navegador, sesión persistente y resolución de challenges de Tennis
  Abstract, con pantalla interactiva cuando sea necesaria. Obtener sus datos,
  métricas e históricos de jugadores es la **máxima prioridad de adquisición**.
  La autorización de acceso del proveedor sigue registrada como declaración
  del operador, no como verificación independiente.
- Esta autorización sustituye, **solo para Tennis Abstract**, las restricciones
  anteriores de transporte exclusivamente estático y de no usar navegador para
  resolver challenges. No cambia las reglas de TennisRatio, Tennis Explorer,
  Sackmann, Match Charting ni CS2. La documentación antigua que describa un
  cliente solo estático refleja la implementación anterior, no limita esta
  autorización nueva.
- Se priorizan las capacidades de Scrapling: transporte HTTP, modo navegador,
  sesiones persistentes y manejo de challenges. Se permite recurrir a
  Selenium/Playwright si es necesario y se justifica técnicamente. La ejecución
  de JavaScript necesaria para cargar la página o completar el challenge queda
  permitida dentro del navegador; los históricos descargados siguen pasando
  por un parser de datos, nunca por evaluación de código de la fuente.
- El navegador puede completar el challenge y conservar las cookies/tokens
  válidos emitidos por el servidor para esa sesión. Si requiere intervención
  humana, se abre una pantalla visible y se informa al operador. Se usa un
  perfil dedicado al proyecto, no el perfil personal del navegador; cookies y
  tokens no se imprimen en logs ni se incorporan a Git.
- La recuperación debe integrarse con el cliente HTTP consolidado y conservar
  validación de URLs, robots y su excepción autorizada, cadencia por host,
  caché, procedencia y timestamps. Se mantienen los límites de peticiones y
  `Retry-After`: resolver un challenge no autoriza a ignorar un límite 429.
  No se desactiva globalmente la detección WAF ni se crean bucles ilimitados
  de intentos; se distingue challenge recuperable, límite de peticiones y
  error de contenido. Un fallo conserva el progreso y el último dato válido.
- La prioridad de adquisición no altera los contratos anti-fugas, las bases
  de datos sagradas ni la puerta de promoción. No se inventan fechas, IDs o
  estadísticas para completar cobertura. Las filas incompletas se conservan
  en cuarentena, sin rellenar campos, conforme a la aprobación del usuario;
  se continúa con las filas válidas y los demás jugadores cuando el acceso
  y el esquema validado lo permitan.
- Cualquier implementación de esta vía debe declarar sus dependencias directas,
  sincronizar el lock si cambia, probar recuperación/fallo y verificar el
  arranque real. Debe distinguirse siempre **autorizado**, **implementado** y
  **verificado con datos reales**: esta actualización de reglas no demuestra
  por sí sola que un challenge concreto se haya resuelto.

## Documentación

- Todo debe quedar documentado.
- Cada módulo y cada función deben incluir un docstring.
- Cada script debe tener una cabecera que explique qué hace, qué recibe y cómo se ejecuta.
- `README.md` debe mantenerse actualizado en cada fase.

## Implementación

- El lenguaje del proyecto es Python.
- El código debe ser modular y testeable, con tests.
- No se permiten scripts monolíticos.

## Regla de oro anti-fugas

- Para predecir un partido con fecha `D`, solo se puede utilizar información anterior a `D`.
- Esta restricción debe respetarse desde el diseño y durante todas las fases posteriores.

## Arquitectura y límites internos

Estas reglas complementan las anteriores y el `AGENTS.md` de la raíz; no
reemplazan ninguna restricción de alcance ni excepción ya documentada.

- `src/` contiene la lógica reusable del dominio; `scripts/` contiene
  entrypoints finos; `tests/` verifica los contratos; `data/`, `models/` y
  `BBDD/` contienen artefactos con procedencia explícita.
- `src/tennis_explorer/` adquiere y parsea la cartelera; `src/player_mapping/`
  resuelve identidades; `src/elo/` mantiene estado causal; `src/features/`
  construye features; `src/modeling/` entrena/evalúa; `src/daily_pipeline/`
  orquesta inferencia y `src/operations/` es la única puerta a la BBDD operativa.
- `run_tennis.ps1` y `scripts/daily_predictions.py` son rutas de arranque reales.
  La lógica de negocio no se duplica en PowerShell, WEB ni el wrapper raíz.
- TENNIS no importa código ni consulta BBDD/modelos de CS2. La web compartida
  consume únicamente la salida pública y la evidencia oficial de TENNIS.
- Las rutas se resuelven desde la configuración del proyecto, no mediante un
  directorio de trabajo implícito. Todo contrato nuevo se documenta y prueba.

## Contrato anti-fugas completo

- Para un partido de fecha `D`, tanto la fecha efectiva del hecho como el momento
  en que la fuente lo hizo disponible deben ser estrictamente `< D`. Si no se
  puede demostrar disponibilidad, el dato se excluye o se marca como ausente.
- Elo general y por superficie, rankings, forma, estadísticas, mapping y mercado
  se consultan as-of. El estado se emite antes de observar el resultado; ningún
  resultado de `D` o posterior se usa para una feature de `D`.
- Añadir partidos o snapshots futuros no puede cambiar un rating, feature,
  fold o predicción histórica. Toda modificación de estas capas conserva una
  prueba de regresión append-future/as-of.
- Entrenamiento, selección de features e hiperparámetros, calibración,
  validación y test usan splits estrictamente temporales. Entrenamiento precede
  a calibración y esta precede al test; no se permite shuffle ni K-fold aleatorio.
- Imputación, escalado, selección y calibración se ajustan solo sobre el bloque
  de entrenamiento correspondiente, nunca sobre el dataset completo.
- La semilla fija de producción es `42` para orientación, modelos y cualquier
  operación estocástica, y debe pasarse explícitamente. Un test puede variar la
  semilla únicamente cuando ese sea el comportamiento bajo prueba.

## Datos operativos sagrados

- Las tablas `predictions`, `observations` y `settlements` de
  `BBDD/tennis.sqlite3` son evidencia sagrada e inmutable/append-only. Sus
  triggers de protección no se desactivan, eliminan, reemplazan ni eluden.
- Está prohibido ejecutar `UPDATE` o `DELETE` sobre ese triplete, reescribir su
  historia en una migración, editar la SQLite manualmente o convertir una
  predicción retrospectiva en oficial.
- Las únicas escrituras permitidas pasan por las APIs sancionadas de
  `src.operations.store` y la orquestación diaria: nuevas predicciones y
  observaciones se añaden; los settlements se crean desde la primera predicción
  oficial válida y una observación conciliada, respetando idempotencia y
  trazabilidad.
- Una corrección o ampliación se representa con una fila nueva por la vía
  sancionada o en una tabla lateral con claves y auditoría. Las migraciones solo
  pueden ser aditivas y deben conservar los triggers existentes.
- Si una necesidad exige mutar datos ya registrados, se detiene el trabajo y se
  pide una decisión explícita al usuario; no se implementa un bypass temporal.

## Runtime y dependencias

- El único runtime admitido es **CPython 3.13**.
- Toda dependencia directa de runtime, tests o tooling se declara en
  `TENNIS/requirements.txt`; no basta con el lock, CI, `pyproject.toml` ni una
  dependencia transitiva.
- Un cambio de `requirements.txt` exige regenerar `requirements.lock.txt` bajo
  Python 3.13 y validar `pip check`, imports y boot real. No se instalan paquetes
  oportunistamente desde el código.
- Antes de cerrar un cambio deben pasar `ruff check`, `ruff format --check`,
  `mypy`, `pytest tests/`, el smoke determinista, cobertura de imports y el boot
  real de `run_tennis.ps1`/`scripts/daily_predictions.py` según corresponda.

## Roles obligatorios de Codex

### Rol: arquitecto

- [ ] El cambio está en la capa propietaria y respeta la dirección de
      adquisición → identidad/estado → features → modelo → operaciones/salida.
- [ ] Los entrypoints son adaptadores finos y los contratos, rutas y artefactos
      permanecen explícitos, portables, documentados y testeados.
- [ ] No introduje dependencias con CS2 ni lógica de dominio en WEB/orquestación.
- [ ] Toda persistencia nueva es aditiva/lateral y usa `src.operations` sin
      debilitar los triggers del triplete sagrado.

### Rol: revisor anti-fugas

- [ ] Verifiqué fecha efectiva y disponibilidad `< D` de cada feature.
- [ ] Elo/estado se emiten antes de observar el resultado y append-future no
      cambia ningún valor histórico.
- [ ] Entrenamiento, calibración, selección, preprocessing y evaluación son
      temporales y fold-locales, sin shuffle.
- [ ] La semilla de producción es `42` y está propagada explícitamente.
- [ ] Predicciones y settlements operativos no retroalimentan Elo, features ni
      reentreno. Solo resultados terminales de fuente, leídos sin mutar la BBDD
      sagrada y bajo el handoff fijo, pueden actualizar Elo desde `D+1`; ningún
      resultado entra como información prepartido de su propio día.

### Rol: revisor de dependencias

- [ ] Cada import de terceros está declarado directamente en
      `TENNIS/requirements.txt`.
- [ ] Requirements y lock están sincronizados y pasan `pip check` en Python 3.13.
- [ ] Pasan ruff lint/formato, mypy, pytest, smoke determinista, cobertura de
      imports y boot de los entrypoints reales en un entorno limpio.
