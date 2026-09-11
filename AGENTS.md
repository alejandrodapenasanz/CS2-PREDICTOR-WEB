# Reglas permanentes del repositorio

Estas reglas se aplican a todo el repositorio. Los `AGENTS.md` de cada
componente añaden restricciones locales: una modificación debe satisfacer ambos
niveles. Si dos contratos parecen incompatibles, se detiene el trabajo y se
pregunta antes de editar.

## Arquitectura y límites

- `CS2/` y `TENNIS/` son dominios independientes: cada uno es dueño de sus
  datos, modelos, código, pruebas, entorno y manifiestos de dependencias. No se
  importan módulos ni se consultan bases de datos del otro dominio.
- `CS2/SCRAPER/hltv-scraper-api/` es un componente aislado de adquisición de
  CS2, con su propio entorno y sus propios `requirements.txt` y lock. Entrega
  datos al pipeline mediante sus contratos publicados; el modelo no depende de
  la implementación interna del scraper.
- `WEB/`, `TELEGRAM/`, `.github/` y el `start.ps1` raíz son capas compartidas de
  presentación, publicación u orquestación. No contienen lógica de features,
  entrenamiento ni mutaciones directas de las bases de datos de dominio.
- El `start.ps1` raíz solo coordina entrypoints reales. Los entrypoints de cada
  componente siguen siendo los responsables de inicializar su entorno, validar
  entradas y propagar fallos; no se duplican pipelines en el wrapper raíz.
- Toda integración entre capas usa un contrato explícito, documentado y
  testeado. No se acoplan componentes mediante imports cruzados, rutas ocultas,
  el directorio de trabajo actual o lectura oportunista de artefactos internos.
- El código nuevo debe ser modular, tipable y testeable. Las decisiones de
  arquitectura y los cambios de contrato se reflejan en la documentación del
  componente afectado.

## Contrato anti-fugas y reproducibilidad

- Para un partido con fecha `D`, una feature solo puede usar hechos cuya fecha
  efectiva y fecha de disponibilidad sean estrictamente anteriores a `D`.
  `D` y cualquier fecha posterior están prohibidas, aunque el dato exista hoy.
- Excepción estrecha para **CS2 opening odds**: una cuota de apertura observada
  puede estar disponible durante `D` únicamente si conserva un
  `captured_at_utc` demostrable y este es estrictamente anterior al
  `kickoff_utc` publicado del partido. La feature usa siempre la primera captura
  válida, de-vigada a suma 1. Sin ambos timestamps fiables se aplica la regla
  civil `< D`. Esta excepción nunca alcanza cuotas live, de cierre, proxies ni
  snapshots capturados al inicio o después del partido.
- Si una fuente no permite demostrar cuándo estaba disponible un dato, ese dato
  no entra en la feature. Con granularidad diaria no se inventa orden intradía:
  el estado se congela al inicio del día y los resultados de ese día se
  incorporan después de emitir todas sus features.
- Ratings y estados acumulados se emiten antes de observar el resultado del
  partido. Añadir partidos futuros no puede cambiar features, ratings ni
  predicciones históricas; esta invariancia debe tener una prueba de regresión.
- Todo split de entrenamiento, calibración, selección de features,
  hiperparámetros, validación y test es temporal: entrenamiento en el pasado y
  evaluación en el futuro. Se prohíben el shuffle aleatorio y los K-fold
  aleatorios para evaluar el modelo.
- Imputadores, escaladores, calibradores, selectores y cualquier transformación
  aprendida se ajustan solo con el bloque de entrenamiento de cada fold.
- La semilla canónica y fija de producción es `42`. Toda librería o algoritmo
  estocástico debe recibirla explícitamente; no se admite aleatoriedad sin
  semilla ni cambiarla silenciosamente. Un test puede usar otra semilla solo si
  el propio objeto de la prueba es comprobar el comportamiento frente a ella.
- Todo cambio en datos, features, ratings o evaluación incluye una prueba
  anti-fugas temporal proporcional al riesgo.

## Runtime y dependencias

- El runtime único es **CPython 3.13** en local, scripts, CI y contenedores.
- Toda dependencia directa de ejecución, tests o tooling se declara en el
  `requirements.txt` del componente que la importa. No basta con añadirla al
  lock, al CI, al `pyproject.toml` o confiar en que llegue de forma transitiva.
- Cuando existe `requirements.lock.txt`, se regenera desde su
  `requirements.txt` en el mismo cambio y se valida con `pip check` bajo Python
  3.13. CS2, TENNIS y el scraper mantienen manifiestos separados.
- Se prohíben instalaciones oportunistas desde el código y fallbacks que oculten
  una dependencia ausente. La ruta real de arranque debe poder instalar sus
  requisitos declarados e importar/arrancar sus entrypoints.
- Antes de cerrar un cambio, el componente afectado debe pasar sus puertas:
  `ruff check`, `ruff format --check`, `mypy`, `pytest`, smoke determinista y
  cobertura de imports/arranque real.

## Datos sagrados

- `CS2` considera sagrada la tabla `prediction_ledger`. No se modifica con SQL
  manual, scripts ad hoc, backfills destructivos ni migraciones que reescriban o
  borren su historia. Solo se usan las vías operativas sancionadas por CS2 para
  registrar una predicción prepartido y completar su ciclo de vida.
- `TENNIS` considera sagrado el triplete `predictions`, `observations` y
  `settlements`, protegido por triggers. Nunca se desactivan, eliminan ni eluden
  esos triggers, y nunca se hace `UPDATE` o `DELETE` sobre esas tablas.
- Una ampliación de cualquiera de estos contratos debe ser **aditiva**. La
  información nueva se registra con nuevas filas por la API sancionada o en una
  tabla lateral con claves foráneas y trazabilidad. No se corrige el pasado
  sobrescribiéndolo.
- Copiar, restaurar, migrar o reparar una base de datos no autoriza a alterar
  estos historiales. Ante una necesidad no cubierta por una vía sancionada, se
  para y se pide una decisión explícita al usuario.

## Roles obligatorios de Codex

Antes de dar un cambio por terminado, Codex debe recorrer y dejar satisfechos
los tres checklists siguientes.

### Rol: arquitecto

- [ ] Identifiqué el componente dueño de cada cambio y respeté sus límites.
- [ ] Conservé la dirección de dependencias y usé contratos explícitos entre
      adquisición, persistencia, modelado, presentación y orquestación.
- [ ] Evité duplicar lógica de dominio en launchers, CI, WEB o TELEGRAM.
- [ ] Actualicé pruebas y documentación cuando cambió un contrato observable.
- [ ] La solución es mínima, modular y compatible con los entrypoints reales.

### Rol: revisor anti-fugas

- [ ] Para cada feature identifiqué fecha efectiva y fecha de disponibilidad, y
      ambas son `< D` para el partido predicho.
- [ ] Ratings/estados se emiten antes de observar el resultado y existe cobertura
      frente a la adición de datos futuros.
- [ ] Todos los folds y pasos aprendidos respetan el orden temporal, sin shuffle.
- [ ] La semilla de producción es `42` y queda propagada explícitamente.
- [ ] No alteré datos sagrados ni usé resultados, cuotas live/cierre u otros
      artefactos posteriores como información prepartido.
- [ ] Si usé opening odds de CS2 capturadas en `D`, acredité
      `captured_at_utc < kickoff_utc`, primera captura válida y de-vig; sin esa
      evidencia la fila siguió el corte civil `< D` o el régimen sin odds.

### Rol: revisor de dependencias

- [ ] Cada import no estándar pertenece al componente correcto y su dependencia
      directa figura en el `requirements.txt` correspondiente.
- [ ] Sincronicé y comprobé el lock cuando cambió `requirements.txt`.
- [ ] Verifiqué instalación, `pip check` e imports con CPython 3.13.
- [ ] Ejecuté lint, formato, tipos, tests, smoke determinista y boot real del
      componente, sin apoyarme en paquetes transitivos o globales.
