# Formato y ronda en inferencia diaria

Contrato `schedule_match_format_v1`. Es un adaptador de cartelera a las
features existentes, no una familia nueva ni un cambio del modelo activo.

## Evidencia y corte temporal

TennisRatio sí publica el nivel exacto en `data-level`, la ronda en el bloque
`.match-time` de cada tarjeta y en `.tournament-meta` del grupo concreto de
partidos. Las filas compactas heredan solo la ronda de su propio grupo. Un
conflicto entre tarjetas y cabecera invalida esa propagación. No se consulta
el resultado del partido ni el histórico del perfil para determinar su ronda.

El parser conserva `tournament_level_source`, `round_source` y
`round_evidence`, sin cambiar el antiguo `tour_level` de presentación.
La lectura causal selecciona únicamente lotes **publicados y capturados antes
del día D**. El vector recibe campos `feature_*` de esa observación, nunca de
la cartelera actual de D. Se registran URL, SHA-256, captura, versión del
adaptador y motivos de resolución en el CSV diario.

Para snapshots antiguos cuyo parser descartó estos campos,
`tennisratio.agenda_context.restore_agenda_context` consulta la SQLite lateral
en `mode=ro`, localiza los bytes originales por su SHA-256 y los vuelve a
parsear. Verifica hash, identidad, torneo y fecha; conserva la captura original.
No reescribe observaciones ni crea una supuesta captura antigua. Raw eliminado,
corrupto o identidad contradictoria: ausencia explícita y aviso.

## Semántica compatible con entrenamiento

| Evidencia de individuales | `best_of` |
| --- | ---: |
| Grand Slam masculino, ronda inequívoca de cuadro principal | 5 |
| Grand Slam femenino, cuadro principal o previa `Q1`–`Q3` | 3 |
| ATP/WTA/Challenger/ITF estándar, régimen moderno desde 2009 | 3 |
| Previa masculina de Slam sin regla específica, formato especial o nivel desconocido | nulo |

No se deduce un formato por ciudad, resultado, marcador o cantidad de sets
jugados. Next Gen, exhibiciones, dobles, juniors, silla de ruedas y Davis Cup
no reciben el supuesto genérico. Una etiqueta `Qualies` con `Final` no se
convierte en una final de cuadro principal.

El [reglamento Grand Slam 2026, I.L](https://www.itftennis.com/media/5986/grand-slam-rulebook-2026-f2.pdf)
acredita la distinción masculina de cuadro principal. Como comprobación de
compatibilidad, los CSV Sackmann locales de 2026 contienen 254 partidos
masculinos `G/5` y 254 femeninos `G/3`. En ATP, los niveles A/M de esos CSV
usan 3 sets; en WTA los niveles I/P/PM también usan 3.

`Final → F`, `Semi-finals → SF`, `Quarter-final → QF`, `Round of 128/64/32/16
→ R128/R64/R32/R16`, `Round robin → RR`, `Qualifying round n → Qn`.
Los códigos Sackmann ya canónicos se conservan. `First round`, `Qualification`
sin número o una etiqueta desconocida **no** permiten inventar el tamaño del
cuadro. Entrenamiento conserva la ronda raw Sackmann y esta normalización
solo adapta la fuente de producción.

Ausencia significa `None` en el vector, con `best_of_missing`/`round_missing`
en la confianza y motivos explícitos en `best_of_reason`/`round_reason`.
Se conserva el preprocesado ajustado durante entrenamiento: imputación
numérica con los indicadores que ese ajuste haya aprendido y categoría
`__MISSING__` para categóricas. No se añaden indicadores nuevos a un bundle
ya entrenado ni se cambia su dimensionalidad. No se usa un default de formato
en el adaptador. Elo, modelos, calibración y tablas sagradas quedan intactos.

## Verificación reproducible sin escrituras operativas

Desde `TENNIS/`:

```powershell
.\.venv\Scripts\python.exe scripts\audit_match_format.py
.\.venv\Scripts\python.exe scripts\audit_match_format.py --date 2026-09-03 --replay-at 2026-09-03T00:00:00+00:00
.\.venv\Scripts\python.exe scripts\run_quality_gates.py
```

La auditoría usa el pipeline real con `publish=False`: no publica predicciones,
no concilia resultados y no activa modelos. Una repetición histórica se
etiqueta `diagnostic_replay_not_official`, nunca se inserta en el ledger.
El pipeline puede regenerar su caché derivada de superficies.

El resumen normal de `scripts/daily_predictions.py`, invocado desde los
launchers, muestra cobertura de ambos campos y cuántas predicciones suben de
nivel. Los porcentajes usan como denominador las predicciones efectivamente
calculadas; el total de la cartelera se informa por separado. Sin muestra: N/A.

## Resultado observado el 5 de septiembre de 2026

Repetición causal del 3 de septiembre, con modelo activo y snapshots reales
guardados antes de D: **42 partidos, 41 predicciones**. Los 41 tienen formato
y ronda (100% de predicciones, 97,62% de la cartelera). Antes del arreglo,
ambos campos se enviaban nulos. 23 usan regla Slam de cuadro principal y
18 la regla de circuito estándar. Confianza: **41 MEDIUM → MEDIUM**; se
eliminan flags de ausencia, pero otros avisos impiden subir de nivel.

En la auditoría inicial del 5 de septiembre, antes del arreglo JSD:
**0 filas disponibles en la agenda publicada**,
porcentajes N/A. La inferencia sin publicación arranca, pero no hay una
medición de cobertura de hoy. El último lote fuente publicado es del día 2.
El arranque del día 5 terminó en `tennisratio_update_failed`: el cliente A1
rechaza una respuesta HTTP 200 con cartelera y con el marcador
`/cdn-cgi/challenge-platform/scripts/jsd/main.js`. Este cambio no desactiva ni
modifica el cortacircuitos. El problema de adquisición queda separado y no se
oculta declarando cobertura histórica como si fuera actual.

Tras autorizar y corregir la detección JSD ese mismo día, el actualizador
Scrapling publicó 35 partidos. La inferencia real sin publicación verificó:

| Fecha | Cartelera | Predicciones válidas | `best_of` / `round` rellenos | Confianza |
| --- | ---: | ---: | ---: | --- |
| 2026-09-05 | 22 | 16 | 0/16 (0%) | 16 MEDIUM → MEDIUM |
| 2026-09-06 | 13 | 12 | 12/12 (100%) | 12 MEDIUM → MEDIUM |

Las 16 predicciones de hoy conservan ambos flags: la captura fue durante D,
por lo que no puede rellenar las features de hoy retrospectivamente. Para
mañana, el mismo lote sí acredita captura y publicación anteriores a D:
9 predicciones usan formato Slam principal y 3 circuito estándar. Es una
comprobación operativa adicional del corte as-of, no una excepción nueva.
Véase [corrección de adquisición y verificación](http_acquisition.md).

Puertas locales finales: Ruff lint y ratchet de formato verdes (82 archivos
protegidos; 94 pendientes heredados, sin crecimiento), mypy en 15 módulos,
**508 tests y 262 subtests**, cobertura de imports, `pip check` y smoke de
16 imports/7 entrypoints deterministas. Los 32 tests específicos incluyen
cambio de publicación futura, igualdad del contexto congelado, rechazo de
captura durante D, SHA incorrecto, conflicto de grupo y paso efectivo de los
valores a la inferencia. Modelo activo conservado: `d080b15c…`.

Revisión de roles: adquisición/adaptación/inferencia separadas; ningún cambio
matemático en Elo, calibración o entrenamiento; ningún import nuevo externo;
sin escrituras en predictions/observations/settlements ni modificaciones de
sus triggers.
