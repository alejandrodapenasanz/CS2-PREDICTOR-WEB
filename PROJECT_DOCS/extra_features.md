# Extra features: activacion automatica y fuentes pendientes

Ultima actualizacion: 2026-07-10

Este archivo lista las familias extendidas, su cobertura point-in-time y las fuentes que todavia no permiten construir una feature valida. Las familias implementadas no requieren switches: `MODEL/train.py` las activa automaticamente al superar el umbral y registra el resultado en `feature_policies`.

Regla general: una feature solo debe entrar al modelo o a un calibrador productivo si existe muestra point-in-time suficiente, mejora log loss/Brier en walk-forward expansivo y no introduce fuga temporal.

## Resumen de estado

| Feature | Estado actual | Activacion minima sugerida |
|---|---|---|
| HLTV Betting Analytics | Auto-gate 100/120 | 120+ partidos cerrados point-in-time con Analytics |
| Contexto HLTV `Maps` | Auto-gate 65/200; LAN 7/50, online 58/50 | 200+ y 50 por entorno |
| Model B con odds de apertura | 98 partidos unicos; evaluacion separada | 120+ partidos cerrados con odds de apertura |
| Stats individuales point-in-time | Auto-gate 88/200 | 200+ con cobertura de ambos equipos |
| Map/player side stats L5/L10/L20 | Auto-gate 20/200 | 200+ con historial para ambos equipos |
| Modelo composicional Bo3 por mapas | Disenado; no implementado | Modelo de mapa + veto predictivo |
| Veto/picks/bans predictivo | Veto real guardado en completados; no predicho prepartido | Historial suficiente de vetos por equipo |
| Ranking historico HLTV/Valve | Auto-gate 86/200 con join as-of | 200+ con ranking previo de ambos equipos |
| Fatiga/travel real | Fatiga simple en flags; travel desconocido | Localizacion/evento/LAN y calendario fiable |
| Roster/stand-ins como feature fuerte | Auto-gate 61/200 con join as-of | 200+ con roster previo completo |

## 1. HLTV Betting Analytics

Fuente: pagina `/betting/analytics/...` de HLTV.

Ya existe:

- Captura en cada scrape futuro cuando HLTV la expone.
- Persistencia en `DAILY_SNAPSHOTS/runs/<run>/analytics/*.json`.
- Archivo raw en `raw_snapshots(kind='match_analytics')`.
- Parser de mapas/insights y columnas `analytics_*` preparadas.

Preparado para activacion automatica (100/120):

- `analytics_map_win_pct_diff`
- `analytics_first_pick_pct_diff`
- `analytics_first_ban_pct_diff`
- `analytics_map_played_diff`
- `analytics_insight_score_diff`
- `analytics_available`
- `analytics_common_maps`
- `analytics_map_rows_total`
- `analytics_insight_total`

Motivo:

- Muestra todavia pequena para entrenamiento sin overfitting.
- Debe garantizarse que `captured_at` sea anterior o igual al dia del partido.

Criterio de activacion:

- 120+ partidos cerrados con Analytics point-in-time.
- Mejora de log loss/Brier en walk-forward.
- SHAP revisado para comprobar que no canibaliza o aprende ruido.

## 2. Contexto competitivo desde HLTV `Maps`

Fuente: caja `Maps` de HLTV, por ejemplo:

- `Best of 3 (LAN)`
- `Swiss round 4 (teams with a 2-1 record). Winner advances to playoffs.`
- `Losing team is eliminated.`
- `Grand final`
- notas de sustitucion/stand-in

Ya existe:

- Parser `DAILY_SNAPSHOTS/match_context.py`.
- Guardado en snapshots, assets y SQLite:
  - `matches.environment`
  - `matches.stage`
  - `matches.stage_detail`
  - `matches.incentive_label`
  - `matches.context_json`
- Visualizacion en detalle de partido de la web.
- Flags como `LAN_MATCH`, `ONLINE_MATCH`, `WINNER_ADVANCES`, `ELIMINATION_MATCH`, `SUBSTITUTION_NOTE`.
- Diagnostico automatico con `MODEL/analyze_context_calibration.py`.

Preparado para activacion automatica (65/200; falta llegar a 50 LAN):

- Calibrador contextual de confianza.
- Features directas del Modelo A:
  - `context_is_lan`
  - `context_environment_known`
  - `context_high_stakes`
  - `context_winner_advances`
  - `context_loser_eliminated`
  - `context_substitution_note`
  - one-hot de `stage`

Motivo:

- Ahora hay 65 muestras cerradas con contexto: 7 LAN y 58 online. Sigue siendo diagnostico hasta superar todos los umbrales.
- Los subgrupos pequenos pueden enganar: finales, league, semis, eliminacion, etc.

Criterio de activacion:

- 200+ partidos cerrados con contexto point-in-time.
- Subgrupos relevantes con 30+ muestras.
- Walk-forward expansivo mejora log loss/Brier.
- No activar si solo mejora accuracy pero empeora calibracion.

## 3. Model B: stats + odds de apertura

Fuente: odds guardadas por bookmaker con timestamp.

Ya existe:

- Tabla `odds` con `opening`, `live`, `closing`.
- `staking` y EV usan odds.
- Evaluacion separada de Model B en `MODEL/train.py`.
- Politica de mercado como prior prudente en `decision_prob_team1`.

No activado aun como modelo productivo:

- `opening_odds_prob_centered`
- `opening_odds_confidence`
- `opening_bookmaker_count_log`
- movimientos de linea como feature fuerte
- drift apertura -> ultima cuota
- consenso multi-book como feature del modelo

Motivo:

- Riesgo de que el modelo copie mercado y pierda interpretabilidad.
- Poca muestra con odds point-in-time.
- Closing odds pueden contener informacion tardia y no deben usarse como feature prepartido.

Criterio de activacion:

- 120+ partidos cerrados con odds de apertura.
- Comparar Modelo A vs Modelo B por walk-forward.
- Usar apertura, nunca cierre, para feature prepartido.
- Mercado puede seguir siendo benchmark y EV/staking aunque no entre al Modelo A.

## 4. Stats individuales point-in-time de jugadores

Fuente: `/stats/players/compare/...` y stats HLTV por ventanas.

Ya existe:

- Captura por `start.ps1` de `past3months`, `past6months`, `past12months` y ano actual cuando HLTV lo devuelve.
- Persistencia en `player_stat_snapshots`.
- Uso en web/rosters:
  - rating
  - KPR/DPR/APR
  - KAST
  - Impact
  - ADR
  - Round Swing
  - Multi-kill rating
  - AWP KPR
  - HS %
  - Opening KPR/DPR
  - Flash assists
- Agregados de equipo mostrados:
  - media
  - maximo/estrella
  - minimo/weak link
  - desviacion/spread
  - star gap
  - weak-link gap

Preparado para activacion automatica (88/200):

- Las columnas agregadas ya estan conectadas al entrenamiento y scoring, protegidas por el auto-gate.
- Comparacion pairwise completa jugador vs jugador como matriz de matchups.
- Sinergia entre jugadores o dependencia de estrella.

Motivo:

- Se necesita asegurar point-in-time: no pegar stats actuales a partidos pasados.
- Las ventanas de HLTV deben estar asociadas al dia del snapshot.
- Muestra cerrada con snapshots reales aun pequena.

Criterio de activacion:

- Feature store por jugador/equipo con `captured_at` implementado en SQLite.
- En backtest, usar solo snapshots existentes antes del partido.
- Evaluar por walk-forward y por cobertura.

## 5. Box score por mapa y forma L5/L10/L20

Fuente: `map_player_stats` y `map_player_side_stats`.

Ya existe:

- Captura de partidos completados con assets HLTV.
- Tablas SQLite:
  - `maps`
  - `veto`
  - `match_lineups`
  - `map_player_stats`
  - `map_player_side_stats`
- Algunas features historicas derivadas de assets ya existen parcialmente:
  - `asset_map_winrate_diff`
  - `asset_ct_round_winrate_diff`
  - `asset_t_round_winrate_diff`
  - `asset_rating_l5_diff`
  - `asset_rating_l10_diff`
  - `asset_rating_l20_diff`
  - `asset_adr_l10_diff`
  - `asset_kast_l10_diff`
  - `asset_opening_diff_l10_diff`

No implementado completamente:

- Forma L5/L10/L20 por jugador actual real.
- Forma L5/L10/L20 por lineup de 5 jugadores.
- Separacion por mapa y lado CT/T con cobertura suficiente.
- Agregados con decay temporal.
- Features de pistol/eco/force-buy, si se consigue fuente fiable.

Motivo:

- Solo hay assets para una parte del master diario.
- Falta backfill grande de mapstats historicos o acumulacion suficiente.

Criterio de activacion:

- Cobertura suficiente por equipo/jugador/mapa.
- Separar "no hay datos" de "valor cero".
- Validacion por subgrupo: equipos con alta vs baja cobertura.

## 6. Map pool, veto y modelo composicional Bo3

Fuente: veto real de completados, stats de mapa, HLTV Analytics y resultados por mapa.

Ya existe:

- Estimacion prepartido de map pool para web.
- Veto real guardado en completados capturados.
- `map_pool_advantage_team1` como senal operativa.

No implementado:

- Modelo predictivo de veto/picks/bans.
- Probabilidad por mapa `P(team gana mapa X)`.
- Modelo composicional Bo3:
  - `P(2-0)`
  - `P(2-1)`
  - probabilidad de serie a partir de mapas esperados.
- H2H por mapa.
- Pick/ban tendencies por equipo y por formato.

Motivo:

- Requiere mucha mas cobertura de vetos y mapstats.
- El veto real muchas veces no se conoce antes del partido.
- Hay riesgo de fuga si se usa veto real postpartido para predecir.

Criterio de activacion:

- Backfill o acumulacion de vetos historicos.
- Modelo de veto entrenado solo con datos anteriores al partido.
- Comparar serie directa vs composicional por log loss.

## 7. Ranking historico y fuerza del evento

Fuente: rankings HLTV/Valve, evento, prize pool, fase, tier.

Ya existe:

- Snapshots de rankings actuales HLTV/Valve.
- Contexto de evento basico.
- Volatilidad historica del evento aproximada.

Implementado parcialmente y protegido por auto-gate (86/200):

- Join de ranking exacto "as of match date" para los snapshots ya acumulados.
- Fuerza del evento/tier aprendida.
- Prize pool normalizado.
- Region/event strength.
- Campo de invitacional/qualifier/closed qualifier/playoff con codificacion estable.

Motivo:

- Los snapshots actuales no bastan para partidos historicos.
- Se necesita backfill de rankings por fecha o fuente historica fiable.

Criterio de activacion:

- Ranking point-in-time por semana/fecha.
- Evitar usar ranking actual para partidos pasados.
- Validar contra Glicko/Elo para ver si aporta senal incremental.

## 8. Roster, stand-ins y lineup real

Fuente: perfiles de equipo, box score, notas HLTV y roster history.

Ya existe:

- `team_rosters` con snapshots.
- `match_lineups` desde mapstats completados.
- Flags de roster reciente y stand-in risk.
- Notas `SUBSTITUTION_NOTE` desde HLTV `Maps`.

Implementado parcialmente y protegido por auto-gate (61/200):

- Dias desde el primer snapshot observado del roster actual, tamano y riesgo de roster incompleto.
- Mapas jugados juntos por el quinteto actual.
- Penalizacion aprendida por stand-in.
- Cambio de IGL/coach/rol.
- Transfer windows / roster lock.

Motivo:

- Los perfiles capturados no siempre tienen roster completo.
- El lineup real se sabe mejor despues de jugar el mapa.
- Stand-in puede aparecer en nota textual no estandar.

Criterio de activacion:

- Lineup point-in-time fiable.
- Validar subgrupo "stand-in note" y "recent roster change".
- No inferir roles internos si no hay fuente.

## 9. Fatiga, calendario y travel

Fuente: calendario propio, eventos, LAN/online y geografia.

Ya existe:

- `fatigue_last24_team*`
- `fatigue_last48_team*`
- `same_day_total`
- flags de back-to-back y densidad de calendario.
- `online_lan` en controles/fatigue.

No implementado completamente:

- Travel real entre LANs.
- Jet lag por region/pais.
- Duracion real del partido anterior.
- Hora local del equipo.
- Fatiga por varios mapas jugados el mismo dia.
- Descanso entre series en horas exactas con zona horaria fiable.

Motivo:

- HLTV no siempre da ubicacion operacional suficiente.
- Requiere geocodificar evento/equipo y horarios exactos.

Criterio de activacion:

- Fuente fiable de ubicacion del evento y equipos.
- Diferenciar online de LAN.
- Validar si la senal mejora log loss sin castigar tiers con datos incompletos.

## 10. Line movement, consenso y eficiencia de mercado

Fuente: odds por bookmaker y timestamp.

Ya existe:

- Odds por proveedor.
- Apertura, live y cierre cuando se capturan.
- Drift y consenso como controles.
- Staking usa EV/Kelly con cuota disponible.

No implementado como modelo:

- Curva temporal de movimiento de linea.
- Features por tipo de casa/bookmaker.
- Consenso ponderado por liquidez.
- Steam move / sharp move.
- Comparacion contra closing line value.

Motivo:

- HLTV no da liquidez ni limites.
- Todavia hay pocas odds historicas multi-book.

Criterio de activacion:

- Mas odds point-in-time.
- Separar benchmark/EV de prediccion estadistica.
- No usar cierre como feature prepartido.

## 11. Favorite upset history

Fuente: odds prepartido guardadas y resultados cerrados.

Ya existe:

- Web muestra cuantas veces el favorito de mercado perdio en ventana reciente.
- `favorite_upset_90d` como control/explicacion.

No implementado en Modelo A:

- Feature aprendida de upset por equipo cuando era favorito.
- Interaccion con odds, tier, stage y formato.

Motivo:

- Muestra con odds baja.
- Riesgo de sobreajustar a equipos con pocos partidos.

Criterio de activacion:

- Suficiente historial por equipo como favorito de mercado.
- Regularizacion fuerte o smoothing bayesiano.
- Validar por log loss, no solo por detectar upsets.

## 12. Patch, map pool y cambios de metajuego

Fuente: calendario de Valve/HLTV, cambios de mapas, cambios de economia/rating.

Ya existe:

- Filtro de era CS2.
- Documentacion de riesgos por cambios de rating/map pool.

No implementado:

- Feature de version/parche.
- Feature de map pool activo por fecha.
- Decay o reseteo parcial tras parche gordo.
- Indicador de cambio de formula HLTV/rating.

Motivo:

- Requiere calendario estructurado de cambios.
- Dificil atribuir efectos con poca muestra alrededor del parche.

Criterio de activacion:

- Tabla `game_versions` o equivalente.
- Ablation por periodos antes/despues.
- Validar si reduce drift de calibracion.

## 13. Integridad competitiva y sanciones oficiales

Fuente: ESIC, IBIA, comunicados oficiales.

Ya existe:

- Politica documentada: no inferir match fixing por cuenta propia.

No implementado:

- Tabla `sanctions`.
- Limpieza automatica del entrenamiento por periodo sancionado.
- Flag oficial por equipo/jugador sancionado.

Motivo:

- Requiere fuente oficial y fechas exactas.
- Riesgo legal/etico si se infiere sin sancion publicada.

Criterio de activacion:

- Solo sanciones oficiales.
- Guardar fuente y rango temporal.
- Excluir/flaggear solo el periodo afectado.

## 14. Datos externos no disponibles ahora

Estas features podrian ser utiles, pero no deben inventarse:

- lesiones o enfermedad de jugadores
- problemas internos del equipo
- scrims/practice results
- liquidez real y limites de casas
- informacion privada de mercado
- cambios de rol no publicados
- estado mental/motivacion interna

Estado:

- No implementadas.
- Solo se anaden si hay fuente publica fiable, timestamp y politica anti-fuga.

## Orden recomendado de implementacion

1. Acumular muestra; la activacion de familias implementadas ya es automatica.
2. Revisar `MODEL/results/REPORT.md` y `feature_policies` despues de cada entreno.
3. Backfill de ranking historico HLTV/Valve para ampliar cobertura as-of.
4. Modelo B con odds de apertura cuando haya 120+ cerrados con odds.
5. Modelo composicional Bo3 por mapas cuando haya suficiente veto/mapstats.
6. Añadir travel/parches/sanciones solo cuando exista fuente point-in-time fiable.
