# Reglas permanentes de TELEGRAM

Estas reglas se aplican a todo el árbol `TELEGRAM/`.

- Todo el publicador de Telegram vive dentro de `TELEGRAM/`. No se crea, edita ni borra ningún archivo fuera de esta carpeta.
- El canal operativo es `@cs2DailyPicks` y el bot publicador es `@CS2PredictorPublisherBot`.
- El canal es un canal de difusión: los suscriptores solo leen. No se vincula un grupo de discusión.
- El bot recibe únicamente el permiso de administrador **Post Messages**. No necesita borrar mensajes, editar el canal, invitar usuarios ni añadir administradores.
- El token del bot es secreto. Solo puede existir en `TELEGRAM/.env`, que está ignorado por Git. Nunca se escribe en código, pruebas, documentación, logs ni argumentos de línea de comandos.
- Todo token expuesto debe revocarse en `@BotFather` antes de ejecutar un envío real. No se reutiliza aunque su propietario considere seguro el contexto donde se publicó.
- Cada jornada genera exactamente dos publicaciones en inglés: una para **BEST OPPORTUNITIES**, con todos los partidos elegibles de ese día, y otra para **TODAY'S OTHER PICKS**, con el resto de partidos de ese día.
- BEST OPPORTUNITIES exige además `prediction.decision_confidence > 0.65`, sin redondear: probabilidad ajustada del ganador, no la probabilidad cruda del modelo. El 65% exacto queda en OTHER PICKS. Se conserva la elegibilidad exportada por CS2; no se recalcula la lógica de rosters/modelo en Telegram. El mismo filtro se aplica a `daily_report.txt`.
- Cada pronóstico muestra `Confidence: HIGH/MEDIUM/LOW`, procedente de `prediction.estimate_confidence_level`; no se deduce del porcentaje de victoria. Si falta o es null, muestra `NOT AVAILABLE`; un valor no reconocido falla explícitamente. Los cambios de formato no reinician la idempotencia ni reenvían mensajes confirmados.
- Una ejecución real que termine correctamente, incluido el no-op idempotente, reemplaza atómicamente `TELEGRAM/daily_report.txt`. El archivo contiene todas las oportunidades elegibles nuevas de la fecha solicitada y de fechas posteriores disponibles en el último run, con fecha, URL canónica de HLTV, ganador previsto e invitación sin emojis a `@cs2DailyPicks`.
- Cada partido que aparece en `daily_report.txt` se registra por ID en SQLite. Un partido ya registrado nunca vuelve a incluirse en ejecuciones posteriores, aunque haya desaparecido del archivo reemplazado.
- Las probabilidades son estimaciones del modelo, no garantías. El texto nunca presenta una apuesta como resultado cierto.
- Los envíos deben ser idempotentes: reejecutar la misma jornada no debe duplicar mensajes ya confirmados.
- `--dry-run` no realiza ninguna petición de red, no abre la base de estado y no escribe `daily_report.txt`; solo muestra las dos publicaciones y una vista previa del informe.
- Las pruebas automatizadas no dependen de Telegram ni de ninguna otra red.
- Todo módulo y toda función deben tener docstring. Los scripts deben documentar propósito, entradas y forma de ejecución.
- Ante un contrato de datos ambiguo o un cambio en la salida de CS2, se falla con un error explícito; no se inventan partidos, equipos, ganadores ni probabilidades.
- Excepción autorizada: se puede editar únicamente el `start.ps1` de la raíz para orquestar `TELEGRAM/run_telegram.ps1`; esta excepción no autoriza modificar `CS2/start.ps1`, `TENNIS/` ni ningún otro archivo fuera de `TELEGRAM/`.
