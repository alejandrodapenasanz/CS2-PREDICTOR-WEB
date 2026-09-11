# Integración del lanzador raíz

El usuario autorizó expresamente integrar ambos deportes en el `start.ps1` de
la raíz. La implementación no modifica `CS2/start.ps1`: el lanzador raíz actúa
únicamente como orquestador y resuelve todas las rutas desde `$PSScriptRoot`.

Ejecución diaria conjunta desde cualquier ruta:

```powershell
& 'C:\dev\CS2-Predictor\start.ps1'
```

El orden es:

1. ejecutar `CS2/start.ps1` con todos los argumentos recibidos;
2. detenerse y devolver su código si CS2 falla;
3. ejecutar `TENNIS/run_tennis.ps1` sin reenviarle flags exclusivos de CS2;
4. devolver el código de salida de tenis.

El reentreno conjunto explícito se puede seguir solicitando con una sola orden:

```powershell
& 'C:\dev\CS2-Predictor\start.ps1' -Retrain
```

`-Retrain` llega al pipeline de CS2 y también a `run_tennis.ps1`; por tanto,
CS2 reentrena su modelo. TENNIS actualiza fuentes, reconstruye Elo/features y
entrena o reutiliza por fingerprint sus modelos masculino y femenino en todo
arranque completo, también con `start.ps1` sin argumentos. En TENNIS el flag se
conserva como alias compatible. El contrato del wrapper usa los nombres
canónicos `-Retrain`, `-DryRun` y `-WhatIf`, sin abreviaturas ni formas
booleanas con dos puntos.

Los demás argumentos se reenvían únicamente a CS2. En particular,
`-DryRun`/`-WhatIf` ejecutan el diagnóstico de CS2 y omiten tenis para no
convertir una comprobación en una ejecución real. Tras un flujo ordinario,
tenis regenera `WEB/data.js` al final, de modo que el dashboard conserva CS2 y
añade las predicciones oficiales de tenis.

Tenis también puede ejecutarse de forma independiente:

```powershell
& 'C:\dev\CS2-Predictor\TENNIS\run_tennis.ps1'
& 'C:\dev\CS2-Predictor\TENNIS\run_tennis.ps1' -Retrain
```

Cada arranque operativo completo ejecuta primero `update_sources.py` y después
`scripts/update_tennisratio.py` con el transporte Scrapling estático, para que
el remapeo lateral vea el Sackmann recién actualizado. El resto de Elo,
features y modelo continúa después del refresco TennisRatio. El script decide
con la fecha civil real si la actualización diaria ya se completó, por lo que
no descarga TennisRatio una segunda vez. Si devuelve un fallo real,
`run_tennis.ps1` conserva y propaga exactamente ese código; la política de
conservar un snapshot anterior válido pertenece al actualizador Python.

El transporte usa la sesión estática recomendada porque el HTML público ya
incluye `window.playerData` y `window.matchesData`: pooling/cookies, TLS fija de
Chrome, cabeceras coherentes de navegador, tres intentos de red con espera
determinista, límites de tamaño, marcadores de body, allowlist HTTPS y rechazo
de redirects. HTTP/3 queda explícitamente desactivado: la prueba real con
impersonación produjo timeout de handshake. No se arrancan Chromium/Patchright,
resolución Cloudflare, proxies ni selector adaptativo porque aquí no existe JS,
challenge ni selector cambiante que lo justifique; añadirlos empeoraría tiempo,
dependencias y superficie de fallo sin obtener más datos.

La disponibilidad queda ligada a la descarga real: datos observados el día
`D` no se habilitan para `D` ni se retrofechan, y solo pueden consumirse desde
`D+1`. `-Date` selecciona la jornada de predicción, pero no suplanta el reloj
del actualizador.

El launcher hace la comprobación automáticamente y como máximo una vez al día.
Para independizarla del resto de deportes existe una ruta estrecha que no
reentrena, no predice, no abre la BBDD operativa y no regenera la WEB:

```powershell
& 'C:\dev\CS2-Predictor\TENNIS\run_tennis.ps1' -UpdateOnly
```

La tarea diaria opcional se administra con un script separado; no se registra
como efecto colateral de un arranque normal:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'C:\dev\CS2-Predictor\TENNIS\scripts\manage_tennisratio_daily_task.ps1' -Mode Install
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'C:\dev\CS2-Predictor\TENNIS\scripts\manage_tennisratio_daily_task.ps1' -Mode Install -DailyTime 07:30
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'C:\dev\CS2-Predictor\TENNIS\scripts\manage_tennisratio_daily_task.ps1' -Mode Status
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'C:\dev\CS2-Predictor\TENNIS\scripts\manage_tennisratio_daily_task.ps1' -Mode Remove
```

`Install` registra o reemplaza por nombre
`CS2-Predictor-TennisRatio-Daily`, usa una acción absoluta hacia
`run_tennis.ps1 -UpdateOnly`, hora local 06:00 por defecto,
`StartWhenAvailable`, requisito de red y `IgnoreNew` frente a solapamientos.
Un fallo se reintenta hasta tres veces a intervalos de 30 minutos. Como solo se
publica un batch completo e idempotente, el límite de 12 horas permite que el
primer inventario termine sin convertir un bootstrap largo en un bucle; un
éxito no duplica la descarga ni sus hechos laterales.
No solicita ni guarda credenciales: usa el token interactivo del usuario
actual, por lo que una ejecución perdida se recupera al volver a iniciar sesión
pero la tarea no corre mientras ese usuario no esté conectado. La tarea real
solo existe después de ejecutar explícitamente `-Mode Install`.

Cada invocación real de `TENNIS/run_tennis.ps1`, tanto independiente como
iniciada por el wrapper raíz, abre antes del bootstrap un transcript bajo
`TENNIS/logs/run_tennis_<UTC>_<pid>.log`. El archivo conserva stdout, stderr,
la excepción no controlada cuando exista y una línea final `RUN_END` con estado,
código de salida e instantes UTC. El transcript propio de `CS2/start.ps1`
termina antes de que el wrapper ejecute TENNIS, por lo que los errores de tenis
deben diagnosticarse en este log y no en `CS2/PIPELINE/logs/start_*.log`.

Los transcripts son artefactos operativos locales y `TENNIS/logs/` está
excluido de Git. Un fallo mantiene el código del entrypoint que falló; el
registro no convierte errores en ejecuciones correctas.

Para revertir únicamente la integración habría que restaurar el lanzador raíz
para que llame solo a `CS2/start.ps1`; no es necesario modificar ningún archivo
dentro de `CS2/`.
