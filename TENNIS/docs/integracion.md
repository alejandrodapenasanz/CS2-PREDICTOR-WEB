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

El reentreno semanal conjunto se solicita con una sola orden:

```powershell
& 'C:\dev\CS2-Predictor\start.ps1' -Retrain
```

`-Retrain` llega al pipeline de CS2 y también a `run_tennis.ps1`; por tanto,
CS2 reentrena su modelo y tenis actualiza fuentes, reconstruye Elo/features y
reentrena los modelos masculino y femenino antes de predecir el día.
Para desactivarlo se omite el flag; el contrato del wrapper usa los nombres
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

Para revertir únicamente la integración habría que restaurar el lanzador raíz
para que llame solo a `CS2/start.ps1`; no es necesario modificar ningún archivo
dentro de `CS2/`.
