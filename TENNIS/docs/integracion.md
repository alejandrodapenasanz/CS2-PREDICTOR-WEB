# Integración no invasiva con `start.ps1`

El archivo `start.ps1` de la raíz no forma parte del proyecto de tenis y no se
modifica automáticamente. Para lanzar tenis después de que termine el flujo de
CS:GO, añade tú mismo esta línea exacta al final de `start.ps1`:

```powershell
& (Join-Path $PSScriptRoot 'TENNIS\run_tennis.ps1')
```

`$PSScriptRoot` apunta al directorio donde vive `start.ps1`; por eso la llamada
funciona aunque el lanzador se invoque desde otra ruta. `run_tennis.ps1`
localiza `TENNIS/`, activa `TENNIS/.venv` y ejecuta la predicción de la fecha
local actual. El pipeline reutiliza la caché diaria de Tennis Explorer si ya
existe y publica un CSV timestamped en `TENNIS/data/processed/predictions/`.

Para revertir la integración, elimina únicamente esa misma línea de
`start.ps1`. No es necesario borrar ni modificar ningún archivo de CS:GO.

La llamada integrada hereda la política de la sesión que ya está ejecutando
`start.ps1`. Si se prueba `run_tennis.ps1` directamente y Windows bloquea los
scripts, puede validarse sin cambiar la política permanente:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\TENNIS\run_tennis.ps1
```

La integración no actualiza ni reentrena el histórico automáticamente. Cuando
llegue un commit nuevo de las fuentes, sigue primero el ciclo documentado en el
README: actualizar fuentes, reconstruir Elo, reconstruir features y reentrenar.
