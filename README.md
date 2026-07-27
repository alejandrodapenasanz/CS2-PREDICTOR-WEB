# Multi-Sport Predictor

Repositorio preparado para alojar predictores de distintos deportes sin
mezclar datos, modelos ni pipelines.

## Estructura

- `CS2/`: dominio completo de Counter-Strike 2.
- `WEB/`: dashboard compartido entre deportes.
- `.github/`: automatizacion comun del repositorio.
- `pyproject.toml`: agregador de tests y calidad para todos los dominios.

La documentacion tecnica de CS2 vive en `CS2/README.md` y `CS2/PROJECT.md`.

El comando habitual sigue siendo compatible desde la raiz:

```powershell
.\start.ps1
.\start.ps1 -Retrain
```

Tambien puede ejecutarse directamente:

```powershell
.\CS2\start.ps1
```
