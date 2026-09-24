# Multi-Sport Predictor

## Arranque portable (VAULT)

Código de esta versión + carpeta privada `VAULT/` en la raíz + `./start.ps1`.
Los launchers localizan automáticamente datos/modelos y preparan sus entornos
Python 3.13. No hay que editar rutas ni copiar archivos entre componentes.
VAULT contiene credenciales: no se publica en Git y se copia con los procesos
detenidos. Contrato, verificación y contexto técnico integral:
[DOCS/contexto.md](DOCS/contexto.md).

Repositorio preparado para alojar predictores de distintos deportes sin
mezclar datos, modelos ni pipelines.

## Estructura

- `CS2/`: dominio completo de Counter-Strike 2.
- `TENNIS/`: dominio completo de tenis masculino y femenino.
- `TELEGRAM/`: publicación de picks de CS2 y reporte diario idempotente.
- `WEB/`: dashboard compartido entre deportes.
- `VAULT/`: estado privado portable; excluido de Git.
- `DOCS/contexto.md`: referencia técnica transversal y contrato VAULT.
- `.github/`: automatizacion comun del repositorio.
- `pyproject.toml`: agregador de tests y calidad para todos los dominios.

La documentacion tecnica de CS2 vive en `CS2/README.md` y `CS2/PROJECT.md`.

> **¿Retomas el proyecto (IA o humano)? Empieza por
> [`DOCS/contexto.md`](DOCS/contexto.md)**: arquitectura, datos, modelos,
> anti-fugas, operación, VAULT y limitaciones verificadas de todos los componentes.
> Lee también `AGENTS.md` y las reglas del componente antes de modificar código.
> Los informes históricos y detalles específicos siguen en `CS2/DOCS/` y
> `TENNIS/docs/`.

El comando habitual sigue siendo compatible desde la raiz:

```powershell
.\start.ps1
.\start.ps1 -Retrain
```

Tambien puede ejecutarse directamente:

```powershell
.\CS2\start.ps1
```
