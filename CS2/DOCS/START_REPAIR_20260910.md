# Reparación de arranque CS2 — 2026-09-10

El log `PIPELINE/logs/start_20260910_084907.log` termina en `check_retrain.py`:
`PermissionError` al hacer `Path.is_symlink()` sobre el `metadata.json` de
`20260824_125852Z`. El `try` anterior solo cubría la lectura del JSON, no los
`stat/lstat` anteriores. No era un fallo del scraper ni un rechazo del modelo.

La política ahora captura `OSError` alrededor de la inspección completa de cada
entrada, omite solo la entrada ilegible y continúa. El JSON añade
`registry_warnings` cuando procede; `CS2/start.ps1` lo muestra como WARN. Un
residuo no cambia el cutoff ni autoriza una promoción. No se cambian ACL,
propietarios, modelos, punteros ni datos históricos.

En la verificación actual aparecen dos entradas inaccesibles:
`20260824_160514Z` y `20260828_064216Z`. Se avisa de ambas y se conserva el corte
legible 2026-08-23. La decisión real devuelve código 0, 616 nuevas etiquetas,
umbral 100 y `should_train=true`. El próximo arranque puede por tanto reentrenar;
ese entrenamiento seguirá sujeto a la puerta habitual.

También había `InconsistentVersionWarning`: el lock instalaba scikit-learn 1.9.0
para artefactos serializados con 1.8.0. Se fija **1.8.0 en requirements.txt**, se
regenera el lock con hashes bajo CPython 3.13 y se reaprovisiona el entorno por
`Ensure-ModelPython`, la misma vía de arranque. Se conserva un entorno anterior;
no se reserializa producción ni se ocultan advertencias. Véase el
[contrato de persistencia de scikit-learn](https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations).

Verificación realizada después de reparar el entorno:

- Python 3.13.15, scikit-learn 1.8.0, `pip check` y lock exacto: OK.
- Vivo `20260810_064827Z` y `last_good` `20260803_064101Z`: cargan sin avisos de
  versión y producen probabilidades finitas, reproducibles a tolerancia 1e-12.
- `Invoke-Native` + `MODEL/check_retrain.py`, la llamada que fallaba: código 0.
- Ruff lint/formato, mypy, cobertura de imports y boots CLI: OK.
- 350 tests de la puerta y 33 adicionales: **383 aprobados**, 4 omitidos por
  condiciones existentes y 17 subtests. Smoke determinista: OK, 160 filas.
- Nuevas regresiones simulan permisos denegados en directorio, metadatos,
  `stat`, lectura y modelo; los punteros de fixture permanecen intactos.
- Ledger: 1.187 filas, SHA-256
  `857b68a92f94f14542978cd376bc596885863c2d5b8bf6c6ad0c1c61390632ba`,
  igual al comenzar y terminar esta reparación; sin errores de claves foráneas.
- No se volvió a ejecutar el scraping, la ingesta real ni la publicación web.
  Los temporales de herramientas/pruebas son prescindibles y se eliminan.

Para repetir el arranque completo desde cualquier directorio:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\dev\CS2-Predictor\start.ps1
```

`ExecutionPolicy` se aplica únicamente al proceso invocado; no cambia la política
permanente de Windows. La reparación de CS2 también se aplica al launcher
`CS2/start.ps1`, al que delega el de la raíz.

Revisión: cambio en MODEL para la política, launcher solo para mostrar avisos;
sin cambios de features ni datos sagrados. La única modificación de dependencias
es el pin compatible y su lock regenerado. No se ha forzado una promoción.
