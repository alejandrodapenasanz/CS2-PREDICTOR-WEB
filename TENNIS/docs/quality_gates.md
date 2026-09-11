# Puertas locales de calidad

## Contrato ejecutable

El componente exige CPython 3.13. Desde `TENNIS/`, la puerta única es:

```powershell
.\.venv\Scripts\python.exe scripts\run_quality_gates.py
```

El runner falla en la primera puerta roja y ejecuta, en este orden:

1. `ruff check src scripts tests`;
2. el ratchet de `ruff format --check`;
3. Mypy sobre el primer conjunto de módulos críticos tipados;
4. la suite completa mediante `pytest tests` con temporales derivados bajo
   `.cache/pytest/`;
5. cobertura estática de imports frente a `requirements.txt`;
6. imports de runtime y doble boot determinista (`PYTHONHASHSEED=42`) de cada
   entrypoint real invocado por `run_tennis.ps1`.

Ningún smoke ejecuta ingesta, red, predicciones ni escrituras SQLite: cada
entrypoint arranca únicamente con `--help`.

## Adopción gradual y deuda conocida

La configuración inicial es deliberadamente razonable, no estricta:

- Ruff bloquea errores `E4`, `E7`, `E9` y `F`; `F401` se pospone porque hay
  reexports públicos y tests con imports de registro.
- `scripts/build_elo.py:F841` queda registrado: valida la cuarentena de
  identidades pero no consume la variable. Corregirlo requiere una decisión
  funcional y reconstrucción causal, no un autofix de estilo.
- El formato protege todos los archivos nuevos y ya conformes. Los archivos
  heredados pendientes están enumerados uno a uno en
  `quality/ruff_format_baseline.txt`. La puerta falla si aparece deuda nueva o
  si una entrada ya formateada no se elimina de la lista.
- Mypy comienza por diez módulos de contratos, configuración, temporalidad,
  esquemas y splits. La auditoría inicial del árbol completo detectó 78 errores
  aproximados, concentrados en conversiones desde `object`/pandas, casts
  redundantes, Literals, anotaciones de arrays y payloads de artefactos. Se
  ampliará `files` únicamente cuando cada módulo quede limpio.
- El siguiente endurecimiento de Ruff pendiente es activar `I` (orden de
  imports): la auditoría inicial encontró 157 bloques heredados.

La deuda puede reducirse, nunca crecer silenciosamente. Los cambios de formato
en módulos incluidos en fingerprints de Elo/features/modelos deben coordinarse
con la reconstrucción sancionada de esos artefactos.

## Cobertura de imports

El chequeo se puede ejecutar aislado:

```powershell
.\.venv\Scripts\python.exe scripts\check_import_coverage.py
```

Analiza el AST de `src/` y `scripts/`, ignora stdlib y módulos locales, resuelve
alias como `bs4 -> beautifulsoup4` y `sklearn -> scikit-learn`, y falla si una
raíz externa no está declarada directamente. El test negativo crea una copia
temporal del manifiesto, elimina `pandas`, comprueba el fallo y restaura la
copia; nunca modifica el manifiesto real.
