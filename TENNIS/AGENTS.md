# Reglas permanentes del proyecto TENNIS

Estas reglas se aplican en todas las fases y sesiones del proyecto.

## Alcance del repositorio

- Todo el proyecto vive dentro de la carpeta `TENNIS/` situada en la raíz del repositorio.
- Nunca se debe crear, editar ni borrar nada fuera de `TENNIS/`.
- En particular, no se debe modificar `start.ps1` ni ningún código de CS:GO existente en la raíz.

### Excepción explícita y limitada de la fase 9

- El usuario autorizó expresamente que la fase 9 adapte de forma mínima
  `WEB/build_web.py`, `WEB/index.html` y el `WEB/data.js` generado para mostrar
  los partidos de tenis, su ganador previsto, probabilidad y fiabilidad.
- Esta excepción no autoriza ningún rediseño general de la web ni cambios en
  `start.ps1`, `CS2/` o cualquier otro archivo situado fuera de `TENNIS/`.

## Dudas y ambigüedades

- Ante cualquier duda o ambigüedad —incluidos formatos de datos, decisiones de diseño o detalles no especificados— hay que detenerse y preguntar al usuario.
- No se deben inventar nombres de columnas, rutas ni supuestos. Es preferible formular una pregunta que adoptar un supuesto silencioso.

## Documentación

- Todo debe quedar documentado.
- Cada módulo y cada función deben incluir un docstring.
- Cada script debe tener una cabecera que explique qué hace, qué recibe y cómo se ejecuta.
- `README.md` debe mantenerse actualizado en cada fase.

## Implementación

- El lenguaje del proyecto es Python.
- El código debe ser modular y testeable, con tests.
- No se permiten scripts monolíticos.

## Regla de oro anti-fugas

- Para predecir un partido con fecha `D`, solo se puede utilizar información anterior a `D`.
- Esta restricción debe respetarse desde el diseño y durante todas las fases posteriores.
