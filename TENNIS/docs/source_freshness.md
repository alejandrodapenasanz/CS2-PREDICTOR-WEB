# Actualización y vigencia de las fuentes

## Objetivo

El histórico descargado en una fecha concreta no basta para producir
predicciones diarias indefinidamente. La arquitectura separa tres fuentes con
funciones distintas y conserva la revisión o el instante de adquisición de cada
una. Ninguna fuente auxiliar se concatena silenciosamente con el histórico
canónico.

## Histórico Sackmann archivado

El mirror
[`Aneeshers/tennis-sackmann-archive`](https://github.com/Aneeshers/tennis-sackmann-archive)
es la fuente canónica de resultados de la fase 2. En cada ejecución se consulta
la revisión actual de `main`:

- si el commit no cambió, la ejecución es un *no-op*;
- si aparecen archivos o cambian blobs, solo se descargan los objetos afectados;
- los bytes se validan por tamaño y SHA de blob Git;
- un archivo local alterado fuera del proceso no se sobrescribe sin
  `--force`;
- se conserva un manifiesto por commit además del manifiesto activo.

Este mecanismo detecta publicaciones nuevas, pero no puede garantizar que el
mantenedor del mirror vuelva a actualizarlo. El snapshot inicial del proyecto
es del 25 de junio de 2026 y contiene datos parciales de 2026.

## Match Charting Project

[`JeffSackmann/tennis_MatchChartingProject`](https://github.com/JeffSackmann/tennis_MatchChartingProject)
es una fuente auxiliar *crowdsourced* con metadatos, puntos y estadísticas
detalladas de una muestra selectiva de partidos. No sustituye a los históricos
ATP/WTA:

- no contiene IDs Sackmann;
- `Player 1` es quien sirvió primero, no el ganador;
- no ofrece una cobertura exhaustiva ni una cadencia diaria garantizada;
- sus filas no se añaden al universo canónico de partidos.

La fase 2 descarga sus CSV en un espacio de nombres independiente y fija cada
ejecución a un commit. La integración de estadísticas requerirá el mapeo de
jugadores y partidos de la fase 5. Una coincidencia deberá ser unívoca y
auditable; las coincidencias ausentes o ambiguas no se resolverán por
aproximación silenciosa.

La auditoría del commit
`2c59eef194967e688b69e73df344184a06322cd8` encontró excepciones en los dos
CSV de metadatos:

- `charting-m-matches.csv`: 7.566 filas, 7.565 `match_id` únicos y 2 filas con
  13 campos frente a los 15 de la cabecera;
- `charting-w-matches.csv`: 4.080 filas, 4.073 `match_id` únicos y 9 filas con
  11 o 13 campos frente a los 15 esperados;
- hay un `match_id` duplicado masculino y siete femeninos; las fechas
  incorporadas en los propios identificadores siguen siendo válidas;
- los otros 38 CSV descargados no presentan diferencias entre el número de
  campos de sus filas y su cabecera.

No se han rellenado, unido ni eliminado esas filas. Los bytes quedan disponibles
para una decisión explícita durante el mapeo; hasta entonces los metadatos MCP
no tienen loader canónico.

Los datos del propio partido solo podrán contribuir a predicciones de partidos
posteriores. Además de la fecha deportiva, se conservará la fecha en que el dato
estuvo disponible para el proyecto.

## Tennis Abstract

La actualización web aprobada se limita a estos dos informes públicos:

- <https://www.tennisabstract.com/reports/atp_elo_ratings.html>
- <https://www.tennisabstract.com/reports/wta_elo_ratings.html>

Política de acceso:

- una ejecución ordinaria como máximo cada 24 horas;
- exactamente un `GET` secuencial por género, dos solicitudes totales;
- peticiones condicionales mediante `If-Modified-Since` y `If-None-Match`;
- agente de usuario identificable;
- sin reintentos durante esa ejecución ante `403`, `429` o errores `5xx`;
- conservación del HTML crudo, URL, hash, cabeceras HTTP e instante UTC;
- nunca solicitar `/jsfrags/`, `/jsmatches/` ni `/jsplayers/`, que están
  excluidos por <https://www.tennisabstract.com/robots.txt>.

Los informes se actualizan aproximadamente cada semana. Son una referencia viva
y un benchmark externo, no una fuente completa de resultados diarios. Sus
ratings no reemplazan el Elo propio de la fase 3 y no se aplican
retroactivamente a fechas anteriores a su adquisición.

El usuario del proyecto informa de autorización expresa de Tennis Abstract
para este uso académico y no comercial, condicionada a mantener una cadencia
baja que no perturbe el servicio. Esta declaración se registra como procedencia
del permiso y no como verificación independiente. Los snapshots son locales y
no se redistribuyen. Cualquier ampliación de alcance o uso comercial exigiría
una nueva autorización del propietario mediante su
[página oficial de contacto](https://www.tennisabstract.com/blog/tennis-abstract-contact/).

## Regla temporal común

Toda observación externa debe conservar al menos:

- `source`;
- revisión Git o URL;
- `retrieved_at_utc`;
- hash del contenido;
- ruta cruda de procedencia.

Para predecir en el instante `T`, solo puede emplearse una observación con
`retrieved_at_utc < T`. La fecha del partido no demuestra por sí sola cuándo
estuvieron disponibles sus datos. La ejecución automática y el horario concreto
se conectarán al pipeline diario en la fase 8.
