# Informe de ejecución de la fase 4

## Ejecución auditada

La captura real corresponde al **2026-07-30** y procede de:

```text
https://www.tennisexplorer.com/matches/?day=30&month=07&type=all&year=2026
```

Se adquirió una sola vez a las `2026-07-30T09:48:28Z`. El HTML ocupa
`928679` bytes y su SHA-256 es:

```text
2910901e884a468c0f0ae3ef55d0032bb44748bc6026752c878f63075f9758ec
```

La ejecución final de `scripts/scrape_tennis_explorer.py` reutilizó esa caché
validada y, por tanto, generó **cero solicitudes HTTP**.

## Cobertura por género y nivel

| Género | `tour_level` | Partidos |
|---|---|---:|
| `F` | `ITF` | 122 |
| `F` | `WTA` | 21 |
| `M` | `ATP` | 13 |
| `M` | `Challenger` | 24 |
| `M` | `ITF` | 133 |
| **Total** |  | **313** |

La página completa también contenía dobles y 60 partidos UTR. Quedaron fuera
por los marcadores estructurales observados, no por deduplicación posterior.

## Estado, enlaces y cuotas

| `status` | Partidos |
|---|---:|
| `finished` | 76 |
| `scheduled` | 228 |
| `unknown` | 9 |

La captura no contenía etiquetas explícitas de `walkover`, `cancelled` o
`in_progress`. Los nueve casos que no podían distinguirse sin inferencias se
conservaron como `unknown`.

Los 313 partidos tenían enlaces de ficha para ambos jugadores. Cinco filas
carecían de la cuota de cada lado; 308 contenían ambas cuotas. Los tests cubren
también el contrato permitido para un jugador sin enlace y para cuotas vacías.

## Superficie recuperada en fase 8

Una auditoría posterior del mismo HTML localizó la superficie explícita en el
catálogo semanal incluido en la página. El parser la une por
`tournament_href`, sin nuevas solicitudes:

| Segmento | Partidos con superficie | Total |
|---|---:|---:|
| ATP masculino | 13 | 13 |
| Challenger masculino | 24 | 24 |
| ITF masculino | 0 | 133 |
| WTA femenino | 21 | 21 |
| ITF femenino | 122 | 122 |
| **Total** | **180** | **313** |

Los únicos valores observados fueron `Hard` y `Clay`. Los Futures masculinos
siguen nulos porque la cabecera agregada no publica enlace ni superficie.

## Limitaciones observadas

- Los Futures masculinos están agregados como `Futures 2026`; Tennis Explorer
  no publica una sede, un enlace de torneo ni una superficie enlazable.
- `Targu Mures` no contiene marcador ITF. Se clasifica como WTA mediante la
  convención confirmada para `wta-women` sin marcador ITF.
- El HTML diario no permite separar con seguridad un walkover, una retirada,
  un partido en juego y algunos resultados parciales si no aparece texto de
  estado explícito.
- El texto anidado `Live streams` no es un estado deportivo y se ignora.
- No se consulta el widget externo de EnetScores ni se realizan peticiones por
  jugador o partido.

## Acceso de bajo ruido

El cliente actual usa `scrapling[fetchers]==0.4.12` exclusivamente como
transporte HTTP estático, con cabeceras fijas, impersonación y funciones
stealth desactivadas, una sola tentativa y cero redirecciones. Bajo la
autorización académica/no comercial reportada, la excepción de `robots.txt`
queda limitada por código a la URL diaria canónica `type=all`; por ello una
fecha nueva genera un GET y una fecha cacheada, cero.

Un `403`, `429` o desafío inequívoco abre
`data/raw/tennis_explorer/policy/circuit_breaker.json`. El archivo no caduca:
las fechas no cacheadas quedan bloqueadas antes de crear el transporte hasta
que el operador revise la causa y lo retire manualmente. El proyecto no usa
proxies, rotación, navegadores ni resolución de desafíos.

## Validación

La suite completa se ejecutó desde `TENNIS/`:

```powershell
python -m unittest discover -s tests -v
```

Resultado tras integrar el transporte estático de Scrapling: **86 tests
correctos**. De ellos, 32 corresponden directamente a Tennis Explorer y son
enteramente offline: 14 validan el parser contra el snapshot real, 13 validan
el cliente, la excepción limitada de `robots.txt`, caché, integridad, lock,
pausa, cortacircuitos y parada ante bloqueos, y 5 validan el adaptador
Scrapling sin sigilo ni reintentos. Otros 7 tests comprueban el detector de
releases estables de Scrapling sin actualizar el proyecto.
