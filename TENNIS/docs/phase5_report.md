# Informe de ejecución de la fase 5

## Ejecución auditada

La validación real se ejecutó sobre la cartelera cacheada de Tennis Explorer
del `2026-07-30`. El snapshot contiene 313 partidos de singles, fue adquirido
el `2026-07-30T09:48:28Z` y tiene SHA-256
`2910901e884a468c0f0ae3ef55d0032bb44748bc6026752c878f63075f9758ec`.
La ejecución reutilizó ese HTML y emitió cero peticiones de red.

`data/overrides.csv` estaba vacío, por lo que todos los IDs de esta ejecución
proceden de coincidencias automáticas exactas. Se persistieron 569 claves
únicas `(gender, slug)`: 255 femeninas y 314 masculinas.

## Cobertura por género y nivel

La cobertura de participantes cuenta ambos lados del partido. La cobertura de
partidos exige que los dos participantes estén resueltos.

| Género | Nivel | Partidos | Participantes | Automáticos | Automáticos (%) | Partidos mapeados | Partidos mapeados (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| F | ITF | 122 | 244 | 224 | 91,80 | 102 | 83,61 |
| F | WTA | 21 | 42 | 38 | 90,48 | 17 | 80,95 |
| M | ATP | 13 | 26 | 23 | 88,46 | 10 | 76,92 |
| M | Challenger | 24 | 48 | 48 | 100,00 | 24 | 100,00 |
| M | ITF | 133 | 266 | 247 | 92,86 | 115 | 86,47 |
| **Total** | **Todos** | **313** | **626** | **580** | **92,65** | **268** | **85,62** |

No hubo overrides manuales, por lo que los 268 partidos mapeados fueron también
partidos completamente automáticos. Los 313 partidos se conservaron: los 45
que carecían de uno o ambos IDs tienen `mapping_status="unmapped"`.

Ejemplos observados en la salida:

| Tennis Explorer | Slug | Género | `player_id` | Nombre Sackmann | Método |
|---|---|---|---:|---|---|
| `Michelsen A.` | `michelsen-a98bb` | M | 210506 | Alex Michelsen | `automatic_name` |
| `Mannarino A.` | `mannarino-a7108` | M | 105173 | Adrian Mannarino | `automatic_name` |
| `Jodar R.` | `jodar` | M | 212588 | Rafael Jodar | `automatic_name` |
| `Nishikori K.` | `nishikori` | M | 105453 | Kei Nishikori | `automatic_name` |
| `Cerundolo F.` | `cerundolo` | M | 202103 | Francisco Cerundolo | `automatic_name` |

## No resueltos

Quedaron 46 apariciones de participante sin ID, agrupadas en 45 identidades
pendientes. `Jones E.` con slug `jones-cff84` aparece dos veces en la misma
cartelera; la cola la conserva como una única identidad y una única fecha
observada.

Por diagnóstico:

| Género | Motivo | Identidades |
|---|---|---:|
| F | `ambiguous_active_candidates` | 17 |
| F | `no_active_candidate` | 6 |
| M | `ambiguous_active_candidates` | 9 |
| M | `no_active_candidate` | 13 |
| **Total** | **Todos** | **45** |

La lista completa es:

| Género | Nivel | Nombre | Slug | Motivo | Candidatos |
|---|---|---|---|---|---|
| F | ITF | Alame R. | alame-efff1 | ambiguous_active_candidates | 266383\|266384\|270281 |
| F | ITF | Black B. | black-a5c31 | ambiguous_active_candidates | 259607\|265657\|266682 |
| F | ITF | Garcia Cid M. | garcia-cid | ambiguous_active_candidates | 225850\|259776 |
| F | ITF | Kim E. | kim-ac772 | ambiguous_active_candidates | 222190\|232885\|259772\|260093\|260123\|270290 |
| F | ITF | Lee E. | lee-d4739 | ambiguous_active_candidates | 215501\|221269\|222546\|260127\|270183 |
| F | ITF | Lee H. | lee-534c9 | ambiguous_active_candidates | 269747\|270112\|270449 |
| F | ITF | Marron Baruqui M. | marron-baruqui | no_active_candidate | — |
| F | ITF | Menendez Zozaya E. | menendez-zozaya | no_active_candidate | — |
| F | ITF | Nguyen A. | nguyen-cfb33 | ambiguous_active_candidates | 260787\|269715 |
| F | ITF | Park S. | park-aa1d7 | ambiguous_active_candidates | 216213\|220667\|221268 |
| F | ITF | Petkovic A. | petkovic-1cadf | ambiguous_active_candidates | 252571\|253668\|267525 |
| F | ITF | Saavedra Carrera Z. | saavedra-carrera-d45af | no_active_candidate | — |
| F | ITF | Sato H. | sato-92070 | ambiguous_active_candidates | 216156\|221141 |
| F | ITF | Stevens A. | stevens-ea018 | ambiguous_active_candidates | 264278\|265599 |
| F | ITF | Symons O. | symons-a0066 | no_active_candidate | — |
| F | ITF | Taylor L. | taylor-9bf7d | ambiguous_active_candidates | 259576\|266485 |
| F | ITF | Tikhonova A. | tikhonova | ambiguous_active_candidates | 221073\|270431 |
| F | ITF | Uchijima M. | uchijima-39346 | ambiguous_active_candidates | 220416\|264134 |
| F | ITF | Werner C. | werner-0c6a8 | no_active_candidate | — |
| F | ITF | Yang Y. | yang-7c89b | ambiguous_active_candidates | 215820\|222390\|269728 |
| F | WTA | Jones E. | jones-cff84 | ambiguous_active_candidates | 200992\|263644 |
| F | WTA | Sobolieva A. | sobolieva | no_active_candidate | — |
| F | WTA | Uchijima M. | uchijima | ambiguous_active_candidates | 220416\|264134 |
| M | ATP | Boyer T. | boyer-64ede | ambiguous_active_candidates | 207729\|208142 |
| M | ATP | Nakashima B. | nakashima-68876 | ambiguous_active_candidates | 206909\|210416 |
| M | ATP | Wong C. | wong-d3ead | ambiguous_active_candidates | 124022\|208597\|209409 |
| M | ITF | Borisov D. | borisov-5c139 | no_active_candidate | — |
| M | ITF | Chang A. | chang-ddf68 | ambiguous_active_candidates | 144081\|212959 |
| M | ITF | Da Costa R. | da-costa-6dc81 | no_active_candidate | — |
| M | ITF | Delaney J. | delaney | ambiguous_active_candidates | 117359\|202334 |
| M | ITF | Dellien M. | dellien-e101c | no_active_candidate | — |
| M | ITF | Huszar A. | huszar | no_active_candidate | — |
| M | ITF | Juszczak P. | juszczak-e14d8 | no_active_candidate | — |
| M | ITF | Lima E. | lima-eba65 | ambiguous_active_candidates | 211478\|212457 |
| M | ITF | Marek W. | marek-eb437 | no_active_candidate | — |
| M | ITF | McFadzean L. | mcfadzean | ambiguous_active_candidates | 213796\|214043 |
| M | ITF | Nakashima B. | nakashima-baff8 | ambiguous_active_candidates | 206909\|210416 |
| M | ITF | Nirundorn T. | nirundorn-9d4b6 | no_active_candidate | — |
| M | ITF | Petre S. | petre | no_active_candidate | — |
| M | ITF | Piening C. | piening | no_active_candidate | — |
| M | ITF | Rovai S. | rovai-0eea1 | no_active_candidate | — |
| M | ITF | Salazar D. | salazar-a04be | no_active_candidate | — |
| M | ITF | Shin S. | shin-72677 | no_active_candidate | — |
| M | ITF | Sik M. | sik | no_active_candidate | — |
| M | ITF | Tsitsipas P. | tsitsipas-557e1 | ambiguous_active_candidates | 202065\|210409 |

Estas filas no se deben corregir por intuición. Las colisiones exigen verificar
el perfil y añadir un override motivado; las ausencias pueden corresponder a
jugadores nuevos o a falta de actividad en el snapshot histórico.

## Advertencia de frescura

La cartelera es del 30 de julio, pero los CSV parciales de 2026 usados para
determinar actividad llegan solamente hasta:

| Familia Sackmann | Última `tourney_date` |
|---|---|
| ATP principal | 2026-05-25 |
| ATP qualifying/Challenger | 2026-06-01 |
| ATP Futures/ITF | 2026-06-01 |
| WTA principal | 2026-05-25 |
| WTA qualifying/ITF | 2026-06-02 |

Este desfase se expone y no se tapa. En particular, parte de los 19 casos
`no_active_candidate` puede deberse a jugadores cuya primera actividad ocurrió
después del último dato Sackmann disponible. Deben revisarse contra una fuente
actualizada antes de crear overrides.

## Idempotencia y validación

La misma fecha se ejecutó dos veces. Después de la segunda ejecución:

- `player_mappings` seguía teniendo 569 filas;
- `mapping_audit` seguía teniendo 569 eventos;
- el SHA-256 de `unresolved_players.csv` seguía siendo
  `4b66f28b0d7ea3cb4a08b421583e377ecbf636e304ec212f09dced31e4dc8d48`.

Por tanto, una fecha repetida no recalculó slugs, no duplicó auditoría y no
incrementó la cola. La suite completa de tests se ejecuta con:

```powershell
python -m unittest discover -s tests -v
```

Resultado definitivo: **127 tests ejecutados, 127 correctos**.
