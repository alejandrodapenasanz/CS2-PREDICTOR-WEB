"""Normalización auditada del nivel de torneo para las features de contexto.

El histórico Sackmann conserva ``tourney_level`` como un código opaco. Este
módulo lo acompaña con una categoría amplia útil para modelado, sin perder
nunca el código original. La clasificación histórica está cerrada sobre las
combinaciones ``(source_family, tourney_level)`` observadas en el manifiesto
activo de la fase 2 (commit
``83733587353df8a41f2fd4f516147d5aa83f5a8d``). Las familias diarias
``tennis_explorer`` y ``tennisratio`` conservan por separado sus cuatro
etiquetas literales; nunca las presentan como códigos Sackmann.

No se emplea una regla comodín: si una futura actualización incorpora una
combinación distinta, :func:`normalize_tour_level` lanza un error para que se
audite antes de usarla. Las categorías ``Challenger`` y ``WTA Tour`` son
deliberadamente amplias:

* ``C`` femenino reúne circuitos secundarios históricos y WTA 125 modernos.
* ``W`` femenino representa eventos de tour históricos, aunque el snapshot
  contiene 31 filas de Buenos Aires 125 de 2024 bajo ese código.

Ambas decisiones conservan ``raw_level`` para que el modelo pueda distinguir
el código fuente y para permitir una revisión posterior sin perder
información. Exhibiciones, juniors y Juegos Olímpicos quedan en ``Other``; el
filtro de elegibilidad Elo excluye por separado ``E`` y ``J``.
"""

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias


CanonicalTourLevel: TypeAlias = Literal[
    "Grand Slam",
    "ATP Tour",
    "WTA Tour",
    "Challenger",
    "ITF",
    "Team",
    "Other",
]


class UnknownTourLevelError(ValueError):
    """Indica una combinación de familia y nivel que aún no fue auditada."""


@dataclass(frozen=True, slots=True)
class TourLevelContext:
    """Contiene el nivel original y su categoría contextual conservadora.

    Attributes:
        raw_level: Código ``tourney_level`` exacto de Sackmann.
        canonical_level: Categoría amplia destinada a features.
        source_family: Familia exacta del archivo Sackmann de procedencia.
    """

    raw_level: str
    canonical_level: CanonicalTourLevel
    source_family: str


_GRAND_SLAM: Final[CanonicalTourLevel] = "Grand Slam"
_ATP_TOUR: Final[CanonicalTourLevel] = "ATP Tour"
_WTA_TOUR: Final[CanonicalTourLevel] = "WTA Tour"
_CHALLENGER: Final[CanonicalTourLevel] = "Challenger"
_ITF: Final[CanonicalTourLevel] = "ITF"
_TEAM: Final[CanonicalTourLevel] = "Team"
_OTHER: Final[CanonicalTourLevel] = "Other"


# Inventario cerrado: 54 combinaciones históricas y 8 etiquetas web diarias.
_CANONICAL_BY_SOURCE_AND_RAW: Final[
    dict[tuple[str, str], CanonicalTourLevel]
] = {
    # ATP, cuadro principal.
    ("atp_main", "A"): _ATP_TOUR,
    ("atp_main", "D"): _TEAM,
    ("atp_main", "F"): _ATP_TOUR,
    ("atp_main", "G"): _GRAND_SLAM,
    ("atp_main", "M"): _ATP_TOUR,
    ("atp_main", "O"): _OTHER,
    # ATP, qualifying de tour y cuadros principales Challenger.
    ("atp_qual_chall", "A"): _ATP_TOUR,
    ("atp_qual_chall", "C"): _CHALLENGER,
    ("atp_qual_chall", "G"): _GRAND_SLAM,
    ("atp_qual_chall", "M"): _ATP_TOUR,
    # ATP Futures/ITF.
    ("atp_futures", "S"): _ITF,
    ("atp_futures", "15"): _ITF,
    ("atp_futures", "25"): _ITF,
    # WTA, cuadro principal. CC y los importes +H son circuitos ITF.
    ("wta_main", "35+H"): _ITF,
    ("wta_main", "50+H"): _ITF,
    ("wta_main", "CC"): _ITF,
    ("wta_main", "D"): _TEAM,
    ("wta_main", "E"): _OTHER,
    ("wta_main", "F"): _WTA_TOUR,
    ("wta_main", "G"): _GRAND_SLAM,
    ("wta_main", "I"): _WTA_TOUR,
    ("wta_main", "J"): _OTHER,
    ("wta_main", "O"): _OTHER,
    ("wta_main", "P"): _WTA_TOUR,
    ("wta_main", "PM"): _WTA_TOUR,
    ("wta_main", "T1"): _WTA_TOUR,
    ("wta_main", "T2"): _WTA_TOUR,
    ("wta_main", "T3"): _WTA_TOUR,
    ("wta_main", "T4"): _WTA_TOUR,
    ("wta_main", "T5"): _WTA_TOUR,
    ("wta_main", "W"): _WTA_TOUR,
    # WTA qualifying, WTA 125 y circuitos ITF.
    ("wta_qual_itf", "10"): _ITF,
    ("wta_qual_itf", "100"): _ITF,
    ("wta_qual_itf", "15"): _ITF,
    ("wta_qual_itf", "20"): _ITF,
    ("wta_qual_itf", "25"): _ITF,
    ("wta_qual_itf", "35"): _ITF,
    ("wta_qual_itf", "40"): _ITF,
    ("wta_qual_itf", "50"): _ITF,
    ("wta_qual_itf", "60"): _ITF,
    ("wta_qual_itf", "75"): _ITF,
    ("wta_qual_itf", "80"): _ITF,
    ("wta_qual_itf", "C"): _CHALLENGER,
    ("wta_qual_itf", "E"): _OTHER,
    ("wta_qual_itf", "G"): _GRAND_SLAM,
    ("wta_qual_itf", "I"): _WTA_TOUR,
    ("wta_qual_itf", "P"): _WTA_TOUR,
    ("wta_qual_itf", "PM"): _WTA_TOUR,
    ("wta_qual_itf", "T1"): _WTA_TOUR,
    ("wta_qual_itf", "T2"): _WTA_TOUR,
    ("wta_qual_itf", "T3"): _WTA_TOUR,
    ("wta_qual_itf", "T4"): _WTA_TOUR,
    ("wta_qual_itf", "T5"): _WTA_TOUR,
    ("wta_qual_itf", "W"): _WTA_TOUR,
    # Tennis Explorer diario: etiquetas literales, no códigos Sackmann.
    ("tennis_explorer", "ATP"): _ATP_TOUR,
    ("tennis_explorer", "WTA"): _WTA_TOUR,
    ("tennis_explorer", "Challenger"): _CHALLENGER,
    ("tennis_explorer", "ITF"): _ITF,
    # TennisRatio diario: vocabulario visible igual, procedencia independiente.
    ("tennisratio", "ATP"): _ATP_TOUR,
    ("tennisratio", "WTA"): _WTA_TOUR,
    ("tennisratio", "Challenger"): _CHALLENGER,
    ("tennisratio", "ITF"): _ITF,
}


def _validate_exact_token(value: object, field_name: str) -> str:
    """Valida un identificador fuente sin normalizar ni perder información.

    Args:
        value: Valor que debe ser texto no vacío y sin espacios periféricos.
        field_name: Nombre empleado en los mensajes de validación.

    Returns:
        El mismo texto recibido, sin transformaciones.

    Raises:
        TypeError: Si el valor no es una cadena.
        ValueError: Si está vacío o contiene espacios periféricos.
    """

    if not isinstance(value, str):
        raise TypeError(f"{field_name} debe ser str.")
    if not value:
        raise ValueError(f"{field_name} no puede estar vacío.")
    if value != value.strip():
        raise ValueError(
            f"{field_name} debe conservar el token fuente exacto sin "
            "espacios periféricos."
        )
    return value


def normalize_tour_level(
    raw_level: str,
    source_family: str,
) -> TourLevelContext:
    """Clasifica una combinación Sackmann previamente auditada.

    Args:
        raw_level: Código original ``tourney_level``. Se conserva literalmente.
        source_family: Familia de archivo original añadida durante la ingesta.

    Returns:
        Contexto con código, categoría canónica y familia de procedencia.

    Raises:
        TypeError: Si cualquiera de los argumentos no es texto.
        ValueError: Si un token está vacío o tiene espacios periféricos.
        UnknownTourLevelError: Si la combinación no figura en el inventario
            auditado. Este error intencional evita clasificaciones silenciosas
            cuando cambia la fuente.
    """

    validated_raw = _validate_exact_token(raw_level, "raw_level")
    validated_family = _validate_exact_token(
        source_family,
        "source_family",
    )
    key = (validated_family, validated_raw)
    try:
        canonical = _CANONICAL_BY_SOURCE_AND_RAW[key]
    except KeyError as exc:
        raise UnknownTourLevelError(
            "Combinación de nivel Sackmann no auditada: "
            f"source_family={validated_family!r}, "
            f"raw_level={validated_raw!r}. Debe revisarse antes de construir "
            "features."
        ) from exc
    return TourLevelContext(
        raw_level=validated_raw,
        canonical_level=canonical,
        source_family=validated_family,
    )
