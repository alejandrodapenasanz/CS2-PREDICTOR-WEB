"""Orientación A/B estable y reproducible para resultados históricos.

La decisión no usa un generador pseudoaleatorio secuencial. Cada partido se
orienta independientemente mediante SHA-256, por lo que reordenar filas o
añadir partidos futuros no modifica ninguna orientación ya calculada.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Final


DEFAULT_ORIENTATION_SEED: Final[int] = 42
_SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")


class OrientationError(ValueError):
    """Indica que la identidad o los participantes no permiten orientar."""


@dataclass(frozen=True)
class Orientation:
    """Resultado auditable de asignar ganador y perdedor a los lados A/B.

    ``swapped`` se refiere al orden original ganador/perdedor. Si es falso,
    A es el ganador y ``y=1``; si es verdadero, A es el perdedor y ``y=0``.
    Por definición, la etiqueta coincide con que A sea el ganador original.
    La propiedad estadística que debe aproximarse al 50 % es la asignación del
    ganador al lado A, no una supuesta independencia entre ``y`` y ese hecho.
    """

    player_a_id: int
    player_b_id: int
    y: int
    swapped: bool
    orientation_hash: str

    def __post_init__(self) -> None:
        """Protege la coherencia interna de instancias construidas manualmente."""

        for field_name, player_id in (
            ("player_a_id", self.player_a_id),
            ("player_b_id", self.player_b_id),
        ):
            _validate_player_id(player_id, field_name=field_name)
        if self.player_a_id == self.player_b_id:
            raise OrientationError("Los lados A y B deben ser jugadores distintos.")
        if isinstance(self.y, bool) or self.y not in {0, 1}:
            raise OrientationError("y debe ser el entero 0 o 1.")
        if not isinstance(self.swapped, bool):
            raise OrientationError("swapped debe ser booleano.")
        expected_y = 0 if self.swapped else 1
        if self.y != expected_y:
            raise OrientationError(
                "y no es coherente con swapped: y=1 solo cuando A es el ganador."
            )
        _validate_source_hash(
            self.orientation_hash,
            field_name="orientation_hash",
        )


def _validate_player_id(player_id: int, *, field_name: str) -> None:
    """Exige un identificador entero no negativo y excluye booleanos."""

    if (
        isinstance(player_id, bool)
        or not isinstance(player_id, int)
        or player_id < 0
    ):
        raise OrientationError(
            f"{field_name} debe ser un entero no negativo."
        )


def _validate_source_hash(value: str, *, field_name: str) -> None:
    """Exige la representación hexadecimal minúscula completa de SHA-256."""

    if (
        not isinstance(value, str)
        or _SHA256_PATTERN.fullmatch(value) is None
    ):
        raise OrientationError(
            f"{field_name} debe ser un SHA-256 hexadecimal minúsculo de 64 caracteres."
        )


def _orientation_digest(
    *,
    seed: int,
    gender: str,
    source_record_hash: str,
) -> str:
    """Serializa la identidad canónicamente y devuelve su SHA-256."""

    payload = json.dumps(
        (seed, gender, source_record_hash),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def orient_match(
    winner_id: int,
    loser_id: int,
    *,
    gender: str,
    source_record_hash: str,
    seed: int = DEFAULT_ORIENTATION_SEED,
) -> Orientation:
    """Asigna los participantes a A/B de forma estable y genera la etiqueta.

    La clave de decisión es el SHA-256 de la serialización JSON compacta de
    ``(seed, gender, source_record_hash)``. El bit menos significativo decide
    si se invierte el orden original. La función no depende del orden del
    DataFrame ni del número de partidos procesados.
    """

    _validate_player_id(winner_id, field_name="winner_id")
    _validate_player_id(loser_id, field_name="loser_id")
    if winner_id == loser_id:
        raise OrientationError("winner_id y loser_id deben ser distintos.")
    if gender not in {"M", "F"}:
        raise OrientationError("gender debe ser exactamente 'M' o 'F'.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise OrientationError("seed debe ser un entero.")
    _validate_source_hash(
        source_record_hash,
        field_name="source_record_hash",
    )

    orientation_hash = _orientation_digest(
        seed=seed,
        gender=gender,
        source_record_hash=source_record_hash,
    )
    swapped = bool(int(orientation_hash[-1], 16) & 1)
    if swapped:
        return Orientation(
            player_a_id=loser_id,
            player_b_id=winner_id,
            y=0,
            swapped=True,
            orientation_hash=orientation_hash,
        )
    return Orientation(
        player_a_id=winner_id,
        player_b_id=loser_id,
        y=1,
        swapped=False,
        orientation_hash=orientation_hash,
    )
