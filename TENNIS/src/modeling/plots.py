"""Gráficos reproducibles de fiabilidad probabilística."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from ..config import PROJECT_ROOT


_MATPLOTLIB_CACHE = PROJECT_ROOT / ".cache" / "matplotlib"
_MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CACHE))

import matplotlib  # noqa: E402


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


class PlotError(RuntimeError):
    """Indica que no se pudo construir o guardar una figura."""


def _ensure_project_path(path: Path) -> Path:
    """Restringe la salida gráfica al proyecto ``TENNIS/``."""

    resolved = Path(path).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise PlotError(f"La figura debe quedar dentro de TENNIS/: {resolved}.")
    return resolved


def save_reliability_plot(
    reliability: pd.DataFrame,
    *,
    gender: str,
    output_path: Path,
) -> Path:
    """Guarda curvas raw/Platt para logística y LightGBM.

    Args:
        reliability: Bins producidos por
            :func:`reliability_curve_quantile`.
        gender: ``M`` o ``F``.
        output_path: PNG de destino dentro de ``TENNIS/``.

    Returns:
        Ruta absoluta de la figura guardada.
    """

    if gender not in {"M", "F"}:
        raise ValueError("gender debe ser exactamente 'M' o 'F'.")
    required = {"gender", "model", "mean_predicted", "observed_rate", "count"}
    missing = sorted(required.difference(reliability.columns))
    if missing:
        raise PlotError(f"Faltan columnas de fiabilidad: {missing}.")
    subset = reliability.loc[reliability["gender"] == gender]
    if subset.empty:
        raise PlotError(f"No hay bins de fiabilidad para {gender}.")
    output = _ensure_project_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharex=True, sharey=True)
    specifications = (
        ("Regresión logística", "logistic_raw", "logistic_platt"),
        ("LightGBM", "lightgbm_raw", "lightgbm_platt"),
    )
    for axis, (title, raw_name, calibrated_name) in zip(
        axes, specifications
    ):
        axis.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            linestyle="--",
            color="#555555",
            linewidth=1.2,
            label="Calibración perfecta",
        )
        for name, label, color, marker in (
            (raw_name, "Sin calibrar", "#d95f02", "o"),
            (calibrated_name, "Platt", "#1b9e77", "s"),
        ):
            curve = subset.loc[subset["model"] == name].sort_values("bin")
            if curve.empty:
                raise PlotError(f"Falta la curva {name} para {gender}.")
            axis.plot(
                curve["mean_predicted"],
                curve["observed_rate"],
                color=color,
                marker=marker,
                linewidth=1.8,
                markersize=5,
                label=label,
            )
        axis.set_title(title)
        axis.set_xlabel("Probabilidad media predicha")
        axis.grid(alpha=0.25)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Frecuencia observada de victoria de A")
    axes[1].legend(loc="upper left", fontsize=9)
    fig.suptitle(
        "Curvas de fiabilidad temporales — "
        + ("hombres" if gender == "M" else "mujeres")
    )
    fig.tight_layout()
    try:
        fig.savefig(output, dpi=170, bbox_inches="tight")
    except OSError as exc:
        raise PlotError(f"No se pudo guardar {output}.") from exc
    finally:
        plt.close(fig)
    return output
