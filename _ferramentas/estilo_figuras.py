"""
estilo_figuras.py — Paleta e estilo compartilhados pelas figuras dos PDFs.

Usado pelos scripts `figuras_teoria.py` de cada módulo (em `_fontes/<tema>/
<modulo>/`), que geram os PNGs referenciados pelos `teoria.md` via
`![legenda](figuras/nome.png)`. Mantém a mesma paleta já usada nos
notebooks do curso, para que figura de PDF e figura de notebook pareçam
parte do mesmo material.

Salva sempre em 150 DPI — `md2pdf.py` assume essa resolução ao calcular o
tamanho da figura embutida na página.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AZUL = "#1F5C8B"
VERDE = "#1E6B4F"
VERMELHO = "#9C2B2B"
ROXO = "#5B3E86"
AMBAR = "#8A6100"
CINZA = "#5A5A5A"
CINZA_CLARO = "#D5D9DE"

DPI = 150


def aplica_estilo() -> None:
    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.edgecolor": CINZA,
        "axes.labelcolor": "#1C1C1C",
        "text.color": "#1C1C1C",
        "xtick.color": CINZA,
        "ytick.color": CINZA,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def salva(fig, destino: str | Path) -> None:
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  figura salva: {destino}")


aplica_estilo()
