"""Gera as figuras do teoria.pdf do módulo 03-feature-engineering."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ / "_ferramentas"))
from estilo_figuras import AZUL, VERDE, VERMELHO, ROXO, AMBAR, CINZA, salva  # noqa: E402

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(11)
MODULO_REL = Path(__file__).resolve().parent.relative_to(RAIZ / "_fontes")
DESTINO = RAIZ / MODULO_REL / "figuras"

# ---------------------------------------------------------------------------
# 1. log1p comprime a assimetria: receita bruta vs. log1p(receita)
# ---------------------------------------------------------------------------
receita = rng.lognormal(mean=8.5, sigma=1.1, size=4000)
receita_log = np.log1p(receita)
skew_bruta = stats.skew(receita)
skew_log = stats.skew(receita_log)
print(f"skew receita bruta:  {skew_bruta:.2f}")
print(f"skew log1p(receita): {skew_log:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
axes[0].hist(receita, bins=60, color=VERMELHO, alpha=0.85)
axes[0].set_title(f"Receita bruta\nassimetria (skewness) = {skew_bruta:.2f}", fontsize=10.5)
axes[0].set_xlabel("receita (R$)"); axes[0].set_ylabel("frequência")
axes[1].hist(receita_log, bins=60, color=AZUL, alpha=0.85)
axes[1].set_title(f"log1p(receita)\nassimetria (skewness) = {skew_log:.2f}", fontsize=10.5)
axes[1].set_xlabel("log(1 + receita)"); axes[1].set_ylabel("frequência")
fig.suptitle("A transformação log1p comprime a cauda longa e aproxima a\n"
             "distribuição de uma normal, reduzindo a assimetria em ~90%",
             fontsize=11.5, fontweight="bold", y=1.03)
salva(fig, DESTINO / "log-transformacao-receita.png")

# ---------------------------------------------------------------------------
# 2. Encoding cíclico: hora do dia como ponto no círculo unitário
# ---------------------------------------------------------------------------
horas = np.arange(24)
sin_h = np.sin(2 * np.pi * horas / 24)
cos_h = np.cos(2 * np.pi * horas / 24)

fig, ax = plt.subplots(figsize=(6.8, 6.8))
tt = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(tt), np.sin(tt), color=CINZA, lw=1, ls=":", alpha=0.6)
ax.scatter(cos_h, sin_h, s=90, color=AZUL, zorder=5)
for h in [0, 6, 12, 18, 23]:
    ax.annotate(f"{h}h", (cos_h[h], sin_h[h]), textcoords="offset points",
                xytext=(10, 6), fontsize=10, fontweight="bold", color=VERMELHO)
ax.plot([cos_h[23], cos_h[0]], [sin_h[23], sin_h[0]], color=VERDE, lw=2.4,
        label="distância real entre 23h e 0h: quase zero")
ax.set_aspect("equal")
ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35)
ax.set_title("Encoding cíclico: 23h e 0h ficam geometricamente\n"
             "vizinhas — como deveriam", fontsize=11.5)
ax.legend(fontsize=8.5, loc="lower center")
ax.set_xlabel(r"$\cos(2\pi \cdot hora/24)$"); ax.set_ylabel(r"$\sin(2\pi \cdot hora/24)$")
salva(fig, DESTINO / "encoding-ciclico-hora.png")

# ---------------------------------------------------------------------------
# 3. RFM: dispersão recência x frequência, cor = monetário, segmentos
# ---------------------------------------------------------------------------
n = 300
recencia = rng.gamma(2.0, 25, n)
frequencia = np.clip(30 - recencia * 0.25 + rng.normal(0, 4, n), 0.3, None)
monetario = np.clip(frequencia * rng.uniform(80, 220, n) + rng.normal(0, 200, n), 50, None)

fig, ax = plt.subplots(figsize=(8.5, 6))
sc = ax.scatter(recencia, frequencia, c=monetario, cmap="viridis", s=32, alpha=0.85)
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("monetário (R$/ano)")
ax.axvline(30, color=CINZA, lw=1, ls="--")
ax.axhline(15, color=CINZA, lw=1, ls="--")
ax.text(6, 32, "campeões\n(R baixo, F alto)", fontsize=9, color=VERDE, fontweight="bold")
ax.text(75, 32, "ex-fiéis em risco\n(R alto, F alto)", fontsize=9, color=AMBAR, fontweight="bold")
ax.text(75, 2, "perdidos\n(R alto, F baixo)", fontsize=9, color=VERMELHO, fontweight="bold")
ax.set_xlabel("recência (dias desde a última compra)")
ax.set_ylabel("frequência (compras / ano)")
ax.set_title("RFM: os três eixos de uma base de clientes, e onde cada\n"
             "segmento clássico aparece no espaço recência × frequência", fontsize=11.5)
salva(fig, DESTINO / "rfm-dispersao.png")

# ---------------------------------------------------------------------------
# 4. Explosão combinatória de PolynomialFeatures(degree=2)
# ---------------------------------------------------------------------------
from math import comb
ps = np.arange(2, 121)
Ns = np.array([comb(p + 2, 2) - 1 for p in ps])

fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.plot(ps, Ns, color=ROXO, lw=2.2)
for p_marca in [20, 50, 100]:
    n_marca = comb(p_marca + 2, 2) - 1
    ax.scatter([p_marca], [n_marca], color=VERMELHO, zorder=5, s=45)
    ax.annotate(f"p={p_marca} → {n_marca:,}".replace(",", ".") + " features",
                (p_marca, n_marca), textcoords="offset points", xytext=(-10, 12),
                fontsize=9, ha="right", color=VERMELHO, fontweight="bold")
ax.set_xlabel("número de features originais (p)")
ax.set_ylabel("features após PolynomialFeatures(degree=2)")
ax.set_title(r"$N = \binom{p+2}{2} - 1$: a explosão combinatória de gerar"
             "\ntodas as interações de grau 2 sem filtro", fontsize=11.5)
salva(fig, DESTINO / "explosao-features-polinomiais.png")

print("done")
