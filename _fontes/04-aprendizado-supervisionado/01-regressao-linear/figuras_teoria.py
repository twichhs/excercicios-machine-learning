"""Gera as figuras do teoria.pdf do módulo 01-regressao-linear.

Rodar a partir da raiz do repositório:
    python3 _fontes/04-aprendizado-supervisionado/01-regressao-linear/figuras_teoria.py
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ / "_ferramentas"))
from estilo_figuras import AZUL, VERDE, VERMELHO, ROXO, AMBAR, CINZA, salva  # noqa: E402

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)
MODULO_REL = Path(__file__).resolve().parent.relative_to(RAIZ / "_fontes")
DESTINO = RAIZ / MODULO_REL / "figuras"

# ---------------------------------------------------------------------------
# 1. Ajuste e resíduos: o que a regressão minimiza, literalmente
# ---------------------------------------------------------------------------
n = 22
x = np.linspace(1, 10, n)
y = 2 + 1.4 * x + rng.normal(0, 1.6, n)
beta1, beta0 = np.polyfit(x, y, 1)
y_hat = beta0 + beta1 * x

fig, ax = plt.subplots(figsize=(7.2, 4.6))
for xi, yi, yhi in zip(x, y, y_hat):
    ax.plot([xi, xi], [yi, yhi], color=VERMELHO, lw=1.1, alpha=0.7, zorder=1)
ax.scatter(x, y, color=AZUL, s=42, zorder=3, label="observações ($y_i$)")
ax.plot(x, y_hat, color="#12314F", lw=2.2, zorder=2, label=r"reta ajustada ($\hat y$)")
ax.scatter([], [], color=VERMELHO, marker="_", s=200, label="resíduo ($y_i-\\hat y_i$)")
idx_destaque = 15
ax.annotate("resíduo", xy=(x[idx_destaque], (y[idx_destaque] + y_hat[idx_destaque]) / 2),
            xytext=(x[idx_destaque] + 0.9, (y[idx_destaque] + y_hat[idx_destaque]) / 2 - 1.5),
            fontsize=9.5, color=VERMELHO,
            arrowprops=dict(arrowstyle="->", color=VERMELHO, lw=1.2))
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("A reta que minimiza a soma dos resíduos ao quadrado")
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
salva(fig, DESTINO / "ajuste-e-residuos.png")

# ---------------------------------------------------------------------------
# 2. Diagnóstico de resíduos: quatro assinaturas visuais
# ---------------------------------------------------------------------------
n = 250
x2 = rng.uniform(-3, 3, n)

casos = {}
casos["saudável\n(sem padrão)"] = rng.normal(0, 1, n)
casos["não-linearidade\n(padrão em U)"] = 1.1 * x2 ** 2 - 2.2 + rng.normal(0, 0.7, n)
casos["heterocedasticidade\n(funil)"] = rng.normal(0, 0.08 + 0.85 * np.abs(x2), n)
_erro_assimetrico = rng.exponential(1.3, n) - 1.3
casos["não-normalidade\n(cauda assimétrica)"] = _erro_assimetrico

fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.4), sharex=True)
for ax, (titulo, residuo) in zip(axes, casos.items()):
    ax.scatter(x2, residuo, s=7, alpha=0.45, color=AZUL)
    ax.axhline(0, color=VERMELHO, lw=1.6)
    ax.set_title(titulo, fontsize=9.5)
    ax.set_xlabel("previsto")
axes[0].set_ylabel("resíduo")
fig.suptitle("Resíduo vs. previsto: cada violação do pressuposto tem uma assinatura visual",
             fontsize=11, fontweight="bold", y=1.04)
salva(fig, DESTINO / "diagnostico-de-residuos.png")

# ---------------------------------------------------------------------------
# 3. Decomposição da variância: SQT = SQE + SQR (Pitágoras nas observações)
# ---------------------------------------------------------------------------
n = 18
x3 = np.linspace(0, 10, n)
y3 = 3 + 1.2 * x3 + rng.normal(0, 2.0, n)
b1, b0 = np.polyfit(x3, y3, 1)
y3_hat = b0 + b1 * x3
y3_bar = y3.mean()

fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=True)
especificacoes = [
    ("SQ Total\n$(y_i - \\bar y)^2$", y3, y3_bar, VERMELHO),
    ("SQ Explicada\n$(\\hat y_i - \\bar y)^2$", y3_hat, y3_bar, VERDE),
    ("SQ Residual\n$(y_i - \\hat y_i)^2$", y3, y3_hat, AMBAR),
]
for ax, (titulo, alvo, referencia, cor) in zip(axes, especificacoes):
    ax.scatter(x3, y3, color=AZUL, s=28, zorder=3, alpha=0.85)
    ax.plot(x3, y3_hat, color="#12314F", lw=1.8, zorder=2)
    ax.axhline(y3_bar, color=CINZA, lw=1.2, ls="--", zorder=1)
    if np.isscalar(referencia):
        for xi, ai in zip(x3, alvo):
            ax.plot([xi, xi], [ai, referencia], color=cor, lw=1.6, alpha=0.8, zorder=1)
    else:
        for xi, ai, bi in zip(x3, alvo, referencia):
            ax.plot([xi, xi], [ai, bi], color=cor, lw=1.6, alpha=0.8, zorder=1)
    ax.set_title(titulo, fontsize=10)
    ax.set_xlabel("x")
axes[0].set_ylabel("y")
fig.suptitle(r"$R^2 = \mathrm{SQE}/\mathrm{SQT}$ — o quanto da dispersão total a reta explica",
             fontsize=11, fontweight="bold", y=1.03)
salva(fig, DESTINO / "decomposicao-variancia.png")

print("done")
