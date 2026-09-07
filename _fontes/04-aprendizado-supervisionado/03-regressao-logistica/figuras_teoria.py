"""Gera as figuras do teoria.pdf do módulo 03-regressao-logistica."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ / "_ferramentas"))
from estilo_figuras import AZUL, VERDE, VERMELHO, ROXO, AMBAR, CINZA, salva  # noqa: E402

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression

rng = np.random.default_rng(3)
MODULO_REL = Path(__file__).resolve().parent.relative_to(RAIZ / "_fontes")
DESTINO = RAIZ / MODULO_REL / "figuras"

# ---------------------------------------------------------------------------
# 1. Por que uma reta falha para prever probabilidade
# ---------------------------------------------------------------------------
n = 120
x = np.concatenate([rng.normal(2, 1.3, n // 2), rng.normal(7, 1.3, n // 2)])
logit_real = 1.4 * (x - 4.5)
prob_real = 1 / (1 + np.exp(-logit_real))
y = (rng.random(n) < prob_real).astype(int)

modelo_linear = LinearRegression().fit(x.reshape(-1, 1), y)
modelo_log = LogisticRegression().fit(x.reshape(-1, 1), y)

xs = np.linspace(x.min() - 1, x.max() + 1, 300)
pred_linear = modelo_linear.predict(xs.reshape(-1, 1))
pred_log = modelo_log.predict_proba(xs.reshape(-1, 1))[:, 1]

fig, ax = plt.subplots(figsize=(8.5, 5))
ax.scatter(x, y, s=22, alpha=0.55, color=CINZA, zorder=3, label="observações (0 ou 1)")
ax.plot(xs, pred_linear, color=VERMELHO, lw=2.2, label="regressão linear (extrapola < 0 e > 1)")
ax.plot(xs, pred_log, color=AZUL, lw=2.4, label="regressão logística (sigmoide)")
ax.axhline(0, color=CINZA, lw=0.8, ls=":"); ax.axhline(1, color=CINZA, lw=0.8, ls=":")
ax.fill_between(xs, -0.35, 0, color=VERMELHO, alpha=0.08)
ax.fill_between(xs, 1, 1.35, color=VERMELHO, alpha=0.08)
ax.set_ylim(-0.35, 1.35)
ax.set_xlabel("x"); ax.set_ylabel("y / probabilidade prevista")
ax.set_title("Uma reta produz probabilidades impossíveis (< 0 ou > 1); a sigmoide não")
ax.legend(fontsize=9, loc="center right")
salva(fig, DESTINO / "reta-vs-sigmoide.png")

# ---------------------------------------------------------------------------
# 2. Fronteira de decisão linear em 2D
# ---------------------------------------------------------------------------
n = 300
X2 = rng.normal(0, 1, (n, 2))
beta_2d = np.array([0.0, 2.2, -1.6])
logit2 = beta_2d[0] + X2 @ beta_2d[1:]
prob2 = 1 / (1 + np.exp(-logit2))
y2 = (rng.random(n) < prob2).astype(int)
modelo2 = LogisticRegression().fit(X2, y2)

xx, yy = np.meshgrid(np.linspace(-3, 3, 250), np.linspace(-3, 3, 250))
grade = np.column_stack([xx.ravel(), yy.ravel()])
prob_grade = modelo2.predict_proba(grade)[:, 1].reshape(xx.shape)

fig, ax = plt.subplots(figsize=(7, 6))
contorno = ax.contourf(xx, yy, prob_grade, levels=20, cmap="RdBu_r", alpha=0.65)
ax.contour(xx, yy, prob_grade, levels=[0.5], colors="black", linewidths=2.2)
ax.scatter(X2[y2 == 0, 0], X2[y2 == 0, 1], s=14, color=AZUL, alpha=0.75, label="classe 0")
ax.scatter(X2[y2 == 1, 0], X2[y2 == 1, 1], s=14, color=VERMELHO, alpha=0.75, label="classe 1")
plt.colorbar(contorno, label="P(y=1)")
ax.set_title("A fronteira P=0,5 (linha preta) é sempre um hiperplano — reta em 2D")
ax.legend(fontsize=8, loc="upper left")
salva(fig, DESTINO / "fronteira-decisao.png")

# ---------------------------------------------------------------------------
# 3. Odds ratio: a mesma mudança de chances move a probabilidade de forma
#    diferente dependendo do ponto de partida
# ---------------------------------------------------------------------------
p0 = np.linspace(0.02, 0.98, 300)
odds0 = p0 / (1 - p0)
razao = 2.0  # "as chances dobram"
p1 = (odds0 * razao) / (1 + odds0 * razao)
delta = p1 - p0

fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(p0, delta, color=ROXO, lw=2.4)
ax.axvline(0.5, color=CINZA, lw=1, ls="--")
ax.fill_between(p0, 0, delta, color=ROXO, alpha=0.12)
ax.set_xlabel("probabilidade de partida")
ax.set_ylabel("variação na probabilidade quando as chances DOBRAM")
ax.set_title("Dobrar as chances (odds ratio = 2) desloca MENOS a probabilidade\n"
             "perto dos extremos, e MAIS perto de 50%")
salva(fig, DESTINO / "odds-vs-probabilidade.png")

print("done")
