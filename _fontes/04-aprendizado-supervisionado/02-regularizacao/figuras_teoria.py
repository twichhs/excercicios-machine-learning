"""Gera as figuras do teoria.pdf do módulo 02-regularizacao."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ / "_ferramentas"))
from estilo_figuras import AZUL, VERDE, VERMELHO, ROXO, AMBAR, CINZA, salva  # noqa: E402

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(2)
MODULO_REL = Path(__file__).resolve().parent.relative_to(RAIZ / "_fontes")
DESTINO = RAIZ / MODULO_REL / "figuras"

# ---------------------------------------------------------------------------
# 1. Geometria: por que L1 zera coeficientes e L2 não
# ---------------------------------------------------------------------------
n = 150
X = StandardScaler().fit_transform(rng.normal(0, 1, (n, 2)))
beta_real = np.array([2.5, 0.3])
y = X @ beta_real + rng.normal(0, 1, n)
beta_mq = np.linalg.lstsq(X, y, rcond=None)[0]


def soma_quadrados(b1_flat, b2_flat, X, y):
    pred = np.outer(X[:, 0], b1_flat) + np.outer(X[:, 1], b2_flat)
    residuo = y[:, None] - pred
    return np.sum(residuo ** 2, axis=0)


eixo_b1, eixo_b2 = np.linspace(-1, 4, 150), np.linspace(-3, 3, 150)
b1_grid, b2_grid = np.meshgrid(eixo_b1, eixo_b2)
sqr = soma_quadrados(b1_grid.ravel(), b2_grid.ravel(), X, y).reshape(b1_grid.shape)

# solucoes regularizadas de verdade (sklearn) -- o raio de cada regiao eh
# definido para bater EXATAMENTE com a norma da solucao encontrada, porque
# a dualidade Lagrangiana garante que a solucao regularizada eh o ponto de
# tangencia entre as elipses e a regiao de raio igual a norma da propria
# solucao. Sem isso, desenhar um raio arbitrario faz o ponto cair fora da
# regiao -- geometricamente incoerente.
beta_ridge = Ridge(alpha=2.6, fit_intercept=False).fit(X, y).coef_
beta_lasso = Lasso(alpha=0.62, fit_intercept=False).fit(X, y).coef_

raio_ridge = np.linalg.norm(beta_ridge, 2)
raio_lasso = np.linalg.norm(beta_lasso, 1)

theta = np.linspace(0, 2 * np.pi, 200)
circulo_x, circulo_y = raio_ridge * np.cos(theta), raio_ridge * np.sin(theta)
losango_x = np.array([raio_lasso, 0, -raio_lasso, 0, raio_lasso])
losango_y = np.array([0, raio_lasso, 0, -raio_lasso, 0])

fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
for ax, (cx, cy, nome, cor, beta_reg, rotulo_reg) in zip(axes, [
    (circulo_x, circulo_y, "Ridge (L2) — região circular", VERDE, beta_ridge,
     "solução Ridge"),
    (losango_x, losango_y, "Lasso (L1) — região em losango", ROXO, beta_lasso,
     "solução Lasso\n(β₂ = 0 exatamente)"),
]):
    ax.contour(b1_grid, b2_grid, sqr, levels=18, cmap="Blues", alpha=0.6)
    ax.plot(cx, cy, color=cor, lw=2.4)
    ax.fill(cx, cy, color=cor, alpha=0.15)
    ax.scatter([beta_mq[0]], [beta_mq[1]], color=VERMELHO, s=55, zorder=5,
               label="mínimos quadrados\n(sem restrição)")
    ax.scatter([beta_reg[0]], [beta_reg[1]], color="black", marker="*", s=180,
               zorder=6, label=rotulo_reg)
    ax.axhline(0, color=CINZA, lw=0.7); ax.axvline(0, color=CINZA, lw=0.7)
    ax.set_xlabel(r"$\beta_1$"); ax.set_ylabel(r"$\beta_2$")
    ax.set_title(nome, fontsize=10.5)
    ax.set_aspect("equal"); ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(-2, 4); ax.set_ylim(-3, 3)
fig.suptitle("O ótimo restrito é onde a menor elipse toca a região —\n"
             "o losango tem quinas sobre os eixos, o círculo não",
             fontsize=11.5, fontweight="bold", y=1.05)
salva(fig, DESTINO / "geometria-l1-l2.png")

# ---------------------------------------------------------------------------
# 2. Caminho de encolhimento dos coeficientes: Ridge vs. Lasso
# ---------------------------------------------------------------------------
n, p = 200, 5
X2 = StandardScaler().fit_transform(rng.normal(0, 1, (n, p)))
beta2 = np.array([3.0, -2.0, 0.0, 0.0, 1.5])
y2 = X2 @ beta2 + rng.normal(0, 1, n)

lambdas = np.logspace(-2, 2.3, 60)
caminho_ridge = np.array([Ridge(alpha=lam).fit(X2, y2).coef_ for lam in lambdas])
caminho_lasso = np.array([Lasso(alpha=lam / 40, max_iter=5000).fit(X2, y2).coef_
                          for lam in lambdas])

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
for ax, caminho, nome in [(axes[0], caminho_ridge, "Ridge (L2)"),
                          (axes[1], caminho_lasso, "Lasso (L1)")]:
    for j in range(p):
        cor = VERDE if beta2[j] == 0 else AZUL
        estilo = "--" if beta2[j] == 0 else "-"
        ax.plot(lambdas, caminho[:, j], estilo, color=cor, lw=1.8,
                label=f"β{j+1} (real={beta2[j]})")
    ax.axhline(0, color=CINZA, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("força da penalidade (escala log)")
    ax.set_title(nome, fontsize=11)
    ax.legend(fontsize=7.5, loc="upper right")
axes[0].set_ylabel("coeficiente")
fig.suptitle("Ridge encolhe suavemente para perto de 0; Lasso chega a EXATAMENTE 0",
             fontsize=11.5, fontweight="bold", y=1.02)
salva(fig, DESTINO / "caminho-encolhimento.png")

# ---------------------------------------------------------------------------
# 3. Trade-off viés-variância: erro de treino e de teste vs. força da penalidade
# ---------------------------------------------------------------------------
n_rep = 80
n_treino_sim, p_sim = 30, 15  # poucas observações relativas a p -> overfitting real e visível
lambdas3 = np.logspace(-2, 2.5, 25)
erro_treino, erro_teste = [], []
for lam in lambdas3:
    et, ete = [], []
    for rep in range(n_rep):
        rng_r = np.random.default_rng(rep)
        Xr = StandardScaler().fit_transform(rng_r.normal(0, 1, (n_treino_sim, p_sim)))
        betar = rng_r.normal(0, 1.3, p_sim)
        yr = Xr @ betar + rng_r.normal(0, 1, n_treino_sim)
        Xte = StandardScaler().fit_transform(rng_r.normal(0, 1, (300, p_sim)))
        yte = Xte @ betar + rng_r.normal(0, 1, 300)
        modelo = Ridge(alpha=lam).fit(Xr, yr)
        et.append(np.mean((yr - modelo.predict(Xr)) ** 2))
        ete.append(np.mean((yte - modelo.predict(Xte)) ** 2))
    erro_treino.append(np.mean(et))
    erro_teste.append(np.mean(ete))

fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(lambdas3, erro_treino, color=VERMELHO, lw=2, label="erro de treino")
ax.plot(lambdas3, erro_teste, color=AZUL, lw=2, label="erro em dados novos (teste)")
melhor_lambda = lambdas3[np.argmin(erro_teste)]
ax.axvline(melhor_lambda, color=CINZA, ls="--", lw=1.3,
           label=f"melhor λ ≈ {melhor_lambda:.2f}")
ax.set_xscale("log")
ax.set_xlim(1e-2, 15)
ax.set_ylim(0, 6)
ax.set_xlabel(r"força da penalidade $\lambda$ (escala log)")
ax.set_ylabel("erro quadrático médio")
ax.set_title("O erro de teste tem um mínimo em algum λ > 0 — nunca em λ = 0\n"
             "(eixos recortados para ampliar a região do mínimo)", fontsize=11.5)
ax.legend(fontsize=9)
salva(fig, DESTINO / "vies-variancia-lambda.png")

print("done")
