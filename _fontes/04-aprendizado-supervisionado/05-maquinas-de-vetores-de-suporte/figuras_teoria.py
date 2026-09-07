"""Gera as figuras do teoria.pdf do módulo 05-maquinas-de-vetores-de-suporte."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ / "_ferramentas"))
from estilo_figuras import AZUL, VERDE, VERMELHO, ROXO, AMBAR, CINZA, salva  # noqa: E402

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_circles

rng = np.random.default_rng(5)
MODULO_REL = Path(__file__).resolve().parent.relative_to(RAIZ / "_fontes")
DESTINO = RAIZ / MODULO_REL / "figuras"

# ---------------------------------------------------------------------------
# 1. Margem máxima e vetores de suporte
# ---------------------------------------------------------------------------
n = 30
X0 = rng.normal([2, 2], 0.55, (n, 2))
X1 = rng.normal([6, 6], 0.55, (n, 2))
X = np.vstack([X0, X1])
y = np.concatenate([-np.ones(n), np.ones(n)])
modelo = SVC(kernel="linear", C=1000).fit(X, y)
w, b = modelo.coef_[0], modelo.intercept_[0]
sv = modelo.support_vectors_

xx = np.linspace(0, 8, 100)
yy_c = -(w[0] * xx + b) / w[1]
margem = 1 / np.linalg.norm(w)
offset = margem * np.sqrt(1 + (w[0] / w[1]) ** 2)

fig, ax = plt.subplots(figsize=(7.5, 6.5))
ax.scatter(X0[:, 0], X0[:, 1], s=45, color=AZUL, label="classe A")
ax.scatter(X1[:, 0], X1[:, 1], s=45, color=VERMELHO, label="classe B")
ax.scatter(sv[:, 0], sv[:, 1], s=170, facecolors="none", edgecolors="black",
           linewidths=1.8, label="vetores de suporte")
ax.plot(xx, yy_c, color="black", lw=2.2, label="fronteira de decisão")
ax.plot(xx, yy_c + offset, color=CINZA, lw=1.3, ls="--")
ax.plot(xx, yy_c - offset, color=CINZA, lw=1.3, ls="--", label="margem")
ax.fill_between(xx, yy_c - offset, yy_c + offset, color=AMBAR, alpha=0.08)
ax.annotate("a 'estrada' mais larga\npossível entre as classes",
            xy=(4, 4), xytext=(0.3, 6.6), fontsize=9.5,
            arrowprops=dict(arrowstyle="->", color=CINZA))
ax.set_xlim(0, 8); ax.set_ylim(0, 8); ax.set_aspect("equal")
ax.set_title("Margem máxima: só os vetores de suporte determinam a fronteira")
ax.legend(fontsize=8, loc="lower right")
salva(fig, DESTINO / "margem-maxima.png")

# ---------------------------------------------------------------------------
# 2. Efeito de C: margem rígida vs. suave
# ---------------------------------------------------------------------------
n = 70
X0 = rng.normal([2, 2], 1.15, (n, 2))
X1 = rng.normal([5, 5], 1.15, (n, 2))
Xs = np.vstack([X0, X1])
ys = np.concatenate([-np.ones(n), np.ones(n)])
xg, yg = np.meshgrid(np.linspace(-1, 8, 220), np.linspace(-1, 8, 220))

fig, axes = plt.subplots(1, 3, figsize=(14.5, 5))
for ax, C in zip(axes, [0.02, 1, 200]):
    m = SVC(kernel="linear", C=C).fit(Xs, ys)
    Z = m.decision_function(np.column_stack([xg.ravel(), yg.ravel()])).reshape(xg.shape)
    ax.contourf(xg, yg, Z, levels=[-1e9, 0, 1e9], colors=[AZUL, VERMELHO], alpha=0.15)
    ax.contour(xg, yg, Z, levels=[-1, 0, 1], colors="black",
               linestyles=["--", "-", "--"], linewidths=[1, 2, 1])
    ax.scatter(Xs[ys == -1, 0], Xs[ys == -1, 1], s=20, color=AZUL, alpha=0.7)
    ax.scatter(Xs[ys == 1, 0], Xs[ys == 1, 1], s=20, color=VERMELHO, alpha=0.7)
    ax.scatter(m.support_vectors_[:, 0], m.support_vectors_[:, 1], s=90,
               facecolors="none", edgecolors="black", linewidths=1.4)
    ax.set_title(f"C={C}  ({len(m.support_vectors_)} vetores de suporte)", fontsize=10.5)
fig.suptitle("C pequeno: margem larga, tolera violações | C grande: margem estreita, "
             "ajusta-se ao treino", fontsize=11.5, fontweight="bold", y=1.03)
salva(fig, DESTINO / "efeito-de-c.png")

# ---------------------------------------------------------------------------
# 3. O truque do kernel: quando uma reta não resolve, RBF resolve
# ---------------------------------------------------------------------------
X, y = make_circles(n_samples=300, noise=0.08, factor=0.4, random_state=0)
xx2, yy2 = np.meshgrid(np.linspace(-1.5, 1.5, 220), np.linspace(-1.5, 1.5, 220))
grade = np.column_stack([xx2.ravel(), yy2.ravel()])

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5))
for ax, kernel, titulo in zip(axes, ["linear", "rbf"],
                              ["Kernel linear: fracassa", "Kernel RBF: resolve"]):
    m = SVC(kernel=kernel, C=1.0, gamma="scale").fit(X, y)
    Z = m.decision_function(grade).reshape(xx2.shape)
    ax.contourf(xx2, yy2, Z, levels=20, cmap="RdBu_r", alpha=0.6)
    ax.contour(xx2, yy2, Z, levels=[0], colors="black", linewidths=2)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=14, color=AZUL)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=14, color=VERMELHO)
    ax.set_title(titulo, fontsize=11)
    ax.set_aspect("equal")
fig.suptitle("Círculos concêntricos: nenhuma reta separa as classes, mas o kernel RBF\n"
             "encontra a fronteira circular sem nunca calcular features explícitas",
             fontsize=11, fontweight="bold", y=1.05)
salva(fig, DESTINO / "kernel-trick.png")

print("done")
