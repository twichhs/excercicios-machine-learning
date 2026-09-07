"""Gera as figuras do teoria.pdf do módulo 04-knn-e-naive-bayes."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ / "_ferramentas"))
from estilo_figuras import AZUL, VERDE, VERMELHO, ROXO, AMBAR, CINZA, salva  # noqa: E402

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.neighbors import KNeighborsClassifier

rng = np.random.default_rng(4)
MODULO_REL = Path(__file__).resolve().parent.relative_to(RAIZ / "_fontes")
DESTINO = RAIZ / MODULO_REL / "figuras"

# ---------------------------------------------------------------------------
# 1. k-NN: o algoritmo inteiro, ilustrado
# ---------------------------------------------------------------------------
n = 40
classe_a = rng.normal([2, 2], 0.9, (n, 2))
classe_b = rng.normal([5, 5], 0.9, (n, 2))
consulta = np.array([3.8, 3.4])

todos = np.vstack([classe_a, classe_b])
rotulos = np.array(["A"] * n + ["B"] * n)
distancias = np.linalg.norm(todos - consulta, axis=1)
k = 7
indices_vizinhos = np.argsort(distancias)[:k]
raio_k = distancias[indices_vizinhos].max()
votos_a = np.sum(rotulos[indices_vizinhos] == "A")
votos_b = k - votos_a

fig, ax = plt.subplots(figsize=(7, 6.4))
ax.scatter(classe_a[:, 0], classe_a[:, 1], s=35, color=AZUL, alpha=0.5, label="classe A")
ax.scatter(classe_b[:, 0], classe_b[:, 1], s=35, color=VERMELHO, alpha=0.5, label="classe B")
ax.scatter(todos[indices_vizinhos, 0], todos[indices_vizinhos, 1], s=90,
           facecolors="none", edgecolors="black", linewidths=1.8, zorder=4,
           label=f"os {k} vizinhos mais próximos")
circulo = Circle(consulta, raio_k, fill=False, color=AMBAR, lw=2, ls="--", zorder=3)
ax.add_patch(circulo)
ax.scatter(*consulta, s=180, color=AMBAR, marker="*", zorder=5, label="ponto novo (consulta)")
vencedor = "A" if votos_a > votos_b else "B"
ax.annotate(f"votos: {votos_a}×A, {votos_b}×B\n→ classificado como {vencedor}",
            xy=consulta, xytext=(consulta[0] + 1.3, consulta[1] - 2.0), fontsize=9.5,
            arrowprops=dict(arrowstyle="->", color=CINZA))
ax.set_aspect("equal")
ax.set_title(f"k-NN com k={k}: o círculo delimita os vizinhos que votam")
ax.legend(fontsize=8, loc="upper left")
salva(fig, DESTINO / "knn-algoritmo.png")

# ---------------------------------------------------------------------------
# 2. Efeito de k na fronteira de decisão: k pequeno vs. k grande
# ---------------------------------------------------------------------------
n = 300
X = rng.uniform(-3, 3, (n, 2))
prob_real = 1 / (1 + np.exp(-(X[:, 0] ** 2 - X[:, 1])))
y = (rng.random(n) < prob_real).astype(int)

xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
grade = np.column_stack([xx.ravel(), yy.ravel()])

fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
for ax, k_val, rotulo in zip(axes, [1, 15, 150],
                             ["k=1 (memoriza o treino)", "k=15 (equilíbrio)",
                              "k=150 (suaviza demais)"]):
    modelo = KNeighborsClassifier(n_neighbors=k_val).fit(X, y)
    pred_grade = modelo.predict(grade).reshape(xx.shape)
    ax.contourf(xx, yy, pred_grade, levels=1, cmap="RdBu_r", alpha=0.5)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=8, color=AZUL, alpha=0.6)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=8, color=VERMELHO, alpha=0.6)
    ax.set_title(rotulo, fontsize=10.5)
fig.suptitle("k pequeno: fronteira irregular (alta variância) | "
             "k grande: fronteira suave demais (alto viés)",
             fontsize=11.5, fontweight="bold", y=1.03)
salva(fig, DESTINO / "efeito-de-k.png")

# ---------------------------------------------------------------------------
# 3. Naive Bayes Gaussiano: distribuições por classe e o ponto de decisão
# ---------------------------------------------------------------------------
x_grid = np.linspace(-2, 14, 400)


def gaussiana(x, media, dp):
    return (1 / (dp * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - media) / dp) ** 2)


media_a, dp_a, prior_a = 3.0, 1.6, 0.6
media_b, dp_b, prior_b = 8.0, 2.0, 0.4
verossim_a = gaussiana(x_grid, media_a, dp_a) * prior_a
verossim_b = gaussiana(x_grid, media_b, dp_b) * prior_b

fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(x_grid, verossim_a, color=AZUL, lw=2.2, label=r"$P(x|A) \cdot P(A)$")
ax.plot(x_grid, verossim_b, color=VERMELHO, lw=2.2, label=r"$P(x|B) \cdot P(B)$")
ax.fill_between(x_grid, verossim_a, color=AZUL, alpha=0.15)
ax.fill_between(x_grid, verossim_b, color=VERMELHO, alpha=0.15)
# ponto de cruzamento (fronteira de decisao em 1D)
diferenca = verossim_a - verossim_b
cruzamentos = np.where(np.diff(np.sign(diferenca)))[0]
for c in cruzamentos:
    ax.axvline(x_grid[c], color=CINZA, lw=1.3, ls="--")
ax.set_xlabel("x (uma feature contínua)"); ax.set_ylabel("densidade × prior")
ax.set_title("Naive Bayes classifica pelo maior valor — o ponto onde as\n"
             "curvas se cruzam é a fronteira de decisão em 1D")
ax.legend(fontsize=9.5)
salva(fig, DESTINO / "naive-bayes-gaussiano.png")

print("done")
