"""Gera as figuras do teoria.pdf do módulo 06-arvores-de-decisao."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ / "_ferramentas"))
from estilo_figuras import AZUL, VERDE, VERMELHO, ROXO, AMBAR, CINZA, salva  # noqa: E402

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

rng = np.random.default_rng(6)
MODULO_REL = Path(__file__).resolve().parent.relative_to(RAIZ / "_fontes")
DESTINO = RAIZ / MODULO_REL / "figuras"

# ---------------------------------------------------------------------------
# 1. Uma árvore pequena e a partição do espaço que ela produz, lado a lado
# ---------------------------------------------------------------------------
n = 200
renda = rng.uniform(1, 10, n)
idade = rng.uniform(20, 65, n)
# padrão em "xadrez": precisa das DUAS features para separar bem --
# nenhum corte único em renda OU idade sozinho resolve
aprovado = (((renda > 5.5) & (idade > 42)) | ((renda <= 5.5) & (idade <= 42)))
aprovado = aprovado.astype(int)
X = np.column_stack([renda, idade])

arvore = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, aprovado)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.6),
                         gridspec_kw={"width_ratios": [1.3, 1]})
plot_tree(arvore, ax=axes[0], feature_names=["renda", "idade"],
          class_names=["negado", "aprovado"], filled=True, fontsize=9, impurity=False,
          rounded=True)
axes[0].set_title("A árvore: perguntas binárias em sequência", fontsize=11)

xx, yy = np.meshgrid(np.linspace(1, 10, 300), np.linspace(20, 65, 300))
pred_grade = arvore.predict(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
axes[1].contourf(xx, yy, pred_grade, levels=1, cmap="RdBu_r", alpha=0.45)
axes[1].scatter(X[aprovado == 0, 0], X[aprovado == 0, 1], s=18, color=AZUL,
                label="negado", alpha=0.75)
axes[1].scatter(X[aprovado == 1, 0], X[aprovado == 1, 1], s=18, color=VERMELHO,
                label="aprovado", alpha=0.75)
axes[1].set_xlabel("renda"); axes[1].set_ylabel("idade")
axes[1].set_title("A mesma árvore: retângulos no espaço de features", fontsize=11)
axes[1].legend(fontsize=8)
fig.suptitle("Cada corte da árvore é uma linha reta ALINHADA A UM EIXO no espaço original",
             fontsize=11.5, fontweight="bold", y=1.02)
salva(fig, DESTINO / "arvore-e-particao.png")

# ---------------------------------------------------------------------------
# 2. Overfitting: profundidade pequena vs. grande vs. sem limite
# ---------------------------------------------------------------------------
n = 300
X2 = rng.uniform(-3, 3, (n, 2))
prob = 1 / (1 + np.exp(-(X2[:, 0] ** 2 - X2[:, 1])))
y2 = (rng.random(n) < prob).astype(int)
xx2, yy2 = np.meshgrid(np.linspace(-3, 3, 250), np.linspace(-3, 3, 250))
grade = np.column_stack([xx2.ravel(), yy2.ravel()])

fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
for ax, prof, rotulo in zip(axes, [2, 5, None],
                            ["profundidade=2\n(underfitting)",
                             "profundidade=5\n(equilíbrio)",
                             "sem limite\n(memoriza o treino)"]):
    modelo = DecisionTreeClassifier(max_depth=prof, random_state=0).fit(X2, y2)
    pred_grade = modelo.predict(grade).reshape(xx2.shape)
    ax.contourf(xx2, yy2, pred_grade, levels=1, cmap="RdBu_r", alpha=0.5)
    ax.scatter(X2[y2 == 0, 0], X2[y2 == 0, 1], s=8, color=AZUL, alpha=0.6)
    ax.scatter(X2[y2 == 1, 0], X2[y2 == 1, 1], s=8, color=VERMELHO, alpha=0.6)
    n_folhas = modelo.get_n_leaves()
    ax.set_title(f"{rotulo}\n({n_folhas} folhas)", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.86])
fig.suptitle("Sem limite de profundidade, a árvore cria uma folha para cada\n"
             "irregularidade do treino — inclusive ruído", fontsize=11.5,
             fontweight="bold", y=0.99)
salva(fig, DESTINO / "overfitting-profundidade.png")

print("done")
