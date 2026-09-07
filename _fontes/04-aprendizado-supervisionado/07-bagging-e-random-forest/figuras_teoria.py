"""Gera as figuras do teoria.pdf do módulo 07-bagging-e-random-forest."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ / "_ferramentas"))
from estilo_figuras import AZUL, VERDE, VERMELHO, ROXO, AMBAR, CINZA, salva  # noqa: E402

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

rng = np.random.default_rng(7)
MODULO_REL = Path(__file__).resolve().parent.relative_to(RAIZ / "_fontes")
DESTINO = RAIZ / MODULO_REL / "figuras"

# ---------------------------------------------------------------------------
# 1. Bagging reduz variância: muitas árvores ruidosas, a média é estável
# ---------------------------------------------------------------------------
n = 40
x = np.sort(rng.uniform(0, 10, n))
verdade = lambda t: np.sin(t) * 2 + 0.15 * t
y = verdade(x) + rng.normal(0, 0.9, n)
x_grade = np.linspace(0, 10, 300).reshape(-1, 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

# painel 1: árvores individuais, cada uma numa amostra bootstrap
preds = []
for b in range(25):
    idx = rng.integers(0, n, n)
    arvore = DecisionTreeRegressor(max_depth=4, random_state=b).fit(x[idx].reshape(-1, 1), y[idx])
    pred = arvore.predict(x_grade)
    preds.append(pred)
    axes[0].plot(x_grade, pred, color=AZUL, alpha=0.12, lw=1)
axes[0].scatter(x, y, s=22, color=CINZA, zorder=5, label="dados de treino")
axes[0].plot(x_grade, verdade(x_grade.ravel()), color="black", lw=2, ls="--", label="função real")
axes[0].set_title("25 árvores, cada uma numa amostra\nbootstrap diferente (max_depth=4)", fontsize=10.5)
axes[0].legend(fontsize=8, loc="upper left")

# painel 2: a média das 25 árvores vs. função real
preds = np.array(preds)
media = preds.mean(axis=0)
axes[1].scatter(x, y, s=22, color=CINZA, zorder=5, label="dados de treino")
axes[1].plot(x_grade, verdade(x_grade.ravel()), color="black", lw=2, ls="--", label="função real")
axes[1].plot(x_grade, media, color=VERMELHO, lw=2.4, label="média das 25 árvores (bagging)")
axes[1].set_title("A média das 25 árvores: muito mais\npróxima da função real", fontsize=10.5)
axes[1].legend(fontsize=8, loc="upper left")
for ax in axes:
    ax.set_xlabel("x"); ax.set_ylabel("y")
fig.suptitle("Cada árvore individual é ruidosa; a média de muitas reduz a variância "
             "sem aumentar o viés", fontsize=11.5, fontweight="bold", y=1.02)
salva(fig, DESTINO / "bagging-reduz-variancia.png")

# ---------------------------------------------------------------------------
# 2. O piso de correlação: Var(média) vs. B, para diferentes rho
# ---------------------------------------------------------------------------
sigma2 = 1.0
Bs = np.arange(1, 201)
fig, ax = plt.subplots(figsize=(8, 5.5))
for rho, cor, rotulo in zip([0.0, 0.1, 0.3, 0.6],
                            [VERDE, AZUL, AMBAR, VERMELHO],
                            [r"$\rho=0$ (árvores independentes)", r"$\rho=0{,}1$ (Random Forest, típico)",
                             r"$\rho=0{,}3$", r"$\rho=0{,}6$ (bagging simples, típico)"]):
    var_media = rho * sigma2 + (1 - rho) * sigma2 / Bs
    ax.plot(Bs, var_media, color=cor, lw=2.2, label=rotulo)
    ax.axhline(rho * sigma2, color=cor, lw=1, ls=":", alpha=0.6)
ax.set_xlabel("número de árvores (B)")
ax.set_ylabel(r"Var$\left(\frac{1}{B}\sum \hat{f}_b\right)$")
ax.set_title(r"Aumentar B tem retorno decrescente — o piso é $\rho\sigma^2$,"
             "\nque só reduzir a correlação entre árvores ataca", fontsize=11)
ax.legend(fontsize=8.5)
salva(fig, DESTINO / "piso-de-correlacao.png")

# ---------------------------------------------------------------------------
# 3. MDI vs. permutação: o viés de cardinalidade em ação
# ---------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

n = 600
util = rng.normal(0, 1, n)
alvo = (util + rng.normal(0, 1.2, n) > 0).astype(int)   # sinal real, mas fraco
ruido_baixa_card = rng.integers(0, 3, n)          # 3 categorias, sem relação com o alvo
ruido_alta_card = rng.integers(0, 550, n)          # quase um ID, sem relação real

X = np.column_stack([util, ruido_baixa_card, ruido_alta_card])
nomes = ["feature útil\n(sinal real, fraco)", "categórica\n(3 níveis, ruído)", "quase-ID\n(550 níveis, ruído)"]

X_tr, X_te, y_tr, y_te = train_test_split(X, alvo, test_size=0.4, random_state=0)
modelo = RandomForestClassifier(n_estimators=300, random_state=0).fit(X_tr, y_tr)
mdi = modelo.feature_importances_

perm = permutation_importance(modelo, X_te, y_te, n_repeats=30, random_state=0)
perm_media = np.clip(perm.importances_mean, 0, None)

fig, ax = plt.subplots(figsize=(8.5, 5.5))
largura = 0.35
posicoes = np.arange(len(nomes))
ax.bar(posicoes - largura / 2, mdi / mdi.sum(), largura, color=AMBAR, label="MDI (impureza)")
ax.bar(posicoes + largura / 2, perm_media / perm_media.sum(), largura, color=AZUL, label="permutação")
ax.set_xticks(posicoes); ax.set_xticklabels(nomes, fontsize=9.5)
ax.set_ylabel("importância normalizada")
ax.set_title("A feature 'quase-ID' (só ruído) recebe importância alta por MDI,\n"
             "mas quase zero por permutação em dados de validação", fontsize=11)
ax.legend(fontsize=9)
salva(fig, DESTINO / "mdi-vs-permutacao.png")

print("done")
