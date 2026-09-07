"""Gera as figuras do teoria.pdf do módulo 08-boosting."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ / "_ferramentas"))
from estilo_figuras import AZUL, VERDE, VERMELHO, ROXO, AMBAR, CINZA, salva  # noqa: E402

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

rng = np.random.default_rng(8)
MODULO_REL = Path(__file__).resolve().parent.relative_to(RAIZ / "_fontes")
DESTINO = RAIZ / MODULO_REL / "figuras"

# ---------------------------------------------------------------------------
# 1. Ajuste sequencial: a soma de árvores rasas se aproxima da função real
# ---------------------------------------------------------------------------
n = 60
x = np.sort(rng.uniform(0, 10, n))
verdade = lambda t: np.sin(t) * 2 + 0.15 * t
y = verdade(x) + rng.normal(0, 0.4, n)
x_grade = np.linspace(0, 10, 300).reshape(-1, 1)

fig, axes = plt.subplots(1, 4, figsize=(15, 4))
for ax, M in zip(axes, [1, 5, 20, 100]):
    modelo = GradientBoostingRegressor(n_estimators=M, max_depth=2, learning_rate=0.3,
                                       random_state=0).fit(x.reshape(-1, 1), y)
    pred = modelo.predict(x_grade)
    ax.scatter(x, y, s=16, color=CINZA, alpha=0.7, zorder=5)
    ax.plot(x_grade, verdade(x_grade.ravel()), color="black", lw=1.6, ls="--")
    ax.plot(x_grade, pred, color=VERMELHO, lw=2.2)
    ax.set_title(f"M={M} árvores", fontsize=11)
    ax.set_xlabel("x")
axes[0].set_ylabel("y")
fig.suptitle("Cada árvore soma uma pequena correção — a soma se aproxima\n"
             "gradualmente da função real conforme M cresce", fontsize=11.5,
             fontweight="bold", y=1.05)
salva(fig, DESTINO / "ajuste-sequencial.png")

# ---------------------------------------------------------------------------
# 2. Overfitting sem early stopping: perda de treino vs. validação por rodada
# ---------------------------------------------------------------------------
n = 800
X = rng.normal(0, 1, (n, 10))
logit = X[:, 0] * 1.4 - X[:, 1] * 0.8 + 0.5 * X[:, 2] * X[:, 3]
prob = 1 / (1 + np.exp(-logit))
y = (rng.random(n) < prob).astype(int)
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.3, random_state=0)

M_max = 400
modelo = GradientBoostingClassifier(n_estimators=M_max, max_depth=3, learning_rate=0.15,
                                    random_state=0).fit(X_tr, y_tr)
perda_treino = []
perda_val = []
for pred_tr, pred_val in zip(modelo.staged_predict_proba(X_tr), modelo.staged_predict_proba(X_val)):
    perda_treino.append(log_loss(y_tr, pred_tr))
    perda_val.append(log_loss(y_val, pred_val))
perda_treino = np.array(perda_treino)
perda_val = np.array(perda_val)
rodada_otima = np.argmin(perda_val)

fig, ax = plt.subplots(figsize=(8.5, 5.5))
rounds = np.arange(1, M_max + 1)
ax.plot(rounds, perda_treino, color=AZUL, lw=2, label="perda de treino")
ax.plot(rounds, perda_val, color=VERMELHO, lw=2, label="perda de validação")
ax.axvline(rodada_otima + 1, color=CINZA, lw=1.5, ls=":")
ax.annotate(f"ponto ótimo de early stopping\n(rodada {rodada_otima + 1})",
            xy=(rodada_otima + 1, perda_val[rodada_otima]),
            xytext=(rodada_otima + 60, perda_val[rodada_otima] + 0.18),
            fontsize=9.5, arrowprops=dict(arrowstyle="->", color=CINZA))
ax.set_xlabel("número de árvores (rodadas de boosting)")
ax.set_ylabel("log-loss")
ax.set_title("Sem early stopping, a perda de treino cai para sempre — mas a de\n"
             "validação piora depois de um certo ponto: overfitting real", fontsize=11)
ax.legend(fontsize=9.5)
salva(fig, DESTINO / "overfitting-sem-early-stopping.png")

# ---------------------------------------------------------------------------
# 3. Efeito da taxa de aprendizado (shrinkage)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.5))
for eta, cor in zip([0.5, 0.1, 0.02], [VERMELHO, AZUL, VERDE]):
    modelo = GradientBoostingClassifier(n_estimators=M_max, max_depth=3, learning_rate=eta,
                                        random_state=0).fit(X_tr, y_tr)
    perda_val_eta = [log_loss(y_val, p) for p in modelo.staged_predict_proba(X_val)]
    ax.plot(rounds, perda_val_eta, color=cor, lw=2.2, label=rf"$\eta$={eta}")
ax.set_xlabel("número de árvores (rodadas de boosting)")
ax.set_ylabel("log-loss de validação")
ax.set_title(r"$\eta$ grande converge rápido, mas passa do ponto ótimo e overfita"
             "\n" r"cedo; $\eta$ pequeno precisa de mais árvores, mas permanece estável", fontsize=11)
ax.legend(fontsize=10)
salva(fig, DESTINO / "efeito-taxa-aprendizado.png")

print("done")
