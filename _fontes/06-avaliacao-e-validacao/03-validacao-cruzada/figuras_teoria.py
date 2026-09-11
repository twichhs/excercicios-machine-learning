"""Gera as figuras do teoria.pdf do módulo 03-validacao-cruzada."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ / "_ferramentas"))
from estilo_figuras import AZUL, VERDE, VERMELHO, ROXO, AMBAR, CINZA, salva  # noqa: E402

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit,
                                     cross_val_score, LeaveOneOut)
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier

rng = np.random.default_rng(63)
MODULO_REL = Path(__file__).resolve().parent.relative_to(RAIZ / "_fontes")
DESTINO = RAIZ / MODULO_REL / "figuras"

# ---------------------------------------------------------------------------
# 1. Esquemas de divisão, lado a lado
# ---------------------------------------------------------------------------
n = 60
y_vis = (rng.random(n) < 0.3).astype(int)
grupos_vis = np.repeat(np.arange(12), 5)
esquemas = [("KFold (embaralhado)", KFold(5, shuffle=True, random_state=0), None),
            ("StratifiedKFold", StratifiedKFold(5, shuffle=True, random_state=0), y_vis),
            ("GroupKFold (12 grupos)", GroupKFold(5), grupos_vis),
            ("TimeSeriesSplit", TimeSeriesSplit(5), None)]
cmap = ListedColormap(["#FFFFFF", AZUL, VERMELHO])
fig, axes = plt.subplots(len(esquemas), 1, figsize=(11, 8.6))
for ax, (nome, cv, extra) in zip(axes, esquemas):
    matriz = np.zeros((5, n))
    kw = {"groups": grupos_vis} if "Group" in nome else {}
    alvo = y_vis if extra is not None and "Group" not in nome else None
    for i, (tr, te) in enumerate(cv.split(np.zeros((n, 1)), alvo if alvo is not None else y_vis, **kw)):
        matriz[i, tr] = 1
        matriz[i, te] = 2
    ax.imshow(matriz, aspect="auto", cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    ax.set_yticks(range(5)); ax.set_yticklabels([f"fold {i + 1}" for i in range(5)], fontsize=8)
    ax.set_xticks([]); ax.grid(False)
    ax.set_title(nome, fontsize=10.5, loc="left")
    if "Strat" in nome:
        for j in np.where(y_vis == 1)[0]:
            ax.plot(j, 5.1, marker="^", color=AMBAR, ms=4, clip_on=False)
    if "Group" in nome:
        for g in range(1, 12):
            ax.axvline(g * 5 - 0.5, color=CINZA, lw=0.6)
axes[-1].set_xlabel("índice da observação (em TimeSeriesSplit, a ordem é o tempo)")
fig.suptitle("Azul = treino, vermelho = validação, branco = não usado. Triângulos: exemplos da classe rara",
             fontsize=11, fontweight="bold")
fig.tight_layout()
salva(fig, DESTINO / "esquemas-de-divisao.png")

# ---------------------------------------------------------------------------
# 2. Viés e variância do estimador de CV para diferentes K
# ---------------------------------------------------------------------------
def gera(n, g):
    X = g.normal(0, 1, (n, 5))
    logit = X @ np.array([1.2, -0.8, 0.5, 0, 0])
    return X, (g.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

g_big = np.random.default_rng(999)
X_big, y_big = gera(200000, g_big)
n_amostra, reps = 80, 150
res = {k: [] for k in ["K=2", "K=5", "K=10", "LOO"]}
verdade = []
for r in range(reps):
    g = np.random.default_rng(r)
    X, y = gera(n_amostra, g)
    modelo = LogisticRegression(C=1.0)
    verdade.append(1 - modelo.fit(X, y).score(X_big[:20000], y_big[:20000]))  # erro do modelo final
    for nome, cv in [("K=2", KFold(2, shuffle=True, random_state=r)),
                     ("K=5", KFold(5, shuffle=True, random_state=r)),
                     ("K=10", KFold(10, shuffle=True, random_state=r)),
                     ("LOO", LeaveOneOut())]:
        res[nome].append(1 - cross_val_score(modelo, X, y, cv=cv).mean())
fig, ax = plt.subplots(figsize=(9, 4.8))
dados = [np.array(verdade)] + [np.array(v) for v in res.values()]
rotulos = ["erro real do\nmodelo final"] + list(res.keys())
bp = ax.boxplot(dados, tick_labels=rotulos, patch_artist=True, widths=0.55)
for patch, cor in zip(bp["boxes"], [CINZA, VERMELHO, AMBAR, VERDE, AZUL]):
    patch.set_facecolor(cor); patch.set_alpha(0.45)
ax.axhline(np.mean(verdade), color="black", ls=":", lw=1.2)
ax.set_ylabel("erro de classificação")
ax.set_title(f"{reps} amostras de n={n_amostra}: a CV de cada amostra estima o erro com muito ruído;\n"
             "K pequeno é pessimista (treina com menos dados)", fontsize=11)
salva(fig, DESTINO / "vies-e-variancia-da-cv.png")
for nome, v in res.items():
    v = np.array(v)
    print(f"{nome}: média={v.mean():.3f} (viés vs real {v.mean() - np.mean(verdade):+.3f}), dp={v.std():.3f}, "
          f"corr com erro real={np.corrcoef(v, verdade)[0, 1]:+.2f}")
print(f"erro real médio={np.mean(verdade):.3f}, dp={np.std(verdade):.3f}")

# ---------------------------------------------------------------------------
# 3. Seleção de atributos fora da CV: acurácia alta em ruído puro
# ---------------------------------------------------------------------------
n_amostras, n_feat = 50, 5000
X_ruido = rng.normal(0, 1, (n_amostras, n_feat))
y_ruido = np.r_[np.zeros(25), np.ones(25)].astype(int)
ks = [5, 10, 20, 50, 100, 500]
fora, dentro = [], []
cv = StratifiedKFold(5, shuffle=True, random_state=0)
for k in ks:
    sel = SelectKBest(f_classif, k=k).fit(X_ruido, y_ruido)       # ERRADO: usa todos os rótulos
    fora.append(cross_val_score(KNeighborsClassifier(3), sel.transform(X_ruido), y_ruido, cv=cv).mean())
    pipe = make_pipeline(SelectKBest(f_classif, k=k), KNeighborsClassifier(3))
    dentro.append(cross_val_score(pipe, X_ruido, y_ruido, cv=cv).mean())
fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.plot(ks, fora, "o-", color=VERMELHO, lw=2.2, label="seleção ANTES da CV (vazamento)")
ax.plot(ks, dentro, "o-", color=VERDE, lw=2.2, label="seleção DENTRO de cada fold (Pipeline)")
ax.axhline(0.5, color="black", ls=":", label="acaso (ruído puro)")
ax.set_xscale("log"); ax.set_ylim(0.2, 1.02)
ax.set_xlabel("nº de atributos selecionados (de 5.000 atributos de ruído puro)")
ax.set_ylabel("acurácia estimada pela CV"); ax.legend(fontsize=9)
ax.set_title("50 amostras, 5.000 atributos sem nenhuma relação com o rótulo:\n"
             "selecionar atributos fora da CV fabrica um modelo 'excelente'", fontsize=11)
salva(fig, DESTINO / "selecao-fora-da-cv.png")
print("seleção fora:", np.round(fora, 3), "dentro:", np.round(dentro, 3))

# ---------------------------------------------------------------------------
# 4. A maldição do vencedor: o melhor escore de CV entre M configurações iguais
# ---------------------------------------------------------------------------
Ms = [1, 2, 5, 10, 20, 50, 100, 200]
verdadeira, dp_cv = 0.80, 0.02
sims = rng.normal(verdadeira, dp_cv, (5000, max(Ms)))
medias_max = [sims[:, :M].max(axis=1).mean() for M in Ms]
p95 = [np.percentile(sims[:, :M].max(axis=1), 95) for M in Ms]
fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.plot(Ms, medias_max, "o-", color=VERMELHO, lw=2.2, label="melhor escore de CV (média)")
ax.fill_between(Ms, verdadeira, p95, color=VERMELHO, alpha=0.1, label="até o percentil 95")
ax.axhline(verdadeira, color="black", ls=":", label="desempenho real de TODAS as configurações")
ax.set_xscale("log"); ax.set_xlabel("número de configurações testadas (M)")
ax.set_ylabel("acurácia"); ax.legend(fontsize=9)
ax.set_title("M configurações com o MESMO desempenho real (0,80) e ruído de CV de 0,02:\n"
             "o vencedor sempre parece melhor do que é", fontsize=11)
salva(fig, DESTINO / "maldicao-do-vencedor.png")
print("otimismo do melhor:", dict(zip(Ms, np.round(np.array(medias_max) - verdadeira, 4))))
print("done")
