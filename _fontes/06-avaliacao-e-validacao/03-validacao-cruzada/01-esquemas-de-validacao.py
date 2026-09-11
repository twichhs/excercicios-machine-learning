# %% [markdown]
# # Esquemas de validação
#
# **Tema:** Avaliação e Validação de Modelos › Validação Cruzada
#
# Este notebook implementa o K-Fold do zero, mede o ruído da partição (e o
# que a repetição faz com ele), compara o teste t ingênuo com o corrigido
# de Nadeau e Bengio e, principalmente, constrói dois casos em que o
# esquema errado dá uma estimativa muito otimista: dados agrupados por
# paciente e dados com ordem temporal.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as t_student
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, mean_absolute_error
from sklearn.model_selection import (KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit,
                                     RepeatedStratifiedKFold, cross_val_score)

rng = np.random.default_rng(631)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
AZUL, VERDE, VERMELHO, ROXO, AMBAR = "#1F5C8B", "#1E6B4F", "#9C2B2B", "#5B3E86", "#8A6100"
print("pronto")

# %% [markdown]
# ## 1. K-Fold do zero

# %%
def kfold_do_zero(n, k, semente=0):
    idx = np.random.default_rng(semente).permutation(n)
    folds = np.array_split(idx, k)
    for i in range(k):
        yield np.concatenate([folds[j] for j in range(k) if j != i]), folds[i]

n = 600
X = rng.normal(0, 1, (n, 6))
y = (rng.random(n) < 1 / (1 + np.exp(-(X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2])))).astype(int)
modelo = LogisticRegression()
nossos = [roc_auc_score(y[te], modelo.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1])
          for tr, te in kfold_do_zero(n, 5)]
print(f"AUC por fold (do zero): {np.round(nossos, 3)} -> média {np.mean(nossos):.4f}")
cada_ponto_uma_vez = np.sort(np.concatenate([te for _, te in kfold_do_zero(n, 5)]))
print("cada observação aparece exatamente uma vez na validação?", np.array_equal(cada_ponto_uma_vez, np.arange(n)))

# %% [markdown]
# ## 2. O ruído da partição e o efeito da repetição
#
# Rodamos o 5-fold com 200 sementes diferentes (mesmos dados, mesmo
# modelo) e comparamos com 5 repetições de 5-fold.

# %%
unico = [cross_val_score(modelo, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=s),
                         scoring="roc_auc").mean() for s in range(200)]
repetido = [cross_val_score(modelo, X, y, scoring="roc_auc",
                            cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=s)).mean()
            for s in range(60)]
print(f"5-fold simples   : AUC entre {np.min(unico):.4f} e {np.max(unico):.4f} (dp {np.std(unico):.4f})")
print(f"5x5-fold repetido: AUC entre {np.min(repetido):.4f} e {np.max(repetido):.4f} (dp {np.std(repetido):.4f})")

# %% [markdown]
# A repetição encolhe o ruído da **partição** (a "sorte" de quem cai em
# qual fold). Ela não encolhe o ruído da **amostra**: todos esses números
# vêm dos mesmos 600 dados.

# %% [markdown]
# ## 3. Comparando dois modelos: teste t ingênuo vs. corrigido
#
# Logística contra Random Forest nos mesmos folds (10 repetições de
# 10-fold = 100 avaliações pareadas).

# %%
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0)
rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=0)
d = []
for tr, te in cv.split(X, y):
    auc_lr = roc_auc_score(y[te], modelo.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1])
    auc_rf = roc_auc_score(y[te], rf.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1])
    d.append(auc_lr - auc_rf)
d = np.array(d)
J, razao = len(d), 1 / 9
t_ing = d.mean() / (d.std(ddof=1) / np.sqrt(J))
t_cor = d.mean() / np.sqrt((1 / J + razao) * d.var(ddof=1))
print(f"diferença média de AUC (logística - floresta): {d.mean():+.4f} em {J} avaliações")
print(f"teste t ingênuo  : t = {t_ing:6.2f}, p-valor = {2 * t_student.sf(abs(t_ing), J - 1):.4f}")
print(f"teste t corrigido: t = {t_cor:6.2f}, p-valor = {2 * t_student.sf(abs(t_cor), J - 1):.4f}")

# %% [markdown]
# O ingênuo trata as 100 avaliações como independentes e encontra uma
# significância que o corrigido não sustenta. O exemplo numérico do
# `teoria.pdf` ($\bar{d}=0{,}012$, $s_d=0{,}020$, $J=10$):

# %%
for nome, se in [("ingênuo", 0.020 / np.sqrt(10)), ("corrigido", np.sqrt((0.1 + 1 / 9) * 0.020 ** 2))]:
    tt = 0.012 / se
    print(f"{nome:>9s}: t = {tt:.2f}, p-valor = {2 * t_student.sf(tt, 9):.3f}")

# %% [markdown]
# ## 4. Grupos: pacientes que o modelo já viu
#
# 200 pacientes, 10 consultas cada, e uma doença crônica: o diagnóstico é
# **do paciente**, o mesmo em todas as consultas. Cada paciente tem uma
# "assinatura" fisiológica própria (um deslocamento fixo nas medições), e
# só uma das oito medições carrega um sinal fraco da doença. O modelo
# precisa prever a doença de pacientes **novos**.

# %%
n_pac, n_cons = 200, 10
paciente = np.repeat(np.arange(n_pac), n_cons)
assinatura = rng.normal(0, 1.5, (n_pac, 8))                 # característica fixa de cada paciente
doente_pac = (rng.random(n_pac) < 0.4).astype(int)
Xg = assinatura[paciente] + rng.normal(0, 0.5, (n_pac * n_cons, 8))
Xg[:, 0] += 0.8 * doente_pac[paciente]                      # o único sinal "fisiológico", fraco
yg = doente_pac[paciente]

floresta = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=0)
auc_kfold = cross_val_score(floresta, Xg, yg, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                            scoring="roc_auc").mean()
auc_grupo = cross_val_score(floresta, Xg, yg, cv=GroupKFold(5), groups=paciente,
                            scoring="roc_auc").mean()
# "produção": 100 pacientes completamente novos
pac_novos = rng.normal(0, 1.5, (100, 8))
doente_novos = (rng.random(100) < 0.4).astype(int)
id_novo = np.repeat(np.arange(100), n_cons)
X_novo = pac_novos[id_novo] + rng.normal(0, 0.5, (1000, 8))
X_novo[:, 0] += 0.8 * doente_novos[id_novo]
y_novo = doente_novos[id_novo]
auc_prod = roc_auc_score(y_novo, floresta.fit(Xg, yg).predict_proba(X_novo)[:, 1])
print(f"CV com K-Fold estratificado: AUC = {auc_kfold:.3f}")
print(f"CV com GroupKFold          : AUC = {auc_grupo:.3f}")
print(f"'produção' (pacientes novos): AUC = {auc_prod:.3f}")

# %% [markdown]
# **Leitura esperada:** com K-Fold comum, as consultas de um paciente
# ficam dos dois lados; a floresta reconhece a assinatura do paciente e
# "lembra" seu diagnóstico — AUC quase perfeita. Com `GroupKFold`, a
# estimativa despenca para perto do que acontece com pacientes novos, em
# que só o sinal fisiológico fraco ajuda. A CV certa não melhora o modelo;
# ela para de mentir sobre ele.

# %% [markdown]
# ## 5. Tempo: o futuro no treino
#
# Vendas diárias com tendência e mudança de regime. O modelo prevê as
# vendas do dia a partir de features do próprio dia (dia da semana,
# promoção) e do índice de tempo.

# %%
n_dias = 1000
tempo = np.arange(n_dias)
dia_sem = tempo % 7
promo = (rng.random(n_dias) < 0.15).astype(int)
nivel = 100 + 0.08 * tempo + 25 * (tempo > 700)            # tendência + degrau no dia 700
vendas = nivel + 12 * (dia_sem >= 5) + 30 * promo + rng.normal(0, 8, n_dias)
Xt = np.column_stack([tempo, dia_sem, promo])
floresta_r = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0)
mae_embaralhado = -cross_val_score(floresta_r, Xt, vendas, cv=KFold(5, shuffle=True, random_state=0),
                                   scoring="neg_mean_absolute_error").mean()
mae_temporal = -cross_val_score(floresta_r, Xt, vendas, cv=TimeSeriesSplit(5),
                                scoring="neg_mean_absolute_error").mean()
# "produção": treinar até o dia 899 e prever os 100 dias seguintes
floresta_r.fit(Xt[:900], vendas[:900])
mae_prod = mean_absolute_error(vendas[900:], floresta_r.predict(Xt[900:]))
print(f"K-Fold embaralhado: MAE = {mae_embaralhado:.2f}")
print(f"TimeSeriesSplit   : MAE = {mae_temporal:.2f}")
print(f"'produção' (100 dias seguintes): MAE = {mae_prod:.2f}")

# %% [markdown]
# Com K-Fold embaralhado, cada dia de validação tem vizinhos no tempo no
# treino — a floresta interpola entre ontem e amanhã. Em produção, não
# existe amanhã no treino, e árvores não extrapolam tendências (tema 4). O
# `TimeSeriesSplit` reproduz essa dificuldade — e aqui chega a ser
# pessimista, porque seus primeiros folds treinam com pouco histórico e um
# deles atravessa o degrau do dia 700. Mas erra para o lado seguro; a
# estimativa embaralhada erra para o lado que custa caro.

# %% [markdown]
# ## O que levar deste notebook
#
# - A CV tem ruído de partição; a repetição o reduz, sem reduzir o ruído da
#   amostra.
# - Para comparar modelos na CV, use o teste t corrigido — o ingênuo
#   fabrica significância.
# - Grupos e tempo exigem esquemas próprios; a pergunta é sempre "a
#   validação imita a produção?".
#
# → Próximo: **Validação aninhada**.
