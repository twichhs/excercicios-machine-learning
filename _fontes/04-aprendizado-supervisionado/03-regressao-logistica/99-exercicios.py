# %% [markdown]
# # Exercícios — Regressão Logística
#
# **Tema:** Aprendizado Supervisionado › Regressão Logística
#
# Resolva antes de olhar o gabarito. **Dificuldade:** 🟢 base · 🟡 aplicação ·
# 🔴 síntese

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score

rng = np.random.default_rng(909)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
AZUL, VERDE, VERMELHO, ROXO, AMBAR = "#1F5C8B", "#1E6B4F", "#9C2B2B", "#5B3E86", "#8A6100"
print("ambiente pronto")

# %% [markdown]
# ---
# ## Exercício 1 🟢 — Sigmoide e logit na mão
#
# Calcule, sem usar nenhuma função pronta do scipy/sklearn: (a) a
# probabilidade correspondente a um logit de 1,5; (b) o odds ratio
# correspondente a um coeficiente de 0,4; (c) o logit correspondente a uma
# probabilidade de 0,2.

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 1

# %%
import math

logit_a = 1.5
prob_a = 1 / (1 + math.exp(-logit_a))
print(f"(a) probabilidade para logit=1,5: {prob_a:.4f}")

coef_b = 0.4
odds_ratio_b = math.exp(coef_b)
print(f"(b) odds ratio para coeficiente=0,4: {odds_ratio_b:.4f}")

prob_c = 0.2
logit_c = math.log(prob_c / (1 - prob_c))
print(f"(c) logit para probabilidade=0,2: {logit_c:.4f}")

# %% [markdown]
# ---
# ## Exercício 2 🟢 — "Dobrar as chances" não é "dobrar a probabilidade"
#
# Para um evento com probabilidade de base de 0,10, calcule a nova
# probabilidade se as chances (odds) dobrarem. Depois calcule para uma
# probabilidade de base de 0,50. Compare o quanto a probabilidade mudou em
# cada caso.

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 2

# %%
def nova_probabilidade(p0, razao_odds):
    odds0 = p0 / (1 - p0)
    odds1 = odds0 * razao_odds
    return odds1 / (1 + odds1)


for p0 in [0.10, 0.50]:
    p1 = nova_probabilidade(p0, 2.0)
    print(f"base={p0:.2f}  ->  nova probabilidade={p1:.4f}  "
          f"(mudança de {p1-p0:+.4f}, NÃO de +{p0:.2f})")

# %% [markdown]
# **Resposta:** dobrar as chances a partir de 10% leva a ~18% (mudança de
# +8pp, não +10pp que seria "dobrar"). A partir de 50%, leva a ~67% (mudança
# de +17pp). Em nenhum caso a probabilidade dobra — só as chances dobram, por
# definição da operação.

# %% [markdown]
# ---
# ## Exercício 3 🟡 — Detectando separação perfeita
#
# Gere um dataset onde uma feature separa perfeitamente as classes (todos os
# valores acima de um limiar são classe 1, abaixo são classe 0, sem
# sobreposição). Ajuste `LogisticRegression(C=np.inf)` (sem penalidade) e
# observe o coeficiente. Repita com `C=1.0` (penalidade L2 padrão) e compare
# a magnitude.

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 3

# %%
x = np.concatenate([rng.normal(-5, 1, 100), rng.normal(5, 1, 100)])
y = np.concatenate([np.zeros(100), np.ones(100)])

modelo_sem_penalidade = LogisticRegression(C=np.inf, max_iter=1000).fit(x.reshape(-1, 1), y)
modelo_regularizado = LogisticRegression(C=1.0).fit(x.reshape(-1, 1), y)

print(f"coeficiente SEM penalidade: {modelo_sem_penalidade.coef_[0][0]:.2f}   "
      f"(convergiu em {modelo_sem_penalidade.n_iter_[0]} iterações)")
print(f"coeficiente COM penalidade L2 (C=1.0): {modelo_regularizado.coef_[0][0]:.4f}")

# %% [markdown]
# **Resposta esperada:** sem penalidade, o coeficiente sai bem maior que com
# L2 (2,7 contra 1,4, aproximadamente). Note que o solver (`lbfgs`) para em
# poucas iterações mesmo sem penalidade — ele "acha" que convergiu porque o
# gradiente numérico chega a zero por *underflow* de ponto flutuante muito
# antes do verdadeiro ótimo, que é infinito. O valor de 2,7 reportado não é
# "o" coeficiente correto — é só onde a precisão numérica do solver desistiu
# de melhorar. O notebook-guia (`01-logistica-do-zero`) mostra isso de forma
# mais crua: gradiente descendente manual, sem critério de parada
# inteligente, faz o coeficiente crescer indefinidamente enquanto você deixar
# rodar. Com L2, o coeficiente fica estável e bem definido, ao custo de um
# pouco de viés — a razão prática pela qual regularização é a resposta
# padrão a esse problema, não "rodar por mais iterações".

# %% [markdown]
# ---
# ## Exercício 4 🟡 — Comparando limiares diferentes de decisão
#
# Usando um modelo treinado num problema desbalanceado (~8% positivos),
# calcule precisão e recall nos limiares 0,3, 0,5 e 0,7. Confirme que a soma
# de precisão e recall não segue um padrão fixo — cada limiar tem seu próprio
# trade-off (tema 3, módulo 5).

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 4

# %%
n = 3000
x1 = rng.normal(0, 1, n)
x2 = rng.normal(0, 1, n)
logit = -2.5 + 1.5 * x1 - 1.0 * x2
prob = 1 / (1 + np.exp(-logit))
y = (rng.random(n) < prob).astype(int)

X_treino, X_teste, y_treino, y_teste = train_test_split(
    np.column_stack([x1, x2]), y, test_size=0.3, stratify=y, random_state=0)
modelo = LogisticRegression().fit(X_treino, y_treino)
probs_teste = modelo.predict_proba(X_teste)[:, 1]

print(f"taxa de positivos no teste: {y_teste.mean():.2%}\n")
print(f"{'limiar':>8s} {'precisão':>10s} {'recall':>10s}")
print("-" * 30)
for limiar in [0.3, 0.5, 0.7]:
    pred_limiar = (probs_teste >= limiar).astype(int)
    print(f"{limiar:>8.1f} {precision_score(y_teste, pred_limiar, zero_division=0):>10.4f} "
          f"{recall_score(y_teste, pred_limiar, zero_division=0):>10.4f}")

# %% [markdown]
# ---
# ## Exercício 5 🔴 — Auditoria: o coeficiente parece grande demais
#
# Um colega treina um modelo de churn e reporta: "achei uma feature com odds
# ratio de 45 — clientes com essa característica têm 45x mais chance de
# cancelar!". Investigue: construa um cenário onde isso acontece por
# vazamento de dados (tema 3) em vez de um efeito real, e mostre como
# diagnosticar a diferença.

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 5

# %%
n = 2000
uso_mensal = rng.normal(50, 15, n)
churn = (rng.random(n) < (0.15 - 0.002 * uso_mensal)).astype(int)
churn = np.clip(churn, 0, 1)
# vazamento: feature só é preenchida (valor alto) DEPOIS que o cliente já cancelou
dias_desde_ultimo_acesso = np.where(churn == 1, rng.uniform(60, 120, n), rng.uniform(0, 30, n))

X_com_vazamento = np.column_stack([uso_mensal, dias_desde_ultimo_acesso])
modelo_vazado = LogisticRegression().fit(X_com_vazamento, churn)
odds_ratios = np.exp(modelo_vazado.coef_[0])

print(f"odds ratio de 'uso_mensal'               : {odds_ratios[0]:.4f}")
print(f"odds ratio de 'dias_desde_ultimo_acesso'  : {odds_ratios[1]:.4f}")

auc_com_vazamento = roc_auc_score(churn, modelo_vazado.predict_proba(X_com_vazamento)[:, 1])
modelo_sem_vazamento = LogisticRegression().fit(uso_mensal.reshape(-1, 1), churn)
auc_sem_vazamento = roc_auc_score(
    churn, modelo_sem_vazamento.predict_proba(uso_mensal.reshape(-1, 1))[:, 1])

print(f"\nAUC COM a feature suspeita : {auc_com_vazamento:.4f}")
print(f"AUC SEM a feature suspeita : {auc_sem_vazamento:.4f}")

# %% [markdown]
# **Diagnóstico:** um odds ratio muito alto (dezenas ou centenas) é motivo de
# suspeita, não de comemoração — a pergunta certa é sempre "essa feature
# existiria, com esse valor, no momento real da previsão?". Aqui,
# `dias_desde_ultimo_acesso` só assume valores altos **depois** do
# cancelamento — é o alvo disfarçado. O sintoma combinado (odds ratio
# extremo + AUC excessivamente alto) é o padrão a procurar, exatamente como
# no tema 3, módulo 4.

# %% [markdown]
# ---
# ## Fechamento
#
# - Sigmoide e logit são as duas metades de uma mesma transformação — vale a
#   pena calculá-las na mão até virarem intuição.
# - "As chances mudaram X%" nunca é o mesmo que "a probabilidade mudou X%" —
#   a diferença é maior perto de 50% e menor nos extremos.
# - Separação perfeita quebra o ajuste sem regularização — coeficientes
#   enormes e avisos de convergência são o sintoma.
# - O limiar de decisão muda completamente o trade-off entre precisão e
#   recall — escolha-o com intenção, não por padrão.
# - Odds ratio extremo é um sinal de alerta de vazamento até prova em
#   contrário, do mesmo jeito que um AUC excessivamente alto.
#
# → Próximo módulo: **k-NN e Naive Bayes**, dois modelos com filosofias
# opostas de como aprender com os dados.
