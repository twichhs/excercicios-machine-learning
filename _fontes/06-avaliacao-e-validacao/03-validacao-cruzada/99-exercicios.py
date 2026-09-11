# %% [markdown]
# # Exercícios — Validação Cruzada
#
# **Tema:** Avaliação e Validação de Modelos › Validação Cruzada
#
# Resolva antes de olhar o gabarito. **Dificuldade:** 🟢 base · 🟡 aplicação ·
# 🔴 síntese

# %%
import numpy as np
import pandas as pd
from scipy.stats import t as t_student
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (StratifiedKFold, GroupKFold, StratifiedGroupKFold,
                                     RepeatedStratifiedKFold, GridSearchCV, cross_val_score)
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(636)
print("ambiente pronto")

# %% [markdown]
# ---
# ## Dataset de trabalho: atrito de funcionários em uma rede de lojas
#
# 3.000 registros mensais de 600 funcionários (5 meses cada), em 40 lojas.
# O alvo `saiu` indica se o funcionário pediu demissão nos 3 meses
# seguintes. Algumas lojas têm gerentes difíceis (atrito alto); cada
# funcionário tem características próprias estáveis. Há também 200 colunas
# `pesquisa_*` de uma pesquisa de clima — quase todas sem relação nenhuma
# com o alvo.

# %%
n_func, n_meses, n_lojas = 600, 5, 40
func = np.repeat(np.arange(n_func), n_meses)
loja_do_func = rng.integers(0, n_lojas, n_func)
loja = loja_do_func[func]
efeito_loja = rng.normal(0, 0.8, n_lojas)
perfil = rng.normal(0, 1, (n_func, 4))                     # traços estáveis de cada funcionário
salario_rel = perfil[:, 0][func] + rng.normal(0, 0.3, len(func))
horas_extras = np.clip(10 + 6 * perfil[:, 1][func] + rng.normal(0, 3, len(func)), 0, None)
tempo_casa = np.clip(24 + 12 * perfil[:, 2][func] + np.tile(np.arange(n_meses), n_func), 1, None)
avaliacao = perfil[:, 3][func] + rng.normal(0, 0.5, len(func))
logit = (-1.6 - 0.7 * salario_rel + 0.06 * (horas_extras - 10) - 0.02 * (tempo_casa - 24)
         + efeito_loja[loja])
risco_func = 1 / (1 + np.exp(-logit))
saiu_func = (rng.random(n_func) < pd.Series(risco_func).groupby(func).mean().to_numpy()).astype(int)
saiu = saiu_func[func]                                     # o desfecho é do funcionário
pesquisa = rng.normal(0, 1, (len(func), 200))
X = np.column_stack([salario_rel, horas_extras, tempo_casa, avaliacao, pesquisa])
colunas = ["salario_rel", "horas_extras", "tempo_casa", "avaliacao"] + [f"pesquisa_{i}" for i in range(200)]
print(f"{len(func)} registros | {n_func} funcionários | {n_lojas} lojas | taxa de saída: {saiu.mean():.1%}")

# %% [markdown]
# ---
# ## Exercício 1 🟢 — O teste t corrigido na mão
#
# Numa comparação em 5-fold repetido 4 vezes ($J = 20$), o modelo A supera o
# B em AUC por $\bar{d} = 0{,}008$, com $s_d = 0{,}015$. Calcule o t e o
# p-valor do teste ingênuo e do corrigido de Nadeau e Bengio
# ($n_{te}/n_{tr} = 1/4$). Conclua.

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 1

# %%
J, d_bar, s_d, razao = 20, 0.008, 0.015, 1 / 4
t_ing = d_bar / (s_d / np.sqrt(J))
t_cor = d_bar / np.sqrt((1 / J + razao) * s_d ** 2)
for nome, tt in [("ingênuo", t_ing), ("corrigido", t_cor)]:
    print(f"{nome:>9s}: t = {tt:.2f}, p-valor bilateral = {2 * t_student.sf(abs(tt), J - 1):.4f}")

# %% [markdown]
# **Por quê:** o ingênuo usa $s_d/\sqrt{20} \approx 0{,}0034$ e dá $t \approx 2{,}39$
# (p ≈ 0,03, "significativo"). O corrigido usa
# $\sqrt{(0{,}05 + 0{,}25) \times 0{,}015^2} \approx 0{,}0082$ e dá $t \approx 0{,}98$
# (p ≈ 0,34). Repare que o termo $n_{te}/n_{tr} = 0{,}25$ é cinco vezes
# maior que $1/J$: é ele que domina, e ele não diminui com mais repetições.
# A vantagem de A não está demonstrada.

# %% [markdown]
# ---
# ## Exercício 2 🟢 — Quem é o "grupo" aqui?
#
# Estime a AUC de uma Random Forest com três esquemas: (a) `StratifiedKFold`
# por registro; (b) `GroupKFold` por **funcionário**; (c) `GroupKFold` por
# **loja**. Use só as 4 primeiras colunas. Qual esquema corresponde a qual
# pergunta de negócio?

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 2

# %%
X4 = X[:, :4]
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=-1)
esquemas = {"(a) registro": (StratifiedKFold(5, shuffle=True, random_state=0), None),
            "(b) funcionário": (GroupKFold(5), func),
            "(c) loja": (GroupKFold(5), loja)}
for nome, (cv, grupos) in esquemas.items():
    auc = cross_val_score(rf, X4, saiu, cv=cv, groups=grupos, scoring="roc_auc").mean()
    print(f"{nome:>16s}: AUC = {auc:.3f}")

# %% [markdown]
# **Por quê:** (a) valida em meses de funcionários que o modelo já viu
# (os outros 4 meses do mesmo funcionário estão no treino, com o mesmo
# desfecho) — a floresta reconhece o perfil e "lembra" se ele saiu: AUC
# inflada, sem pergunta de negócio correspondente. (b) responde "como o
# modelo se sai em funcionários **novos** das lojas que já conhecemos?" —
# a pergunta de quem vai usar o modelo na rede atual. (c) responde "como se
# sai numa loja **nova**?" — a pergunta de quem vai expandir a rede. (c)
# costuma ser a mais baixa, porque o efeito da loja (o gerente) não pode
# ser aprendido para uma loja que o modelo nunca viu.

# %% [markdown]
# ---
# ## Exercício 3 🟡 — Selecionando perguntas da pesquisa
#
# Um analista selecionou as 10 colunas `pesquisa_*` mais associadas ao
# alvo (teste F) usando a base inteira, juntou com as 4 colunas de RH e
# estimou a AUC de uma regressão logística com `GroupKFold` por
# funcionário. (a) Reproduza o resultado dele. (b) Refaça com a seleção
# dentro do `Pipeline`. (c) Compare com o modelo só com as 4 colunas de RH.
# Alguma pergunta da pesquisa ajuda de fato?

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 3

# %%
gkf = GroupKFold(5)
Xp = X[:, 4:]
sel = SelectKBest(f_classif, k=10).fit(Xp, saiu)                        # ERRADO: usa todos os rótulos
X_errado = np.column_stack([X4, sel.transform(Xp)])
auc_errado = cross_val_score(LogisticRegression(max_iter=3000), X_errado, saiu, cv=gkf,
                             groups=func, scoring="roc_auc").mean()

from sklearn.compose import ColumnTransformer
pre = ColumnTransformer([("rh", "passthrough", list(range(4))),
                         ("pesq", SelectKBest(f_classif, k=10), list(range(4, X.shape[1])))])
auc_certo = cross_val_score(make_pipeline(pre, LogisticRegression(max_iter=3000)), X, saiu, cv=gkf,
                            groups=func, scoring="roc_auc").mean()
auc_rh = cross_val_score(LogisticRegression(max_iter=3000), X4, saiu, cv=gkf, groups=func,
                         scoring="roc_auc").mean()
print(f"(a) seleção com a base inteira : AUC = {auc_errado:.3f}")
print(f"(b) seleção dentro do Pipeline : AUC = {auc_certo:.3f}")
print(f"(c) só as 4 colunas de RH      : AUC = {auc_rh:.3f}")

# %% [markdown]
# **Por quê:** as 200 perguntas são ruído, mas com 600 funcionários algumas
# se correlacionam com a saída por acaso. Selecioná-las com todos os rótulos
# embute nelas a informação dos folds de validação, e a AUC sobe. Com a
# seleção dentro do `Pipeline`, a vantagem some — e o modelo com as 10
# perguntas fica **igual ou pior** que o só de RH, porque adiciona ruído.
# A resposta honesta à pergunta do analista é "nenhuma pergunta da pesquisa
# ajuda". Note que o vazamento aparece mesmo com `GroupKFold`: o esquema de
# divisão estava certo; o erro estava no que aconteceu **antes** dele.

# %% [markdown]
# ---
# ## Exercício 4 🟡 — O otimismo do tuning
#
# Faça uma busca em grade para um `HistGradientBoostingClassifier`
# (`learning_rate` em [0,03; 0,1; 0,3], `max_leaf_nodes` em [7; 31],
# `min_samples_leaf` em [5; 50]) com `GroupKFold` por funcionário, usando
# as 4 colunas de RH. Compare o `best_score_` com a validação aninhada (laço
# externo também por funcionário). Quanto do número "melhor" era seleção?

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 4

# %%
grade = {"learning_rate": [0.03, 0.1, 0.3], "max_leaf_nodes": [7, 31], "min_samples_leaf": [5, 50]}
hgb = HistGradientBoostingClassifier(max_iter=150, random_state=0)
busca = GridSearchCV(hgb, grade, cv=GroupKFold(5), scoring="roc_auc")
busca.fit(X4, saiu, groups=func)
print(f"best_score_ da busca: {busca.best_score_:.4f} com {busca.best_params_}")

externos = []
for tr, te in GroupKFold(5).split(X4, saiu, groups=func):
    b = GridSearchCV(hgb, grade, cv=GroupKFold(4), scoring="roc_auc")
    b.fit(X4[tr], saiu[tr], groups=func[tr])
    externos.append(roc_auc_score(saiu[te], b.predict_proba(X4[te])[:, 1]))
print(f"validação aninhada  : {np.mean(externos):.4f} (folds: {np.round(externos, 3)})")
print(f"otimismo do best_score_: {busca.best_score_ - np.mean(externos):+.4f}")

# %% [markdown]
# **Por quê:** com 12 configurações, a vencedora é, em parte, a que o ruído
# favoreceu; a aninhada estima o procedimento completo "boosting + busca".
# O otimismo aqui é pequeno (poucas configurações e um sinal real), da
# ordem de centésimos de AUC — mas cresce com o tamanho da grade. Note um
# detalhe de implementação: com grupos, **os dois laços** precisam respeitar
# os grupos, e `GridSearchCV.fit` recebe `groups=` para repassá-los ao
# `GroupKFold` interno. Um laço interno sem grupos reintroduziria o
# vazamento por funcionário na escolha dos hiperparâmetros.

# %% [markdown]
# ---
# ## Exercício 5 🔴 — O protocolo de validação para o comitê
#
# O comitê de RH quer usar o modelo para priorizar conversas de retenção
# nas lojas **atuais**, e perguntou: "o boosting é melhor que a regressão
# logística?". Monte o protocolo completo e responda: esquema de divisão
# justificado, repetição para reduzir o ruído de partição (use
# `StratifiedGroupKFold` com sementes diferentes), comparação pareada das
# AUCs com o teste t corrigido e a recomendação final.

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 5

# %%
from sklearn.preprocessing import StandardScaler
logistica = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
boosting = HistGradientBoostingClassifier(max_iter=150, random_state=0, **busca.best_params_)
d, n_te_sobre_n_tr = [], 1 / 4
for semente in range(4):
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=semente)
    for tr, te in cv.split(X4, saiu, groups=func):
        a_log = roc_auc_score(saiu[te], logistica.fit(X4[tr], saiu[tr]).predict_proba(X4[te])[:, 1])
        a_gb = roc_auc_score(saiu[te], boosting.fit(X4[tr], saiu[tr]).predict_proba(X4[te])[:, 1])
        d.append(a_gb - a_log)
d = np.array(d)
J = len(d)
t_cor = d.mean() / np.sqrt((1 / J + n_te_sobre_n_tr) * d.var(ddof=1))
t_ing = d.mean() / (d.std(ddof=1) / np.sqrt(J))
print(f"diferença média de AUC (boosting - logística): {d.mean():+.4f} em {J} folds "
      f"(boosting vence em {np.mean(d > 0):.0%} deles)")
print(f"t ingênuo = {t_ing:.2f} (p = {2 * t_student.sf(abs(t_ing), J - 1):.3f}) | "
      f"t corrigido = {t_cor:.2f} (p = {2 * t_student.sf(abs(t_cor), J - 1):.3f})")

# %% [markdown]
# **Como avaliar sua resposta:** o esquema é por **funcionário**
# (`StratifiedGroupKFold`), porque o uso é em funcionários novos das lojas
# atuais — não por loja (que responderia a outra pergunta) nem por registro
# (que vaza). A repetição com 4 sementes reduz o ruído de partição, e a
# comparação é **pareada** (os dois modelos nos mesmos folds). O teste t
# corrigido é o que decide: se o p-valor corrigido não for pequeno, a
# resposta ao comitê é "não há evidência de que o boosting seja melhor" — e
# a recomendação pende para a logística, que é mais simples de explicar a um
# comitê de RH e de auditar quanto a vieses (tema 11). Uma boa resposta
# também registra o que **não** foi feito: as perguntas da pesquisa ficaram
# de fora porque o exercício 3 mostrou que não ajudam, e os hiperparâmetros
# do boosting vieram de uma busca com grupos. Se a diferença fosse
# significativa, a recomendação ainda deveria traduzir o ganho de AUC em
# conversas de retenção a mais por mês — a métrica que o comitê entende.

# %% [markdown]
# ---
# ## Fechamento
#
# - O termo $n_{te}/n_{tr}$ do teste corrigido domina e não encolhe com
#   repetições.
# - "Qual é o grupo?" é uma pergunta de negócio: funcionário novo e loja
#   nova são validações diferentes.
# - Seleção com todos os rótulos fabrica sinal, mesmo com o esquema de
#   divisão certo.
# - O melhor escore de uma busca é otimista; com grupos, os dois laços da
#   aninhada respeitam os grupos.
# - Um protocolo defensável: esquema que imita a produção, repetição,
#   comparação pareada, teste corrigido, métrica de negócio.
#
# → Próximo módulo: **Viés, Variância e Curvas de Aprendizado**.
