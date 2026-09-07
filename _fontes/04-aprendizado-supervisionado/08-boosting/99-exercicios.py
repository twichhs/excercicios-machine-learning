# %% [markdown]
# # Exercícios — Boosting
#
# **Tema:** Aprendizado Supervisionado › Boosting
#
# Resolva antes de olhar o gabarito. **Dificuldade:** 🟢 base · 🟡 aplicação ·
# 🔴 síntese

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score, log_loss
import xgboost as xgb

rng = np.random.default_rng(1616)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
AZUL, VERDE, VERMELHO, ROXO, AMBAR = "#1F5C8B", "#1E6B4F", "#9C2B2B", "#5B3E86", "#8A6100"
print("ambiente pronto")

# %% [markdown]
# ---
# ## Exercício 1 🟢 — O resíduo é o gradiente negativo
#
# Para $y=[3, 7, 2, 9]$ e uma previsão atual $F=[4, 5, 4, 8]$, calcule o
# resíduo $y - F$ (o gradiente negativo da perda quadrática) e confirme que
# uma árvore treinada para prever esse resíduo, somada a $F$, reduz o erro
# quadrático total.

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 1

# %%
y = np.array([3.0, 7.0, 2.0, 9.0])
F = np.array([4.0, 5.0, 4.0, 8.0])
residuo = y - F
print(f"resíduo (gradiente negativo): {residuo}")

erro_antes = np.sum((y - F) ** 2)
# uma "árvore perfeita" preveria exatamente o resíduo
F_depois = F + 1.0 * residuo  # eta=1.0, árvore perfeita
erro_depois = np.sum((y - F_depois) ** 2)
print(f"erro quadrático ANTES : {erro_antes:.2f}")
print(f"erro quadrático DEPOIS: {erro_depois:.2f}  (deveria ser 0 com árvore perfeita e eta=1)")

# %% [markdown]
# ---
# ## Exercício 2 🟢 — Comparando learning_rate x n_estimators para o MESMO ajuste
#
# Treine `GradientBoostingClassifier` com (a) `learning_rate=0.3,
# n_estimators=50` e (b) `learning_rate=0.03, n_estimators=500` (10x menor
# taxa, 10x mais árvores). Compare o desempenho no teste.

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 2

# %%
n = 2000
X = rng.normal(0, 1, (n, 6))
y = (rng.random(n) < 1 / (1 + np.exp(-(X @ rng.normal(0, 1, 6))))).astype(int)
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=0)

for nome, lr, n_est in [("rápido (lr=0.3, 50 árvores)", 0.3, 50),
                        ("lento (lr=0.03, 500 árvores)", 0.03, 500)]:
    modelo = GradientBoostingClassifier(
        learning_rate=lr, n_estimators=n_est, max_depth=3, random_state=0).fit(
        X_treino, y_treino)
    auc = roc_auc_score(y_teste, modelo.predict_proba(X_teste)[:, 1])
    print(f"{nome:<32s} AUC={auc:.4f}")

# %% [markdown]
# **Resposta esperada:** os dois costumam chegar a desempenho parecido — a
# ideia central de `teoria.pdf` de que $\eta$ pequeno + mais árvores é
# equivalente, em ajuste final, a $\eta$ maior + menos árvores, mas com uma
# trajetória de treino mais suave e geralmente mais robusta.

# %% [markdown]
# ---
# ## Exercício 3 🟡 — Profundidade da árvore em boosting: por que raso é melhor
#
# Compare `GradientBoostingClassifier` com `max_depth=1` (só um corte por
# árvore — os chamados *decision stumps*), `max_depth=3` e `max_depth=10`,
# todos com o mesmo `n_estimators`. Meça overfitting (gap treino-teste) em
# cada caso.

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 3

# %%
print(f"{'max_depth':>10s} {'AUC treino':>12s} {'AUC teste':>12s} {'gap':>8s}")
print("-" * 46)
for profundidade in [1, 3, 10]:
    modelo = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=profundidade,
        random_state=0).fit(X_treino, y_treino)
    auc_treino = roc_auc_score(y_treino, modelo.predict_proba(X_treino)[:, 1])
    auc_teste = roc_auc_score(y_teste, modelo.predict_proba(X_teste)[:, 1])
    print(f"{profundidade:>10d} {auc_treino:>12.4f} {auc_teste:>12.4f} "
          f"{auc_treino-auc_teste:>8.4f}")

# %% [markdown]
# **Resposta esperada:** `max_depth=10` deve mostrar o maior gap
# treino-teste (cada árvore já é complexa o suficiente para decorar padrões
# específicos do treino, e o boosting sequencial amplifica isso). Árvores
# rasas (`max_depth=1` a `3`) são "aprendizes fracos" de propósito — o ganho
# vem de combinar muitas delas em sequência, não de cada uma ser poderosa
# sozinha.

# %% [markdown]
# ---
# ## Exercício 4 🟡 — Early stopping evita overfitting mesmo com learning_rate alto
#
# Repita o experimento de "overfitting sem early stopping" do notebook-guia,
# mas agora com `learning_rate=0.5` (bem mais agressivo). Confirme que early
# stopping ainda encontra um ponto de parada razoável, mesmo com uma taxa de
# aprendizado que sozinha tenderia a overfitar rápido.

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 4

# %%
X_treino2, X_resto2, y_treino2, y_resto2 = train_test_split(X, y, test_size=0.4, random_state=1)
X_val2, X_teste2, y_val2, y_teste2 = train_test_split(
    X_resto2, y_resto2, test_size=0.5, random_state=1)

modelo_agressivo = xgb.XGBClassifier(
    n_estimators=500, learning_rate=0.5, max_depth=4, random_state=0,
    eval_metric="logloss", early_stopping_rounds=15)
modelo_agressivo.fit(X_treino2, y_treino2, eval_set=[(X_val2, y_val2)], verbose=False)

print(f"learning_rate=0.5: parou em {modelo_agressivo.best_iteration + 1} árvores "
      f"(de 500 permitidas)")
auc_agressivo = roc_auc_score(y_teste2, modelo_agressivo.predict_proba(X_teste2)[:, 1])
print(f"AUC no teste: {auc_agressivo:.4f}")

# %% [markdown]
# ---
# ## Exercício 5 🔴 — Construindo um relatório de comparação completo
#
# Para o desafio de classificação do tema (`_desafios/desafio-classificacao.csv`),
# construa um pipeline completo (tratamento de nulos, encoding, escalonamento
# quando fizer sentido) e compare pelo menos 3 modelos deste tema — por
# exemplo, regressão logística regularizada, Random Forest e XGBoost com
# early stopping — usando PR-AUC (tema 3, módulo 5, pela classe rara) como
# métrica principal. Reporte qual você recomendaria e por quê.

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 5

# %%
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score

caminho_desafio = Path("../_desafios/dados/desafio_classificacao.csv")
if caminho_desafio.exists():
    df = pd.read_csv(caminho_desafio)
    X = df.drop(columns=["id_cliente", "inadimplente"])
    y = df["inadimplente"]

    colunas_numericas = X.select_dtypes(include=np.number).columns.tolist()
    colunas_categoricas = X.select_dtypes(exclude=np.number).columns.tolist()

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=0)

    pre = ColumnTransformer([
        ("num", Pipeline([("imputa", SimpleImputer(strategy="median")),
                          ("escalona", StandardScaler())]), colunas_numericas),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), colunas_categoricas),
    ])

    modelos_finais = {
        "regressão logística (balanced)": Pipeline([
            ("preparo", pre),
            ("modelo", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]),
        "random forest (balanced)": Pipeline([
            ("preparo", pre),
            ("modelo", RandomForestClassifier(
                n_estimators=300, class_weight="balanced", random_state=0)),
        ]),
    }

    print(f"{'modelo':<32s} {'PR-AUC':>10s}")
    print("-" * 44)
    for nome, pipeline in modelos_finais.items():
        pipeline.fit(X_treino, y_treino)
        probs = pipeline.predict_proba(X_teste)[:, 1]
        pr_auc = average_precision_score(y_teste, probs)
        print(f"{nome:<32s} {pr_auc:>10.4f}")

    # XGBoost separado, porque lida nativamente com nulos e categóricas via
    # encoding numérico simples aqui (poderia usar enable_categorical=True)
    X_treino_num = pd.get_dummies(X_treino, columns=colunas_categoricas, drop_first=True)
    X_teste_num = pd.get_dummies(X_teste, columns=colunas_categoricas, drop_first=True)
    X_teste_num = X_teste_num.reindex(columns=X_treino_num.columns, fill_value=0)

    peso_classe_positiva = (y_treino == 0).sum() / (y_treino == 1).sum()
    modelo_xgb_final = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.1, max_depth=4, random_state=0,
        eval_metric="aucpr", scale_pos_weight=peso_classe_positiva,
    ).fit(X_treino_num, y_treino)
    pr_auc_xgb = average_precision_score(
        y_teste, modelo_xgb_final.predict_proba(X_teste_num)[:, 1])
    print(f"{'XGBoost (scale_pos_weight)':<32s} {pr_auc_xgb:>10.4f}")
else:
    print("Dataset do desafio não encontrado neste ambiente — rode o notebook "
          "'_desafios/desafio-classificacao.ipynb' primeiro.")

# %% [markdown]
# **Como avaliar sua resposta:** não existe um "vencedor" universal
# garantido — o que importa é o processo: (1) split antes de qualquer
# preparo, (2) `Pipeline` para evitar vazamento, (3) tratamento explícito do
# desbalanceamento (`class_weight` ou `scale_pos_weight`), (4) métrica
# adequada à classe rara (PR-AUC, não acurácia), e (5) uma recomendação que
# leve em conta não só o desempenho, mas o custo de manutenção e a
# interpretabilidade exigida pelo caso de uso.

# %% [markdown]
# ---
# ## Fechamento
#
# - O resíduo, para perda quadrática, é literalmente o gradiente negativo —
#   a base de todo o algoritmo de Gradient Boosting.
# - Taxa de aprendizado pequena com mais árvores tende ao mesmo ajuste que
#   taxa grande com menos árvores, com uma trajetória mais estável.
# - Árvores rasas (aprendizes fracos) são intencionais em boosting — a força
#   vem da sequência, não de cada árvore isolada.
# - Early stopping com validação separada é o mecanismo central que torna
#   boosting seguro de usar com muitas árvores disponíveis.
# - Um pipeline honesto de comparação de modelos exige toda a disciplina dos
#   temas 2 e 3 combinada — não existe atalho para uma comparação confiável.
#
# → Este é o fim do tema **Aprendizado Supervisionado**. Os próximos temas
# tratam de aprendizado sem rótulo (tema 5), da disciplina de validação que
# sustenta todo número reportado até aqui (tema 6), e das arquiteturas mais
# profundas que constroem sobre estes fundamentos (temas 7 e 8).
