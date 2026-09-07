# %% [markdown]
# # XGBoost e LightGBM
#
# **Tema:** Aprendizado Supervisionado › Boosting
#
# Este notebook compara `sklearn.GradientBoostingClassifier`, `XGBoost` e
# `LightGBM` no mesmo problema — velocidade de treino, desempenho, e o
# comportamento nativo com valores faltantes que nenhum dos modelos
# anteriores deste tema oferece.

# %%
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb

rng = np.random.default_rng(55)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
AZUL, VERDE, VERMELHO, ROXO, AMBAR = "#1F5C8B", "#1E6B4F", "#9C2B2B", "#5B3E86", "#8A6100"
print("pronto")

# %% [markdown]
# ## 1. Um dataset de tamanho moderado, para comparar velocidade e desempenho

# %%
n = 6000
X_informativo = rng.normal(0, 1, (n, 8))
beta = rng.normal(0, 1.2, 8)
logit = X_informativo @ beta
y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
X = np.column_stack([X_informativo, rng.normal(0, 1, (n, 4))])  # + 4 features de ruído

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=0)

# %% [markdown]
# ## 2. Comparando tempo de treino e desempenho

# %%
resultados = {}

inicio = time.perf_counter()
modelo_sklearn = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=3, random_state=0).fit(X_treino, y_treino)
tempo_sklearn = time.perf_counter() - inicio
auc_sklearn = roc_auc_score(y_teste, modelo_sklearn.predict_proba(X_teste)[:, 1])
resultados["sklearn GBM"] = (tempo_sklearn, auc_sklearn)

inicio = time.perf_counter()
modelo_xgb = xgb.XGBClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=3, random_state=0,
    eval_metric="logloss").fit(X_treino, y_treino)
tempo_xgb = time.perf_counter() - inicio
auc_xgb = roc_auc_score(y_teste, modelo_xgb.predict_proba(X_teste)[:, 1])
resultados["XGBoost"] = (tempo_xgb, auc_xgb)

inicio = time.perf_counter()
modelo_lgb = lgb.LGBMClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=3, random_state=0, verbosity=-1
).fit(X_treino, y_treino)
tempo_lgb = time.perf_counter() - inicio
auc_lgb = roc_auc_score(y_teste, modelo_lgb.predict_proba(X_teste)[:, 1])
resultados["LightGBM"] = (tempo_lgb, auc_lgb)

print(f"{'modelo':<14s} {'tempo de treino (s)':>20s} {'ROC-AUC':>10s}")
print("-" * 46)
for nome, (tempo, auc) in resultados.items():
    print(f"{nome:<14s} {tempo:>20.3f} {auc:>10.4f}")

# %% [markdown]
# **Leitura esperada:** XGBoost e LightGBM tendem a treinar mais rápido que
# o Gradient Boosting do `sklearn` no mesmo número de árvores — o efeito da
# busca de corte por histograma do `teoria.pdf` — com desempenho comparável
# ou melhor, graças à regularização e ao boosting de segunda ordem.

# %% [markdown]
# ## 3. Tratamento nativo de valores faltantes
#
# Diferente de todos os modelos anteriores deste tema (que exigem imputação
# explícita, tema 3), XGBoost e LightGBM aprendem, durante o treino, para
# que lado do corte mandar um valor ausente.

# %%
X_com_nulos = X_treino.copy()
mascara_nulos = rng.random(X_com_nulos.shape) < 0.15  # 15% de nulos espalhados
X_com_nulos[mascara_nulos] = np.nan

print(f"fração de nulos no dataset: {np.isnan(X_com_nulos).mean():.2%}")

# XGBoost e LightGBM aceitam NaN diretamente — sem imputação (tema 3)!
modelo_xgb_com_nulos = xgb.XGBClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=3, random_state=0,
    eval_metric="logloss").fit(X_com_nulos, y_treino)
auc_xgb_com_nulos = roc_auc_score(
    y_teste, modelo_xgb_com_nulos.predict_proba(X_teste)[:, 1])

print(f"AUC do XGBoost treinado DIRETAMENTE com 15% de nulos (sem imputar): "
      f"{auc_xgb_com_nulos:.4f}")
print(f"AUC do XGBoost original (sem nulos, para referência)              : "
      f"{auc_xgb:.4f}")

# %% [markdown]
# ## 4. Importância de features: o mesmo cuidado do módulo anterior

# %%
importancia_xgb = pd.Series(
    modelo_xgb.feature_importances_,
    index=[f"informativa_{i}" for i in range(8)] + [f"ruido_{i}" for i in range(4)]
).sort_values(ascending=False)

print("importância de features (XGBoost, ganho médio):")
print(importancia_xgb.round(4).to_string())
print("\nAs 8 primeiras (informativas) deveriam dominar o ranking sobre as 4 de ruído —")
print("mas, como no módulo anterior, vale confirmar com importância por permutação")
print("antes de qualquer decisão de negócio, não confiar cegamente neste ranking.")

# %% [markdown]
# ## O que levar deste notebook
#
# - XGBoost e LightGBM entregam desempenho igual ou melhor que Gradient
#   Boosting clássico, tipicamente com treino mais rápido — o efeito
#   mensurável dos avanços de engenharia do `teoria.pdf`.
# - Tratamento nativo de nulos é uma vantagem prática real: dá para treinar
#   direto em dados com valores faltantes, sem o pipeline de imputação do
#   tema 3 — embora entender POR QUE os dados faltam continue importante.
# - Importância de features em boosting merece o mesmo ceticismo do módulo
#   anterior — confirme com permutação antes de agir sobre o ranking.
#
# → Próximo: **Tuning e early stopping**, fechando o módulo com o fluxo
# correto de ajuste de hiperparâmetros.
