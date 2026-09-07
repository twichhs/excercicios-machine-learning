# %% [markdown]
# # Desafio — Regressão
#
# **Tema:** Aprendizado Supervisionado
#
# Como o desafio de classificação: este notebook só entrega um dataset
# artificial e sujo, sem gabarito. Volte a ele depois de cada módulo de
# regressão (linear, Ridge/Lasso/ElasticNet, árvores, Random Forest,
# boosting) e compare o desempenho de cada modelo nos mesmos dados.
#
# ## O problema de negócio (fictício)
#
# Uma seguradora quer prever `custo_sinistro` (o valor, em reais, que um
# sinistro de seguro residencial vai custar) a partir de características do
# imóvel e do segurado. O alvo é contínuo, positivo, e com cauda longa
# (a maioria dos sinistros é barata; poucos são catastroficamente caros).
#
# ## O que o dataset esconde de propósito
#
# - **Alvo com cauda pesada** — considere se/quando transformar (log?).
# - **Relação não-linear** entre pelo menos uma feature e o alvo.
# - **Outliers genuínos** em `valor_imovel` (imóveis de altíssimo padrão).
# - **Dados faltantes** em duas colunas, mecanismos diferentes.
# - **Categórica com uma categoria rara.**
# - **Duas features colineares.**
# - **Escalas muito diferentes** entre as features numéricas.
# - **Colunas de ruído puro.**
# - **Um identificador** que não é feature.
#
# ## Sugestão de roteiro (não obrigatório)
#
# 1. Rode a célula abaixo — ela salva uma cópia em
#    `_desafios/dados/desafio_regressao.csv`.
# 2. EDA, tratamento de nulos/outliers, feature engineering, encoding e
#    escalonamento (tema 3) antes de qualquer modelo.
# 3. Treine o modelo do módulo que você acabou de estudar e avalie com MAE,
#    RMSE e R² (tema 6 aprofunda essas métricas) — e não esqueça de checar se
#    o alvo pede uma transformação antes de modelar.

# %%
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(7)

n = 6000

# --- variáveis informativas de verdade ---------------------------------
area_construida_m2 = rng.gamma(shape=5, scale=25, size=n).clip(30, None)
idade_imovel_anos = rng.exponential(15, n).clip(0, 90)
valor_imovel = (area_construida_m2 * rng.normal(3800, 700, n)).clip(50000, None)
# imóveis de altíssimo padrão -- outliers REAIS, não erro
idx_alto_padrao = rng.choice(n, size=int(0.01 * n), replace=False)
valor_imovel[idx_alto_padrao] *= rng.uniform(5, 12, len(idx_alto_padrao))

distancia_corpo_bombeiros_km = rng.exponential(6, n).clip(0.2, 60)
n_sinistros_anteriores = rng.poisson(0.3, n)

# --- feature colinear (quase redundante) ----------------------------------
area_construida_m2_v2 = area_construida_m2 * rng.uniform(0.97, 1.03, n)

# --- categóricas ------------------------------------------------------------
tipo_imovel = rng.choice(["casa", "apartamento", "sobrado", "kitnet", "cobertura_duplex"],
                         n, p=[0.42, 0.38, 0.12, 0.07, 0.01])
regiao = rng.choice(["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"],
                    n, p=[0.08, 0.20, 0.10, 0.42, 0.20])

# --- ruído puro -------------------------------------------------------------
ruido_1 = rng.normal(0, 1, n)
ruido_2 = rng.integers(0, 4, n)

# --- identificador (NÃO é feature) ------------------------------------------
id_apolice = np.arange(50000, 50000 + n)

# --- alvo: combinação REAL e NÃO-LINEAR das variáveis informativas ----------
risco_base = (
    120                                                 # custo administrativo mínimo de qualquer sinistro
    + 0.02 * valor_imovel / 1000                        # em milhares de reais do imóvel
    + 800 * n_sinistros_anteriores ** 1.8               # efeito não-linear, forte
    - 40 * np.log1p(distancia_corpo_bombeiros_km)       # quanto mais perto, mais caro reparar rápido (efeito fictício)
    + 15 * idade_imovel_anos
)
custo_sinistro = np.exp(rng.normal(np.log1p(risco_base.clip(1, None)), 0.5, n)) - 1
custo_sinistro = custo_sinistro.clip(30, None).round(2)

df = pd.DataFrame({
    "id_apolice": id_apolice,
    "area_construida_m2": area_construida_m2.round(1),
    "area_construida_m2_v2": area_construida_m2_v2.round(1),
    "idade_imovel_anos": idade_imovel_anos.round(1),
    "valor_imovel": valor_imovel.round(2),
    "distancia_corpo_bombeiros_km": distancia_corpo_bombeiros_km.round(2),
    "n_sinistros_anteriores": n_sinistros_anteriores,
    "tipo_imovel": tipo_imovel,
    "regiao": regiao,
    "feature_ruido_1": ruido_1.round(4),
    "feature_ruido_2": ruido_2,
    "custo_sinistro": custo_sinistro,
})

# --- nulos: um padrão provavelmente MCAR, outro provavelmente MAR ---------
idx_nulo_mcar = rng.choice(n, size=int(0.05 * n), replace=False)
df.loc[idx_nulo_mcar, "idade_imovel_anos"] = np.nan

prob_nulo_mar = 0.02 + 0.3 * (df["tipo_imovel"] == "cobertura_duplex")
idx_nulo_mar = rng.random(n) < prob_nulo_mar
df.loc[idx_nulo_mar, "distancia_corpo_bombeiros_km"] = np.nan

# --- embaralha as linhas e salva ------------------------------------------
df = df.sample(frac=1, random_state=0).reset_index(drop=True)

destino = Path("dados/desafio_regressao.csv")
destino.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(destino, index=False)

print(f"dataset salvo em: {destino.resolve()}")
print(f"shape: {df.shape}")
print(f"custo_sinistro — média: {df['custo_sinistro'].mean():.2f}   "
      f"mediana: {df['custo_sinistro'].median():.2f}   "
      f"máximo: {df['custo_sinistro'].max():.2f}")
df.head()
