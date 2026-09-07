# %% [markdown]
# # Encoding de variáveis categóricas
#
# **Tema:** Preparação de Dados › Encoding, Escala e Vazamento de Dados
#
# One-hot, ordinal, frequência e target encoding — cada um aplicado ao mesmo
# dataset, com foco especial no ponto mais perigoso deste notebook: como
# target encoding vaza a resposta para dentro da própria feature se calculado
# sem cuidado, e como corrigir isso com validação cruzada out-of-fold.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(15)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
AZUL, VERDE, VERMELHO, ROXO, AMBAR = "#1F5C8B", "#1E6B4F", "#9C2B2B", "#5B3E86", "#8A6100"
print("pronto")

# %% [markdown]
# ## 1. One-hot: nominal, sem ordem

# %%
regiao = pd.Series(rng.choice(["Norte", "Sul", "Sudeste", "Nordeste"], 10))

ohe = OneHotEncoder(sparse_output=False, drop="first")
regiao_onehot = ohe.fit_transform(regiao.to_frame())
print("categorias:", ohe.categories_[0])
print("\ncom drop='first' (evita a dependência linear exata do tema 2):")
print(pd.DataFrame(regiao_onehot, columns=ohe.get_feature_names_out()).head())

# %% [markdown]
# ## 2. Ordinal: quando a ordem é real (e o erro de usar em nominal)

# %%
nivel_educacao = pd.Series(
    rng.choice(["fundamental", "médio", "superior", "pós-graduação"], 10))
ordem_correta = ["fundamental", "médio", "superior", "pós-graduação"]

ordinal = OrdinalEncoder(categories=[ordem_correta])
codificado_correto = ordinal.fit_transform(nivel_educacao.to_frame())
print("ordinal CORRETO (ordem real de escolaridade):")
print(pd.DataFrame({"original": nivel_educacao, "código": codificado_correto.ravel()}))

# o erro: usar ordinal (alfabética, por padrão) numa variável SEM ordem real
regiao_ordinal_errado = OrdinalEncoder().fit_transform(regiao.to_frame())
print("\nordinal ERRADO em variável NOMINAL (região) — ordem alfabética arbitrária:")
print(pd.DataFrame({"original": regiao, "código (sem sentido)": regiao_ordinal_errado.ravel()}))
print("\nO modelo passaria a achar 'Norte' (0) mais perto de 'Nordeste' (1) do que")
print("de 'Sudeste' (3) — uma relação de distância inventada pela ordem alfabética.")

# %% [markdown]
# ## 3. Frequência: simples e sem explosão de dimensionalidade

# %%
n = 5000
produto = rng.choice([f"produto_{i}" for i in range(200)], n,
                     p=np.concatenate([[0.3, 0.2, 0.1], np.full(197, 0.4/197)]))

freq = pd.Series(produto).value_counts(normalize=True)
produto_freq_encoded = pd.Series(produto).map(freq)

print(f"cardinalidade de 'produto': {len(freq)}")
print(f"\ntop 3 produtos por frequência:\n{freq.head(3)}")
print(f"\nprimeiras linhas codificadas por frequência:")
print(pd.DataFrame({"produto": produto[:5], "freq_encoded": produto_freq_encoded[:5].values}))

# %% [markdown]
# ## 4. Target encoding: o jeito ERRADO primeiro (para expor o vazamento)

# %%
n = 3000
categoria = rng.choice([f"cat_{i}" for i in range(30)], n)
# 5 categorias RARAS de propósito, para expor o problema com poucos dados
categorias_raras = [f"cat_rara_{i}" for i in range(5)]
indices_raros = rng.choice(n, 15, replace=False)
categoria = categoria.astype(object)
for idx, cat_rara in zip(indices_raros, rng.choice(categorias_raras, 15)):
    categoria[idx] = cat_rara

# o alvo tem alguma relação real com a categoria, mas é majoritariamente ruído
efeito_categoria = {c: rng.normal(0, 0.3) for c in np.unique(categoria)}
logit = np.array([efeito_categoria[c] for c in categoria]) + rng.normal(0, 1, n)
alvo = (logit > np.median(logit)).astype(int)

df = pd.DataFrame({"categoria": categoria, "alvo": alvo})

# ERRADO: média do alvo por categoria, calculada COM a própria linha incluída
media_por_categoria = df.groupby("categoria")["alvo"].transform("mean")
print("Nas categorias raras (poucas observações), o encoding 'errado' se aproxima")
print("perigosamente do próprio valor do alvo:")
comparacao = df.loc[df["categoria"].isin(categorias_raras)].copy()
comparacao["encoding_vazado"] = media_por_categoria.loc[comparacao.index]
print(comparacao.sort_values("categoria").to_string(index=False))

# %% [markdown]
# **O problema fica óbvio:** para uma categoria com 1 única observação, a
# "média da categoria" **é** o próprio valor do alvo daquela linha — o
# encoding aprendeu de cor a resposta que deveria prever.

# %% [markdown]
# ## 5. Target encoding: a versão correta (out-of-fold)

# %%
def target_encoding_out_of_fold(categorias, y, n_splits=5, seed=0):
    """Para cada linha, usa a média do alvo calculada SEM o fold dela."""
    encoded = np.zeros(len(categorias), dtype=float)
    media_global = y.mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    categorias = pd.Series(categorias).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    for idx_treino, idx_val in kf.split(categorias):
        medias_treino = y.iloc[idx_treino].groupby(categorias.iloc[idx_treino]).mean()
        encoded[idx_val] = categorias.iloc[idx_val].map(medias_treino).fillna(media_global)
    return encoded


encoding_correto = target_encoding_out_of_fold(df["categoria"], df["alvo"])
df["encoding_correto"] = encoding_correto

comparacao2 = df.loc[df["categoria"].isin(categorias_raras)].copy()
print("mesma comparação, agora com o encoding out-of-fold:")
print(comparacao2.sort_values("categoria")[["categoria", "alvo", "encoding_correto"]]
      .to_string(index=False))

# %% [markdown]
# ## 6. Medindo o efeito real: desempenho vazado vs. honesto
#
# A prova definitiva: um modelo treinado com o encoding vazado deve parecer
# bom demais na validação, e essa vantagem deve ser majoritariamente artefato
# — vamos comparar contra `sklearn.preprocessing.TargetEncoder`, que já
# implementa a lógica out-of-fold corretamente.

# %%
# encoding VAZADO: fit no dataset inteiro, sem separar fold nenhum
encoding_vazado_completo = df.groupby("categoria")["alvo"].transform("mean")

X_vazado = encoding_vazado_completo.to_numpy().reshape(-1, 1)
X_correto_manual = df["encoding_correto"].to_numpy().reshape(-1, 1)

te_sklearn = TargetEncoder(random_state=0)
X_correto_sklearn = te_sklearn.fit_transform(df[["categoria"]], df["alvo"])

for nome, X in [("VAZADO (média com a própria linha)", X_vazado),
               ("correto (out-of-fold, manual)", X_correto_manual),
               ("correto (sklearn TargetEncoder)", X_correto_sklearn)]:
    auc = cross_val_score(LogisticRegression(), X, df["alvo"], cv=5, scoring="roc_auc").mean()
    print(f"{nome:<38s} AUC (5-fold) = {auc:.4f}")

# %% [markdown]
# **Leitura esperada:** a versão vazada tende a parecer artificialmente
# melhor na validação cruzada — porque mesmo dentro da validação, a feature já
# "sabe" a resposta correta de cada linha do fold de teste (ela foi calculada
# usando o dataset inteiro antes da divisão). As versões corretas (manual e
# `sklearn`) mostram o desempenho real, sem essa vantagem artificial.

# %% [markdown]
# ## O que levar deste notebook
#
# - One-hot para nominal, ordinal para categorias com ordem real — trocar os
#   dois inventa ou destrói relações de distância que o modelo vai levar a
#   sério.
# - Frequência é uma alternativa simples e barata para alta cardinalidade.
# - Target encoding é poderoso e o mais fácil de implementar com vazamento —
#   a versão sem esquema out-of-fold usa a própria linha para calcular sua
#   própria feature.
# - `sklearn.preprocessing.TargetEncoder` já resolve isso corretamente; a
#   versão manual deste notebook existe para deixar o mecanismo transparente.
#
# → Próximo: **Escalonamento**, e como a presença de outliers decide qual
# escalonador usar.
