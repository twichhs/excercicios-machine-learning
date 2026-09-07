# %% [markdown]
# # Exercícios — Encoding, Escala e Vazamento de Dados
#
# **Tema:** Preparação de Dados › Encoding, Escala e Vazamento de Dados
#
# Resolva antes de olhar o gabarito. **Dificuldade:** 🟢 base · 🟡 aplicação ·
# 🔴 síntese

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold, StratifiedGroupKFold

rng = np.random.default_rng(2468)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
AZUL, VERDE, VERMELHO, ROXO, AMBAR = "#1F5C8B", "#1E6B4F", "#9C2B2B", "#5B3E86", "#8A6100"
print("ambiente pronto")

# %% [markdown]
# ---
# ## Exercício 1 🟢 — Escolhendo o encoding certo
#
# Para cada variável, diga se usaria one-hot, ordinal, frequência ou target
# encoding, e por quê:
#
# 1. `estado_civil` (solteiro, casado, divorciado, viúvo) — 4 categorias.
# 2. `cep` — 8.000 valores únicos, sem ordem.
# 3. `classificacao_risco` (baixo, médio, alto, crítico).
# 4. `id_vendedor` — 3.000 vendedores, e você quer capturar "taxa histórica de
#    conversão daquele vendedor" na feature.

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 1

# %%
respostas = {
    1: "One-hot — nominal, poucas categorias (4), sem ordem natural.",
    2: "Frequência (ou hashing) — altíssima cardinalidade sem ordem; one-hot "
       "criaria 8.000 colunas.",
    3: "Ordinal — ordem real e conhecida (baixo < médio < alto < crítico).",
    4: "Target encoding com esquema out-of-fold — a media histórica do alvo "
       "por vendedor é exatamente a informação de negócio desejada, mas "
       "cardinalidade alta demais para one-hot e a resposta É o alvo.",
}
for k, v in respostas.items():
    print(f"{k}. {v}\n")

# %% [markdown]
# ---
# ## Exercício 2 🟢 — RobustScaler vs. StandardScaler na prática
#
# Gere uma variável com 3% de outliers extremos. Escalone com
# `StandardScaler` e `RobustScaler`. Compare a posição do percentil 50 (a
# mediana) depois de cada escalonamento — ela deveria ficar perto de 0 nos
# dois casos?

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 2

# %%
n = 2000
dados = rng.normal(100, 15, n)
idx_outliers = rng.choice(n, int(0.03 * n), replace=False)
dados[idx_outliers] *= rng.uniform(4, 8, len(idx_outliers))

padrao = StandardScaler().fit_transform(dados.reshape(-1, 1)).ravel()
robusto = RobustScaler().fit_transform(dados.reshape(-1, 1)).ravel()

print(f"mediana ORIGINAL           : {np.median(dados):.2f}")
print(f"mediana após StandardScaler: {np.median(padrao):.4f}")
print(f"mediana após RobustScaler  : {np.median(robusto):.4f}")

# %% [markdown]
# **Resposta:** a mediana do `RobustScaler` fica exatamente em 0 por
# construção (ele subtrai a mediana). A do `StandardScaler` fica deslocada de
# 0 — porque ele subtrai a **média**, que os outliers puxam para cima,
# deixando a mediana (o "centro" real dos dados) abaixo de zero na escala
# resultante.

# %% [markdown]
# ---
# ## Exercício 3 🟡 — Encontrando o vazamento em um trecho de código
#
# O código abaixo mede a acurácia de um modelo. Encontre os DOIS vazamentos
# de pré-processamento nele.
#
# ```python
# X_todos = pd.get_dummies(df, columns=["categoria"])
# X_todos["idade"] = SimpleImputer(strategy="mean").fit_transform(X_todos[["idade"]])
# X_todos["renda"] = StandardScaler().fit_transform(X_todos[["renda"]])
# X_treino, X_teste, y_treino, y_teste = train_test_split(X_todos, y, test_size=0.2)
# modelo.fit(X_treino, y_treino)
# ```

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 3

# %%
print("Vazamento 1: SimpleImputer().fit_transform() é chamado no dataset")
print("COMPLETO (X_todos), antes do split. A média usada para imputar 'idade'")
print("inclui as linhas que deveriam virar teste.")
print()
print("Vazamento 2: o mesmo problema com StandardScaler em 'renda' — média e")
print("desvio calculados com o dataset inteiro, não só o treino.")
print()
print("(pd.get_dummies também deveria, a rigor, aprender as categorias só no")
print("treino — uma categoria nova no teste que não apareceu no treino deve")
print("ser tratada explicitamente, não silenciosamente incorporada ao ajuste.)")
print()
print("A correção estrutural: envolver tudo em um Pipeline/ColumnTransformer,")
print("fazer o split ANTES de qualquer fit, e deixar cross_val_score ou")
print("fit/predict cuidarem da ordem correta automaticamente.")

# %% [markdown]
# ---
# ## Exercício 4 🟡 — Medindo o vazamento de target encoding em categorias raras
#
# Construa uma variável categórica com 200 categorias, sendo 150 delas com
# **apenas 1 ou 2 observações**. Compare o AUC de um modelo usando target
# encoding SEM esquema out-of-fold vs. COM `sklearn.preprocessing.TargetEncoder`
# (que já inclui a correção). A diferença deve ser bem mais visível aqui do
# que no notebook-guia (onde as categorias raras eram poucas).

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 4

# %%
n = 3000
# 150 categorias raras (1-2 obs cada) + 50 categorias comuns (o resto dos dados)
categorias_comuns = [f"comum_{i}" for i in range(50)]
categorias_raras = [f"rara_{i}" for i in range(150)]

linhas_raras = np.concatenate([
    np.repeat(categorias_raras[:100], 1),   # 100 categorias com 1 observação
    np.repeat(categorias_raras[100:], 2),   # 50 categorias com 2 observações
])
categoria = np.concatenate([
    linhas_raras, rng.choice(categorias_comuns, n - len(linhas_raras)),
])
rng.shuffle(categoria)

alvo = (rng.random(n) < 0.3).astype(int)  # SEM relação real com a categoria
df = pd.DataFrame({"categoria": categoria, "alvo": alvo})

# ERRADO: média por categoria COM a própria linha
encoding_vazado = df.groupby("categoria")["alvo"].transform("mean").to_numpy().reshape(-1, 1)

# CERTO: sklearn TargetEncoder (out-of-fold internamente)
encoding_correto = TargetEncoder(random_state=0).fit_transform(df[["categoria"]], df["alvo"])

for nome, X in [("vazado", encoding_vazado), ("TargetEncoder (correto)", encoding_correto)]:
    auc = cross_val_score(LogisticRegression(), X, df["alvo"], cv=5, scoring="roc_auc").mean()
    print(f"{nome:<26s} AUC = {auc:.4f}")

print("\nComo o alvo foi gerado SEM NENHUMA relação real com a categoria, o AUC")
print("correto deve ficar perto de 0.5. Um AUC muito acima disso na versão")
print("vazada é a evidência mais clara possível de vazamento: o modelo está")
print("'aprendendo' um padrão que estatisticamente não existe.")

# %% [markdown]
# ---
# ## Exercício 5 🔴 — Construindo o pipeline honesto de ponta a ponta
#
# Um dataset tem: `idade` (numérica, com nulos), `renda` (numérica, com
# outliers), `plano` (categórica nominal) e `cliente_id` (várias linhas por
# cliente — o mesmo cliente aparece em datas diferentes). Construa um
# `Pipeline` completo (imputação robusta a outliers para renda, escalonamento
# adequado, encoding, modelo) e valide com o esquema de validação cruzada
# correto dado que existe repetição de cliente.

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 5

# %%
n_clientes = 150
registros = []
for cid in range(n_clientes):
    n_obs = rng.integers(1, 5)
    plano_cliente = rng.choice(["básico", "padrão", "premium"])
    for _ in range(n_obs):
        idade = rng.normal(40, 12)
        renda = rng.lognormal(np.log(4000), 0.5)
        if rng.random() < 0.05:
            renda *= rng.uniform(5, 10)  # outlier genuíno
        registros.append({"cliente_id": cid, "idade": idade, "renda": renda,
                          "plano": plano_cliente})

df = pd.DataFrame(registros)
df.loc[rng.choice(len(df), 30, replace=False), "idade"] = np.nan
alvo = (rng.random(len(df)) < (0.2 + 0.15 * (df["plano"] == "premium"))).astype(int)
grupos = df["cliente_id"]

pre_processador = ColumnTransformer([
    ("idade", Pipeline([
        ("imputa", SimpleImputer(strategy="median")),
        ("escalona", StandardScaler()),
    ]), ["idade"]),
    ("renda", Pipeline([
        # RobustScaler: renda tem outliers genuínos, não queremos que
        # dominem a escala (módulo 2 + esta seção)
        ("escalona", RobustScaler()),
    ]), ["renda"]),
    ("plano", OneHotEncoder(drop="first"), ["plano"]),
])

pipeline = Pipeline([
    ("preparo", pre_processador),
    ("modelo", LogisticRegression()),
])

# ERRADO: KFold comum ignora que o mesmo cliente aparece em várias linhas
auc_errado = cross_val_score(
    pipeline, df.drop(columns="cliente_id"), alvo,
    cv=KFold(5, shuffle=True, random_state=0), scoring="roc_auc").mean()

# CERTO: StratifiedGroupKFold respeita que cada cliente fica só de um lado
auc_correto = cross_val_score(
    pipeline, df.drop(columns="cliente_id"), alvo,
    cv=StratifiedGroupKFold(5, shuffle=True, random_state=0), groups=grupos,
    scoring="roc_auc").mean()

print(f"AUC com KFold comum (vazamento de grupo)     : {auc_errado:.4f}")
print(f"AUC com StratifiedGroupKFold (correto)       : {auc_correto:.4f}")

# %% [markdown]
# **O pipeline resolve o vazamento de pré-processamento por construção**
# (imputação, escalonamento e encoding são ajustados só no treino de cada
# fold). Mas **não resolve sozinho o vazamento de grupo** — isso exige
# escolher o esquema de validação certo (`StratifiedGroupKFold`), passando
# `groups` explicitamente. As duas disciplinas são independentes e as duas
# são necessárias.

# %% [markdown]
# ---
# ## Fechamento
#
# - A escolha de encoding (one-hot, ordinal, frequência, target) depende do
#   tipo de variável e da cardinalidade — não existe um "encoding padrão"
#   universal.
# - Target encoding sem esquema out-of-fold é o vazamento mais fácil de
#   cometer por acidente nesta etapa, e o mais fácil de medir (basta comparar
#   com um alvo sem relação real, como no exercício 4).
# - `Pipeline`/`ColumnTransformer` eliminam vazamento de pré-processamento;
#   `StratifiedGroupKFold`/`GroupKFold` eliminam vazamento de grupo — são
#   duas disciplinas complementares, não uma substituindo a outra.
#
# → Próximo módulo: **Dados Desbalanceados**, o último deste tema, sobre o que
# fazer quando a classe de interesse é rara.
