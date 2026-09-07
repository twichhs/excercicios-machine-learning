# %% [markdown]
# # Exercícios — Feature Engineering
#
# **Tema:** Preparação de Dados › Feature Engineering
#
# Resolva antes de olhar o gabarito. **Dificuldade:** 🟢 base · 🟡 aplicação ·
# 🔴 síntese

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

rng = np.random.default_rng(789)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
AZUL, VERDE, VERMELHO, ROXO, AMBAR = "#1F5C8B", "#1E6B4F", "#9C2B2B", "#5B3E86", "#8A6100"
print("ambiente pronto")

# %% [markdown]
# ---
# ## Exercício 1 🟢 — Log ou não?
#
# Gere duas variáveis: `A` normal (média 50, desvio 10) e `B` log-normal
# (mesma ordem de grandeza). Para cada uma, calcule a assimetria antes e
# depois de aplicar `log1p`. Em qual delas o log ajuda, e em qual atrapalha?

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 1

# %%
from scipy import stats

A = rng.normal(50, 10, 3000).clip(1, None)
B = rng.lognormal(np.log(50), 0.6, 3000)

for nome, v in [("A (normal)", A), ("B (log-normal)", B)]:
    assimetria_antes = stats.skew(v)
    assimetria_depois = stats.skew(np.log1p(v))
    print(f"{nome}: assimetria original = {assimetria_antes:+.3f}   "
          f"assimetria após log1p = {assimetria_depois:+.3f}")

# %% [markdown]
# **Resposta:** o log ajuda em `B` (reduz a assimetria, aproximando de uma
# distribuição simétrica) e **piora** `A` (uma normal já é simétrica; aplicar
# log nela introduz assimetria onde não havia). A régua certa não é "sempre
# aplicar log em variáveis positivas" — é medir a assimetria antes e depois e
# aplicar só quando ajuda.

# %% [markdown]
# ---
# ## Exercício 2 🟢 — Encoding cíclico na mão
#
# Para os meses do ano (1 a 12), calcule as coordenadas seno/cosseno.
# Verifique numericamente que a distância cíclica entre dezembro (12) e
# janeiro (1) é pequena, e a distância cíclica entre janeiro (1) e julho (7)
# é grande.

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 2

# %%
def coordenadas_ciclicas(valor, periodo):
    ang = 2 * np.pi * valor / periodo
    return np.sin(ang), np.cos(ang)


def distancia_ciclica(v1, v2, periodo):
    s1, c1 = coordenadas_ciclicas(v1, periodo)
    s2, c2 = coordenadas_ciclicas(v2, periodo)
    return np.sqrt((s1 - s2) ** 2 + (c1 - c2) ** 2)


dez_jan = distancia_ciclica(12, 1, 12)
jan_jul = distancia_ciclica(1, 7, 12)
print(f"distância cíclica dezembro-janeiro: {dez_jan:.4f}")
print(f"distância cíclica janeiro-julho   : {jan_jul:.4f}")
print(f"\ndistância LINEAR ingênua dezembro-janeiro: {abs(12-1)}")
print(f"distância LINEAR ingênua janeiro-julho   : {abs(1-7)}")
print("\nA codificação linear diz que dez-jan estão MUITO longe (11) e jan-jul")
print("perto (6) — exatamente o oposto da realidade do calendário.")

# %% [markdown]
# ---
# ## Exercício 3 🟡 — Caçando o vazamento em uma média móvel
#
# O código abaixo tem um vazamento temporal. Encontre-o, explique por que é
# um vazamento, e corrija.
#
# ```python
# vendas_futuras = serie["vendas"].rolling(3, center=True).mean()
# ```

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 3

# %%
n_dias = 100
serie = pd.DataFrame({"vendas": rng.normal(100, 10, n_dias)})

# a versão do enunciado, reproduzida
vendas_com_vazamento = serie["vendas"].rolling(3, center=True).mean()
print("com center=True, a janela do dia t inclui os dias t-1, t E t+1:")
print(serie["vendas"].iloc[8:13].to_string())
print("\nmédia móvel centrada (VAZADA):")
print(vendas_com_vazamento.iloc[8:13].to_string())

# %% [markdown]
# **O vazamento:** `center=True` faz a janela do dia $t$ incluir $t-1$, $t$ e
# $t+1$ — ou seja, a feature do dia $t$ usa o valor de vendas de **amanhã**,
# que não existiria no momento real de uma previsão. A correção é usar
# `shift(1)` antes de uma janela **não-centrada** (olhando só para trás):

# %%
vendas_corrigida = serie["vendas"].shift(1).rolling(3).mean()
print("versão corrigida (só passado):")
print(vendas_corrigida.iloc[8:13].to_string())

# %% [markdown]
# ---
# ## Exercício 4 🟡 — Filtro vs. wrapper: quando eles discordam
#
# Construa um dataset com uma feature `xor_a` e `xor_b` cuja combinação (XOR)
# prevê perfeitamente o alvo, mas cada uma isoladamente tem informação mútua
# baixa com o alvo. Calcule a informação mútua individual de cada uma. Depois
# treine um modelo usando as duas juntas mais a interação (`xor_a * xor_b`
# não ajuda em XOR — pense em outra feature derivada que ajude, ou use uma
# árvore). Explique por que um filtro univariado sozinho descartaria as duas
# features, apesar delas serem, juntas, perfeitamente preditivas.

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 4

# %%
n = 2000
xor_a = rng.integers(0, 2, n)
xor_b = rng.integers(0, 2, n)
alvo_xor = (xor_a != xor_b).astype(int)  # XOR

mi_a = mutual_info_classif(xor_a.reshape(-1, 1), alvo_xor, random_state=0)[0]
mi_b = mutual_info_classif(xor_b.reshape(-1, 1), alvo_xor, random_state=0)[0]
mi_conjunta = mutual_info_classif(
    np.column_stack([xor_a, xor_b]), alvo_xor, random_state=0)

print(f"informação mútua de xor_a sozinha : {mi_a:.4f}")
print(f"informação mútua de xor_b sozinha : {mi_b:.4f}")
print(f"informação mútua conjunta (ambas) : {mi_conjunta}")

from sklearn.tree import DecisionTreeClassifier

acuracia_arvore = cross_val_score(
    DecisionTreeClassifier(max_depth=3, random_state=0),
    np.column_stack([xor_a, xor_b]), alvo_xor, cv=5).mean()
print(f"\nacurácia de uma árvore usando as DUAS features: {acuracia_arvore:.4f}")

# %% [markdown]
# **Explicação:** cada feature isolada tem informação mútua baixíssima com o
# alvo (XOR é o exemplo clássico de interação pura: nenhuma variável, sozinha,
# diz quase nada). Um filtro univariado — que avalia cada feature isolada —
# descartaria as duas por parecerem irrelevantes. Uma árvore de decisão
# (wrapper implícito, ao considerar as features juntas durante o treino)
# encontra a estrutura perfeitamente. Essa é a limitação central de qualquer
# método de filtro: ele é cego a relações que só existem na **combinação** de
# features.

# %% [markdown]
# ---
# ## Exercício 5 🔴 — Construindo um pipeline de features de cliente sem vazamento
#
# Dado um histórico de pedidos (`cliente_id`, `dia`, `valor`), construa as
# features `n_pedidos_ultimos_30_dias` e `valor_medio_ultimos_30_dias` para
# cada pedido, usando **apenas** pedidos anteriores dentro da janela de 30
# dias (sem incluir o próprio pedido). Valide que a primeira linha de cada
# cliente sempre tem `n_pedidos_ultimos_30_dias = 0`.

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 5

# %%
n_clientes = 50
pedidos = []
for cid in range(n_clientes):
    n_ped = rng.integers(5, 20)
    dias = np.sort(rng.choice(np.arange(180), n_ped, replace=False))
    valores = rng.lognormal(np.log(80), 0.4, n_ped)
    for d, v in zip(dias, valores):
        pedidos.append({"cliente_id": cid, "dia": d, "valor": v})

df = pd.DataFrame(pedidos).sort_values(["cliente_id", "dia"]).reset_index(drop=True)


def features_janela_movel(grupo, janela_dias=30):
    n_pedidos_janela = []
    valor_medio_janela = []
    dias_arr = grupo["dia"].to_numpy()
    valores_arr = grupo["valor"].to_numpy()
    for i in range(len(grupo)):
        dia_atual = dias_arr[i]
        # só pedidos ESTRITAMENTE anteriores, dentro da janela
        mascara = (dias_arr < dia_atual) & (dias_arr >= dia_atual - janela_dias)
        n_pedidos_janela.append(mascara.sum())
        valor_medio_janela.append(valores_arr[mascara].mean() if mascara.any() else np.nan)
    grupo = grupo.copy()
    grupo["n_pedidos_ultimos_30_dias"] = n_pedidos_janela
    grupo["valor_medio_ultimos_30_dias"] = valor_medio_janela
    return grupo


df_com_features = df.groupby("cliente_id", group_keys=False).apply(
    features_janela_movel, include_groups=False)
df_com_features["cliente_id"] = df["cliente_id"]

primeira_linha_por_cliente = df_com_features.groupby("cliente_id").head(1)
print(f"todas as primeiras linhas têm n_pedidos_ultimos_30_dias == 0? "
      f"{(primeira_linha_por_cliente['n_pedidos_ultimos_30_dias'] == 0).all()}")

print(f"\nexemplo (cliente 0):")
print(df_com_features[df_com_features["cliente_id"] == 0]
      [["dia", "valor", "n_pedidos_ultimos_30_dias", "valor_medio_ultimos_30_dias"]]
      .head(6).to_string(index=False))

# %% [markdown]
# **Validação central:** a primeira linha de cada cliente tem
# `n_pedidos_ultimos_30_dias = 0` por construção — não existe pedido anterior a
# ele. Se essa checagem falhasse, seria sinal quase certo de vazamento (a
# janela, sem querer, contando o próprio pedido).

# %% [markdown]
# ---
# ## Fechamento
#
# - Transformações numéricas devem ser justificadas por medição (assimetria
#   antes/depois), não aplicadas por hábito.
# - Encoding cíclico corrige uma distorção geométrica real em variáveis de
#   calendário — a diferença entre codificação linear e circular é
#   mensurável.
# - `rolling(center=True)` e qualquer janela que não exclua explicitamente o
#   presente/futuro é vazamento temporal — a assinatura de código a procurar é
#   sempre um `shift(1)` antes de qualquer agregação de janela.
# - Filtros univariados são cegos a interações puras (XOR é o caso extremo);
#   wrappers e modelos baseados em árvore capturam o que o filtro não vê.
# - Toda feature de "histórico até aqui" exige uma checagem de sanidade: a
#   primeira observação de cada entidade deve sempre refletir "sem histórico".
#
# → Próximo módulo: **Encoding, Escala e Vazamento de Dados**, formalizando com
# `Pipeline` a disciplina de nunca deixar o teste vazar para o treino.
