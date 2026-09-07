# %% [markdown]
# # Vazamento de dados: anatomia de um desastre
#
# **Tema:** Preparação de Dados › Encoding, Escala e Vazamento de Dados
#
# Este notebook fecha o módulo reunindo as quatro formas de vazamento da
# tabela do `teoria.pdf` num único fluxo, e mostra a solução estrutural:
# `Pipeline` e `ColumnTransformer`, que tornam o vazamento de pré-processamento
# estruturalmente difícil de cometer por acidente.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedGroupKFold, KFold

rng = np.random.default_rng(99)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
AZUL, VERDE, VERMELHO, ROXO, AMBAR = "#1F5C8B", "#1E6B4F", "#9C2B2B", "#5B3E86", "#8A6100"
print("pronto")

# %% [markdown]
# ## 1. Vazamento de alvo: a feature que É o alvo disfarçado
#
# Um cenário clássico: prever `inadimplente`, com uma feature
# `valor_em_cobranca` que só é preenchida (não-zero) depois que o cliente já
# está em processo de cobrança — ou seja, depois que ele já é inadimplente.

# %%
n = 3000
renda = rng.lognormal(np.log(4000), 0.5, n)
inadimplente = (rng.random(n) < 0.15).astype(int)
# vazamento: só existe valor em cobrança PARA QUEM JÁ é inadimplente
valor_em_cobranca = np.where(inadimplente == 1, rng.uniform(500, 5000, n), 0.0)

df_vazado = pd.DataFrame({"renda": renda, "valor_em_cobranca": valor_em_cobranca,
                          "inadimplente": inadimplente})

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

modelo_com_vazamento = LogisticRegression()
X_com_vazamento = df_vazado[["renda", "valor_em_cobranca"]]
probs_vazado = cross_val_predict(modelo_com_vazamento, X_com_vazamento,
                                 df_vazado["inadimplente"], cv=5, method="predict_proba")[:, 1]
auc_vazado = roc_auc_score(df_vazado["inadimplente"], probs_vazado)

modelo_sem_vazamento = LogisticRegression()
X_sem_vazamento = df_vazado[["renda"]]
probs_sem = cross_val_predict(modelo_sem_vazamento, X_sem_vazamento,
                              df_vazado["inadimplente"], cv=5, method="predict_proba")[:, 1]
auc_sem = roc_auc_score(df_vazado["inadimplente"], probs_sem)

print(f"AUC COM 'valor_em_cobrança' (vazamento de alvo): {auc_vazado:.4f}")
print(f"AUC SEM a feature vazada (só renda)            : {auc_sem:.4f}")
print("\nUm AUC próximo de 1.0 deveria ser sempre motivo de SUSPEITA, não de")
print("comemoração — é raro um problema de negócio real ser tão fácil assim.")

# %% [markdown]
# ## 2. Vazamento de grupo: a mesma entidade em treino e teste
#
# Múltiplas leituras de sensor por máquina. Se linhas da mesma máquina caem em
# treino e teste, o modelo pode "memorizar" a máquina em vez de aprender o
# padrão geral de falha.

# %%
n_maquinas = 40
leituras_por_maquina = 40
registros = []
for maquina_id in range(n_maquinas):
    # cada máquina tem uma "assinatura" própria de ruído de base, MAIS forte
    # que o efeito real de falha -- o cenário onde memorizar a máquina "engana"
    assinatura = rng.normal(0, 4)
    vai_falhar = rng.random() < 0.3
    for _ in range(leituras_por_maquina):
        vibracao = rng.normal(5 + assinatura + (1.5 if vai_falhar else 0), 0.6)
        registros.append({"maquina_id": maquina_id, "vibracao": vibracao, "falha": int(vai_falhar)})

df_sensores = pd.DataFrame(registros)

X = df_sensores[["vibracao"]]
y = df_sensores["falha"]
grupos = df_sensores["maquina_id"]

# Um modelo baseado em DISTÂNCIA expõe o problema com mais clareza: se outras
# leituras da MESMA máquina estão no treino, elas ficam quase coladas (em
# valor) à leitura de teste — o vizinho mais próximo "entrega" a resposta.
modelo_sensor = KNeighborsClassifier(n_neighbors=3)

# ERRADO: KFold comum, ignorando que várias linhas são da MESMA máquina
auc_kfold_comum = cross_val_score(
    modelo_sensor, X, y, cv=KFold(5, shuffle=True, random_state=0),
    scoring="roc_auc").mean()

# CERTO: StratifiedGroupKFold garante que uma máquina inteira fica OU no
# treino OU no teste (GroupKFold puro), preservando a proporção de falhas em
# cada fold (evita folds sem nenhum exemplo de uma das classes).
auc_groupkfold = cross_val_score(
    modelo_sensor, X, y, cv=StratifiedGroupKFold(5, shuffle=True, random_state=0),
    groups=grupos, scoring="roc_auc").mean()

print(f"AUC com KFold comum (vazamento de grupo)      : {auc_kfold_comum:.4f}")
print(f"AUC com StratifiedGroupKFold (sem vazamento)  : {auc_groupkfold:.4f}")
print("\nCom KFold comum, outras leituras DA MESMA máquina (mesma assinatura de")
print("ruído) ficam no treino — o k-NN 'reconhece' a máquina pela proximidade")
print("do valor, não pelo padrão real de falha, inflando o desempenho.")

# %% [markdown]
# ## 3. A solução estrutural: `Pipeline` + `ColumnTransformer`
#
# Construímos um pipeline completo (imputação + encoding + escalonamento +
# modelo) e mostramos que `cross_val_score` aplicado a ele nunca vaza — cada
# etapa é ajustada só no treino de cada fold, automaticamente.

# %%
n = 2000
idade = rng.normal(40, 12, n)
idade[rng.choice(n, 100, replace=False)] = np.nan  # nulos
renda = rng.lognormal(np.log(4000), 0.6, n)
regiao = rng.choice(["Norte", "Sul", "Sudeste", "Nordeste"], n)
alvo = (0.02 * idade + 0.0003 * renda + rng.normal(0, 1, n) > 2).astype(int)

df_completo = pd.DataFrame({"idade": idade, "renda": renda, "regiao": regiao})

colunas_numericas = ["idade", "renda"]
colunas_categoricas = ["regiao"]

pre_processador = ColumnTransformer([
    ("numericas", Pipeline([
        ("imputa", SimpleImputer(strategy="median")),
        ("escalona", StandardScaler()),
    ]), colunas_numericas),
    ("categoricas", OneHotEncoder(drop="first", handle_unknown="ignore"), colunas_categoricas),
])

pipeline_completo = Pipeline([
    ("preparo", pre_processador),
    ("modelo", LogisticRegression()),
])

auc_pipeline = cross_val_score(pipeline_completo, df_completo, alvo, cv=5,
                               scoring="roc_auc").mean()
print(f"AUC do pipeline completo (imputação + encoding + escalonamento + modelo, "
      f"sem vazamento por construção): {auc_pipeline:.4f}")

# %% [markdown]
# **O que aconteceu por baixo dos panos:** a cada um dos 5 folds da validação
# cruzada, o `SimpleImputer` calculou a mediana **só no treino daquele fold**,
# o `StandardScaler` calculou média/desvio **só no treino daquele fold**, e o
# `OneHotEncoder` aprendeu as categorias **só no treino daquele fold** — e cada
# transformação já ajustada foi aplicada ao fold de validação sem re-ajustar.
# Nenhuma linha de código extra foi necessária para garantir isso: é o
# comportamento padrão do `Pipeline`.

# %% [markdown]
# ## 4. Comparando com o erro clássico: pré-processar antes de dividir

# %%
# ERRADO: imputar e escalonar o dataset INTEIRO antes de qualquer validação
df_pre_processado_errado = df_completo.copy()
df_pre_processado_errado["idade"] = SimpleImputer(strategy="median").fit_transform(
    df_pre_processado_errado[["idade"]])
df_pre_processado_errado[colunas_numericas] = StandardScaler().fit_transform(
    df_pre_processado_errado[colunas_numericas])
df_pre_processado_errado = pd.get_dummies(df_pre_processado_errado, columns=["regiao"],
                                          drop_first=True)

auc_errado = cross_val_score(LogisticRegression(), df_pre_processado_errado, alvo,
                             cv=5, scoring="roc_auc").mean()

print(f"AUC com preparo ERRADO (fit no dataset inteiro): {auc_errado:.4f}")
print(f"AUC com Pipeline (fit por fold, sem vazamento) : {auc_pipeline:.4f}")
print(f"\ndiferença: {auc_errado - auc_pipeline:+.4f}")
print("Neste exemplo a diferença numérica é pequena (dataset grande, split")
print("aleatório) — mas o princípio se mantém: o número errado é sempre")
print("otimista, nunca pessimista, e a diferença cresce quanto menor o dataset")
print("e quanto mais 'informação global' cada transformação incorporar.")

# %% [markdown]
# ## O que levar deste notebook
#
# - Vazamento de alvo: pergunte sempre "essa feature existiria, com esse
#   valor, no momento real da previsão?" — um AUC próximo de 1.0 é motivo de
#   suspeita, não de confiança.
# - Vazamento de grupo: quando várias linhas pertencem à mesma entidade,
#   `GroupKFold` (não `KFold`) é obrigatório.
# - `Pipeline` + `ColumnTransformer` tornam o vazamento de pré-processamento
#   estruturalmente improvável — cada etapa é ajustada só no treino de cada
#   fold, automaticamente, sem depender de disciplina manual.
# - O vazamento sempre INFLA o desempenho medido — nunca o reduz. Um resultado
#   bom demais para ser verdade quase sempre não é.
#
# → Próximo: o notebook de **exercícios** do módulo.
