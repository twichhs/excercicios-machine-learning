# %% [markdown]
# # Desafio — Classificação
#
# **Tema:** Aprendizado Supervisionado
#
# Este notebook **não ensina nada** — ele só entrega um dataset artificial,
# sujo e desbalanceado, para você aplicar sozinho tudo que aprender ao longo
# deste tema (e do tema 3, Preparação de Dados). Não há gabarito. A ideia é
# que você volte a este mesmo dataset depois de cada módulo — regressão
# logística, k-NN, SVM, árvores, Random Forest, boosting — e compare como
# cada família de modelo se sai exatamente nos mesmos dados.
#
# ## O problema de negócio (fictício)
#
# Uma operadora de cartões quer prever `inadimplente` (1 = vai deixar de
# pagar a fatura nos próximos 3 meses, 0 = vai continuar pagando em dia) a
# partir do histórico do cliente. A classe positiva é rara — a maioria dos
# clientes paga em dia.
#
# ## O que o dataset esconde de propósito
#
# - **Desbalanceamento de classes:** a classe positiva é minoria (~6%).
# - **Dados faltantes:** duas colunas têm nulos, com mecanismos diferentes
#   (um provavelmente MCAR, outro provavelmente MAR — descubra qual é qual).
# - **Outliers genuínos:** alguns valores extremos em `limite_credito` são
#   reais (clientes de alta renda), não erro.
# - **Uma coluna identificadora** (`id_cliente`) que não deveria virar feature.
# - **Uma categoria rara** em `canal_aquisicao`.
# - **Duas features quase redundantes** (multicolinearidade) para você
#   detectar.
# - **Uma feature de escala muito diferente das outras** (`renda_anual`, em
#   reais, contra variáveis em unidades pequenas).
# - **Colunas irrelevantes** (ruído puro) misturadas com as informativas.
#
# Nada disso é anunciado na tabela — faz parte do desafio descobrir.
#
# ## Sugestão de roteiro (não obrigatório)
#
# 1. Rode a célula abaixo. Ela também salva uma cópia em
#    `_desafios/dados/desafio_classificacao.csv`, para você recarregar depois
#    sem precisar rodar este notebook de novo.
# 2. Faça uma EDA rápida (tema 3, módulo 1).
# 3. Trate nulos e outliers com uma decisão justificada (tema 3, módulo 2).
# 4. Construa features, escolha encoding e escalonamento dentro de um
#    `Pipeline` (tema 3, módulos 3 e 4).
# 5. Lide com o desbalanceamento (tema 3, módulo 5).
# 6. Treine o modelo do módulo que você acabou de estudar, valide com
#    métricas adequadas a classes desbalanceadas (precisão, recall, F1,
#    PR-AUC) — nunca só acurácia.

# %%
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(2024)

n = 6000

# --- variáveis informativas de verdade ---------------------------------
idade = rng.normal(40, 11, n).clip(18, 78)
tempo_relacionamento_meses = rng.exponential(36, n).clip(1, None)
utilizacao_limite = rng.beta(2, 5, n)  # fração do limite já usada
n_atrasos_12m = rng.poisson(0.4, n)
renda_anual = rng.lognormal(np.log(48000), 0.55, n)
limite_credito = (renda_anual * rng.uniform(0.15, 0.45, n)).round(2)
# alguns clientes de alta renda com limite bem acima do padrão -- outliers REAIS
idx_vip = rng.choice(n, size=int(0.015 * n), replace=False)
limite_credito[idx_vip] *= rng.uniform(4, 9, len(idx_vip))

# --- feature quase redundante (multicolinearidade) ----------------------
utilizacao_limite_ruido = (utilizacao_limite + rng.normal(0, 0.02, n)).clip(0, 1)

# --- categóricas ----------------------------------------------------------
canal_aquisicao = rng.choice(
    ["agência", "app", "indicação", "parceria_varejo", "evento_promocional"],
    n, p=[0.35, 0.40, 0.15, 0.09, 0.01])
regiao = rng.choice(["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"],
                    n, p=[0.08, 0.20, 0.10, 0.42, 0.20])

# --- colunas de ruído puro (sem relação real com o alvo) ------------------
ruido_1 = rng.normal(0, 1, n)
ruido_2 = rng.integers(0, 5, n)

# --- identificador (NÃO é feature) -----------------------------------------
id_cliente = np.arange(900000, 900000 + n)

# --- alvo: combinação real das variáveis informativas ----------------------
logit = (
    -3.4
    + 2.6 * utilizacao_limite
    + 0.55 * n_atrasos_12m
    - 0.015 * tempo_relacionamento_meses / 12
    - 0.000012 * renda_anual
    + 0.01 * (idade < 25)
)
prob_inadimplencia = 1 / (1 + np.exp(-logit))
inadimplente = (rng.random(n) < prob_inadimplencia).astype(int)

df = pd.DataFrame({
    "id_cliente": id_cliente,
    "idade": idade.round(1),
    "renda_anual": renda_anual.round(2),
    "limite_credito": limite_credito,
    "tempo_relacionamento_meses": tempo_relacionamento_meses.round(1),
    "utilizacao_limite": utilizacao_limite.round(4),
    "utilizacao_limite_v2": utilizacao_limite_ruido.round(4),
    "n_atrasos_12m": n_atrasos_12m,
    "canal_aquisicao": canal_aquisicao,
    "regiao": regiao,
    "feature_ruido_1": ruido_1.round(4),
    "feature_ruido_2": ruido_2,
    "inadimplente": inadimplente,
})

# --- nulos: um padrão provavelmente MCAR, outro provavelmente MAR ---------
idx_nulo_mcar = rng.choice(n, size=int(0.06 * n), replace=False)
df.loc[idx_nulo_mcar, "tempo_relacionamento_meses"] = np.nan

prob_nulo_mar = 0.03 + 0.25 * (df["idade"] > 60)
idx_nulo_mar = rng.random(n) < prob_nulo_mar
df.loc[idx_nulo_mar, "renda_anual"] = np.nan

# --- embaralha as linhas e salva ------------------------------------------
df = df.sample(frac=1, random_state=0).reset_index(drop=True)

destino = Path("dados/desafio_classificacao.csv")
destino.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(destino, index=False)

print(f"dataset salvo em: {destino.resolve()}")
print(f"shape: {df.shape}")
print(f"taxa de inadimplência: {df['inadimplente'].mean():.2%}")
df.head()
