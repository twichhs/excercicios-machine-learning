# %% [markdown]
# # Exercícios — Métricas de Regressão
#
# **Tema:** Avaliação e Validação de Modelos › Métricas de Regressão
#
# Resolva antes de olhar o gabarito. **Dificuldade:** 🟢 base · 🟡 aplicação ·
# 🔴 síntese

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_pinball_loss
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(626)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
AZUL, VERDE, VERMELHO, ROXO, AMBAR = "#1F5C8B", "#1E6B4F", "#9C2B2B", "#5B3E86", "#8A6100"
print("ambiente pronto")

# %% [markdown]
# ---
# ## Dataset de trabalho: demanda diária de bicicletas compartilhadas
#
# Três anos de dados de 40 estações: número de retiradas por estação e
# dia, em função do dia da semana, da temperatura, da chuva e do tipo de
# estação (centro comercial, residencial, parque). A demanda é uma
# contagem com variância que cresce com a média — e nos parques depende
# muito mais do clima.

# %%
n_dias, n_est = 3 * 365, 40
dia = np.repeat(np.arange(n_dias), n_est)
estacao = np.tile(np.arange(n_est), n_dias)
tipo = np.array(["comercial", "residencial", "parque"])[estacao % 3]
dia_semana = dia % 7
temp = 20 + 8 * np.sin(2 * np.pi * dia / 365) + rng.normal(0, 3, len(dia))
chuva = (rng.random(len(dia)) < 0.2).astype(int)
base = np.select([tipo == "comercial", tipo == "residencial"], [60, 35], 25)
efeito_fds = np.select([tipo == "comercial", tipo == "parque"], [-0.5, 0.9], 0.1) * (dia_semana >= 5)
efeito_clima = np.where(tipo == "parque", 0.06, 0.02) * (temp - 20) - np.where(tipo == "parque", 0.9, 0.3) * chuva
media = base * np.exp(efeito_fds + efeito_clima)
retiradas = rng.negative_binomial(5, 5 / (5 + media))
bikes = pd.DataFrame({"dia": dia, "estacao": estacao, "tipo": tipo, "dia_semana": dia_semana,
                      "temperatura": temp.round(1), "chuva": chuva, "retiradas": retiradas})
bikes["tipo_cod"] = bikes["tipo"].map({"comercial": 0, "residencial": 1, "parque": 2})
features = ["tipo_cod", "dia_semana", "temperatura", "chuva"]
treino = bikes["dia"] < 2 * 365
print(bikes.groupby("tipo")["retiradas"].describe().round(1).to_string())

# %% [markdown]
# ---
# ## Exercício 1 🟢 — Métricas na mão
#
# Numa estação, em 4 dias, as retiradas foram $y = [40, 10, 60, 30]$ e a
# previsão foi $\hat{y} = [35, 20, 50, 30]$. Calcule MAE, RMSE, MAPE, WAPE
# e R². Qual dia mais pesa em cada métrica?

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 1

# %%
y = np.array([40, 10, 60, 30], dtype=float)
yh = np.array([35, 20, 50, 30], dtype=float)
e = y - yh
print(f"resíduos: {e}")
print(f"MAE = {np.mean(np.abs(e)):.2f} | RMSE = {np.sqrt(np.mean(e ** 2)):.2f}")
print(f"MAPE = {np.mean(np.abs(e) / y):.1%} | WAPE = {np.abs(e).sum() / y.sum():.1%}")
print(f"R² = {1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2):.3f}")
print("contribuição de cada dia no MAPE:", (np.abs(e) / y / np.sum(np.abs(e) / y)).round(2))

# %% [markdown]
# **Por quê:** resíduos $[5, -10, 10, 0]$: MAE $= 25/4 = 6{,}25$; RMSE
# $= \sqrt{225/4} = 7{,}5$. No MAPE (32%), o dia 2 (real 10, previsto 20)
# tem erro de 100% e responde por mais de três quartos da soma dos erros
# percentuais — um erro de 10 bicicletas num dia fraco pesa muito mais que
# o mesmo erro de 10 no dia forte (dia 3, erro de 17%). O WAPE,
# $25/140 \approx 17{,}9\%$, pondera pelo volume. Com $\bar{y} = 35$, o $R^2$
# é $1 - 225/1.300 \approx 0{,}83$.

# %% [markdown]
# ---
# ## Exercício 2 🟢 — O MAPE prefere quem subestima
#
# Dois modelos para 6 dias com demanda real $y = [20, 25, 30, 35, 40, 45]$:
# o modelo A sempre prevê 10% **acima** do real; o modelo B, 10% **abaixo**.
# Calcule MAPE e MAE de cada um. Depois, o modelo C prevê sempre 30
# (constante): compare o MAPE de C com o de uma constante de 32,5 (a
# mediana). O que você conclui?

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 2

# %%
y2 = np.array([20, 25, 30, 35, 40, 45], dtype=float)
for nome, prev in [("A (+10%)", y2 * 1.1), ("B (-10%)", y2 * 0.9)]:
    print(f"{nome}: MAPE = {np.mean(np.abs(y2 - prev) / y2):.1%} | MAE = {np.mean(np.abs(y2 - prev)):.2f}")
grade = np.linspace(15, 50, 3501)
mapes = [np.mean(np.abs(y2 - c) / y2) for c in grade]
print(f"MAPE da constante 30   : {np.mean(np.abs(y2 - 30) / y2):.2%}")
print(f"MAPE da constante 32,5 : {np.mean(np.abs(y2 - 32.5) / y2):.2%}")
print(f"constante de MAPE mínimo: {grade[np.argmin(mapes)]:.2f} (a mediana é {np.median(y2)})")

# %% [markdown]
# **Por quê:** A e B erram ±10% sobre os mesmos valores, então têm o mesmo
# MAPE (10%) e o mesmo MAE — até aqui, simetria. A assimetria aparece
# quando o modelo precisa escolher **um** valor para dias diferentes: como
# o MAPE divide por $y$, errar nos dias de demanda baixa custa mais do que
# errar nos de demanda alta, e a constante que minimiza o MAPE (30) fica
# **abaixo** da mediana (32,5), puxada para os dias fracos. Um modelo otimizado para MAPE aprende a
# prever baixo — e num sistema de rebalanceamento de bicicletas isso
# significa estações vazias.

# %% [markdown]
# ---
# ## Exercício 3 🟡 — Qual estatística cada perda estima?
#
# Treine três `HistGradientBoostingRegressor` com os dois primeiros anos
# (perdas `squared_error`, `absolute_error` e `poisson`) e avalie no
# terceiro ano com MAE, RMSE e a deviance de Poisson média
# (`mean_poisson_deviance`). Depois, compare a média das previsões de cada
# modelo com a média real do teste. Qual modelo tem viés sistemático, e
# para que lado?

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 3

# %%
from sklearn.metrics import mean_poisson_deviance

X_tr, y_tr = bikes.loc[treino, features], bikes.loc[treino, "retiradas"]
X_te, y_te = bikes.loc[~treino, features], bikes.loc[~treino, "retiradas"]
modelos = {}
for perda in ["squared_error", "absolute_error", "poisson"]:
    modelos[perda] = HistGradientBoostingRegressor(loss=perda, max_iter=300,
                                                   random_state=0).fit(X_tr, y_tr)
linhas = {}
for perda, m in modelos.items():
    p = np.clip(m.predict(X_te), 1e-6, None)
    linhas[perda] = {"MAE": mean_absolute_error(y_te, p), "RMSE": np.sqrt(mean_squared_error(y_te, p)),
                     "deviance Poisson": mean_poisson_deviance(y_te, p),
                     "média prevista": p.mean(), "média real": y_te.mean()}
print(pd.DataFrame(linhas).T.round(3).to_string())

# %% [markdown]
# **Por quê:** a demanda é uma contagem assimétrica à direita (binomial
# negativa), então a mediana condicional fica abaixo da média. O modelo de
# erro absoluto estima a mediana: vence no MAE, mas sua média prevista fica
# **abaixo** da média real — viés sistemático para baixo, que, somado
# sobre 40 estações, subestima a frota necessária. Os modelos de erro
# quadrático e de Poisson estimam a média e acertam o total; o de Poisson,
# cuja perda combina com a natureza do alvo (contagem com variância
# crescente), costuma ter a menor deviance. Para planejar frota, o total
# importa: use uma perda de média.

# %% [markdown]
# ---
# ## Exercício 4 🟡 — Quantas bicicletas deixar em cada estação?
#
# Toda manhã a operadora abastece as estações. Cada retirada que não
# acontece por falta de bicicleta custa R$ 4 de receita perdida; cada
# bicicleta que fica parada o dia inteiro custa R$ 1 (manutenção e capital).
# (a) Qual quantil da demanda deve orientar o abastecimento? (b) Treine um
# modelo quantílico nesse nível e compare o custo no terceiro ano com o de
# abastecer pela previsão da média (modelo de Poisson). (c) Em que tipo de
# estação a diferença é maior, e por quê?

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 4

# %%
C_FALTA, C_SOBRA = 4.0, 1.0
tau = C_FALTA / (C_FALTA + C_SOBRA)
m_q = HistGradientBoostingRegressor(loss="quantile", quantile=tau, max_iter=300,
                                    random_state=0).fit(X_tr, y_tr)

def custo(abastece, demanda):
    return C_FALTA * np.maximum(demanda - abastece, 0) + C_SOBRA * np.maximum(abastece - demanda, 0)

politicas = {"média (Poisson)": np.round(modelos["poisson"].predict(X_te)),
             f"quantil {tau:.0%}": np.round(m_q.predict(X_te))}
res = pd.DataFrame({nome: pd.Series(custo(a, y_te.to_numpy())).groupby(
    bikes.loc[~treino, "tipo"].to_numpy()).mean() for nome, a in politicas.items()})
res.loc["TOTAL (R$/estação/dia)"] = [custo(a, y_te.to_numpy()).mean() for a in politicas.values()]
print(f"(a) tau* = {C_FALTA}/({C_FALTA}+{C_SOBRA}) = {tau:.2f}")
print("(b/c) custo médio por estação e dia:")
print(res.round(2).to_string())

# %% [markdown]
# **Por quê:** (a) é o problema do jornaleiro: faltar custa 4 vezes mais
# que sobrar, então abastece-se pelo quantil de 80%. (b) Abastecer pela
# média falta em cerca de metade dos dias; o quantil reduz o custo total em
# torno de 20%. (c) Em termos relativos, a redução é parecida nos três
# tipos; em reais, é maior nas estações **comerciais**, que têm mais volume
# e — como a variância de uma contagem cresce com a média — mais incerteza
# absoluta. É o mesmo padrão da padaria do notebook-guia: o quantil
# acrescenta uma folga proporcional à incerteza local de cada estação e
# dia, algo que nenhuma regra fixa do tipo "média + x bicicletas" reproduz.

# %% [markdown]
# ---
# ## Exercício 5 🔴 — Um intervalo de 90% em que a operação possa confiar
#
# O time de operações quer, para cada estação e dia, um intervalo de
# retiradas com 90% de cobertura. Use o segundo ano como **calibração** e
# o terceiro como teste (treine só no primeiro ano).
#
# (a) Construa o intervalo conformal simples (previsão de Poisson ±
# quantil dos resíduos absolutos). (b) Construa o intervalo por CQR, com
# modelos quantílicos de 5% e 95%. (c) Compare cobertura e largura média
# **por tipo de estação**. Qual você entregaria ao time, e por quê?

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 5

# %%
alfa = 0.10
ano = bikes["dia"] // 365
Xa, ya = bikes.loc[ano == 0, features], bikes.loc[ano == 0, "retiradas"].to_numpy()
Xc, yc = bikes.loc[ano == 1, features], bikes.loc[ano == 1, "retiradas"].to_numpy()
Xt, yt = bikes.loc[ano == 2, features], bikes.loc[ano == 2, "retiradas"].to_numpy()
tipo_t = bikes.loc[ano == 2, "tipo"].to_numpy()
nc = len(yc)
nivel = np.ceil((nc + 1) * (1 - alfa)) / nc

m_media = HistGradientBoostingRegressor(loss="poisson", max_iter=300, random_state=0).fit(Xa, ya)
q_hat = np.quantile(np.abs(yc - m_media.predict(Xc)), nivel, method="higher")
lo_s, hi_s = m_media.predict(Xt) - q_hat, m_media.predict(Xt) + q_hat

m_lo = HistGradientBoostingRegressor(loss="quantile", quantile=alfa / 2, max_iter=300,
                                     random_state=0).fit(Xa, ya)
m_hi = HistGradientBoostingRegressor(loss="quantile", quantile=1 - alfa / 2, max_iter=300,
                                     random_state=0).fit(Xa, ya)
esc = np.maximum(m_lo.predict(Xc) - yc, yc - m_hi.predict(Xc))
corr = np.quantile(esc, nivel, method="higher")
lo_c, hi_c = m_lo.predict(Xt) - corr, m_hi.predict(Xt) + corr

tab = pd.DataFrame({"tipo": tipo_t,
                    "cobre simples": (yt >= lo_s) & (yt <= hi_s), "largura simples": hi_s - lo_s,
                    "cobre CQR": (yt >= lo_c) & (yt <= hi_c), "largura CQR": hi_c - lo_c})
resumo = tab.groupby("tipo").mean()
resumo.loc["TOTAL"] = tab.drop(columns="tipo").mean()
print(f"q_hat (simples) = {q_hat:.1f} | correção CQR = {corr:+.1f}")
print(resumo.round(3).to_string())

# %% [markdown]
# **Como avaliar sua resposta:** as duas construções cobrem cerca de 90% no
# **total** — é a garantia conformal. Mas o intervalo simples tem a mesma
# largura para uma estação residencial pacata e para uma estação comercial
# movimentada: cobre cerca de 94% nas residenciais e só cerca de 83% nas
# comerciais, onde a demanda é mais volátil. Um time de operações que
# confiar nele vai ser surpreendido exatamente nas estações que mais
# movimentam bicicletas. A CQR distribui
# a cobertura de forma muito mais uniforme entre os tipos de estação, com
# intervalos estreitos onde a demanda é previsível e largos onde não é — é
# a que se entrega. Ressalva honesta: a garantia conformal supõe que o ano
# de teste se comporta como o de calibração; se a demanda crescer (mais
# usuários, estações novas), a cobertura cai, e o intervalo precisa ser
# recalibrado periodicamente com dados recentes — tema do tema 13.

# %% [markdown]
# ---
# ## Fechamento
#
# - MAE, RMSE, MAPE e WAPE pesam os erros de formas diferentes; o MAPE
#   pune desproporcionalmente os dias fracos.
# - A perda escolhe a estatística: erro absoluto → mediana (viés para baixo
#   em alvos assimétricos), quadrático/Poisson → média.
# - Custos assimétricos pedem o quantil $c_u/(c_u + c_o)$.
# - Intervalos de previsão se julgam pela cobertura fora da amostra, de
#   preferência por segmento; a CQR entrega cobertura e adaptatividade.
#
# → Próximo módulo: **Validação Cruzada** — como estimar essas métricas de
# forma honesta.
