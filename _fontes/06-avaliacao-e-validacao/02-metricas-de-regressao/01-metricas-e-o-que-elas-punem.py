# %% [markdown]
# # O que cada métrica pune
#
# **Tema:** Avaliação e Validação de Modelos › Métricas de Regressão
#
# Cada métrica de regressão estima uma estatística diferente do alvo. Este
# notebook confere o exemplo numérico do `teoria.pdf`, encontra a melhor
# previsão constante de cada métrica, expõe as patologias do MAPE e do R²,
# mede o efeito de outliers e termina com o experimento que resume o
# módulo: modelos treinados com perdas diferentes, cada um vencendo na
# métrica com que foi treinado.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor, QuantileRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             mean_absolute_percentage_error, mean_pinball_loss)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

rng = np.random.default_rng(621)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
AZUL, VERDE, VERMELHO, ROXO, AMBAR = "#1F5C8B", "#1E6B4F", "#9C2B2B", "#5B3E86", "#8A6100"
print("pronto")

# %% [markdown]
# ## 1. O exemplo das cinco lojas

# %%
y = np.array([100, 120, 80, 200, 50], dtype=float)
yh = np.array([110, 115, 90, 150, 55], dtype=float)
e = y - yh
print(f"MAE  = {np.mean(np.abs(e)):.2f}  (sklearn {mean_absolute_error(y, yh):.2f})")
print(f"MSE  = {np.mean(e ** 2):.2f} | RMSE = {np.sqrt(np.mean(e ** 2)):.2f}")
print(f"MAPE = {np.mean(np.abs(e / y)):.2%}  (sklearn {mean_absolute_percentage_error(y, yh):.2%})")
print(f"WAPE = {np.abs(e).sum() / np.abs(y).sum():.2%}")
print(f"R²   = {1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2):.3f}  (sklearn {r2_score(y, yh):.3f})")
print(f"participação do maior erro na soma dos quadrados: {e[3] ** 2 / np.sum(e ** 2):.0%}")

# %% [markdown]
# ## 2. A melhor constante de cada métrica
#
# Tempos de entrega com cauda longa. Procuramos, numa grade, a constante
# que minimiza cada métrica e comparamos com a estatística que a teoria
# prevê.

# %%
tempos = rng.lognormal(np.log(35), 0.55, 20000)
grade = np.linspace(10, 90, 1601)
def argmin_const(perda):
    return grade[np.argmin([perda(c) for c in grade])]

ot_mse = argmin_const(lambda c: np.mean((tempos - c) ** 2))
ot_mae = argmin_const(lambda c: np.mean(np.abs(tempos - c)))
ot_mape = argmin_const(lambda c: np.mean(np.abs(tempos - c) / tempos))
ot_rmsle = argmin_const(lambda c: np.mean((np.log1p(tempos) - np.log1p(c)) ** 2))

def mediana_ponderada(x, w):
    o = np.argsort(x)
    acum = np.cumsum(w[o]) / w.sum()
    return x[o][np.searchsorted(acum, 0.5)]

print(f"MSE  : {ot_mse:6.2f} | média                         = {tempos.mean():6.2f}")
print(f"MAE  : {ot_mae:6.2f} | mediana                       = {np.median(tempos):6.2f}")
print(f"MAPE : {ot_mape:6.2f} | mediana ponderada por 1/y     = {mediana_ponderada(tempos, 1 / tempos):6.2f}")
print(f"RMSLE: {ot_rmsle:6.2f} | exp(média de log(1+y)) - 1    = {np.expm1(np.mean(np.log1p(tempos))):6.2f}")

# %% [markdown]
# O MAPE leva a uma previsão bem abaixo da mediana: é a mediana ponderada
# por $1/y$, que dá mais peso às entregas rápidas.

# %% [markdown]
# ## 3. MAPE: assimetria e explosão perto de zero

# %%
print(f"y=10, prevê 30: erro percentual = {abs(10 - 30) / 10:.0%}")
print(f"y=30, prevê 10: erro percentual = {abs(30 - 10) / 30:.0%}  (mesmo erro absoluto de 20)")

# carteira de 1.000 itens: 800 de alto giro e 200 de baixo giro (vendem 0 a 4 unidades)
alto = rng.poisson(500, 800).astype(float)
baixo = rng.poisson(1.0, 200).astype(float)
real = np.r_[alto, baixo]
# previsões com ~8% de erro no alto giro e ~40% no baixo giro (demanda intermitente)
prev = np.r_[alto * rng.normal(1, 0.08, 800), rng.lognormal(0, 0.4, 200)]
eh_baixo = np.r_[np.zeros(800, bool), np.ones(200, bool)]
com_valor = real > 0
ape = np.abs(real - prev) / np.where(com_valor, real, np.nan)
print(f"\nitens com venda zero (MAPE indefinido para eles): {np.sum(~com_valor)}")
print(f"baixo giro: {real[eh_baixo].sum() / real.sum():.2%} do volume")
print(f"MAPE do alto giro: {np.nanmean(ape[~eh_baixo]):.1%} | MAPE do baixo giro: {np.nanmean(ape[eh_baixo]):.1%}")
print(f"MAPE da carteira (sem os zeros): {np.nanmean(ape):.1%} — "
      f"{np.nansum(ape[eh_baixo]) / np.nansum(ape):.0%} da soma vem do baixo giro")
print(f"WAPE da carteira (com os zeros): {np.abs(real - prev).sum() / real.sum():.1%}")

# %% [markdown]
# Os itens de baixo giro somam uma fração minúscula do volume, mas
# respondem por uma fatia desproporcional do MAPE da carteira — e para os
# que não venderam nada o MAPE nem existe. O WAPE, ponderado pelo volume,
# conta a história que o negócio quer ouvir: o erro está onde está o
# dinheiro.

# %% [markdown]
# ## 4. Outliers: mínimos quadrados, Huber e mediana

# %%
x = rng.uniform(0, 10, 120)
yo = 3 + 2 * x + rng.normal(0, 1.5, 120)
idx_out = np.argsort(x)[-10:]
yo_sujo = yo.copy()
yo_sujo[idx_out] -= rng.uniform(15, 30, 10)
X = x.reshape(-1, 1)
for nome, m in [("mínimos quadrados", LinearRegression()), ("Huber", HuberRegressor()),
                ("mediana (MAE)", QuantileRegressor(quantile=0.5, alpha=0, solver="highs"))]:
    limpo = m.fit(X, yo).coef_[0]
    sujo = m.fit(X, yo_sujo).coef_[0]
    print(f"{nome:>18s}: inclinação nos dados limpos = {limpo:.2f} | com 10 outliers = {sujo:.2f}")

# %% [markdown]
# ## 5. R²: faixa, teste e a média de referência

# %%
for lo, hi in [(0, 100), (40, 60), (48, 52)]:
    xr = rng.uniform(lo, hi, 400)
    yr = 10 + xr + rng.normal(0, 5, 400)
    pred = 10 + xr
    print(f"x em [{lo:>2d}, {hi:>3d}]: RMSE = {np.sqrt(np.mean((yr - pred) ** 2)):.2f} | "
          f"R² = {r2_score(yr, pred):.3f}")

# R² negativo: uma árvore sem poda decora o treino e erra mais que a média no teste
xs = rng.uniform(0, 1, (300, 5))
ys = xs[:, 0] + rng.normal(0, 0.5, 300)
X_tr, X_te, y_tr, y_te = train_test_split(xs, ys, test_size=0.5, random_state=0)
arv = DecisionTreeRegressor(random_state=0).fit(X_tr, y_tr)
print(f"\nárvore sem poda: R² treino = {r2_score(y_tr, arv.predict(X_tr)):.3f} | "
      f"R² teste = {r2_score(y_te, arv.predict(X_te)):.3f}")
r2_media_treino = 1 - np.sum((y_te - arv.predict(X_te)) ** 2) / np.sum((y_te - y_tr.mean()) ** 2)
print(f"R² teste com a média do TREINO como referência: {r2_media_treino:.3f}")

# %% [markdown]
# ## 6. Cada modelo vence na métrica com que foi treinado
#
# Tempo de entrega em função da distância, do horário de pico e da chuva,
# com cauda longa. Treinamos o mesmo algoritmo com quatro perdas e
# avaliamos cada um com cinco métricas.

# %%
n = 20000
dist = rng.uniform(0.5, 12, n)
pico = (rng.random(n) < 0.3).astype(int)
chuva = (rng.random(n) < 0.15).astype(int)
mediana_cond = 12 + 3.2 * dist + 8 * pico + 10 * chuva
tempo = mediana_cond * rng.lognormal(0, 0.35, n)
Xd = np.column_stack([dist, pico, chuva])
X_tr, X_te, y_tr, y_te = train_test_split(Xd, tempo, test_size=0.5, random_state=0)

perdas = {"squared_error": {}, "absolute_error": {}, "gamma": {},
          "quantile (0,9)": {"loss": "quantile", "quantile": 0.9}}
previsoes = {}
for nome, kw in perdas.items():
    kw = kw or {"loss": nome}
    previsoes[nome] = HistGradientBoostingRegressor(max_iter=200, random_state=0, **kw) \
        .fit(X_tr, y_tr).predict(X_te)

tabela = pd.DataFrame({
    nome: {"MAE": mean_absolute_error(y_te, p),
           "RMSE": np.sqrt(mean_squared_error(y_te, p)),
           "MAPE": mean_absolute_percentage_error(y_te, p),
           "RMSLE": np.sqrt(np.mean((np.log1p(y_te) - np.log1p(p)) ** 2)),
           "pinball 0,9": mean_pinball_loss(y_te, p, alpha=0.9)}
    for nome, p in previsoes.items()}).T
print("linhas = perda de treino; colunas = métrica de avaliação (menor é melhor)")
print(tabela.round(3).to_string())
print("\nvencedor de cada métrica:", tabela.idxmin().to_dict())

# %% [markdown]
# **Leitura esperada:** os dois modelos que estimam a **média** condicional
# — erro quadrático e Gama (a deviance Gama também leva à média) — ficam
# praticamente empatados no RMSE, com leve vantagem da Gama, que lida
# melhor com ruído multiplicativo. O treinado com erro absoluto vence no
# MAE — e também no MAPE e no RMSLE: com ruído lognormal, a mediana
# coincide com a média geométrica (o alvo do RMSLE) e fica mais perto do
# ótimo do MAPE do que a média. O quantílico vence na pinball de 0,9 e perde
# feio em todo o resto, porque prever o percentil 90 é, de propósito,
# prever alto. Não existe "o melhor modelo" em abstrato: existe o melhor
# modelo **para uma métrica**, e a métrica tem de vir da decisão de negócio.

# %% [markdown]
# ## O que levar deste notebook
#
# - MSE → média, MAE → mediana, MAPE → mediana ponderada por $1/y$ (baixa),
#   RMSLE → média geométrica, pinball → quantil.
# - MAPE explode com itens de baixo giro; WAPE é a alternativa ponderada.
# - Mínimos quadrados cedem a outliers; Huber e mediana resistem.
# - R² depende da faixa dos dados e pode ser negativo no teste.
# - Treine e avalie com a mesma métrica — a métrica que o negócio precisa.
#
# → Próximo: **Erro assimétrico e regressão quantílica**.
