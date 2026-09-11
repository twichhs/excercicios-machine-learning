# %% [markdown]
# # Erro assimétrico e regressão quantílica
#
# **Tema:** Avaliação e Validação de Modelos › Métricas de Regressão
#
# Quando faltar e sobrar custam diferente, a previsão certa é um quantil.
# Este notebook confere que a perda pinball leva ao quantil, resolve o
# problema do jornaleiro para uma padaria (e mede o lucro de prever a média
# em vez do quantil), constrói intervalos de previsão por regressão
# quantílica, mede sua cobertura no teste e, por fim, implementa a previsão
# conformal — simples e quantílica — do zero.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(622)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
AZUL, VERDE, VERMELHO, ROXO, AMBAR = "#1F5C8B", "#1E6B4F", "#9C2B2B", "#5B3E86", "#8A6100"
print("pronto")

# %% [markdown]
# ## 1. A perda pinball leva ao quantil

# %%
def pinball(y, q, tau):
    return np.mean(np.where(y >= q, tau * (y - q), (1 - tau) * (q - y)))

amostra = rng.gamma(2, 50, 50000)
grade = np.linspace(0, 400, 4001)
for tau in [0.1, 0.5, 0.75, 0.95]:
    melhor = grade[np.argmin([pinball(amostra, c, tau) for c in grade])]
    print(f"tau={tau:.2f}: minimizador da pinball = {melhor:7.2f} | quantil empírico = "
          f"{np.quantile(amostra, tau):7.2f}")
print(f"conferência com o sklearn (tau=0,75, c=100): {pinball(amostra, 100, 0.75):.4f} vs "
      f"{mean_pinball_loss(amostra, np.full_like(amostra, 100), alpha=0.75):.4f}")

# %% [markdown]
# ## 2. O problema do jornaleiro: a padaria
#
# Margem de R$ 3 por pão vendido; custo de R$ 1 por pão descartado. O
# nível ótimo é $\tau^* = 3/(3+1) = 0{,}75$. Primeiro, o exemplo do
# `teoria.pdf` com demanda normal de média 200 e desvio 30.

# %%
MARGEM, DESCARTE = 3.0, 1.0
tau_otimo = MARGEM / (MARGEM + DESCARTE)
print(f"tau* = {tau_otimo:.2f} -> produzir {200 + norm.ppf(tau_otimo) * 30:.0f} pães "
      f"(média seria 200)")

def lucro_padaria(producao, demanda):
    vendidos = np.minimum(producao, demanda)
    return MARGEM * vendidos - DESCARTE * (producao - vendidos)

demanda_sim = rng.normal(200, 30, 100_000)
for nome, prod in [("média (200)", 200), ("quantil 75% (220)", 220), ("quantil 90% (238)", 238)]:
    print(f"produzir {nome:>18s}: lucro médio por dia = R$ {lucro_padaria(prod, demanda_sim).mean():.2f}")

# %% [markdown]
# Agora com um modelo: a demanda depende do dia da semana, da temperatura
# e de feriados, e o ruído é maior nos fins de semana. Comparamos três
# políticas de produção avaliadas pelo lucro num ano de teste.

# %%
n = 3 * 365
dia_semana = np.arange(n) % 7
temperatura = 22 + 6 * np.sin(2 * np.pi * np.arange(n) / 365) + rng.normal(0, 2, n)
feriado = (rng.random(n) < 0.03).astype(int)
fim_de_semana = (dia_semana >= 5).astype(int)
media_dem = 180 + 60 * fim_de_semana - 2.5 * (temperatura - 22) + 50 * feriado
demanda = rng.normal(media_dem, 20 + 25 * fim_de_semana).clip(0).round()
Xp = np.column_stack([dia_semana, temperatura, feriado])
treino, teste = np.arange(n) < 2 * 365, np.arange(n) >= 2 * 365

m_media = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(Xp[treino], demanda[treino])
m_q75 = HistGradientBoostingRegressor(loss="quantile", quantile=tau_otimo, max_iter=200,
                                      random_state=0).fit(Xp[treino], demanda[treino])
residuo_sd = np.std(demanda[treino] - m_media.predict(Xp[treino]))
politicas = {"prever a média": m_media.predict(Xp[teste]),
             "média + z*desvio (desvio único)": m_media.predict(Xp[teste]) + norm.ppf(tau_otimo) * residuo_sd,
             "regressão quantílica (0,75)": m_q75.predict(Xp[teste])}
for nome, prod in politicas.items():
    l = lucro_padaria(np.round(prod), demanda[teste])
    print(f"{nome:>32s}: lucro no ano de teste = R$ {l.sum():>9,.0f} | "
          f"falta em {np.mean(prod < demanda[teste]):.0%} dos dias")

# %% [markdown]
# **Leitura esperada:** produzir a média falta em cerca de metade dos dias
# e deixa lucro na mesa. Somar $z \cdot$desvio ajuda, mas usa **um único**
# desvio para todos os dias — pouco nos fins de semana (mais incertos) e
# demais nos dias úteis. A regressão quantílica aprende que o quantil de
# 75% fica mais acima da média justamente nos dias de maior incerteza, e
# tende a dar o maior lucro.

# %% [markdown]
# ## 3. Intervalos de previsão por regressão quantílica
#
# Um intervalo de 80% com os quantis de 10% e 90%. Medimos cobertura e
# largura no treino e no teste.

# %%
n = 6000
x = rng.uniform(0, 10, n)
yq = 20 + 4 * x + rng.normal(0, 1 + 0.8 * x, n)
X = x.reshape(-1, 1)
X_tr, X_te, y_tr, y_te = train_test_split(X, yq, test_size=0.5, random_state=0)
q = {}
for tau in [0.1, 0.9]:
    q[tau] = HistGradientBoostingRegressor(loss="quantile", quantile=tau, max_iter=300,
                                           max_leaf_nodes=63, min_samples_leaf=5,
                                           random_state=0).fit(X_tr, y_tr)
for nome, Xa, ya in [("treino", X_tr, y_tr), ("teste", X_te, y_te)]:
    lo, hi = q[0.1].predict(Xa), q[0.9].predict(Xa)
    print(f"{nome:>6s}: cobertura = {np.mean((ya >= lo) & (ya <= hi)):.1%} (nominal 80%) | "
          f"largura média = {np.mean(hi - lo):.2f} | cruzamentos (q10 > q90) = {np.mean(lo > hi):.1%}")

# %% [markdown]
# Um modelo quantílico flexível (muitas folhas, folhas pequenas) cobre mais
# no treino do que no teste: o intervalo é otimista fora da amostra. Os
# **cruzamentos** — pontos em que o quantil de 10% fica acima do de 90%,
# porque os dois modelos são treinados separadamente — são outro sintoma a
# monitorar.

# %% [markdown]
# ## 4. Previsão conformal por divisão
#
# Separamos um conjunto de calibração. O intervalo é a previsão ± o
# quantil corrigido dos resíduos absolutos de calibração. Alvo: 90%.

# %%
alfa = 0.10
X_tr2, X_cal, y_tr2, y_cal = train_test_split(X_tr, y_tr, test_size=0.4, random_state=1)
modelo = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(X_tr2, y_tr2)
residuos = np.abs(y_cal - modelo.predict(X_cal))
n_cal = len(residuos)
q_hat = np.quantile(residuos, np.ceil((n_cal + 1) * (1 - alfa)) / n_cal, method="higher")
pred = modelo.predict(X_te)
cobre = (y_te >= pred - q_hat) & (y_te <= pred + q_hat)
print(f"conformal simples: q_hat = {q_hat:.2f} | cobertura no teste = {cobre.mean():.1%} (alvo 90%) | "
      f"largura = {2 * q_hat:.2f} em todo ponto")

# %% [markdown]
# ## 5. Regressão quantílica conformalizada (CQR)
#
# Treinamos os quantis de 5% e 95% e corrigimos o intervalo com os escores
# de conformidade $E_i = \max(\hat{q}_{lo}(x_i) - y_i,\ y_i - \hat{q}_{hi}(x_i))$
# calculados na calibração.

# %%
q_lo = HistGradientBoostingRegressor(loss="quantile", quantile=alfa / 2, max_iter=300,
                                     random_state=0).fit(X_tr2, y_tr2)
q_hi = HistGradientBoostingRegressor(loss="quantile", quantile=1 - alfa / 2, max_iter=300,
                                     random_state=0).fit(X_tr2, y_tr2)
escores = np.maximum(q_lo.predict(X_cal) - y_cal, y_cal - q_hi.predict(X_cal))
corr = np.quantile(escores, np.ceil((n_cal + 1) * (1 - alfa)) / n_cal, method="higher")
lo, hi = q_lo.predict(X_te) - corr, q_hi.predict(X_te) + corr
print(f"CQR: correção = {corr:+.2f} | cobertura no teste = {np.mean((y_te >= lo) & (y_te <= hi)):.1%} | "
      f"largura média = {np.mean(hi - lo):.2f}")
lo_q, hi_q = q_lo.predict(X_te), q_hi.predict(X_te)
print(f"quantis sem conformalizar: cobertura = {np.mean((y_te >= lo_q) & (y_te <= hi_q)):.1%}")

faixas = pd.cut(X_te.ravel(), [0, 2.5, 5, 7.5, 10])
comp = pd.DataFrame({"faixa de x": faixas,
                     "cobre (conformal simples)": cobre,
                     "cobre (CQR)": (y_te >= lo) & (y_te <= hi),
                     "largura CQR": hi - lo}).groupby("faixa de x", observed=True).mean()
comp["largura conformal simples"] = 2 * q_hat
print("\n" + comp.round(3).to_string())

# %% [markdown]
# **Leitura esperada:** as duas versões conformais cobrem cerca de 90% no
# total — é a garantia. Mas o conformal simples usa a **mesma largura** em
# todo lugar: cobre demais onde o ruído é pequeno (x baixo) e de menos onde
# é grande (x alto). A CQR herda a adaptatividade dos quantis e distribui a
# cobertura de forma muito mais uniforme entre as faixas, com intervalos
# estreitos onde a incerteza é pequena.

# %% [markdown]
# ## O que levar deste notebook
#
# - A perda pinball no nível $\tau$ leva ao quantil $\tau$.
# - Com custos $c_u$ (faltar) e $c_o$ (sobrar), preveja o quantil
#   $c_u/(c_u + c_o)$ — não a média.
# - Intervalos de previsão se avaliam por cobertura **no teste** e largura.
# - Previsão conformal garante a cobertura com qualquer modelo; a CQR
#   garante a cobertura **e** se adapta à incerteza local.
#
# → Próximo: **Exercícios** do módulo.
