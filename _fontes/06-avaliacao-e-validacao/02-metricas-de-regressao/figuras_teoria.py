"""Gera as figuras do teoria.pdf do módulo 02-metricas-de-regressao."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ / "_ferramentas"))
from estilo_figuras import AZUL, VERDE, VERMELHO, ROXO, AMBAR, CINZA, salva  # noqa: E402

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor, QuantileRegressor

rng = np.random.default_rng(62)
MODULO_REL = Path(__file__).resolve().parent.relative_to(RAIZ / "_fontes")
DESTINO = RAIZ / MODULO_REL / "figuras"

# ---------------------------------------------------------------------------
# 1. Funções de perda em função do resíduo
# ---------------------------------------------------------------------------
e = np.linspace(-4, 4, 400)
huber = np.where(np.abs(e) <= 1, 0.5 * e ** 2, np.abs(e) - 0.5)
tau = 0.9
pinball = np.where(e >= 0, tau * e, (tau - 1) * e)
fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.plot(e, e ** 2, color=VERMELHO, lw=2.2, label="quadrática (MSE)")
ax.plot(e, np.abs(e), color=AZUL, lw=2.2, label="absoluta (MAE)")
ax.plot(e, huber, color=VERDE, lw=2.2, label="Huber (delta = 1)")
ax.plot(e, pinball, color=ROXO, lw=2.2, label="pinball (tau = 0,9)")
ax.set_ylim(0, 6); ax.set_xlabel("resíduo  y - ŷ"); ax.set_ylabel("perda")
ax.set_title("Cada métrica pune o resíduo de um jeito: a quadrática explode nos erros grandes;\n"
             "a pinball pune errar para baixo 9 vezes mais que errar para cima", fontsize=11)
ax.legend()
salva(fig, DESTINO / "funcoes-de-perda.png")

# ---------------------------------------------------------------------------
# 2. A constante ótima de cada métrica numa distribuição assimétrica
# ---------------------------------------------------------------------------
tempos = rng.lognormal(np.log(35), 0.55, 20000)          # tempo de entrega (min)
cs = np.linspace(10, 90, 801)
mse = np.array([np.mean((tempos - c) ** 2) for c in cs])
mae = np.array([np.mean(np.abs(tempos - c)) for c in cs])
mape = np.array([np.mean(np.abs(tempos - c) / tempos) for c in cs])
pin = np.array([np.mean(np.where(tempos >= c, 0.9 * (tempos - c), 0.1 * (c - tempos))) for c in cs])
otimos = {"MSE": cs[mse.argmin()], "MAE": cs[mae.argmin()], "MAPE": cs[mape.argmin()],
          "pinball 0,9": cs[pin.argmin()]}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))
ax1.hist(tempos, bins=120, range=(0, 150), color="#BBBBBB", density=True)
for (nome, c), cor in zip(otimos.items(), [VERMELHO, AZUL, AMBAR, ROXO]):
    ax1.axvline(c, color=cor, lw=2.2, label=f"ótimo de {nome}: {c:.1f} min")
ax1.set_xlabel("tempo de entrega (min)"); ax1.legend(fontsize=9)
ax1.set_title("A melhor previsão constante depende da métrica")
for curva, nome, cor in [(mse / mse.min(), "MSE", VERMELHO), (mae / mae.min(), "MAE", AZUL),
                         (mape / mape.min(), "MAPE", AMBAR), (pin / pin.min(), "pinball 0,9", ROXO)]:
    ax2.plot(cs, curva, color=cor, lw=2, label=nome)
ax2.set_ylim(0.95, 2.0); ax2.set_xlabel("previsão constante c (min)")
ax2.set_ylabel("perda / perda mínima"); ax2.legend(fontsize=9)
ax2.set_title("Cada curva tem o mínimo num lugar diferente")
fig.tight_layout()
salva(fig, DESTINO / "otimo-de-cada-metrica.png")
print({k: round(v, 1) for k, v in otimos.items()},
      f"media={tempos.mean():.1f} mediana={np.median(tempos):.1f} p90={np.quantile(tempos, 0.9):.1f}")

# ---------------------------------------------------------------------------
# 3. Outliers: MSE (MQO) vs Huber vs MAE (mediana condicional)
# ---------------------------------------------------------------------------
x = rng.uniform(0, 10, 120)
y = 3 + 2 * x + rng.normal(0, 1.5, 120)
out = np.argsort(x)[-10:]
y[out] -= rng.uniform(15, 30, 10)   # 10 pontos com erro de registro no fim da faixa
Xm = x.reshape(-1, 1)
grade = np.linspace(0, 10, 100).reshape(-1, 1)
modelos = [("MSE (mínimos quadrados)", LinearRegression(), VERMELHO),
           ("Huber", HuberRegressor(epsilon=1.35), VERDE),
           ("MAE (regressão na mediana)", QuantileRegressor(quantile=0.5, alpha=0, solver="highs"), AZUL)]
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.scatter(x, y, s=14, color=CINZA, alpha=0.7)
ax.scatter(x[out], y[out], s=30, color=VERMELHO, marker="x", label="10 registros errados")
ax.plot(grade, 3 + 2 * grade, "k--", lw=1.3, label="relação verdadeira (inclinação 2)")
for nome, m, cor in modelos:
    m.fit(Xm, y)
    ax.plot(grade, m.predict(grade), color=cor, lw=2.2, label=f"{nome}: inclinação {m.coef_[0]:.2f}")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(fontsize=9)
ax.set_title("Dez registros errados puxam a reta de mínimos quadrados;\nHuber e MAE resistem", fontsize=11)
salva(fig, DESTINO / "outliers-e-perda.png")

# ---------------------------------------------------------------------------
# 4. R² depende da variância de y na amostra
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
for ax, (lo, hi) in zip(axes, [(0, 100), (40, 60)]):
    xr = rng.uniform(lo, hi, 300)
    yr = 10 + 1.0 * xr + rng.normal(0, 5, 300)
    pred = 10 + 1.0 * xr                       # o MESMO modelo (o verdadeiro)
    rmse = np.sqrt(np.mean((yr - pred) ** 2))
    r2 = 1 - np.sum((yr - pred) ** 2) / np.sum((yr - yr.mean()) ** 2)
    ax.scatter(xr, yr, s=10, color=AZUL, alpha=0.6)
    ax.plot([lo, hi], [10 + lo, 10 + hi], color=VERMELHO, lw=2)
    ax.set_title(f"x entre {lo} e {hi}: RMSE = {rmse:.2f}, R² = {r2:.2f}", fontsize=11)
    ax.set_xlabel("x"); ax.set_xlim(-2, 102)
    print(f"R2 faixa {lo}-{hi}: {r2:.3f}, RMSE {rmse:.2f}")
axes[0].set_ylabel("y")
fig.suptitle("O mesmo modelo, o mesmo erro típico: o R² muda só porque a faixa de x mudou",
             fontsize=11.5, fontweight="bold", y=1.02)
salva(fig, DESTINO / "r2-depende-da-faixa.png")

# ---------------------------------------------------------------------------
# 5. Regressão quantílica: intervalos de previsão com variância que cresce
# ---------------------------------------------------------------------------
xq = rng.uniform(0, 10, 1500)
yq = 20 + 4 * xq + rng.normal(0, 1 + 0.8 * xq, 1500)
Xq = xq.reshape(-1, 1)
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(xq, yq, s=6, color=CINZA, alpha=0.5)
preds = {}
for q, cor in [(0.1, AZUL), (0.5, "black"), (0.9, VERMELHO)]:
    m = GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=200, max_depth=2,
                                  learning_rate=0.05, random_state=0).fit(Xq, yq)
    preds[q] = m.predict(Xq)
    ax.plot(grade, m.predict(grade), color=cor, lw=2.2, label=f"quantil {q:.0%}")
cobertura = np.mean((yq >= preds[0.1]) & (yq <= preds[0.9]))
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend()
ax.set_title(f"Regressão quantílica: a faixa 10%–90% se alarga onde o ruído cresce\n"
             f"(cobertura observada no treino: {cobertura:.1%}, nominal 80%)", fontsize=11)
salva(fig, DESTINO / "regressao-quantilica.png")
print(f"cobertura 10-90: {cobertura:.3f}")
print("done")
