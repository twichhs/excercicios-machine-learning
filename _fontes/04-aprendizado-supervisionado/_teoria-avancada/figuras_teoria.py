"""Gera as figuras do livro bônus `teoria-avancada.pdf` (tema 04):
validação, tuning e deploy de modelos. Não é um módulo com notebooks — é um
capítulo extra do tema, então as figuras vão direto em
`04-aprendizado-supervisionado/figuras/`, não numa subpasta de módulo.
"""
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ / "_ferramentas"))
from estilo_figuras import AZUL, VERDE, VERMELHO, ROXO, AMBAR, CINZA, CINZA_CLARO, salva  # noqa: E402

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, LeaveOneOut, cross_val_score,
)

rng = np.random.default_rng(11)
DESTINO = RAIZ / "04-aprendizado-supervisionado" / "figuras"

# ---------------------------------------------------------------------------
# 1. O pipeline real de um modelo em produção (capítulo 1)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11.5, 5.4))
ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis("off")

caixas = [
    ("Dados\nbrutos",                0.9, 3.6, AZUL),
    ("train_test_split\n(cap. 2)",   3.2, 3.6, AZUL),
    ("K-Fold +\ntuning (cap. 3-4)",  5.7, 3.6, ROXO),
    ("Balanceamento\n(tema 3)",      8.2, 3.6, ROXO),
    ("Modelo final\nempacotado",     8.2, 1.2, VERDE),
    ("API + Docker\n(cap. 5)",       5.7, 1.2, VERDE),
    ("Monitoramento\ne retreino",    3.2, 1.2, VERMELHO),
]
centros = {}
for nome, x, y, cor in caixas:
    ax.add_patch(FancyBboxPatch((x - 1.05, y - 0.55), 2.1, 1.1,
                 boxstyle="round,pad=0.06,rounding_size=0.15",
                 linewidth=1.8, edgecolor=cor, facecolor="white", zorder=3))
    ax.text(x, y, nome, ha="center", va="center", fontsize=9.1,
            fontweight="bold", color=cor, zorder=4)
    centros[nome] = (x, y)

seq = [c[0] for c in caixas]
for a, b in zip(seq, seq[1:]):
    x1, y1 = centros[a]; x2, y2 = centros[b]
    if abs(y1 - y2) < 0.01:
        p1 = (x1 + 1.05, y1); p2 = (x2 - 1.05, y2)
    else:
        p1 = (x1, y1 - 0.55); p2 = (x2, y2 + 0.55)
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle="-|>", color=CINZA, lw=1.8))

x1, y1 = centros["Monitoramento\ne retreino"]
x2, y2 = centros["Dados\nbrutos"]
ax.annotate("", xy=(x2, y2 - 0.55), xytext=(x1, y1 - 0.55),
            arrowprops=dict(arrowstyle="-|>", color=VERMELHO, lw=1.6, ls="--",
                             connectionstyle="arc3,rad=0.35"))
ax.text((x1 + x2) / 2, 0.15, "drift detectado → retreina", fontsize=8.3,
        color=VERMELHO, ha="center", style="italic")
ax.set_title("O pipeline real de um modelo em produção — cada seta é um "
              "capítulo deste livro", fontsize=12, fontweight="bold", pad=10)
salva(fig, DESTINO / "pipeline-producao.png")

# ---------------------------------------------------------------------------
# 2. A variância de um único split (capítulo 2)
# ---------------------------------------------------------------------------
X, y = make_classification(n_samples=1200, n_features=12, n_informative=6,
                            weights=[0.5, 0.5], random_state=3)
accs = []
for seed in range(300):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    modelo = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    accs.append(modelo.score(X_te, y_te))
accs = np.array(accs)
media, desvio = accs.mean(), accs.std()
print(f"[fig2] media={media:.4f} desvio={desvio:.4f} min={accs.min():.4f} max={accs.max():.4f}")

fig, ax = plt.subplots(figsize=(8.5, 5.3))
ax.hist(accs, bins=24, color=AZUL, alpha=0.75, edgecolor="white")
ax.axvline(media, color=VERMELHO, lw=2, label=f"média = {media:.3f}")
ax.axvline(media - desvio, color=VERMELHO, lw=1, ls=":")
ax.axvline(media + desvio, color=VERMELHO, lw=1, ls=":",
           label=f"±1 desvio-padrão ({desvio:.3f})")
ax.set_xlabel("acurácia no conjunto de teste (20% dos dados)")
ax.set_ylabel("frequência (300 splits aleatórios diferentes)")
ax.set_title("O MESMO modelo, o MESMO dataset, 300 divisões aleatórias\n"
              "diferentes: a acurácia de um único split é uma loteria", fontsize=11)
ax.legend(fontsize=9)
salva(fig, DESTINO / "variancia-split-unico.png")

# ---------------------------------------------------------------------------
# 3. Erro-padrão da acurácia vs. tamanho do teste (capítulo 2)
# ---------------------------------------------------------------------------
n_vals = np.arange(20, 2001, 5)
fig, ax = plt.subplots(figsize=(8.2, 5.3))
for p, cor in zip([0.95, 0.85, 0.70], [VERDE, AZUL, AMBAR]):
    se = np.sqrt(p * (1 - p) / n_vals)
    ax.plot(n_vals, se, color=cor, lw=2.2, label=f"p = {p:.2f}")
ax.axvline(100, color=CINZA, lw=1, ls=":")
ax.axvline(1000, color=CINZA, lw=1, ls=":")
ax.set_xlabel("tamanho do conjunto de teste (n_teste)")
ax.set_ylabel("erro-padrão da acurácia estimada")
ax.set_title("Quanto menor o conjunto de teste, mais incerta é a acurácia\n"
              r"reportada — o erro cai com $\sqrt{n}$, não com $n$", fontsize=11)
ax.legend(fontsize=9)
salva(fig, DESTINO / "erro-padrao-vs-n.png")

# ---------------------------------------------------------------------------
# 4. Esquema de 5-fold cross-validation (capítulo 3)
# ---------------------------------------------------------------------------
K = 5
fig, ax = plt.subplots(figsize=(9, 4.6))
largura = 1.0
for k in range(K):
    for j in range(K):
        cor = VERMELHO if j == k else AZUL
        alpha = 0.85 if j == k else 0.30
        ax.add_patch(plt.Rectangle((j * largura, (K - 1 - k) * largura),
                                    largura * 0.94, largura * 0.8,
                                    facecolor=cor, alpha=alpha, edgecolor="white"))
    ax.text(-0.35, (K - 1 - k) * largura + 0.4, f"iteração {k + 1}", fontsize=8.6,
            ha="right", va="center", color=CINZA)
ax.set_xlim(-1.9, K * largura); ax.set_ylim(-0.3, K * largura + 0.3)
ax.axis("off")
leg = [mpatches.Patch(facecolor=AZUL, alpha=0.30, label="treino"),
       mpatches.Patch(facecolor=VERMELHO, alpha=0.85, label="teste (validação)")]
ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, 1.1),
          ncol=2, fontsize=9.5, frameon=False)
ax.set_title("5-fold cross-validation: cada linha é uma iteração; o bloco\n"
              "vermelho (teste) muda de posição — todo dado vira teste exatamente uma vez",
              fontsize=10.8, y=1.2)
salva(fig, DESTINO / "kfold-esquema.png")

# ---------------------------------------------------------------------------
# 5. Custo x variância de K, medido de verdade (capítulo 3)
# ---------------------------------------------------------------------------
X2, y2 = make_classification(n_samples=400, n_features=10, n_informative=5, random_state=5)
resultados = []
for Kv in [2, 3, 5, 10, 20]:
    t0 = time.time()
    skf = StratifiedKFold(n_splits=Kv, shuffle=True, random_state=0)
    scores = cross_val_score(LogisticRegression(max_iter=1000), X2, y2, cv=skf)
    dt = time.time() - t0
    resultados.append((Kv, scores.mean(), scores.std(), dt))
    print(f"[fig5] K={Kv:>2}  media={scores.mean():.4f}  desvio={scores.std():.4f}  tempo={dt:.3f}s")

loo = LeaveOneOut()
t0 = time.time()
scores_loo = cross_val_score(LogisticRegression(max_iter=1000), X2, y2, cv=loo)
dt_loo = time.time() - t0
print(f"[fig5] LOOCV  media={scores_loo.mean():.4f}  desvio={scores_loo.std():.4f}  "
      f"tempo={dt_loo:.3f}s  n_fits={len(scores_loo)}")

Ks = [r[0] for r in resultados]
medias = [r[1] for r in resultados]
desvios = [r[2] for r in resultados]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
axes[0].errorbar(Ks, medias, yerr=desvios, fmt="o-", color=AZUL, capsize=4, lw=2)
axes[0].set_xlabel("K (número de folds)")
axes[0].set_ylabel("acurácia (média ± desvio entre folds)")
axes[0].set_title("Mais folds: a média muda pouco,\no desvio entre folds também", fontsize=10.5)
tempos = [r[3] for r in resultados] + [dt_loo]
labels = [str(k) for k in Ks] + ["LOOCV\n(n=400)"]
axes[1].bar(labels, tempos, color=[AZUL] * len(Ks) + [VERMELHO])
axes[1].set_ylabel("tempo de treino (segundos)")
axes[1].set_title("Custo computacional: LOOCV treina\n400 modelos para este dataset", fontsize=10.5)
fig.suptitle("O trade-off real de escolher K, medido neste dataset (não teórico)",
             fontsize=12, y=1.03)
salva(fig, DESTINO / "custo-e-variancia-k.png")

# ---------------------------------------------------------------------------
# 6. Grid search vs. random search: cobertura do espaço (capítulo 4)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
grid_vals = np.linspace(0.15, 0.85, 3)
gx, gy = np.meshgrid(grid_vals, grid_vals)
axes[0].scatter(gx.ravel(), gy.ravel(), s=110, color=AZUL, zorder=3)
for v in grid_vals:
    axes[0].axvline(v, color=CINZA_CLARO, lw=1, zorder=1)
axes[0].set_title("Grid search: 9 avaliações,\napenas 3 valores distintos do\n"
                   "hiperparâmetro importante (eixo x)", fontsize=10)

rng_gr = np.random.default_rng(42)
rx = rng_gr.uniform(0.05, 0.95, 9)
ry = rng_gr.uniform(0.05, 0.95, 9)
axes[1].scatter(rx, ry, s=110, color=VERMELHO, zorder=3)
for v in rx:
    axes[1].axvline(v, color=CINZA_CLARO, lw=0.8, zorder=1)
axes[1].set_title("Random search: 9 avaliações,\n9 valores distintos do\n"
                   "hiperparâmetro importante (eixo x)", fontsize=10)

for ax in axes:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("hiperparâmetro importante (ex.: learning_rate)")
    ax.set_ylabel("hiperparâmetro sem efeito real neste problema")
fig.suptitle("Mesmo orçamento (9 avaliações): random search explora o eixo\n"
             "que importa muito melhor do que grid search", fontsize=12, y=1.04)
salva(fig, DESTINO / "grid-vs-random.png")

# ---------------------------------------------------------------------------
# 7. Probabilidade de acertar a região boa com random search (capítulo 4)
# ---------------------------------------------------------------------------
ns = np.arange(1, 201)
fig, ax = plt.subplots(figsize=(8.2, 5.2))
for gamma, cor in zip([0.01, 0.05, 0.10], [VERMELHO, AZUL, VERDE]):
    p = 1 - (1 - gamma) ** ns
    ax.plot(ns, p, color=cor, lw=2.2, label=fr"$\gamma$ = {gamma:.2f} (top {gamma * 100:.0f}%)")
ax.axhline(0.95, color=CINZA, lw=1, ls=":")
ax.set_xlabel("número de configurações testadas (n)")
ax.set_ylabel(r"P(pelo menos 1 tentativa no top $\gamma$)")
ax.set_title("Random search: quantas tentativas para ter 95% de chance\n"
             "de acertar a região boa do espaço de hiperparâmetros?", fontsize=11)
ax.legend(fontsize=9)
salva(fig, DESTINO / "random-search-probabilidade.png")

# ---------------------------------------------------------------------------
# 8. Overfitting ao conjunto de validação por excesso de tuning (capítulo 4)
# ---------------------------------------------------------------------------
n_val = 300
n_configs = 2000
rng3 = np.random.default_rng(99)
resultados_configs = rng3.binomial(n_val, 0.5, n_configs) / n_val
melhor_ate_agora = np.maximum.accumulate(resultados_configs)
print(f"[fig8] melhor apos 10 configs: {melhor_ate_agora[9]:.4f}; "
      f"apos 100: {melhor_ate_agora[99]:.4f}; apos 1000: {melhor_ate_agora[999]:.4f}; "
      f"apos 2000: {melhor_ate_agora[-1]:.4f} (verdade = 0.5000)")

fig, ax = plt.subplots(figsize=(8.6, 5.3))
ax.plot(np.arange(1, n_configs + 1), melhor_ate_agora, color=VERMELHO, lw=2)
ax.axhline(0.5, color=CINZA, lw=1.4, ls="--", label="desempenho real do modelo (0,50 — puro acaso)")
ax.set_xscale("log")
ax.set_xlabel("número de configurações de hiperparâmetro testadas (escala log)")
ax.set_ylabel("melhor acurácia de validação observada até agora")
ax.set_title("Nenhuma das configurações tem sinal real — a acurácia \"sobe\"\n"
              "só porque testamos muitas e reportamos a melhor", fontsize=11)
ax.legend(fontsize=9)
salva(fig, DESTINO / "overfitting-tuning-validacao.png")

# ---------------------------------------------------------------------------
# 9. Latência de uma API: por que percentis, não média (capítulo 5)
# ---------------------------------------------------------------------------
rng4 = np.random.default_rng(2024)
latencias = rng4.lognormal(mean=np.log(45), sigma=0.5, size=20000)
p50, p95, p99 = np.percentile(latencias, [50, 95, 99])
print(f"[fig9] p50={p50:.1f}ms p95={p95:.1f}ms p99={p99:.1f}ms media={latencias.mean():.1f}ms")

fig, ax = plt.subplots(figsize=(8.6, 5.3))
ax.hist(latencias[latencias < 300], bins=80, color=AZUL, alpha=0.75, edgecolor="white")
for val, cor, nome in [(p50, VERDE, "p50"), (p95, AMBAR, "p95"), (p99, VERMELHO, "p99")]:
    ax.axvline(val, color=cor, lw=2, label=f"{nome} = {val:.0f} ms")
ax.set_xlabel("latência da requisição (ms)")
ax.set_ylabel("número de requisições")
ax.set_title("Latência de uma API de inferência é assimétrica: a média\n"
              "esconde a experiência dos usuários mais azarados", fontsize=11)
ax.legend(fontsize=9.5)
salva(fig, DESTINO / "latencia-percentis.png")

# ---------------------------------------------------------------------------
# 10. Deploy canário / shadow: roteamento de tráfego (capítulo 5)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5)

ax.add_patch(FancyBboxPatch((0.2, 2.1), 1.8, 0.9, boxstyle="round,pad=0.06",
             edgecolor=CINZA, facecolor="white", lw=1.6))
ax.text(1.1, 2.55, "Tráfego\nde produção", ha="center", va="center",
        fontsize=9.2, fontweight="bold")

ax.add_patch(FancyBboxPatch((4.0, 3.4), 2.4, 1.0, boxstyle="round,pad=0.06",
             edgecolor=AZUL, facecolor="white", lw=1.8))
ax.text(5.2, 3.9, "Modelo atual\n(95% do tráfego)", ha="center", va="center",
        fontsize=9, color=AZUL, fontweight="bold")

ax.add_patch(FancyBboxPatch((4.0, 0.7), 2.4, 1.0, boxstyle="round,pad=0.06",
             edgecolor=VERMELHO, facecolor="white", lw=1.8))
ax.text(5.2, 1.2, "Modelo novo\n(5% do tráfego — canário)", ha="center", va="center",
        fontsize=9, color=VERMELHO, fontweight="bold")

ax.add_patch(FancyBboxPatch((7.3, 2.1), 2.4, 0.9, boxstyle="round,pad=0.06",
             edgecolor=VERDE, facecolor="white", lw=1.8))
ax.text(8.5, 2.55, "Comparação\nde métricas", ha="center", va="center",
        fontsize=9, color=VERDE, fontweight="bold")

ax.annotate("", xy=(4.0, 3.75), xytext=(2.0, 2.75),
            arrowprops=dict(arrowstyle="-|>", color=CINZA, lw=1.6,
                             connectionstyle="arc3,rad=0.25"))
ax.annotate("", xy=(4.0, 1.35), xytext=(2.0, 2.35),
            arrowprops=dict(arrowstyle="-|>", color=CINZA, lw=1.6,
                             connectionstyle="arc3,rad=-0.25"))
ax.annotate("", xy=(7.3, 2.75), xytext=(6.4, 3.75),
            arrowprops=dict(arrowstyle="-|>", color=CINZA, lw=1.6,
                             connectionstyle="arc3,rad=-0.25"))
ax.annotate("", xy=(7.3, 2.35), xytext=(6.4, 1.35),
            arrowprops=dict(arrowstyle="-|>", color=CINZA, lw=1.6,
                             connectionstyle="arc3,rad=0.25"))
ax.set_title("Deploy canário: o modelo novo recebe uma fatia pequena do\n"
              "tráfego real antes de assumir 100% — erros custam pouco se aparecerem",
              fontsize=11.5, fontweight="bold")
salva(fig, DESTINO / "canario-shadow.png")

print("done")
