# %% [markdown]
# # Grid e Random Search
#
# **Tema:** Avaliação e Validação de Modelos › Otimização de Hiperparâmetros
#
# Este módulo não tem `teoria.pdf`: a teoria mora aqui, ao lado do código
# que a demonstra. O livro dois do tema 4 (`teoria-avancada.pdf`)
# apresentou o catálogo: grid, random, otimização bayesiana e Hyperband.
# Este notebook trata a busca como o que ela é, **uma otimização de uma
# função ruidosa e cara sob orçamento**, e tira disso as decisões
# práticas: em que escala sortear, por que random vence grid, quando
# parar, como gastar o orçamento com Successive Halving e quanto vale o
# escore do vencedor.
#
# ## Por que este módulo existe
#
# Tuning é a etapa mais cara do ciclo de um modelo, e a que mais se faz
# no automático: alguém copia uma grade de um tutorial, roda por uma
# noite e reporta o melhor número. Três coisas costumam sair erradas:
#
# - a grade gasta quase todo o orçamento em hiperparâmetros que não
#   importam, e em valores mal espaçados dos que importam;
# - a busca para cedo demais ou tarde demais, porque ninguém olhou a
#   curva de "melhor até agora";
# - o escore do vencedor é reportado como se fosse a estimativa de
#   produção, e o módulo 3 já mostrou por que não é.
#
# **Analogia.** Você tem 16 jantares para descobrir o melhor restaurante
# de uma cidade nova. Grid: escolhe 4 ruas e prova 4 restaurantes em
# cada. Se a qualidade depende do bairro e não da rua, você provou só 4
# bairros. Random: sorteia 16 endereços; provou 16 bairros. Successive
# Halving: pede uma entrada em 32 lugares, volta para o prato principal
# nos 16 melhores, para a sobremesa nos 8 melhores. Otimização bayesiana
# (próximo notebook): depois de cada jantar, atualiza um mapa mental de
# "onde deve ser bom" e escolhe o próximo endereço por ele.
#
# ### O que você vai conseguir fazer ao final
#
# - Tratar o escore de CV como uma função objetivo ruidosa e saber o
#   tamanho do ruído antes de comparar configurações.
# - Escolher a escala (linear, log, inteira) de cada hiperparâmetro.
# - Explicar e reproduzir por que random search vence grid search com o
#   mesmo orçamento.
# - Ler a curva "melhor até agora × orçamento" e decidir quando parar.
# - Implementar Successive Halving e saber que tipo de configuração ele
#   mata cedo demais.
# - Reportar o desempenho do vencedor sem o otimismo da seleção.

# %%
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import (GridSearchCV, HalvingRandomSearchCV, ParameterGrid, ParameterSampler,
                                     RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split)
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(661)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
AZUL, VERDE, VERMELHO, ROXO, AMBAR, CINZA = "#1F5C8B", "#1E6B4F", "#9C2B2B", "#5B3E86", "#8A6100", "#5A5A5A"
print("pronto")

# %% [markdown]
# ## 1. A função objetivo é ruidosa e cara
#
# Tuning é maximizar $J(\theta) = \text{CV}(\theta)$, o escore de validação
# cruzada de uma configuração $\theta$. Duas propriedades de $J$ decidem
# tudo o que vem depois:
#
# - **É cara.** Cada avaliação custa $K$ treinos. Orçamento é o número de
#   avaliações, $B$.
# - **É ruidosa.** O módulo 3 mostrou que a CV tem ruído de partição e de
#   amostra. Duas avaliações da **mesma** $\theta$ com sementes diferentes
#   dão números diferentes.
#
# A base de trabalho é um problema de churn sintético com 4.000 clientes,
# 8 features e um boosting. Antes de buscar qualquer coisa, medimos o
# ruído de $J$: a mesma configuração, 20 partições diferentes.

# %%
def gera_churn(n, semente):
    g = np.random.default_rng(semente)
    X = g.normal(0, 1, (n, 8))
    lg = (-0.8 + 1.1 * X[:, 0] - 0.9 * X[:, 1] + 0.8 * X[:, 0] * X[:, 2] + 0.6 * np.sin(2 * X[:, 3])
          + 0.5 * (X[:, 4] > 1) - 0.4 * X[:, 5] ** 2)
    return X, (g.random(n) < 1 / (1 + np.exp(-lg))).astype(int)

X, y = gera_churn(4000, 0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
print(f"treino {len(y_tr)} | teste (só no fim) {len(y_te)} | prevalência {y.mean():.1%}")

def J(params, semente=0, X=X_tr, y=y_tr, k=3):
    """Escore de CV (AUC) de uma configuração: a função objetivo do tuning."""
    m = HistGradientBoostingClassifier(max_iter=150, random_state=0, **params)
    return cross_val_score(m, X, y, cv=StratifiedKFold(k, shuffle=True, random_state=semente),
                           scoring="roc_auc").mean()

config = {"learning_rate": 0.1, "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 0.0}
t0 = time.time()
ruido = np.array([J(config, semente=s) for s in range(20)])
print(f"mesma configuração, 20 partições: AUC entre {ruido.min():.4f} e {ruido.max():.4f} | "
      f"dp = {ruido.std():.4f} | {(time.time() - t0) / 20:.2f} s por avaliação")
RUIDO = ruido.std()

# %% [markdown]
# Guarde esse desvio-padrão. Qualquer diferença entre duas configurações
# menor que uns 2 desvios **não é uma decisão**, é sorteio de partição. A
# consequência prática: no fim da busca, o "melhor" e o quinto melhor
# costumam ser indistinguíveis, e a escolha entre eles deve ser feita por
# outro critério (simplicidade, custo de inferência, estabilidade).

# %% [markdown]
# ## 2. Em que escala sortear
#
# Hiperparâmetros multiplicativos (taxa de aprendizado, `C`, `alpha`,
# `l2_regularization`, `gamma`) vivem em escala **log**: a diferença entre
# 0,001 e 0,01 importa tanto quanto entre 0,1 e 1. Sortear uniforme em
# [0,001; 0,5] desperdiça a maior parte das tentativas acima de 0,05.
#
# | Hiperparâmetro | Escala | Faixa típica |
# | :-- | :-- | :-- |
# | `learning_rate` (boosting) | log | 0,005 a 0,3 |
# | `n_estimators` / `max_iter` | não sortear: early stopping (tema 4, módulo 8) | |
# | `max_depth`, `max_leaf_nodes` | inteira, às vezes log | 3 a 12 / 7 a 255 |
# | `min_samples_leaf` | log-inteira | 1 a 200 |
# | `subsample`, `colsample` | linear | 0,5 a 1,0 |
# | `l2_regularization`, `alpha`, `lambda` | log | $10^{-4}$ a $10^{2}$ |
# | `C` (SVM, logística) | log | $10^{-3}$ a $10^{3}$ |
# | `k` (k-NN) | inteira, log | 1 a 100 |
# | número de neurônios, largura | log-inteira | 16 a 1024 |
#
# Categóricos (função de perda, tipo de kernel) entram como escolha
# uniforme. Hiperparâmetros **condicionais** (o `gamma` só existe com
# kernel RBF) pedem cuidado no grid e são naturais no random.

# %%
n_sorteios = 2000
lin = uniform(0.001, 0.5 - 0.001).rvs(n_sorteios, random_state=1)
log_ = loguniform(0.001, 0.5).rvs(n_sorteios, random_state=1)
decadas = [(0.001, 0.01), (0.01, 0.1), (0.1, 0.5)]
print("fração dos sorteios de learning_rate que cai em cada faixa:")
print(f"{'faixa':>14s} {'uniforme':>10s} {'log-uniforme':>13s}")
for lo, hi in decadas:
    print(f"[{lo:.3f}, {hi:.2f}) {np.mean((lin >= lo) & (lin < hi)):>10.1%} {np.mean((log_ >= lo) & (log_ < hi)):>13.1%}")

# %% [markdown]
# O uniforme gasta 80% do orçamento na faixa [0,1; 0,5), que num boosting
# com 150 iterações é quase sempre a pior. O log-uniforme divide o
# orçamento por igual entre as décadas.

# %% [markdown]
# ## 3. Por que random vence grid: baixa dimensionalidade efetiva
#
# Bergstra e Bengio (2012) observaram que, na maioria dos problemas,
# **poucos** hiperparâmetros importam, e quais importam muda de problema
# para problema. Uma grade de $4 \times 4$ testa 4 valores distintos de
# cada eixo; se só o eixo $x$ importa, foram 4 experimentos úteis
# repetidos 4 vezes. Dezesseis sorteios aleatórios testam 16 valores
# distintos de $x$.
#
# Primeiro numa função sintética, onde podemos repetir o experimento mil
# vezes: $f(x, z) = g(x) + 0{,}05\,h(z) + \text{ruído}$, com $x$ importando e
# $z$ quase não. A região boa é um pico estreito em $x$, e em cada
# repetição o pico fica num lugar diferente, porque na vida real ninguém
# sabe onde ele está.

# %%
def f_sintetica(x, z, centro=0.62, ruido=0.01, g=None):
    g = g or np.random.default_rng()
    return np.exp(-((x - centro) / 0.08) ** 2) + 0.05 * np.sin(6 * z) + g.normal(0, ruido, np.shape(x))

ORCAMENTO = 16
lado = int(np.sqrt(ORCAMENTO))
gx, gz = np.meshgrid(np.linspace(0.05, 0.95, lado), np.linspace(0.05, 0.95, lado))
melhores = {"grid": [], "random": []}
for rep in range(1000):
    g = np.random.default_rng(rep)
    centro = g.uniform(0.1, 0.9)
    melhores["grid"].append(f_sintetica(gx.ravel(), gz.ravel(), centro, g=g).max())
    melhores["random"].append(f_sintetica(g.random(ORCAMENTO), g.random(ORCAMENTO), centro, g=g).max())
melhores = pd.DataFrame(melhores)
print(f"orçamento {ORCAMENTO} | melhor valor encontrado em 1000 repetições (máximo verdadeiro ≈ 1,05):")
print(melhores.describe().loc[["mean", "50%", "min"]].round(3).to_string())
print(f"chegou a 90% do máximo: grid {np.mean(melhores['grid'] > 0.9):.0%} | random {np.mean(melhores['random'] > 0.9):.0%}")
print(f"random vence grid em {np.mean(melhores['random'] > melhores['grid']):.0%} das repetições")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
xx = np.linspace(0, 1, 300)
XX, ZZ = np.meshgrid(xx, xx)
for ax, (nome, px, pz) in zip(axes, [("grid 4×4", gx.ravel(), gz.ravel()),
                                     ("random, 16 sorteios", rng.random(16), rng.random(16))]):
    ax.contourf(XX, ZZ, f_sintetica(XX, ZZ, centro=0.5, ruido=0), levels=20, cmap="Blues", alpha=0.8)
    ax.scatter(px, pz, color=VERMELHO, s=35, zorder=3, edgecolor="white")
    ax.set_xlabel("x (importa)"); ax.set_ylabel("z (quase não importa)"); ax.set_title(nome)
    ax.grid(False)
plt.tight_layout(); plt.show()

# %% [markdown]
# **Leitura esperada:** a grade só acerta o pico quando uma das suas 4
# colunas cai perto dele, e as 4 colunas são as mesmas em todas as
# repetições; os 16 sorteios cobrem o eixo $x$ com 16 valores diferentes
# e chegam perto do máximo com muito mais frequência. A margem de vitória
# do random cresce com a dimensão: com 5 hiperparâmetros e orçamento 32,
# a grade tem 2 valores por eixo.
#
# A conta que justifica o orçamento: se a região boa ocupa uma fração
# $\gamma$ do espaço, a chance de pelo menos um dos $n$ sorteios cair nela
# é $1 - (1 - \gamma)^n$, **independente da dimensão**. Invertendo,
# $n = \ln(1 - P) / \ln(1 - \gamma)$.

# %%
for gama in [0.01, 0.03, 0.05, 0.10]:
    n95 = np.ceil(np.log(1 - 0.95) / np.log(1 - gama))
    print(f"região boa = {gama:4.0%} do espaço -> {n95:4.0f} sorteios para 95% de chance de acertá-la")

# %% [markdown]
# Daí o "60 sorteios" que circula como regra de bolso: é o $n$ para
# $\gamma = 5\%$. Se a região boa for menor, o número sobe rápido.

# %% [markdown]
# ## 4. O mesmo experimento no boosting
#
# Agora com $J$ de verdade. Três hiperparâmetros: `learning_rate` (log),
# `max_leaf_nodes` (inteiro) e `l2_regularization` (log). Grid com
# $3 \times 3 \times 2 = 18$ avaliações contra 18 sorteios, repetido com 5
# sementes para o random (a grade é determinística; o que muda nela é só
# a partição).

# %%
grade = {"learning_rate": [0.01, 0.05, 0.3], "max_leaf_nodes": [7, 31, 127], "l2_regularization": [0.0, 1.0]}
espaco = {"learning_rate": loguniform(0.005, 0.5), "max_leaf_nodes": randint(4, 128),
          "l2_regularization": loguniform(1e-3, 10)}
N_AVAL = len(ParameterGrid(grade))

def fmt(p):
    """Configuração legível: floats com 4 casas, inteiros como inteiros."""
    return "{" + ", ".join(f"{k}={float(v):.4g}" if isinstance(v, (float, np.floating)) else f"{k}={int(v)}"
                           for k, v in p.items()) + "}"

t0 = time.time()
res_grid = [(p, J(p)) for p in ParameterGrid(grade)]
res_rand = {s: [(p, J(p)) for p in ParameterSampler(espaco, n_iter=N_AVAL, random_state=s)] for s in range(5)}
print(f"{N_AVAL} avaliações por busca | {time.time() - t0:.0f} s no total")
melhor_grid = max(res_grid, key=lambda t: t[1])
print(f"grid              : melhor AUC {melhor_grid[1]:.4f} com {fmt(melhor_grid[0])}")
for s, r in res_rand.items():
    m = max(r, key=lambda t: t[1])
    print(f"random (semente {s}): melhor AUC {m[1]:.4f} com {fmt(m[0])}")
print(f"\nruído da CV (seção 1): dp = {RUIDO:.4f}")

# %% [markdown]
# **Leitura esperada:** grade e randoms terminam a menos de dois desvios
# uma da outra: empate técnico, com um espaço de só 3 dimensões e uma
# grade bem posicionada. As configurações vencedoras, por outro lado,
# variam bastante entre sementes. É o retrato do ruído da seção 1: há
# uma **região** boa (taxa de aprendizado entre 0,01 e 0,06, folhas não
# muito grandes, regularização de qualquer tamanho), não um ponto.
# Qualquer método que chegue à região está feito; a diferença entre eles
# é quanto orçamento gastam para chegar, e ela cresce com a dimensão.

# %% [markdown]
# ## 5. Quando parar: a curva "melhor até agora"
#
# A ferramenta que ninguém desenha e todo mundo deveria: o melhor escore
# encontrado em função do número de avaliações. Ela mostra os retornos
# decrescentes e diz onde o orçamento adicional deixa de comprar
# desempenho. Com o random, a curva é fácil de obter em várias sementes,
# porque a ordem dos sorteios é aleatória.

# %%
N_LONGO, N_SEMENTES = 40, 4
res_longo = {s: [J(p) for p in ParameterSampler(espaco, n_iter=N_LONGO, random_state=100 + s)] for s in range(N_SEMENTES)}
curvas = np.array([np.maximum.accumulate(r) for r in res_longo.values()])
media, baixo, alto = curvas.mean(axis=0), curvas.min(axis=0), curvas.max(axis=0)

fig, ax = plt.subplots(figsize=(8.5, 4.4))
ns = np.arange(1, N_LONGO + 1)
for c in curvas:
    ax.plot(ns, c, color=AZUL, alpha=0.25, lw=1)
ax.plot(ns, media, color=AZUL, lw=2.5, label=f"média de {N_SEMENTES} sementes")
ax.fill_between(ns, baixo, alto, color=AZUL, alpha=0.12, label="mín–máx")
ax.axhspan(media[-1] - 2 * RUIDO, media[-1] + 2 * RUIDO, color=CINZA, alpha=0.15, label="±2 dp do ruído da CV")
ax.set_xlabel("avaliações (orçamento gasto)"); ax.set_ylabel("melhor AUC até agora")
ax.set_title("Retornos decrescentes do random search"); ax.legend(fontsize=9, loc="lower right")
plt.tight_layout(); plt.show()
for n in [5, 10, 20, 40]:
    print(f"após {n:2d} avaliações: melhor AUC médio {media[n - 1]:.4f} (pior semente {baixo[n - 1]:.4f})")

# %% [markdown]
# **Leitura esperada:** a curva sobe rápido nas primeiras 10 avaliações
# e entra na faixa do ruído da CV antes das 20. Depois disso, cada
# avaliação nova tem chance pequena de melhorar mais do que o ruído
# permite distinguir. Uma regra prática de parada: quando a curva média
# fica dentro de ±2 dp por mais de $N$ avaliações consecutivas (10 a 20,
# dependendo do custo), o orçamento restante rende mais em outra coisa
# (features, dados, o próximo notebook).

# %% [markdown]
# ## 6. Successive Halving: gastar pouco em quem vai perder
#
# A ideia (Jamieson & Talwalkar, 2016): avaliar **muitas** configurações
# com **pouco** recurso (uma fração dos dados, poucas iterações), manter a
# melhor metade, dobrar o recurso, repetir. Com $n$ configurações
# iniciais e fator de redução $\eta = 2$, o custo total é
# $\approx n \cdot r_0 \cdot \log_2 n$ em vez de $n \cdot R$ (recurso
# cheio para todas). Abaixo, o recurso é a **fração dos dados de treino**:
# 32 configurações começam com 1/16 dos dados; sobrevivem 16 com 1/8, 8
# com 1/4, 4 com 1/2 e 2 com tudo. O custo total é 10 avaliações-cheias
# para triar 32 configurações.

# %%
def successive_halving(configs, X, y, eta=2, fracao_inicial=1 / 16, semente=0):
    g = np.random.default_rng(semente)
    vivos, fracao, custo, historico = list(configs), fracao_inicial, 0.0, []
    while True:
        n_sub = int(len(y) * fracao)
        idx = g.choice(len(y), n_sub, replace=False)
        escores = [J(p, X=X[idx], y=y[idx]) for p in vivos]
        custo += len(vivos) * fracao
        historico.append((fracao, len(vivos), max(escores)))
        if fracao >= 1 or len(vivos) <= eta:
            break
        ordem = np.argsort(escores)[::-1][: max(len(vivos) // eta, 1)]
        vivos, fracao = [vivos[i] for i in ordem], min(fracao * eta, 1.0)
    i_melhor = int(np.argmax(escores))
    return vivos[i_melhor], escores[i_melhor], custo, historico

N_INICIAL = 32
configs = list(ParameterSampler(espaco, n_iter=N_INICIAL, random_state=7))
t0 = time.time()
melhor_sh, auc_sh, custo_sh, hist = successive_halving(configs, X_tr, y_tr)
t_sh = time.time() - t0
print("rodada  fração dos dados  configurações  melhor AUC")
for fr, nv, mx in hist:
    print(f"{hist.index((fr, nv, mx)) + 1:>6d}  {fr:>16.4f}  {nv:>13d}  {mx:>10.4f}")
print(f"\ncusto: {custo_sh:.1f} avaliações-cheias para triar {N_INICIAL} configurações ({t_sh:.0f} s)")
print(f"vencedora: {fmt(melhor_sh)} -> AUC {auc_sh:.4f}")

# a comparação justa: random search com o MESMO custo (10 avaliações cheias)
t0 = time.time()
res_10 = [(p, J(p)) for p in configs[:10]]
m10 = max(res_10, key=lambda t: t[1])
print(f"random com 10 avaliações cheias ({time.time() - t0:.0f} s): melhor AUC {m10[1]:.4f} com {fmt(m10[0])}")

# %% [markdown]
# **Leitura esperada:** com o mesmo custo, o halving olhou 32
# configurações em vez de 10 e chegou a um vencedor equivalente dentro
# do ruído. Neste problema pequeno a vantagem não aparece; ela cresce
# quando o espaço é maior e a região boa mais rara, que é quando triar
# muitas configurações por pouco importa. Note também que o escore da
# última rodada não é comparável com o das primeiras: com 1/16 dos
# dados, todas as AUCs são mais baixas e mais ruidosas.
#
# Há um risco embutido, e ele tem um nome: **configurações que começam
# devagar**. Uma taxa de aprendizado baixa com poucas iterações, ou uma
# regularização forte com poucos dados, parece ruim na primeira rodada e
# é eliminada, mas seria a melhor com recurso cheio. Vamos procurar uma
# vítima: entre as 32, quais foram eliminadas na primeira rodada e qual
# seria a AUC delas com todos os dados?

# %%
primeira_fracao = hist[0][0]
idx0 = np.random.default_rng(0).choice(len(y_tr), int(len(y_tr) * primeira_fracao), replace=False)
esc0 = np.array([J(p, X=X_tr[idx0], y=y_tr[idx0]) for p in configs])
eliminadas = np.argsort(esc0)[: N_INICIAL // 2]
cheias = {i: J(configs[i]) for i in eliminadas}
i_vitima = max(cheias, key=cheias.get)
print(f"melhor AUC com dados cheios entre as 16 eliminadas na 1ª rodada: {cheias[i_vitima]:.4f} "
      f"(vencedora do halving: {auc_sh:.4f})")
print(f"a 'vítima': {fmt(configs[i_vitima])} | AUC com 1/16 dos dados: {esc0[i_vitima]:.4f}")
print("veredito:", "o halving matou cedo uma configuração melhor que a vencedora"
      if cheias[i_vitima] > auc_sh else "nenhuma eliminada era melhor que a vencedora")

# %% [markdown]
# Aconteceu: uma configuração com taxa de aprendizado baixa e pouca
# regularização parecia ruim com 175 amostras (em 150 iterações lentas,
# ela mal saiu do lugar) e, com os dados cheios, supera a vencedora. O
# **Hyperband** (Li et al., 2018) protege contra isso rodando vários
# halvings em paralelo com pontos de partida diferentes ("brackets"): um
# agressivo, com muitas configurações e recurso inicial mínimo, e outros
# progressivamente mais conservadores, até um que dá recurso cheio a
# poucas configurações desde o início. O bracket conservador é a
# seguradora das configurações que começam devagar. No `scikit-learn`,
# `HalvingRandomSearchCV` implementa o halving com `n_samples` ou com um
# hiperparâmetro (por exemplo `max_iter`) como recurso; Hyperband
# completo está no Optuna (`HyperbandPruner`) e no Ray Tune.

# %%
t0 = time.time()
halving_sk = HalvingRandomSearchCV(HistGradientBoostingClassifier(max_iter=150, random_state=0), espaco,
                                   n_candidates=N_INICIAL, factor=2, resource="n_samples", min_resources=175,
                                   cv=StratifiedKFold(3, shuffle=True, random_state=0), scoring="roc_auc",
                                   random_state=7).fit(X_tr, y_tr)
print(f"HalvingRandomSearchCV: melhor AUC {halving_sk.best_score_:.4f} em {time.time() - t0:.0f} s | "
      f"{halving_sk.n_iterations_} rodadas | recurso final {halving_sk.n_resources_[-1]} amostras")

# %% [markdown]
# ## 7. Quanto vale o vencedor
#
# O módulo 3 mostrou a maldição do vencedor: o melhor escore entre muitas
# avaliações ruidosas é otimista. Com o ruído da seção 1 e $M = 40$
# configurações, a aproximação $\sigma\sqrt{2\ln M}$ prevê o tamanho do
# otimismo. A base de teste, tocada uma única vez, dá o número honesto.

# %%
M = N_LONGO
otimismo_previsto = RUIDO * np.sqrt(2 * np.log(M))
melhor_idx = int(np.argmax(res_longo[0]))                     # semente 0 do experimento longo
melhor_cfg = list(ParameterSampler(espaco, n_iter=N_LONGO, random_state=100))[melhor_idx]
modelo_final = HistGradientBoostingClassifier(max_iter=150, random_state=0, **melhor_cfg).fit(X_tr, y_tr)
auc_teste = roc_auc_score(y_te, modelo_final.predict_proba(X_te)[:, 1])
print(f"melhor AUC de CV na busca: {res_longo[0][melhor_idx]:.4f}")
print(f"AUC no teste (tocado uma vez): {auc_teste:.4f}")
print(f"otimismo previsto pela fórmula (σ√(2 ln M), M = {M}): {otimismo_previsto:.4f}")

# %% [markdown]
# A diferença entre o escore da busca e o do teste tem duas partes: o
# otimismo da seleção (previsível) e o ruído da amostra de teste (não).
# O que importa é a disciplina: o número que vai para o relatório é o do
# teste ou o da validação aninhada, nunca o `best_score_`.

# %% [markdown]
# ## 8. O orçamento como restrição de projeto
#
# Uma conta que vale a pena fazer **antes** de rodar qualquer busca.
# Suponha um boosting que leva 40 s por treino numa base de produção,
# 5-fold, e uma máquina de nuvem a R$ 6/hora:

# %%
seg_por_treino, k, custo_hora = 40, 5, 6.0
def custo(n_avaliacoes, fracao_media=1.0):
    horas = n_avaliacoes * k * seg_por_treino * fracao_media / 3600
    return horas, horas * custo_hora
linhas = [("grid 5×5×5×5", 625, 1.0), ("grid 3×3×3×3", 81, 1.0), ("random, 60 sorteios", 60, 1.0),
          ("random, 20 sorteios", 20, 1.0), ("halving, 64 configs (custo ≈ 12 cheias)", 12, 1.0)]
print(f"{'estratégia':>42s} {'treinos':>8s} {'horas':>7s} {'R$':>8s}")
for nome, n, fr in linhas:
    h, r = custo(n, fr)
    print(f"{nome:>42s} {n * k:>8d} {h:>7.1f} {r:>8.0f}")

# %% [markdown]
# A grade "completa" custa um dia e meio de máquina para testar 5 valores
# de cada eixo, dos quais provavelmente dois eixos não importam; numa
# base dez vezes maior, seriam duas semanas. Os 60 sorteios cobrem o
# espaço inteiro por 3 horas. Isso é o que "orçamento
# como restrição de projeto" significa: a escolha do método é uma
# consequência de quanto cada avaliação custa e de quanto se pode gastar,
# não de qual método é "melhor".
#
# **O que ajustar, e em que ordem** (para um boosting; a lógica vale para
# os outros):
#
# 1. Não ajuste `n_estimators`: use early stopping com uma base de
#    validação (tema 4, módulo 8). Ele ajusta de graça e para cada
#    combinação dos outros.
# 2. `learning_rate` e a complexidade da árvore (`max_leaf_nodes` ou
#    `max_depth`, `min_samples_leaf`): os que quase sempre importam.
# 3. Regularização (`l2_regularization`, `subsample`, `colsample`): o
#    segundo bloco; importam quando há sobreajuste.
# 4. O resto: raramente move o ponteiro; deixe no padrão até ter
#    orçamento sobrando.
#
# E o que **não** é hiperparâmetro do modelo, mas deveria entrar na
# busca: as escolhas de pré-processamento (imputação, encoding, seleção
# de features). Num `Pipeline`, elas são hiperparâmetros como os outros,
# e sortear sobre elas evita o vazamento do módulo 3.

# %% [markdown]
# ## O que levar deste notebook
#
# - O escore de CV é uma função objetivo ruidosa e cara. Meça o ruído
#   antes de comparar configurações; diferenças dentro dele não são
#   decisões.
# - Sorteie em escala log o que é multiplicativo; deixe `n_estimators`
#   para o early stopping.
# - Random vence grid porque poucos hiperparâmetros importam; 60 sorteios
#   dão 95% de chance de acertar uma região que ocupa 5% do espaço, em
#   qualquer dimensão.
# - A curva "melhor até agora" mostra quando parar: quando ela entra na
#   faixa do ruído e fica.
# - Successive Halving tria muitas configurações por pouco; mata cedo as
#   que começam devagar, e o Hyperband protege contra isso com brackets.
# - O `best_score_` é otimista; o relatório usa o teste tocado uma vez ou
#   a validação aninhada.
# - O orçamento decide o método. Faça a conta em horas e reais antes de
#   rodar.
#
# → Próximo: **Otimização bayesiana do zero**, para quando cada avaliação
# custa tanto que vale a pena pensar antes de sortear.
