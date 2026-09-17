# %% [markdown]
# # Exercícios — Otimização de Hiperparâmetros
#
# **Tema:** Avaliação e Validação de Modelos › Otimização de Hiperparâmetros
#
# Resolva antes de olhar o gabarito. **Dificuldade:** 🟢 base · 🟡 aplicação ·
# 🔴 síntese

# %%
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import loguniform, randint, norm, t as t_student
from scipy.linalg import cho_factor, cho_solve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (HalvingRandomSearchCV, ParameterGrid, ParameterSampler, StratifiedKFold,
                                     cross_val_score, train_test_split)

rng = np.random.default_rng(666)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
AZUL, VERDE, VERMELHO, ROXO, AMBAR, CINZA = "#1F5C8B", "#1E6B4F", "#9C2B2B", "#5B3E86", "#8A6100", "#5A5A5A"
print("ambiente pronto")

# %% [markdown]
# ---
# ## Dataset de trabalho: conversão de leads
#
# Um time de vendas recebe leads de várias origens e quer saber quais
# têm mais chance de fechar contrato, para priorizar as ligações. Cada
# lead tem 10 features (tamanho da empresa, origem, tempo desde o
# primeiro contato, páginas visitadas, cargo de quem preencheu o
# formulário, e por aí vai, aqui todas numéricas e sintéticas). O modelo
# é um boosting; a métrica é AUC; a validação é 3-fold estratificado. A
# função `J` é a função objetivo do tuning, e cada avaliação dela custa
# uma fração de segundo aqui, mas imagine que custa cinco minutos.

# %%
def gera_leads(n, semente):
    g = np.random.default_rng(semente)
    X = g.normal(0, 1, (n, 10))
    lg = (-1.2 + 0.9 * X[:, 0] + 0.7 * np.tanh(2 * X[:, 1]) - 0.8 * X[:, 2] * (X[:, 3] > 0)
          + 0.5 * np.abs(X[:, 4]) - 0.6 * X[:, 5] + 0.4 * X[:, 6] * X[:, 7])
    return X, (g.random(n) < 1 / (1 + np.exp(-lg))).astype(int)

X, y = gera_leads(4000, 0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
CV = lambda s=0: StratifiedKFold(3, shuffle=True, random_state=s)

def J(params, semente=0, X=X_tr, y=y_tr):
    m = HistGradientBoostingClassifier(max_iter=120, random_state=0, **params)
    return cross_val_score(m, X, y, cv=CV(semente), scoring="roc_auc").mean()

def fmt(p):
    return "{" + ", ".join(f"{k}={float(v):.4g}" if isinstance(v, (float, np.floating)) else f"{k}={int(v)}"
                           for k, v in p.items()) + "}"

espaco = {"learning_rate": loguniform(0.005, 0.5), "max_leaf_nodes": randint(4, 128),
          "min_samples_leaf": randint(5, 100)}
print(f"treino {len(y_tr)} | teste {len(y_te)} | conversão {y.mean():.1%}")

# %% [markdown]
# ---
# ## Exercício 1 🟢 — A conta do orçamento
#
# Em produção, cada treino do boosting leva 40 s, a validação é 5-fold e
# a máquina custa R$ 6 por hora. Calcule horas e reais para: (a) uma
# grade com 4 hiperparâmetros de 5 valores; (b) o número de sorteios do
# random search que dá 95% de chance de acertar uma região boa que
# ocupa 3% do espaço; (c) um Successive Halving com 81 configurações,
# fator 3 e recurso inicial de 1/27 dos dados (custo em avaliações
# cheias: some as rodadas). Qual você propõe e por quê?

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 1

# %%
seg, k, reais_h = 40, 5, 6.0
def custo(avaliacoes_cheias):
    h = avaliacoes_cheias * k * seg / 3600
    return h, h * reais_h

n_grid = 5 ** 4
n_rand = int(np.ceil(np.log(1 - 0.95) / np.log(1 - 0.03)))
rodadas = [(81, 1 / 27), (27, 1 / 9), (9, 1 / 3), (3, 1.0)]
n_halv = sum(n * r for n, r in rodadas)
for nome, n in [("(a) grade 5^4", n_grid), (f"(b) random, {n_rand} sorteios", n_rand), ("(c) halving 81 configs", n_halv)]:
    h, r = custo(n)
    print(f"{nome:>28s}: {n:6.0f} avaliações cheias | {n * k:6.0f} treinos | {h:6.1f} h | R$ {r:6.0f}")

# %% [markdown]
# **Por quê:** a grade custa um dia e meio de máquina e testa 5 valores
# por eixo, dos quais provavelmente dois eixos não importam; numa base
# dez vezes maior, seriam duas semanas. Os 99 sorteios cobrem qualquer
# dimensão por 5,5 horas. O halving olha 81
# configurações por 12 avaliações cheias (40 minutos), mas com o risco
# de eliminar cedo as que começam devagar. A proposta razoável para um
# primeiro passe: random search com 30 a 60 sorteios em escala log, com
# a curva "melhor até agora" decidindo se os 99 valem a pena; halving se
# o orçamento for apertado; e, se 5 minutos por avaliação virarem 50, a
# busca bayesiana do notebook 2.

# %% [markdown]
# ---
# ## Exercício 2 🟢 — Essa diferença é real?
#
# Três configurações saíram de uma busca: A = `{learning_rate: 0.05,
# max_leaf_nodes: 15, min_samples_leaf: 20}`, B = `{learning_rate: 0.1,
# max_leaf_nodes: 31, min_samples_leaf: 10}` e C = `{learning_rate: 0.03,
# max_leaf_nodes: 15, min_samples_leaf: 20}`. (a) Meça o ruído de `J`:
# avalie A com 15 partições diferentes e reporte o desvio-padrão. (b)
# Avalie as três nas **mesmas** 15 partições e aplique o teste t
# corrigido de Nadeau e Bengio (módulo 3) às diferenças A − B e A − C.
# Quais diferenças são decisões?

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 2

# %%
A = {"learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 20}
B = {"learning_rate": 0.1, "max_leaf_nodes": 31, "min_samples_leaf": 10}
C = {"learning_rate": 0.03, "max_leaf_nodes": 15, "min_samples_leaf": 20}
jA, jB, jC = (np.array([J(cfg, semente=s) for s in range(15)]) for cfg in (A, B, C))
print(f"(a) A em 15 partições: média {jA.mean():.4f}, dp {jA.std(ddof=1):.4f}, amplitude {jA.max() - jA.min():.4f}")
n_te_sobre_n_tr = 1 / 2                                   # 3-fold: teste tem metade do tamanho do treino
for nome, d in [("A - B", jA - jB), ("A - C", jA - jC)]:
    t_cor = d.mean() / np.sqrt((1 / len(d) + n_te_sobre_n_tr) * d.var(ddof=1))
    p_val = 2 * t_student.sf(abs(t_cor), len(d) - 1)
    print(f"(b) {nome}: média {d.mean():+.4f} | A vence em {np.mean(d > 0):.0%} das partições | "
          f"t corrigido = {t_cor:.2f}, p = {p_val:.3f}")

# %% [markdown]
# **Por quê:** o desvio-padrão da mesma configuração entre partições é de
# quase meio ponto de AUC, e a amplitude passa de um ponto: milésimos
# entre configurações são ruído. A diferença A − B, de um ponto e meio,
# sobrevive ao teste (A vence em todas as partições; B, com taxa alta e
# folhas pequenas, sobreajusta): é uma decisão. A diferença A − C, de
# milésimos, não sobrevive: as duas são a mesma coisa para efeitos
# práticos, e a escolha entre elas deve usar outro critério (taxa menor
# em C é um pouco mais estável; A treina mais rápido). O teste pareado
# nas mesmas partições é a forma certa de perguntar, e o termo
# $n_{te}/n_{tr}$ (aqui 1/2, porque em 3-fold o teste tem metade do
# treino) impede que repetir as partições fabrique significância.

# %% [markdown]
# ---
# ## Exercício 3 🟡 — Grid, random e quando parar
#
# (a) Com orçamento de 12 avaliações, compare uma grade $3 \times 2 \times 2$
# (`learning_rate` em {0,01; 0,05; 0,3}, `max_leaf_nodes` em {7; 63},
# `min_samples_leaf` em {10; 50}) com 12 sorteios de `espaco`, em 4
# sementes. (b) Rode um random search de 30 sorteios em 3 sementes e
# desenhe a curva "melhor até agora" com a faixa de ±2 desvios do ruído
# medido no exercício 2. Em que orçamento você teria parado?

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 3

# %%
grade = {"learning_rate": [0.01, 0.05, 0.3], "max_leaf_nodes": [7, 63], "min_samples_leaf": [10, 50]}
t0 = time.time()
melhor_grid = max(((p, J(p)) for p in ParameterGrid(grade)), key=lambda t: t[1])
print(f"grade (12): melhor AUC {melhor_grid[1]:.4f} com {fmt(melhor_grid[0])}")
for s in range(4):
    m = max(((p, J(p)) for p in ParameterSampler(espaco, n_iter=12, random_state=s)), key=lambda t: t[1])
    print(f"random (12, semente {s}): melhor AUC {m[1]:.4f} com {fmt(m[0])}")

RUIDO = jA.std(ddof=1)
curvas = np.array([np.maximum.accumulate([J(p) for p in ParameterSampler(espaco, n_iter=30, random_state=50 + s)])
                   for s in range(3)])
media = curvas.mean(axis=0)
print(f"\n{time.time() - t0:.0f} s | melhor até agora (média de 3 sementes) após 5/10/20/30: "
      f"{media[4]:.4f} / {media[9]:.4f} / {media[19]:.4f} / {media[29]:.4f}")

fig, ax = plt.subplots(figsize=(8.5, 4.2))
ns = np.arange(1, 31)
for c in curvas:
    ax.plot(ns, c, color=AZUL, alpha=0.3, lw=1)
ax.plot(ns, media, color=AZUL, lw=2.5, label="média")
ax.axhspan(media[-1] - 2 * RUIDO, media[-1] + 2 * RUIDO, color=CINZA, alpha=0.15, label="±2 dp do ruído")
ax.set_xlabel("avaliações"); ax.set_ylabel("melhor AUC até agora"); ax.legend(loc="lower right")
plt.tight_layout(); plt.show()

# %% [markdown]
# **Por quê:** grade e randoms terminam dentro do ruído um do outro, e as
# configurações vencedoras do random variam, porque a região boa é larga.
# A curva "melhor até agora" entra na faixa de ±2 desvios do valor final
# bem antes das 30 avaliações; a partir dali, cada sorteio a mais tem
# chance pequena de comprar uma melhoria distinguível do ruído. Uma
# parada por volta de 10 a 15 avaliações teria entregado o mesmo modelo
# pela metade do custo. É a regra prática: parar quando a curva média
# fica dentro da faixa por 10 avaliações seguidas.

# %% [markdown]
# ---
# ## Exercício 4 🟡 — Successive Halving e suas vítimas
#
# Implemente o Successive Halving com o recurso "fração dos dados de
# treino": 32 configurações sorteadas de `espaco`, começando com 1/16
# dos dados e dobrando a cada rodada (mantendo a metade melhor). Compare
# com um random search de custo igual (10 avaliações cheias). Depois
# avalie com dados cheios as 16 eliminadas na primeira rodada: alguma
# teria vencido? Que característica ela tem? Feche rodando
# `HalvingRandomSearchCV` com os mesmos parâmetros.

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 4

# %%
def halving(configs, eta=2, fracao=1 / 16, semente=0):
    g = np.random.default_rng(semente)
    vivos, custo = list(configs), 0.0
    while True:
        idx = g.choice(len(y_tr), int(len(y_tr) * fracao), replace=False)
        esc = [J(p, X=X_tr[idx], y=y_tr[idx]) for p in vivos]
        custo += len(vivos) * fracao
        if fracao >= 1 or len(vivos) <= eta:
            break
        ordem = np.argsort(esc)[::-1][: len(vivos) // eta]
        vivos, fracao = [vivos[i] for i in ordem], min(fracao * eta, 1.0)
    i = int(np.argmax(esc))
    return vivos[i], esc[i], custo

configs = list(ParameterSampler(espaco, n_iter=32, random_state=9))
venc, auc_venc, custo_h = halving(configs)
print(f"halving: {fmt(venc)} -> AUC {auc_venc:.4f} | custo {custo_h:.0f} avaliações cheias")
m10 = max(((p, J(p)) for p in configs[:10]), key=lambda t: t[1])
print(f"random de custo igual (10 cheias): {fmt(m10[0])} -> AUC {m10[1]:.4f}")

idx0 = np.random.default_rng(0).choice(len(y_tr), len(y_tr) // 16, replace=False)
esc0 = np.array([J(p, X=X_tr[idx0], y=y_tr[idx0]) for p in configs])
eliminadas = np.argsort(esc0)[:16]
cheias = {i: J(configs[i]) for i in eliminadas}
i_v = max(cheias, key=cheias.get)
print(f"melhor eliminada na 1ª rodada, com dados cheios: {fmt(configs[i_v])} -> AUC {cheias[i_v]:.4f} "
      f"({'melhor' if cheias[i_v] > auc_venc else 'pior'} que a vencedora)")
print(f"taxa de aprendizado das 16 eliminadas: mediana {np.median([configs[i]['learning_rate'] for i in eliminadas]):.3f} | "
      f"das 16 sobreviventes: {np.median([configs[i]['learning_rate'] for i in np.argsort(esc0)[16:]]):.3f}")

hs = HalvingRandomSearchCV(HistGradientBoostingClassifier(max_iter=120, random_state=0), espaco, n_candidates=32,
                           factor=2, resource="n_samples", min_resources=len(y_tr) // 8, cv=CV(), scoring="roc_auc",
                           random_state=9).fit(X_tr, y_tr)
print(f"HalvingRandomSearchCV (recurso inicial 1/8): melhor AUC {hs.best_score_:.4f} com {fmt(hs.best_params_)} | "
      f"recursos por rodada {hs.n_resources_}")

# %% [markdown]
# **Por quê:** o halving triou 32 configurações pelo custo de 10 e
# entregou um vencedor **pior** que o do random de custo igual, e a
# causa está nas eliminadas: a melhor delas, com dados cheios, supera a
# vencedora. As eliminadas na primeira rodada têm, tipicamente, **taxa de
# aprendizado menor** (mediana quatro vezes mais baixa que a das
# sobreviventes): com 1/16 dos dados e 120 iterações, uma taxa baixa não
# chega a aprender, e a configuração parece ruim mesmo quando seria boa
# com recurso cheio. O remédio é o Hyperband (brackets com recursos
# iniciais diferentes) ou, mais simples, um recurso inicial menos
# agressivo quando o espaço inclui taxas de aprendizado baixas: o
# `HalvingRandomSearchCV` começando com 1/8 dos dados chega a um
# resultado equivalente ao do random.

# %% [markdown]
# ---
# ## Exercício 5 🔴 — O protocolo de tuning sob orçamento
#
# O time só pode gastar **15 avaliações** por rodada de tuning (cada uma
# custa 5 minutos na máquina de produção), em duas dimensões:
# `learning_rate` (log em [0,005; 0,5]) e `max_leaf_nodes` (log em
# [4; 128]). (a) Implemente uma busca bayesiana com GP e expected
# improvement (4 avaliações iniciais aleatórias + 11 guiadas) e compare
# com 15 sorteios, em 2 sementes cada. (b) Reporte a AUC da configuração
# vencedora na base de teste, tocada uma única vez, ao lado do escore
# da busca. (c) Escreva o protocolo: método, espaço, esquema de CV, regra
# de parada, o que registrar de cada avaliação e o que vai no relatório.

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 5

# %%
class GP:
    def __init__(self, escala=0.25, sigma_f=0.02, sigma_n=0.005):
        self.escala, self.sigma_f, self.sigma_n = escala, sigma_f, sigma_n
    def kernel(self, A, B):
        return self.sigma_f ** 2 * np.exp(-((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2) / (2 * self.escala ** 2))
    def fit(self, U, y):
        self.U, self.m = U, y.mean()
        self.cho = cho_factor(self.kernel(U, U) + self.sigma_n ** 2 * np.eye(len(y)))
        self.alpha = cho_solve(self.cho, y - self.m); return self
    def predict(self, Us):
        ks = self.kernel(self.U, Us)
        mu = self.m + ks.T @ self.alpha
        var = np.clip(self.sigma_f ** 2 - np.einsum("ij,ij->j", ks, cho_solve(self.cho, ks)), 1e-12, None)
        return mu, np.sqrt(var)

def ei(gp, Uc, f_melhor, xi=0.005):
    mu, sd = gp.predict(Uc); z = (mu - f_melhor - xi) / sd
    return (mu - f_melhor - xi) * norm.cdf(z) + sd * norm.pdf(z)

def decodifica(u):
    return {"learning_rate": float(10 ** (np.log10(0.005) + u[0] * (np.log10(0.5) - np.log10(0.005)))),
            "max_leaf_nodes": int(round(2 ** (2 + u[1] * 5)))}

def J2(u):
    return J(decodifica(u))

def bayesiana(n_ini, n_pas, semente):
    g = np.random.default_rng(semente)
    U = g.random((n_ini, 2)); yv = np.array([J2(u) for u in U]); cand = g.random((3000, 2))
    for _ in range(n_pas):
        u_novo = cand[np.argmax(ei(GP().fit(U, yv), cand, yv.max()))]
        U = np.vstack([U, u_novo]); yv = np.append(yv, J2(u_novo))
    return U, yv

t0 = time.time()
resultados = {}
for s in range(2):
    Ub, yb = bayesiana(4, 11, s)
    g = np.random.default_rng(100 + s); Ur = g.random((15, 2)); yr = np.array([J2(u) for u in Ur])
    resultados[s] = (Ub, yb, Ur, yr)
    print(f"semente {s}: bayesiana melhor AUC {yb.max():.4f} {fmt(decodifica(Ub[np.argmax(yb)]))} | "
          f"random melhor AUC {yr.max():.4f} {fmt(decodifica(Ur[np.argmax(yr)]))}")
print(f"{time.time() - t0:.0f} s")

# (b) o número honesto
Ub, yb, _, _ = resultados[0]
cfg = decodifica(Ub[np.argmax(yb)])
final = HistGradientBoostingClassifier(max_iter=120, random_state=0, **cfg).fit(X_tr, y_tr)
print(f"\nvencedora da bayesiana (semente 0): escore da busca {yb.max():.4f} | "
      f"AUC no teste (uma vez) {roc_auc_score(y_te, final.predict_proba(X_te)[:, 1]):.4f}")

# %% [markdown]
# **Como avaliar sua resposta:** com 15 avaliações, a bayesiana e o
# random chegam a AUCs próximas, e a bayesiana tende a chegar com menos
# variação entre sementes, porque as 11 avaliações guiadas se concentram
# na faixa boa em vez de sortear. A diferença é pequena aqui porque o
# problema tem 2 dimensões e uma região boa larga; ela cresce com o
# custo por avaliação e com a estreiteza da região. Repare no teste: ele
# saiu **acima** do escore da busca. Dois efeitos se somam com sinais
# opostos: o otimismo da seleção (módulo 3) puxa o escore da busca para
# cima, e o pessimismo da CV (cada fold treinou com 2/3 dos dados; o
# modelo final treinou com todos) puxa para baixo. Aqui o segundo venceu.
# Reportar o teste é o que evita ter de adivinhar qual dos dois ganhou.
#
# O protocolo que uma boa resposta escreve: (1) **método**: bayesiana
# (TPE, via Optuna, ou GP) porque cada avaliação custa 5 minutos e o
# orçamento é de 15; com orçamento de 60 ou avaliação de 5 segundos, seria
# random. (2) **Espaço**: taxa de aprendizado e folhas em escala log,
# `max_iter` por early stopping, regularização só se o vão treino-
# validação (módulo 4) pedir. (3) **CV**: estratificada, com a mesma
# semente de partição para todas as avaliações (comparação pareada) e
# repetição das 2 ou 3 finalistas com partições novas. (4) **Parada**:
# orçamento fixo de 15, ou antes se a curva "melhor até agora" ficar
# dentro do ruído por 5 avaliações. (5) **Registro**: cada avaliação com
# configuração, escore por fold, semente, tempo e versão dos dados (tema
# 13). (6) **Relatório**: a AUC do teste tocado uma vez, ao lado do
# escore da busca, com a diferença explicada como otimismo de seleção.

# %% [markdown]
# ---
# ## Fechamento
#
# - Faça a conta do orçamento antes de escolher o método; a grade quase
#   nunca sobrevive à conta.
# - Meça o ruído da função objetivo e compare configurações com teste
#   pareado; diferenças dentro do ruído não são decisões.
# - Random e grid empatam em poucas dimensões; random vence em muitas. A
#   curva "melhor até agora" diz quando parar.
# - Successive Halving tria barato e mata cedo quem começa devagar (taxa
#   de aprendizado baixa); Hyperband ou recurso inicial maior protegem.
# - Bayesiana quando a avaliação é cara; o vencedor é reportado pelo
#   teste, nunca pelo escore da busca.
#
# → Próximo tema: **Séries Temporais**.
