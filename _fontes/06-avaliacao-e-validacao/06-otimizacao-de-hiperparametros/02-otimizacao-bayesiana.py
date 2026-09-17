# %% [markdown]
# # Otimização bayesiana do zero
#
# **Tema:** Avaliação e Validação de Modelos › Otimização de Hiperparâmetros
#
# Grid e random search tratam cada avaliação como se as anteriores não
# existissem. Quando cada avaliação custa minutos ou horas, isso é
# desperdício: depois de dez jantares, você já tem uma ideia de onde a
# comida é boa. Este notebook constrói, do zero, o método que usa essa
# ideia: um **modelo substituto** (processo gaussiano) para o que já foi
# visto, uma **função de aquisição** que decide onde avaliar em seguida,
# e o laço que alterna os dois. Depois compara com random search no
# boosting do notebook anterior, implementa o TPE (o método do Optuna)
# em uma dimensão e fecha com quando usar o quê.
#
# ## A ideia
#
# A busca bayesiana responde a duas perguntas a cada passo:
#
# 1. **O que eu acho que $J(\theta)$ vale em pontos que ainda não
#    avaliei, e com que incerteza?** Um modelo de regressão ajustado às
#    avaliações feitas responde. O modelo precisa devolver uma
#    **incerteza**, não só uma média, e por isso o processo gaussiano é a
#    escolha clássica.
# 2. **Onde vale mais a pena avaliar em seguida?** Onde a média prevista
#    é alta (explorar o que parece bom) ou onde a incerteza é grande
#    (explorar o desconhecido). A função de aquisição pesa os dois.
#
# **Analogia.** Uma empresa de mineração tem orçamento para 20 furos de
# sondagem numa região. Depois de cada furo, o geólogo atualiza um mapa
# de "onde deve ter minério" e um mapa de "onde não sei nada". O próximo
# furo vai para onde o minério esperado é alto **ou** para onde o mapa
# está em branco, porque um furo ali pode revelar um filão inteiro. Furar
# 20 pontos numa grade fixa, ou 20 pontos sorteados, ignoraria tudo o
# que os furos anteriores ensinaram.
#
# ### O que você vai conseguir fazer ao final
#
# - Explicar o que um processo gaussiano devolve e por que a incerteza
#   dele importa.
# - Implementar *expected improvement* e ler o que ela faz a cada passo.
# - Rodar otimização bayesiana em uma e em duas dimensões e comparar com
#   random search sob o mesmo orçamento.
# - Explicar o TPE e por que o Optuna o prefere ao processo gaussiano.
# - Decidir quando a busca bayesiana compensa e quando não.

# %%
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

rng = np.random.default_rng(662)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
AZUL, VERDE, VERMELHO, ROXO, AMBAR, CINZA = "#1F5C8B", "#1E6B4F", "#9C2B2B", "#5B3E86", "#8A6100", "#5A5A5A"
print("pronto")

# %% [markdown]
# ## 1. O processo gaussiano em vinte linhas
#
# Um processo gaussiano (GP) é uma distribuição sobre **funções**. Antes
# de ver dados, ele diz: "a função tem média zero e pontos próximos têm
# valores parecidos". O "parecidos" é o **kernel**; o mais comum é o RBF:
#
# $$k(x, x') = \sigma_f^2 \exp\left(-\frac{\|x - x'\|^2}{2\ell^2}\right)$$
#
# onde $\ell$ (a escala de comprimento) diz a que distância dois pontos
# deixam de se influenciar. Depois de observar $y$ nos pontos $X$ (com
# ruído de variância $\sigma_n^2$), a previsão num ponto novo $x_*$ é
# gaussiana, com média e variância em forma fechada:
#
# $$\mu(x_*) = k_*^\top (K + \sigma_n^2 I)^{-1} y, \qquad
# \sigma^2(x_*) = k(x_*, x_*) - k_*^\top (K + \sigma_n^2 I)^{-1} k_*$$
#
# com $K_{ij} = k(x_i, x_j)$ e $k_* = k(X, x_*)$. Duas leituras: a média é
# uma combinação ponderada dos $y$ observados, com pesos que caem com a
# distância; a variância começa em $\sigma_f^2$ longe de todo ponto
# observado e vai a $\sigma_n^2$ em cima de um. É essa variância que a
# busca vai explorar.
#
# O custo é a inversão de uma matriz $n \times n$ ($O(n^3)$), o que
# limita o GP a algumas centenas de avaliações: exatamente o regime em
# que a busca bayesiana faz sentido.

# %%
class GP:
    def __init__(self, escala=0.2, sigma_f=1.0, sigma_n=0.05):
        self.escala, self.sigma_f, self.sigma_n = escala, sigma_f, sigma_n

    def kernel(self, A, B):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
        return self.sigma_f ** 2 * np.exp(-d2 / (2 * self.escala ** 2))

    def fit(self, X, y):
        self.X, self.media_y = np.atleast_2d(X), y.mean()
        K = self.kernel(self.X, self.X) + self.sigma_n ** 2 * np.eye(len(y))
        self.cho = cho_factor(K)                                  # fatoração de Cholesky: estável e barata
        self.alpha = cho_solve(self.cho, y - self.media_y)        # (K + σ²I)⁻¹ (y - média)
        return self

    def predict(self, Xs):
        Xs = np.atleast_2d(Xs)
        ks = self.kernel(self.X, Xs)
        mu = self.media_y + ks.T @ self.alpha
        v = cho_solve(self.cho, ks)
        var = np.clip(self.sigma_f ** 2 - np.einsum("ij,ij->j", ks, v), 1e-12, None)
        return mu, np.sqrt(var)

# uma função objetivo 1D cara e ruidosa (imagine que cada avaliação leva uma hora)
def objetivo_1d(x, g=rng):
    return (np.sin(3 * x) * np.exp(-x) + 0.6 * np.exp(-((x - 0.75) / 0.08) ** 2) + g.normal(0, 0.03, np.shape(x)))

x_grade = np.linspace(0, 1, 400)[:, None]
verdade = objetivo_1d(x_grade.ravel(), g=np.random.default_rng(0)) * 0 + (
    np.sin(3 * x_grade.ravel()) * np.exp(-x_grade.ravel()) + 0.6 * np.exp(-((x_grade.ravel() - 0.75) / 0.08) ** 2))

X_obs = np.array([[0.1], [0.4], [0.9]])
y_obs = objetivo_1d(X_obs.ravel())
gp = GP(escala=0.15).fit(X_obs, y_obs)
mu, sd = gp.predict(x_grade)

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(x_grade, verdade, color=CINZA, ls="--", lw=1.5, label="função verdadeira (desconhecida)")
ax.plot(x_grade, mu, color=AZUL, lw=2, label="média do GP")
ax.fill_between(x_grade.ravel(), mu - 2 * sd, mu + 2 * sd, color=AZUL, alpha=0.15, label="±2 desvios")
ax.scatter(X_obs, y_obs, color=VERMELHO, s=50, zorder=3, label="avaliações feitas")
ax.set_xlabel("hiperparâmetro (normalizado)"); ax.set_ylabel("escore"); ax.legend(fontsize=9)
ax.set_title("O que o GP acredita depois de 3 avaliações")
plt.tight_layout(); plt.show()

# %% [markdown]
# **Leitura esperada:** em cima dos três pontos a banda é estreita; entre
# eles e nas bordas, larga. A média passa perto da verdade onde há dados
# e volta para a média global onde não há. O pico verdadeiro em
# $x = 0{,}75$ está numa região em que o GP "não sabe", e a banda larga ali
# é o convite para avaliar.

# %% [markdown]
# ## 2. A função de aquisição: expected improvement
#
# Com o melhor valor observado até agora, $f^*$, o ganho de avaliar em
# $x$ é $\max(J(x) - f^*, 0)$. Como $J(x)$ é, para o GP, uma gaussiana
# $\mathcal{N}(\mu, \sigma^2)$, a esperança do ganho tem forma fechada:
#
# $$EI(x) = (\mu - f^* - \xi)\,\Phi(z) + \sigma\,\phi(z), \qquad z = \frac{\mu - f^* - \xi}{\sigma}$$
#
# com $\Phi$ e $\phi$ a distribuição e a densidade da normal padrão, e
# $\xi \geq 0$ um botão de exploração: $\xi$ maior exige melhoria maior
# para valer a pena e empurra a busca para regiões incertas. O primeiro
# termo premia média alta; o segundo, incerteza alta. Onde $\sigma \to 0$
# (em cima de um ponto avaliado), $EI \to 0$: o método nunca reavalia o
# mesmo ponto por conta própria.
#
# A alternativa mais simples é o **limite superior de confiança**,
# $UCB(x) = \mu + \kappa\sigma$: "seja otimista". Funciona bem e tem um
# botão ($\kappa$) mais fácil de interpretar; EI é o padrão porque não
# precisa de escala.

# %%
def expected_improvement(gp, Xc, f_melhor, xi=0.01):
    mu, sd = gp.predict(Xc)
    z = (mu - f_melhor - xi) / sd
    return (mu - f_melhor - xi) * norm.cdf(z) + sd * norm.pdf(z)

XI = 0.01               # mude para 0.1 e veja a busca ficar mais exploradora
ESCALA = 0.15           # escala do kernel: pequena = função "rugosa", grande = suave
N_INICIAL, N_PASSOS = 3, 10

X_obs = np.array([[0.1], [0.4], [0.9]])
y_obs = objetivo_1d(X_obs.ravel())
fig, axes = plt.subplots(N_PASSOS // 2, 2, figsize=(13, 2.6 * (N_PASSOS // 2)))
for passo, ax in enumerate(axes.ravel()):
    gp = GP(escala=ESCALA).fit(X_obs, y_obs)
    mu, sd = gp.predict(x_grade)
    ei = expected_improvement(gp, x_grade, y_obs.max(), xi=XI)
    x_novo = x_grade[np.argmax(ei)]
    ax.plot(x_grade, verdade, color=CINZA, ls="--", lw=1)
    ax.plot(x_grade, mu, color=AZUL, lw=1.6)
    ax.fill_between(x_grade.ravel(), mu - 2 * sd, mu + 2 * sd, color=AZUL, alpha=0.12)
    ax.scatter(X_obs, y_obs, color=VERMELHO, s=25, zorder=3)
    ax2 = ax.twinx(); ax2.plot(x_grade, ei, color=VERDE, lw=1.2); ax2.set_yticks([]); ax2.grid(False)
    ax.axvline(x_novo, color=VERDE, lw=1, ls=":")
    ax.set_title(f"passo {passo + 1}: avalia x = {x_novo[0]:.3f} (melhor até agora {y_obs.max():.3f})", fontsize=9)
    ax.set_yticks([])
    X_obs = np.vstack([X_obs, x_novo]); y_obs = np.append(y_obs, objetivo_1d(x_novo))
plt.tight_layout(); plt.show()
print(f"após {N_INICIAL + N_PASSOS} avaliações: melhor x = {X_obs[np.argmax(y_obs), 0]:.3f}, "
      f"escore {y_obs.max():.3f} | máximo verdadeiro: x = {x_grade[np.argmax(verdade), 0]:.3f}, {verdade.max():.3f}")

# %% [markdown]
# **Leitura esperada:** a linha verde é a EI (escala à direita). Nos
# primeiros passos ela é alta nas regiões vazias, e a busca explora;
# quando um ponto perto de 0,75 devolve um escore alto, a EI concentra
# ali e a busca refina. Em treze avaliações, o pico estreito foi
# encontrado. Um random search com 13 sorteios tem
# $1 - (1 - 0{,}1)^{13} \approx 75\%$ de chance de cair a menos de 0,05 do
# pico, e nenhuma garantia de refinar depois.
#
# Mude `XI` para 0,1: a busca passa mais tempo em regiões incertas antes
# de refinar. Mude `ESCALA` para 0,4: o GP acredita que a função é suave
# e pode "não enxergar" um pico estreito. A escala de comprimento é o
# hiperparâmetro do otimizador de hiperparâmetros, e as bibliotecas a
# estimam por máxima verossimilhança a cada passo.

# %% [markdown]
# ## 3. Duas dimensões, no boosting de verdade
#
# Agora com a função objetivo real: AUC de CV de um boosting em função de
# `learning_rate` (log) e `max_leaf_nodes` (log), no churn do notebook
# anterior. As duas dimensões são normalizadas para $[0, 1]$ antes de
# entrar no GP, que é o que toda biblioteca faz. Orçamento: 5 avaliações
# iniciais aleatórias + 15 guiadas, contra 20 sorteios do random search.
# Repetimos 5 vezes cada um, porque uma corrida só é ruído.

# %%
def gera_churn(n, semente):
    g = np.random.default_rng(semente)
    X = g.normal(0, 1, (n, 8))
    lg = (-0.8 + 1.1 * X[:, 0] - 0.9 * X[:, 1] + 0.8 * X[:, 0] * X[:, 2] + 0.6 * np.sin(2 * X[:, 3])
          + 0.5 * (X[:, 4] > 1) - 0.4 * X[:, 5] ** 2)
    return X, (g.random(n) < 1 / (1 + np.exp(-lg))).astype(int)

X_tr, y_tr = gera_churn(2000, 0)
LIMITES = {"learning_rate": (np.log10(0.005), np.log10(0.5)), "max_leaf_nodes": (np.log2(4), np.log2(128))}

def decodifica(u):
    """Ponto em [0,1]² -> hiperparâmetros reais."""
    lr = 10 ** (LIMITES["learning_rate"][0] + u[0] * (LIMITES["learning_rate"][1] - LIMITES["learning_rate"][0]))
    folhas = int(round(2 ** (LIMITES["max_leaf_nodes"][0] + u[1] * (LIMITES["max_leaf_nodes"][1] - LIMITES["max_leaf_nodes"][0]))))
    return {"learning_rate": lr, "max_leaf_nodes": folhas}

def J(u, semente=0):
    m = HistGradientBoostingClassifier(max_iter=150, random_state=0, **decodifica(u))
    return cross_val_score(m, X_tr, y_tr, cv=StratifiedKFold(3, shuffle=True, random_state=semente),
                           scoring="roc_auc").mean()

def busca_bayesiana(n_inicial, n_passos, semente, xi=0.01):
    g = np.random.default_rng(semente)
    U = g.random((n_inicial, 2)); y = np.array([J(u) for u in U])
    candidatos = g.random((3000, 2))                          # EI maximizada por amostragem densa
    for _ in range(n_passos):
        gp = GP(escala=0.25, sigma_f=0.02, sigma_n=0.005).fit(U, y)
        ei = expected_improvement(gp, candidatos, y.max(), xi=xi)
        u_novo = candidatos[np.argmax(ei)]
        U = np.vstack([U, u_novo]); y = np.append(y, J(u_novo))
    return U, y

def busca_aleatoria(n, semente):
    g = np.random.default_rng(semente)
    U = g.random((n, 2)); return U, np.array([J(u) for u in U])

ORCAMENTO = 20
t0 = time.time()
corridas = {"bayesiana": [busca_bayesiana(5, ORCAMENTO - 5, s) for s in range(5)],
            "random": [busca_aleatoria(ORCAMENTO, s) for s in range(5)]}
print(f"{time.time() - t0:.0f} s para 10 buscas de {ORCAMENTO} avaliações")
for nome, lista in corridas.items():
    finais = [y.max() for _, y in lista]
    print(f"{nome:>10s}: melhor AUC por corrida {np.round(finais, 4)} | média {np.mean(finais):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ns = np.arange(1, ORCAMENTO + 1)
for nome, cor in [("bayesiana", ROXO), ("random", AZUL)]:
    curvas = np.array([np.maximum.accumulate(y) for _, y in corridas[nome]])
    axes[0].plot(ns, curvas.mean(axis=0), color=cor, lw=2.2, label=nome)
    axes[0].fill_between(ns, curvas.min(axis=0), curvas.max(axis=0), color=cor, alpha=0.12)
axes[0].axvline(5, color=CINZA, ls=":", lw=1); axes[0].text(5.2, axes[0].get_ylim()[0] + 0.001, "fim da fase aleatória", fontsize=8, color=CINZA)
axes[0].set_xlabel("avaliações"); axes[0].set_ylabel("melhor AUC até agora"); axes[0].legend()
axes[0].set_title("Melhor até agora, média de 5 corridas")
U_b, y_b = corridas["bayesiana"][0]
sc = axes[1].scatter(U_b[:, 0], U_b[:, 1], c=np.arange(len(y_b)), cmap="viridis", s=60, edgecolor="white", zorder=3)
axes[1].scatter(*U_b[np.argmax(y_b)], marker="*", s=300, color=VERMELHO, zorder=4, label="melhor")
plt.colorbar(sc, ax=axes[1], label="ordem da avaliação")
axes[1].set_xlabel("learning_rate (log, normalizado)"); axes[1].set_ylabel("max_leaf_nodes (log, normalizado)")
axes[1].set_title("Onde a busca bayesiana avaliou (corrida 1)"); axes[1].legend(loc="upper center")
plt.tight_layout(); plt.show()

# %% [markdown]
# **Leitura esperada:** as duas curvas partem iguais (as 5 primeiras
# avaliações da bayesiana são aleatórias) e depois a bayesiana sobe mais
# rápido e chega mais alto, com menos variação entre corridas. No mapa,
# as avaliações guiadas se concentram numa faixa (taxa de aprendizado
# moderada) e ainda visitam cantos vazios de vez em quando. A diferença
# final entre os métodos é pequena em AUC, e é assim que costuma ser: a
# bayesiana chega ao mesmo lugar com **menos avaliações**, o que só
# importa quando cada avaliação é cara.

# %% [markdown]
# ## 4. TPE: o que o Optuna faz
#
# O GP tem dois problemas práticos: custa $O(n^3)$ e lida mal com
# hiperparâmetros inteiros, categóricos e condicionais. O *Tree-structured
# Parzen Estimator* (Bergstra et al., 2011) inverte a pergunta. Em vez de
# modelar $p(y \mid \theta)$, ele divide as avaliações feitas em dois
# grupos, as **boas** (melhores $\gamma$ = 25%) e as **ruins**, e modela
# $\ell(\theta) = p(\theta \mid \text{boa})$ e $g(\theta) = p(\theta \mid \text{ruim})$
# com estimadores de densidade (misturas de gaussianas centradas nas
# avaliações). O próximo ponto é o que maximiza $\ell(\theta) / g(\theta)$:
# "parecido com as boas e diferente das ruins". Bergstra mostrou que isso
# é equivalente a maximizar a EI.
#
# Cada hiperparâmetro tem sua densidade, o que torna inteiros e
# categóricos triviais, e o custo cresce linearmente com $n$. É o padrão
# do Optuna e do Hyperopt. Em 1D, cabe em poucas linhas:

# %%
def tpe_proximo(X_obs, y_obs, g, gama=0.25, n_candidatos=500, largura=0.1):
    corte = np.quantile(y_obs, 1 - gama)
    boas, ruins = X_obs[y_obs >= corte], X_obs[y_obs < corte]
    def densidade(pontos, centros):                            # mistura de gaussianas com uma componente por avaliação
        return np.mean(norm.pdf((pontos[:, None] - centros[None, :]) / largura), axis=1) + 1e-9
    cand = g.random(n_candidatos)
    cand = np.concatenate([cand, np.clip(g.choice(boas, n_candidatos // 2) + g.normal(0, largura, n_candidatos // 2), 0, 1)])
    return cand[np.argmax(densidade(cand, boas) / densidade(cand, ruins))]

g = np.random.default_rng(5)
X_t = g.random(4); y_t = objetivo_1d(X_t)
for _ in range(9):
    x_novo = tpe_proximo(X_t, y_t, g)
    X_t = np.append(X_t, x_novo); y_t = np.append(y_t, objetivo_1d(x_novo))
print(f"TPE em 13 avaliações: melhor x = {X_t[np.argmax(y_t)]:.3f}, escore {y_t.max():.3f} "
      f"(máximo verdadeiro em x = {x_grade[np.argmax(verdade), 0]:.3f})")

# %% [markdown]
# Com o Optuna, o mesmo laço fica assim (não executado aqui, para não
# adicionar uma dependência ao curso):
#
# ```python
# import optuna
#
# def objetivo(trial):
#     params = {"learning_rate": trial.suggest_float("learning_rate", 0.005, 0.5, log=True),
#               "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 4, 128, log=True),
#               "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10, log=True)}
#     modelo = HistGradientBoostingClassifier(max_iter=150, random_state=0, **params)
#     return cross_val_score(modelo, X_tr, y_tr, cv=3, scoring="roc_auc").mean()
#
# estudo = optuna.create_study(direction="maximize",
#                              sampler=optuna.samplers.TPESampler(seed=0),
#                              pruner=optuna.pruners.HyperbandPruner())
# estudo.optimize(objetivo, n_trials=60, timeout=3600)
# estudo.best_params, estudo.best_value
# ```
#
# Três coisas nesse trecho que este módulo ensinou a ler: `log=True` é a
# seção 2 do notebook anterior; o `HyperbandPruner` é a seção 6 dele,
# combinada com o TPE (o que a literatura chama de BOHB); e
# `estudo.best_value` é o `best_score_`, com o mesmo otimismo.

# %% [markdown]
# ## 5. Quando a busca bayesiana compensa
#
# | Situação | Método |
# | :-- | :-- |
# | Avaliação barata (segundos), muitas dimensões | random search, ou halving |
# | Avaliação cara (minutos a horas), até ~20 dimensões, orçamento de dezenas a poucas centenas | bayesiana (TPE ou GP) |
# | Avaliação cara **e** com recurso ajustável (épocas, iterações, dados) | bayesiana + Hyperband (BOHB, Optuna com pruner) |
# | Muitas máquinas em paralelo | random ou halving (trivialmente paralelos); bayesiana precisa de truques para propor vários pontos de uma vez |
# | Hiperparâmetros categóricos e condicionais | TPE, não GP |
#
# Três cuidados que valem para qualquer método bayesiano:
#
# - **Ruído.** A EI assume que uma avaliação alta é real. Com o ruído da
#   CV (notebook anterior, seção 1), um ponto sorte-grande atrai a busca
#   para perto dele. Bibliotecas modelam o ruído ($\sigma_n$ no GP); com
#   ruído alto, vale repetir a CV das candidatas finais.
# - **Fase inicial.** As primeiras avaliações são aleatórias e o modelo
#   substituto só começa a valer com 5 a 10 pontos. Com orçamento de 10,
#   a bayesiana é um random search caro.
# - **Relatório.** O `best_value` é otimista como qualquer máximo de
#   busca. Teste tocado uma vez, ou validação aninhada.

# %% [markdown]
# ## O que levar deste notebook
#
# - A busca bayesiana usa um modelo substituto com incerteza (GP) e uma
#   função de aquisição (EI) que pesa "parece bom" contra "não sei".
# - O GP dá média e desvio em forma fechada; a EI tem forma fechada; o
#   laço inteiro cabe em cinquenta linhas.
# - Contra random search sob o mesmo orçamento, a bayesiana chega ao
#   mesmo lugar com menos avaliações. A vantagem é proporcional ao custo
#   de cada avaliação.
# - TPE modela $p(\theta \mid \text{boa}) / p(\theta \mid \text{ruim})$, escala
#   linearmente e aceita inteiros e categóricos; é o padrão do Optuna.
# - Orçamento pequeno, avaliação barata ou muitas máquinas: random ou
#   halving. Avaliação cara: bayesiana, com pruner se houver recurso
#   ajustável.
#
# → Próximo tema: **Séries Temporais**, onde a validação precisa
# respeitar o tempo e os hiperparâmetros incluem "quanto passado olhar".
