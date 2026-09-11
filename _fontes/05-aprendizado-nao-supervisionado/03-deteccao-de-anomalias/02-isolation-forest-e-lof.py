# %% [markdown]
# # Isolation Forest e LOF
#
# **Tema:** Aprendizado Não Supervisionado › Detecção de Anomalias
#
# Implementamos um Isolation Forest mínimo do zero e conferimos que seu
# escore concorda com o do `scikit-learn`; conferimos a fórmula de $c(n)$;
# calculamos o LOF do zero; e comparamos os métodos do módulo em cenários
# desenhados para que cada um vença ou perca — anomalias globais, anomalias
# locais, alta dimensão e detecção de novidade.

# %%
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

rng = np.random.default_rng(57)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
AZUL, VERDE, VERMELHO, ROXO, AMBAR = "#1F5C8B", "#1E6B4F", "#9C2B2B", "#5B3E86", "#8A6100"
print("pronto")

# %% [markdown]
# ## 1. Um Isolation Forest mínimo, do zero

# %%
def c(n):
    """Comprimento médio de uma busca sem sucesso numa árvore binária com n pontos."""
    if n <= 1:
        return 0.0
    return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n

def constroi_arvore(X, prof, prof_max, gerador):
    if prof >= prof_max or len(X) <= 1:
        return {"folha": True, "n": len(X)}
    f = gerador.integers(X.shape[1])
    lo, hi = X[:, f].min(), X[:, f].max()
    if lo == hi:
        return {"folha": True, "n": len(X)}
    corte = gerador.uniform(lo, hi)
    esq = X[:, f] < corte
    return {"folha": False, "f": f, "corte": corte,
            "esq": constroi_arvore(X[esq], prof + 1, prof_max, gerador),
            "dir": constroi_arvore(X[~esq], prof + 1, prof_max, gerador)}

def caminho(x, no, prof=0):
    if no["folha"]:
        return prof + c(no["n"])     # corrige folhas não isoladas pela profundidade esperada
    filho = no["esq"] if x[no["f"]] < no["corte"] else no["dir"]
    return caminho(x, filho, prof + 1)

class IsolationForestDoZero:
    def __init__(self, n_arvores=100, amostra=256, semente=0):
        self.n_arvores, self.amostra, self.semente = n_arvores, amostra, semente

    def fit(self, X):
        g = np.random.default_rng(self.semente)
        psi = min(self.amostra, len(X))
        prof_max = int(np.ceil(np.log2(psi)))
        self.psi_ = psi
        self.arvores_ = [constroi_arvore(X[g.choice(len(X), psi, replace=False)], 0, prof_max, g)
                         for _ in range(self.n_arvores)]
        return self

    def escore(self, X):
        h = np.array([[caminho(x, a) for a in self.arvores_] for x in X]).mean(axis=1)
        return 2 ** (-h / c(self.psi_))


X = np.vstack([rng.normal(0, 1, (1000, 2)), rng.uniform(-6, 6, (30, 2))])
e_anom = np.r_[np.zeros(1000), np.ones(30)]
t0 = time.perf_counter()
nosso = IsolationForestDoZero(n_arvores=100).fit(X).escore(X)
t_nosso = time.perf_counter() - t0
sk = -IsolationForest(n_estimators=100, random_state=0).fit(X).score_samples(X)
print(f"correlação de Spearman entre os escores: "
      f"{pd.Series(nosso).corr(pd.Series(sk), method='spearman'):.3f}")
print(f"PR-AUC — do zero: {average_precision_score(e_anom, nosso):.3f} | "
      f"sklearn: {average_precision_score(e_anom, sk):.3f}  (tempo do zero: {t_nosso:.1f}s)")
print(f"c(256) = {c(256):.3f}  |  s para E[h]=4: {2 ** (-4 / c(256)):.3f}, "
      f"para E[h]=c(256): {2 ** (-1):.3f}, para E[h]=14: {2 ** (-14 / c(256)):.3f}")

# %% [markdown]
# O escore do `scikit-learn` (`score_samples`) é o **negativo** de $s$ — por
# isso o sinal trocado acima. As duas implementações ordenam os pontos de
# forma muito parecida; a diferença vem só da aleatoriedade das árvores.

# %% [markdown]
# ## 2. contamination não muda o modelo — só o limiar

# %%
for cont in ["auto", 0.01, 0.03, 0.10]:
    m = IsolationForest(n_estimators=200, contamination=cont, random_state=0).fit(X)
    pred = m.predict(X) == -1
    print(f"contamination={str(cont):>5s}: {pred.sum():>4d} alertas | "
          f"escore do 1º ponto = {m.score_samples(X[:1])[0]:.4f}")

# %% [markdown]
# O escore do primeiro ponto é idêntico em todas as linhas; só o número de
# pontos marcados como -1 muda. O limiar é uma decisão sua, não do modelo.

# %% [markdown]
# ## 3. LOF do zero, conferido contra o `scikit-learn`

# %%
def lof_do_zero(X, k):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, idx = nn.kneighbors(X)
    dist, idx = dist[:, 1:], idx[:, 1:]            # remove o próprio ponto
    d_k = dist[:, -1]                               # distância ao k-ésimo vizinho
    reach = np.maximum(d_k[idx], dist)              # reach_k(A, B) = max(d_k(B), d(A, B))
    lrd = 1 / reach.mean(axis=1)
    return lrd[idx].mean(axis=1) / lrd

D = np.vstack([rng.normal([0, 0], 0.25, (200, 2)), rng.normal([5, 0], 1.6, (200, 2)),
               [[1.3, 0.9]]])
lof_nosso = lof_do_zero(D, k=20)
lof_sk = -LocalOutlierFactor(n_neighbors=20).fit(D).negative_outlier_factor_
print(f"diferença máxima entre as implementações: {np.abs(lof_nosso - lof_sk).max():.2e}")
print(f"LOF do ponto suspeito: {lof_nosso[-1]:.2f}  | mediana dos demais: {np.median(lof_nosso[:-1]):.2f}")

# %% [markdown]
# ## 4. Cenários: quem vence onde
#
# Três bases com anomalias rotuladas (só para avaliar — nenhum método usa o
# rótulo): (a) anomalias **globais** espalhadas ao redor de uma nuvem; (b)
# anomalias **locais** perto de um grupo denso, ao lado de um grupo
# esparso; (c) **alta dimensão**: 50 features, das quais as anomalias
# diferem em apenas 3.

# %%
def cenario_global(n=2000, n_a=40):
    Xn = rng.normal(0, 1, (n, 2))
    ang = rng.uniform(0, 2 * np.pi, n_a)
    Xa = np.c_[np.cos(ang), np.sin(ang)] * rng.uniform(4, 6, (n_a, 1))
    return np.vstack([Xn, Xa]), np.r_[np.zeros(n), np.ones(n_a)]

def cenario_local(n=2000, n_a=40):
    denso = rng.normal([0, 0], 0.3, (n // 2, 2))
    esparso = rng.normal([6, 0], 2.0, (n // 2, 2))
    Xa = rng.normal([0, 0], 0.3, (n_a, 2)) + rng.choice([-1, 1], (n_a, 2)) * 1.4
    return np.vstack([denso, esparso, Xa]), np.r_[np.zeros(n), np.ones(n_a)]

def cenario_alta_dim(n=2000, n_a=40, p=50):
    Xn = rng.normal(0, 1, (n, p))
    Xa = rng.normal(0, 1, (n_a, p))
    Xa[:, :3] += rng.choice([-1, 1], (n_a, 3)) * 3.5
    return np.vstack([Xn, Xa]), np.r_[np.zeros(n), np.ones(n_a)]

def escores(X):
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(5, X.shape[1] - 1)).fit(Xs)
    return {
        "Mahalanobis (MCD)": EllipticEnvelope(random_state=0).fit(X).mahalanobis(X),
        "Isolation Forest": -IsolationForest(n_estimators=300, random_state=0).fit(X).score_samples(X),
        "LOF (k=20)": -LocalOutlierFactor(n_neighbors=20).fit(Xs).negative_outlier_factor_,
        "One-Class SVM": -OneClassSVM(nu=0.05, gamma="scale").fit(Xs).score_samples(Xs),
        "erro de reconstrução (PCA)": ((Xs - pca.inverse_transform(pca.transform(Xs))) ** 2).sum(axis=1),
    }

linhas = {}
for nome, gerador in [("global", cenario_global), ("local", cenario_local),
                      ("alta dimensão", cenario_alta_dim)]:
    Xc, yc = gerador()
    linhas[nome] = {m: average_precision_score(yc, e) for m, e in escores(Xc).items()}
print("PR-AUC por cenário (prevalência de anomalias ≈ 2%):")
print(pd.DataFrame(linhas).round(3).to_string())

# %% [markdown]
# **Leitura esperada:** no cenário global, quase todos vão bem. No local, o
# LOF se destaca: os métodos globais confundem as anomalias perto do grupo
# denso com os pontos normais da periferia do grupo esparso. Em alta
# dimensão, o resultado mais importante é o do **Isolation Forest**, que
# cai para o pior lugar: cada corte sorteia uma feature, e como a anomalia
# está em só 3 das 50, quase todos os cortes são gastos em features
# irrelevantes. Os métodos baseados em distância (LOF, One-Class SVM)
# também perdem força, porque as 47 dimensões de ruído diluem a distância.
# Mahalanobis vence porque as normais são exatamente uma gaussiana — o
# cenário ideal dele. Não existe método que vença em todos os cenários: o
# formato do problema escolhe o método, e selecionar features antes de
# detectar anomalias ajuda quase todos eles.

# %% [markdown]
# ## 5. Detecção de novidade: treinar só com o normal
#
# Um detector de novidade aprende o normal com dados limpos e depois julga
# dados novos. Comparamos LOF em modo novidade e One-Class SVM, e o que
# acontece quando o "treino limpo" na verdade está contaminado.

# %%
normal_treino = rng.normal(0, 1, (1000, 4))
teste = np.vstack([rng.normal(0, 1, (500, 4)), rng.normal(0, 1, (25, 4)) + 3.5])
y_teste = np.r_[np.zeros(500), np.ones(25)]
contaminado = np.vstack([normal_treino, rng.normal(0, 1, (80, 4)) + 3.5])  # 7% de anomalias no treino

for nome_treino, treino in [("treino limpo", normal_treino), ("treino contaminado", contaminado)]:
    lof_nov = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(treino)
    ocsvm = OneClassSVM(nu=0.02, gamma="scale").fit(treino)
    print(f"{nome_treino:>18s}: PR-AUC LOF(novelty) = "
          f"{average_precision_score(y_teste, -lof_nov.score_samples(teste)):.3f} | "
          f"One-Class SVM = {average_precision_score(y_teste, -ocsvm.score_samples(teste)):.3f}")

# %% [markdown]
# Com treino contaminado, as anomalias do teste encontram "vizinhos" no
# treino — o aglomerado de anomalias que entrou por engano — e parecem
# normais. Detecção de novidade exige disciplina na curadoria do treino.

# %% [markdown]
# ## 6. Escala: o custo de cada método

# %%
for n in [5_000, 20_000, 80_000]:
    Xg = rng.normal(0, 1, (n, 10))
    tempos = {}
    for nome, f in [("Isolation Forest", lambda: IsolationForest(n_estimators=100, random_state=0).fit(Xg).score_samples(Xg)),
                    ("LOF", lambda: LocalOutlierFactor(n_neighbors=20).fit(Xg).negative_outlier_factor_),
                    ("One-Class SVM", lambda: OneClassSVM(nu=0.05).fit(Xg[:min(n, 20_000)]))]:
        t0 = time.perf_counter()
        f()
        tempos[nome] = time.perf_counter() - t0
    print(f"n={n:>6d}: " + " | ".join(f"{k} {v:6.2f}s" for k, v in tempos.items())
          + ("  (One-Class SVM limitado a 20 mil pontos)" if n > 20_000 else ""))

# %% [markdown]
# ## O que levar deste notebook
#
# - Isolation Forest mede quão fácil é isolar um ponto; o escore normaliza o
#   caminho médio por $c(n)$, e `contamination` só escolhe o limiar.
# - LOF compara a densidade de um ponto com a dos seus vizinhos — é o método
#   para anomalias locais.
# - Cada método tem um cenário em que vence; teste mais de um com os poucos
#   rótulos que existirem.
# - Detecção de novidade só funciona se o treino for de fato limpo.
#
# → Próximo: **Caso real: detecção de fraude**, onde os escores não
# supervisionados encontram um modelo supervisionado.
