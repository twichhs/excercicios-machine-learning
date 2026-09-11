"""Gera as figuras do teoria.pdf do módulo 01-clustering."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ / "_ferramentas"))
from estilo_figuras import AZUL, VERDE, VERMELHO, ROXO, AMBAR, CINZA, salva  # noqa: E402

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(5)
MODULO_REL = Path(__file__).resolve().parent.relative_to(RAIZ / "_fontes")
DESTINO = RAIZ / MODULO_REL / "figuras"
CORES = np.array([AZUL, VERMELHO, VERDE, ROXO, AMBAR, CINZA])

# ---------------------------------------------------------------------------
# 1. Algoritmo de Lloyd: inicialização, duas iterações e convergência
# ---------------------------------------------------------------------------
X, _ = make_blobs(n_samples=300, centers=[[-3, 0], [2, 3], [2.5, -2.5]],
                  cluster_std=1.0, random_state=3)
centroides = np.array([[-3.5, 1.5], [-2.0, 0.5], [4.0, -3.5]])  # inicialização ruim de propósito
historico = [centroides.copy()]
rotulos_hist = []
for _ in range(10):
    d = ((X[:, None, :] - centroides[None, :, :]) ** 2).sum(axis=2)
    rot = d.argmin(axis=1)
    rotulos_hist.append(rot)
    novos = np.array([X[rot == k].mean(axis=0) for k in range(3)])
    historico.append(novos.copy())
    if np.allclose(novos, centroides):
        break
    centroides = novos
n_iter = len(rotulos_hist)

fig, axes = plt.subplots(1, 4, figsize=(16, 4.3), sharex=True, sharey=True)
paineis = [(0, "inicialização"), (1, "iteração 1"), (2, "iteração 2"),
           (n_iter - 1, f"convergência (iteração {n_iter})")]
for ax, (it, titulo) in zip(axes, paineis):
    if titulo == "inicialização":
        ax.scatter(X[:, 0], X[:, 1], s=10, color=CINZA, alpha=0.5)
        c = historico[0]
    else:
        ax.scatter(X[:, 0], X[:, 1], s=10, c=CORES[rotulos_hist[it - 1]], alpha=0.5)
        c = historico[it]
        trilha = np.array(historico[: it + 1])
        for k in range(3):
            ax.plot(trilha[:, k, 0], trilha[:, k, 1], color="black", lw=1, ls=":")
    ax.scatter(c[:, 0], c[:, 1], s=220, marker="X", c=CORES[:3],
               edgecolor="black", linewidth=1.4, zorder=5)
    ax.set_title(titulo, fontsize=11)
fig.suptitle("Algoritmo de Lloyd: atribuir cada ponto ao centróide mais próximo, "
             "mover o centróide para a média do grupo, repetir", fontsize=11.5,
             fontweight="bold", y=1.03)
salva(fig, DESTINO / "kmeans-iteracoes.png")
print(f"Lloyd convergiu em {n_iter} iterações")

# ---------------------------------------------------------------------------
# 2. Cotovelo e silhueta para dados com 4 grupos verdadeiros
# ---------------------------------------------------------------------------
X4, _ = make_blobs(n_samples=600, centers=[[-4, -3], [0, 4], [4, -3], [5, 3]],
                   cluster_std=1.3, random_state=11)
ks = range(2, 10)
inercias, silhuetas = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X4)
    inercias.append(km.inertia_)
    silhuetas.append(silhouette_score(X4, km.labels_))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
ax1.plot(list(ks), inercias, "o-", color=AZUL, lw=2)
ax1.axvline(4, color=VERMELHO, ls="--", lw=1.3)
ax1.set_xlabel("k"); ax1.set_ylabel("inércia (soma dos quadrados intra-grupo)")
ax1.set_title("Método do cotovelo: a queda perde força em k=4")
ax2.plot(list(ks), silhuetas, "o-", color=VERDE, lw=2)
ax2.axvline(4, color=VERMELHO, ls="--", lw=1.3)
ax2.set_xlabel("k"); ax2.set_ylabel("silhueta média")
ax2.set_title("Silhueta: máximo em k=4")
fig.tight_layout()
salva(fig, DESTINO / "cotovelo-e-silhueta.png")
for k, i, s in zip(ks, inercias, silhuetas):
    print(f"k={k}: inércia={i:8.1f}  silhueta={s:.3f}")

# ---------------------------------------------------------------------------
# 3. Onde k-means falha (e o que resolve)
# ---------------------------------------------------------------------------
luas, _ = make_moons(n_samples=500, noise=0.06, random_state=0)
aneis, _ = make_circles(n_samples=500, factor=0.45, noise=0.05, random_state=0)
aniso, _ = make_blobs(n_samples=500, centers=3, random_state=170)
aniso = aniso @ np.array([[0.6, -0.64], [-0.41, 0.85]])

dados = [("duas luas", StandardScaler().fit_transform(luas)),
         ("anéis concêntricos", StandardScaler().fit_transform(aneis)),
         ("grupos alongados", StandardScaler().fit_transform(aniso))]
fig, axes = plt.subplots(2, 3, figsize=(14, 8.6))
for j, (nome, D) in enumerate(dados):
    k = 2 if j < 2 else 3
    rot_km = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(D)
    axes[0, j].scatter(D[:, 0], D[:, 1], s=9, c=CORES[rot_km])
    axes[0, j].set_title(f"k-means — {nome}", fontsize=11)
    if j < 2:
        rot = DBSCAN(eps=0.3, min_samples=8).fit_predict(D)
        cores = np.where(rot[:, None] >= 0, CORES[np.clip(rot, 0, 5)][:, None], "#BBBBBB")[:, 0]
        axes[1, j].scatter(D[:, 0], D[:, 1], s=9, c=cores)
        axes[1, j].set_title(f"DBSCAN — {nome}", fontsize=11)
    else:
        gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=0).fit(D)
        axes[1, j].scatter(D[:, 0], D[:, 1], s=9, c=CORES[gmm.predict(D)])
        axes[1, j].set_title(f"GMM (covariância cheia) — {nome}", fontsize=11)
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("k-means assume grupos convexos e 'redondos'. Quando a forma real é outra,\n"
             "ele corta o espaço em fatias — e um algoritmo com outra premissa acerta",
             fontsize=11.5, fontweight="bold", y=1.0)
fig.tight_layout()
salva(fig, DESTINO / "onde-kmeans-falha.png")

# ---------------------------------------------------------------------------
# 4. Dendrograma: 4 critérios de ligação nos mesmos 18 pontos
# ---------------------------------------------------------------------------
P, _ = make_blobs(n_samples=18, centers=[[0, 0], [5, 1], [2.5, 5]],
                  cluster_std=0.9, random_state=4)
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, metodo in zip(axes, ["single", "complete", "average", "ward"]):
    Z = linkage(P, method=metodo)
    dendrogram(Z, ax=ax, color_threshold=0.55 * Z[-1, 2], above_threshold_color=CINZA,
               no_labels=True)
    ax.set_title(f"ligação '{metodo}'", fontsize=11)
    ax.set_ylabel("distância de fusão" if metodo == "single" else "")
fig.suptitle("Mesmos 18 pontos, quatro regras para medir distância entre grupos: "
             "a altura das fusões e a forma da árvore mudam", fontsize=11.5,
             fontweight="bold", y=1.04)
salva(fig, DESTINO / "dendrograma-ligacoes.png")

# ---------------------------------------------------------------------------
# 5. DBSCAN: pontos-núcleo, de borda e ruído
# ---------------------------------------------------------------------------
Q = np.vstack([rng.normal([0, 0], 0.35, (22, 2)), rng.normal([2.6, 0.4], 0.3, (14, 2)),
               np.array([[1.3, 1.6], [-1.6, 1.4], [3.9, -1.2], [1.25, 0.1]])])
eps, min_pts = 0.5, 4
modelo = DBSCAN(eps=eps, min_samples=min_pts).fit(Q)
nucleo = np.zeros(len(Q), bool)
nucleo[modelo.core_sample_indices_] = True
ruido = modelo.labels_ == -1
borda = ~nucleo & ~ruido
fig, ax = plt.subplots(figsize=(8.5, 5.6))
for i in np.where(nucleo)[0][:6]:
    ax.add_patch(Circle(Q[i], eps, color=AZUL, alpha=0.07))
ax.scatter(Q[nucleo, 0], Q[nucleo, 1], s=60, color=AZUL, label="núcleo", zorder=4)
ax.scatter(Q[borda, 0], Q[borda, 1], s=60, facecolor="white", edgecolor=AZUL,
           linewidth=2, label="borda", zorder=4)
ax.scatter(Q[ruido, 0], Q[ruido, 1], s=70, marker="x", color=VERMELHO, label="ruído", zorder=4)
ax.set_aspect("equal")
ax.set_title(f"DBSCAN com eps={eps}, min_samples={min_pts}: grupos nascem de regiões densas;\n"
             "pontos isolados viram ruído em vez de serem forçados a um grupo", fontsize=11)
ax.legend(loc="upper right")
salva(fig, DESTINO / "dbscan-pontos.png")
print(f"DBSCAN: {nucleo.sum()} núcleo, {borda.sum()} borda, {ruido.sum()} ruído, "
      f"{len(set(modelo.labels_)) - (1 if ruido.any() else 0)} grupos")

# ---------------------------------------------------------------------------
# 6. GMM: elipses e pertinência "suave"
# ---------------------------------------------------------------------------
G = np.vstack([rng.multivariate_normal([0, 0], [[1.6, 1.0], [1.0, 1.2]], 250),
               rng.multivariate_normal([3.2, -0.5], [[0.4, -0.2], [-0.2, 0.6]], 180)])
gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0).fit(G)
prob = gmm.predict_proba(G)[:, 0]
fig, ax = plt.subplots(figsize=(8.5, 5.6))
sc = ax.scatter(G[:, 0], G[:, 1], c=prob, cmap="coolwarm", s=14, vmin=0, vmax=1)
for media, cov in zip(gmm.means_, gmm.covariances_):
    autoval, autovec = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(autovec[1, 1], autovec[0, 1]))
    for n_sd in (1, 2):
        ax.add_patch(Ellipse(media, 2 * n_sd * np.sqrt(autoval[1]), 2 * n_sd * np.sqrt(autoval[0]),
                             angle=ang, fill=False, color="black", lw=1.3, ls="-" if n_sd == 1 else "--"))
fig.colorbar(sc, ax=ax, label="P(componente 1 | x)")
ax.set_aspect("equal")
ax.set_title("GMM: cada componente é uma gaussiana com forma própria (elipses de 1 e 2 dp);\n"
             "pontos na zona de sobreposição recebem probabilidade intermediária", fontsize=11)
salva(fig, DESTINO / "gmm-pertinencia.png")
incertos = ((prob > 0.2) & (prob < 0.8)).mean()
print(f"GMM: {incertos:.1%} dos pontos com pertinência entre 0.2 e 0.8")
print("done")
