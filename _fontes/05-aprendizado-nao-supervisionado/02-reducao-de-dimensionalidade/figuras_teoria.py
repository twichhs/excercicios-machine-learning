"""Gera as figuras do teoria.pdf do módulo 02-reducao-de-dimensionalidade."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ / "_ferramentas"))
from estilo_figuras import AZUL, VERDE, VERMELHO, ROXO, AMBAR, CINZA, salva  # noqa: E402

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, trustworthiness
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(52)
MODULO_REL = Path(__file__).resolve().parent.relative_to(RAIZ / "_fontes")
DESTINO = RAIZ / MODULO_REL / "figuras"
CMAP10 = plt.get_cmap("tab10")

digitos = load_digits()
Xd = digitos.data
yd = digitos.target

# ---------------------------------------------------------------------------
# 1. Maldição da dimensionalidade: contraste entre vizinho mais próximo e mais distante
# ---------------------------------------------------------------------------
dims = [1, 2, 5, 10, 20, 50, 100, 300, 1000]
contraste = []
for d in dims:
    P = rng.uniform(0, 1, (500, d))
    q = rng.uniform(0, 1, d)
    dist = np.sqrt(((P - q) ** 2).sum(axis=1))
    contraste.append((dist.max() - dist.min()) / dist.min())
fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.loglog(dims, contraste, "o-", color=VERMELHO, lw=2)
ax.set_xlabel("número de dimensões (features)")
ax.set_ylabel("(d_max - d_min) / d_min")
ax.set_title("Em alta dimensão, o ponto mais distante fica quase tão perto quanto o mais\n"
             "próximo: a noção de 'vizinho' perde contraste", fontsize=11)
salva(fig, DESTINO / "concentracao-de-distancias.png")
print("contraste:", dict(zip(dims, np.round(contraste, 2))))

# ---------------------------------------------------------------------------
# 2. Variância explicada nos dígitos
# ---------------------------------------------------------------------------
pca = PCA().fit(Xd)
acum = np.cumsum(pca.explained_variance_ratio_)
k90 = int(np.searchsorted(acum, 0.90) + 1)
k95 = int(np.searchsorted(acum, 0.95) + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4))
ax1.bar(range(1, 31), pca.explained_variance_ratio_[:30], color=AZUL)
ax1.set_xlabel("componente"); ax1.set_ylabel("fração da variância")
ax1.set_title("Scree plot: variância de cada componente")
ax2.plot(range(1, 65), acum, color=VERDE, lw=2.2)
for k, alvo, cor in [(k90, 0.90, AMBAR), (k95, 0.95, VERMELHO)]:
    ax2.axhline(alvo, color=cor, ls="--", lw=1)
    ax2.axvline(k, color=cor, ls="--", lw=1)
    ax2.annotate(f"{alvo:.0%} com {k} componentes", (k, alvo), xytext=(k + 5, alvo - 0.12),
                 arrowprops=dict(arrowstyle="->", color=cor), fontsize=9.5)
ax2.set_xlabel("número de componentes"); ax2.set_ylabel("variância acumulada")
ax2.set_title("Variância acumulada (64 pixels originais)")
fig.tight_layout()
salva(fig, DESTINO / "pca-variancia-explicada.png")
print(f"digitos: 90% com {k90}, 95% com {k95} componentes; PC1={pca.explained_variance_ratio_[0]:.3f}")

# ---------------------------------------------------------------------------
# 3. Reconstrução de dígitos com k componentes
# ---------------------------------------------------------------------------
ks = [2, 5, 10, 20, 40, 64]
idx = [0, 1, 3, 7]
fig, axes = plt.subplots(len(idx), len(ks), figsize=(11, 7.4))
for j, k in enumerate(ks):
    p = PCA(n_components=k).fit(Xd)
    rec = p.inverse_transform(p.transform(Xd))
    erro = np.mean((Xd - rec) ** 2)
    for i, a in enumerate(idx):
        axes[i, j].imshow(rec[a].reshape(8, 8), cmap="gray_r")
        axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
        axes[i, j].grid(False)
    axes[0, j].set_title(f"k={k}\nEQM={erro:.1f}", fontsize=10)
fig.suptitle("Reconstrução a partir de k componentes: com 10 a 20 o dígito já é legível",
             fontsize=11.5, fontweight="bold")
fig.tight_layout()
salva(fig, DESTINO / "pca-reconstrucao-digitos.png")

# ---------------------------------------------------------------------------
# 4. PCA vs t-SNE nos dígitos (e confiabilidade da vizinhança)
# ---------------------------------------------------------------------------
Zp = PCA(n_components=2).fit_transform(Xd)
Zt = TSNE(n_components=2, perplexity=30, random_state=0, init="pca").fit_transform(Xd)
tw_p = trustworthiness(Xd, Zp, n_neighbors=10)
tw_t = trustworthiness(Xd, Zt, n_neighbors=10)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
for ax, Z, nome, tw in [(axes[0], Zp, "PCA (2 componentes)", tw_p),
                        (axes[1], Zt, "t-SNE (perplexidade 30)", tw_t)]:
    for dig in range(10):
        m = yd == dig
        ax.scatter(Z[m, 0], Z[m, 1], s=6, color=CMAP10(dig), label=str(dig))
        ax.annotate(str(dig), np.median(Z[m], axis=0), fontsize=13, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, lw=0))
    ax.set_title(f"{nome} — confiabilidade da vizinhança = {tw:.3f}", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Os mesmos 1.797 dígitos de 64 pixels: PCA preserva a variância global, "
             "t-SNE preserva vizinhanças locais", fontsize=11.5, fontweight="bold")
fig.tight_layout()
salva(fig, DESTINO / "pca-vs-tsne-digitos.png")
print(f"trustworthiness: PCA={tw_p:.3f}, t-SNE={tw_t:.3f}")

# ---------------------------------------------------------------------------
# 5. Armadilhas do t-SNE: tamanhos e distâncias entre grupos não significam nada
# ---------------------------------------------------------------------------
A = rng.normal([0, 0], 0.3, (150, 2))
B = rng.normal([3, 0], 2.0, (150, 2))      # grupo 7x mais espalhado
C = rng.normal([25, 0], 0.3, (150, 2))     # grupo MUITO mais longe
D = np.vstack([A, B, C])
cor = np.repeat([AZUL, VERMELHO, VERDE], 150)
fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
axes[0].scatter(D[:, 0], D[:, 1], s=8, c=cor)
axes[0].set_title("dados originais (2D)", fontsize=11)
for ax, perp in zip(axes[1:], [5, 30, 100]):
    Z = TSNE(perplexity=perp, random_state=1, init="pca").fit_transform(D)
    ax.scatter(Z[:, 0], Z[:, 1], s=8, c=cor)
    ax.set_title(f"t-SNE, perplexidade {perp}", fontsize=11)
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Azul compacto, vermelho 7x mais espalhado, verde muito mais distante: o t-SNE "
             "iguala tamanhos e encurta distâncias", fontsize=11.5, fontweight="bold", y=1.04)
salva(fig, DESTINO / "tsne-armadilhas.png")

# ---------------------------------------------------------------------------
# 6. Rolo suíço: linear vs. manifold
# ---------------------------------------------------------------------------
R, t = make_swiss_roll(n_samples=1500, noise=0.2, random_state=0)
fig = plt.figure(figsize=(17, 4.3))
ax = fig.add_subplot(1, 4, 1, projection="3d")
ax.scatter(R[:, 0], R[:, 1], R[:, 2], c=t, cmap="viridis", s=5)
ax.set_title("rolo suíço (3D)", fontsize=11)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
for j, (nome, modelo) in enumerate([("PCA", PCA(2)),
                                    ("Isomap (k=12)", Isomap(n_neighbors=12, n_components=2)),
                                    ("t-SNE (perplexidade 50)", TSNE(perplexity=50, random_state=0,
                                                                     init="pca"))], start=2):
    Z = modelo.fit_transform(R)
    ax = fig.add_subplot(1, 4, j)
    ax.scatter(Z[:, 0], Z[:, 1], c=t, cmap="viridis", s=5)
    ax.set_title(nome, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("PCA achata o rolo (a cor se mistura); Isomap o desenrola seguindo a distância "
             "geodésica", fontsize=11.5, fontweight="bold", y=1.03)
salva(fig, DESTINO / "rolo-suico.png")
print("done")
