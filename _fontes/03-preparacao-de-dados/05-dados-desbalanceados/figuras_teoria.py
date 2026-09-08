"""Gera as figuras do teoria.pdf do módulo 05-dados-desbalanceados."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ / "_ferramentas"))
from estilo_figuras import AZUL, VERDE, VERMELHO, ROXO, AMBAR, CINZA, salva  # noqa: E402

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

rng = np.random.default_rng(11)
MODULO_REL = Path(__file__).resolve().parent.relative_to(RAIZ / "_fontes")
DESTINO = RAIZ / MODULO_REL / "figuras"

# ---------------------------------------------------------------------------
# 1. Baseline "preguiçoso" vs. modelo real: acurácia favorece o inútil
# ---------------------------------------------------------------------------
metricas = ["Acurácia", "Precisão", "Recall", "F1"]
baseline = [0.998, 0.0, 0.0, 0.0]
modelo_real = [0.9905, 0.143, 0.75, 0.240]

fig, ax = plt.subplots(figsize=(8.5, 5.5))
largura = 0.35
posicoes = np.arange(len(metricas))
ax.bar(posicoes - largura / 2, baseline, largura, color=CINZA,
       label="Modelo A: sempre prevê 'não é fraude'")
ax.bar(posicoes + largura / 2, modelo_real, largura, color=VERMELHO,
       label="Modelo B: real, recall de 75%")
for x, v in zip(posicoes - largura / 2, baseline):
    ax.text(x, v + 0.015, f"{v*100:.1f}%", ha="center", fontsize=8.5)
for x, v in zip(posicoes + largura / 2, modelo_real):
    ax.text(x, v + 0.015, f"{v*100:.1f}%", ha="center", fontsize=8.5)
ax.set_xticks(posicoes)
ax.set_xticklabels(metricas, fontsize=10.5)
ax.set_ylim(0, 1.12)
ax.set_ylabel("valor da métrica")
ax.set_title("Acurácia favorece o modelo inútil; precisão, recall e F1\n"
             "revelam qual modelo é realmente melhor", fontsize=11.5)
ax.legend(fontsize=8.6, loc="upper center")
salva(fig, DESTINO / "baseline-fraude-metricas.png")

# ---------------------------------------------------------------------------
# 2. Geometria do SMOTE: pontos reais, vizinhos e sintéticos interpolados
# ---------------------------------------------------------------------------
minoria = np.array([
    [2.0, 5.0], [4.0, 3.0], [3.2, 4.6], [1.4, 3.6], [4.6, 5.2],
])
maioria = rng.normal(loc=[2.8, 4.2], scale=1.8, size=(60, 2))

fig, ax = plt.subplots(figsize=(8, 6.2))
ax.scatter(maioria[:, 0], maioria[:, 1], s=26, color=CINZA, alpha=0.55,
           label="classe majoritária (legítimas)")
ax.scatter(minoria[:, 0], minoria[:, 1], s=70, color=VERMELHO, zorder=5,
           edgecolor="white", linewidth=0.8, label="classe minoritária (fraude)")

nn = NearestNeighbors(n_neighbors=3).fit(minoria)
sinteticos = []
for i, xi in enumerate(minoria):
    _, idx = nn.kneighbors(xi.reshape(1, -1))
    for j in idx[0][1:]:
        xviz = minoria[j]
        for lam in (0.3, 0.6):
            novo = xi + lam * (xviz - xi)
            sinteticos.append(novo)
            ax.plot([xi[0], xviz[0]], [xi[1], xviz[1]], color=AMBAR,
                     lw=0.8, alpha=0.5, zorder=3)
sinteticos = np.array(sinteticos)
ax.scatter(sinteticos[:, 0], sinteticos[:, 1], s=55, color=AMBAR, marker="D",
           zorder=6, edgecolor="white", linewidth=0.6,
           label="pontos sintéticos (SMOTE)")

# destaca o exemplo numérico do texto: xi=(2,5), x_viz=(4,3), lambda=0.3 -> (2.6, 4.4)
xi_txt, xviz_txt = np.array([2.0, 5.0]), np.array([4.0, 3.0])
novo_txt = xi_txt + 0.3 * (xviz_txt - xi_txt)
ax.annotate(r"$x_{novo}=(2{,}6,\ 4{,}4)$" + "\n" + r"($\lambda=0{,}3$)",
            xy=novo_txt, xytext=(novo_txt[0] - 1.9, novo_txt[1] + 1.1),
            fontsize=9, color="#5A4200",
            arrowprops=dict(arrowstyle="->", color="#5A4200", lw=1.1))

ax.set_xlabel("feature 1 (padronizada)")
ax.set_ylabel("feature 2 (padronizada)")
ax.set_title("SMOTE interpola entre pontos minoritários reais e seus vizinhos\n"
             "— cada losango é um exemplo sintético novo ao longo do segmento", fontsize=11)
ax.legend(fontsize=8.4, loc="upper right")
salva(fig, DESTINO / "smote-geometria.png")

# ---------------------------------------------------------------------------
# 3. Fronteira de decisão antes/depois de reamostrar (regressão logística)
# ---------------------------------------------------------------------------
n_maj, n_min = 300, 15
Xmaj = rng.normal(loc=[0, 0], scale=1.4, size=(n_maj, 2))
Xmin = rng.normal(loc=[2.6, 2.6], scale=0.9, size=(n_min, 2))
X = np.vstack([Xmaj, Xmin])
y = np.array([0] * n_maj + [1] * n_min)

# reamostragem simples (oversampling aleatório) só para ilustrar o efeito na fronteira
idx_extra = rng.integers(0, n_min, n_maj - n_min)
X_bal = np.vstack([X, Xmin[idx_extra]])
y_bal = np.concatenate([y, np.ones(len(idx_extra))])

xx, yy = np.meshgrid(np.linspace(-4, 6, 300), np.linspace(-4, 6, 300))
grade = np.column_stack([xx.ravel(), yy.ravel()])

fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
for ax, (Xf, yf, titulo) in zip(axes, [
        (X, y, "Antes: treino no desbalanceamento original\n(20:1)"),
        (X_bal, y_bal, "Depois: treino com oversampling\naleatório (1:1)")]):
    modelo = LogisticRegression().fit(Xf, yf)
    zz = modelo.predict_proba(grade)[:, 1].reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.55)
    ax.contour(xx, yy, zz, levels=[0.5], colors="black", linewidths=1.8)
    ax.scatter(Xf[yf == 0, 0], Xf[yf == 0, 1], s=16, color=CINZA, alpha=0.6,
               label="majoritária")
    ax.scatter(Xf[yf == 1, 0], Xf[yf == 1, 1], s=30, color=VERMELHO,
               edgecolor="white", linewidth=0.5, label="minoritária")
    ax.set_title(titulo, fontsize=10.8)
    ax.legend(fontsize=8, loc="lower right")
fig.suptitle("Reamostrar desloca a fronteira de decisão em direção à classe "
             "majoritária original", fontsize=11.5, y=1.03)
salva(fig, DESTINO / "fronteira-antes-depois-reamostragem.png")

# ---------------------------------------------------------------------------
# 4. Undersampling: aleatório vs. NearMiss-1 (quem fica, quem sai)
# ---------------------------------------------------------------------------
n_maj2, n_min2 = 220, 18
Xmaj2 = rng.normal(loc=[0, 0], scale=1.6, size=(n_maj2, 2))
Xmin2 = rng.normal(loc=[2.4, 2.2], scale=0.8, size=(n_min2, 2))

# random undersampling
idx_rand = rng.choice(n_maj2, size=n_min2, replace=False)
mask_rand = np.zeros(n_maj2, dtype=bool)
mask_rand[idx_rand] = True

# NearMiss-1: mantém os majoritários com menor distância média aos k vizinhos minoritários
nn_min = NearestNeighbors(n_neighbors=3).fit(Xmin2)
dist, _ = nn_min.kneighbors(Xmaj2)
dist_media = dist.mean(axis=1)
idx_nm = np.argsort(dist_media)[:n_min2]
mask_nm = np.zeros(n_maj2, dtype=bool)
mask_nm[idx_nm] = True

fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
for ax, mask, titulo in zip(
        axes, [mask_rand, mask_nm],
        ["Random Undersampling\n(sorteio uniforme dos majoritários mantidos)",
         "NearMiss-1\n(mantém os majoritários mais perto da fronteira)"]):
    ax.scatter(Xmaj2[~mask, 0], Xmaj2[~mask, 1], s=18, color=CINZA, alpha=0.25,
               label="majoritária (descartada)")
    ax.scatter(Xmaj2[mask, 0], Xmaj2[mask, 1], s=32, color=AZUL,
               edgecolor="white", linewidth=0.5, label="majoritária (mantida)")
    ax.scatter(Xmin2[:, 0], Xmin2[:, 1], s=45, color=VERMELHO,
               edgecolor="white", linewidth=0.6, label="minoritária (toda mantida)")
    ax.set_title(titulo, fontsize=10.8)
    ax.legend(fontsize=8, loc="upper left")
fig.suptitle("NearMiss escolhe sistematicamente os majoritários perto da fronteira;\n"
             "o sorteio aleatório espalha a mesma quantidade por todo o espaço", fontsize=11, y=1.05)
salva(fig, DESTINO / "undersampling-random-vs-nearmiss.png")

print("done")
