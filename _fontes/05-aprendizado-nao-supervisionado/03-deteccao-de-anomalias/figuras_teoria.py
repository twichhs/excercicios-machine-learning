"""Gera as figuras do teoria.pdf do módulo 03-deteccao-de-anomalias."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ / "_ferramentas"))
from estilo_figuras import AZUL, VERDE, VERMELHO, ROXO, AMBAR, CINZA, salva  # noqa: E402

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.metrics import precision_score

rng = np.random.default_rng(53)
MODULO_REL = Path(__file__).resolve().parent.relative_to(RAIZ / "_fontes")
DESTINO = RAIZ / MODULO_REL / "figuras"

# ---------------------------------------------------------------------------
# 1. Três tipos de anomalia
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
P = rng.normal(0, 1, (300, 2))
axes[0].scatter(P[:, 0], P[:, 1], s=10, color=CINZA)
axes[0].scatter([3.8, -3.5], [3.2, 3.9], s=90, color=VERMELHO, marker="X")
axes[0].set_title("pontual: longe de tudo", fontsize=11)
dias = np.arange(365 * 2)
temp = 22 + 8 * np.sin(2 * np.pi * (dias - 100) / 365) + rng.normal(0, 1.2, len(dias))
temp[560] = 29.5  # 29,5 graus em pleno inverno: normal no verão, anômalo aqui
axes[1].plot(dias, temp, color=AZUL, lw=0.8)
axes[1].scatter([560], [temp[560]], s=90, color=VERMELHO, marker="X", zorder=5)
axes[1].axhline(29.5, color=CINZA, ls=":", lw=1)
axes[1].set_title("contextual: valor comum, época errada", fontsize=11)
axes[1].set_xlabel("dia")
t = np.arange(600)
sinal = np.sin(2 * np.pi * t / 40) + rng.normal(0, 0.08, len(t))
sinal[300:360] = sinal[300] + rng.normal(0, 0.03, 60)
axes[2].plot(t, sinal, color=AZUL, lw=0.9)
axes[2].axvspan(300, 360, color=VERMELHO, alpha=0.15)
axes[2].set_title("coletiva: cada valor é normal, a sequência não", fontsize=11)
axes[2].set_xlabel("tempo")
salva(fig, DESTINO / "tipos-de-anomalia.png")

# ---------------------------------------------------------------------------
# 2. Mascaramento: z-score clássico vs. z robusto (mediana e MAD)
# ---------------------------------------------------------------------------
normais = rng.normal(200, 20, 900)          # tempo de resposta em ms
lentos = rng.normal(300, 10, 100)           # 10% de requisições lentas (as anomalias)
x = np.concatenate([normais, lentos])
z = (x - x.mean()) / x.std()
mad = np.median(np.abs(x - np.median(x)))
z_rob = (x - np.median(x)) / (1.4826 * mad)
lim_z = x.mean() + 3 * x.std()
lim_rob = np.median(x) + 3 * 1.4826 * mad
fig, ax = plt.subplots(figsize=(9.5, 4.6))
ax.hist(normais, bins=50, color=AZUL, alpha=0.7, label="requisições normais")
ax.hist(lentos, bins=15, color=VERMELHO, alpha=0.8, label="requisições anômalas (10%)")
ax.axvline(lim_z, color=AMBAR, lw=2, ls="--", label=f"média + 3 dp = {lim_z:.0f} ms")
ax.axvline(lim_rob, color=VERDE, lw=2, ls="--", label=f"mediana + 3 MAD robusto = {lim_rob:.0f} ms")
ax.set_xlabel("tempo de resposta (ms)")
ax.set_title("Mascaramento: as próprias anomalias inflam média e desvio-padrão,\n"
             "empurrando o limiar clássico para além delas", fontsize=11)
ax.legend(fontsize=9)
salva(fig, DESTINO / "mascaramento.png")
print(f"mascaramento: z clássico pega {np.mean(z[900:] > 3):.0%} das anômalas; "
      f"z robusto pega {np.mean(z_rob[900:] > 3):.0%}; falsos alarmes robusto: "
      f"{np.mean(z_rob[:900] > 3):.1%}; media={x.mean():.1f} dp={x.std():.1f} "
      f"mediana={np.median(x):.1f} MAD={mad:.1f}")

# ---------------------------------------------------------------------------
# 3. Mahalanobis vs. euclidiana
# ---------------------------------------------------------------------------
S = np.array([[1.0, 0.8], [0.8, 1.0]])
M = rng.multivariate_normal([0, 0], S, 800)
Si = np.linalg.inv(S)
A, B = np.array([1.0, 1.0]), np.array([1.0, -1.0])
dA, dB = A @ Si @ A, B @ Si @ B
lim = chi2.ppf(0.975, 2)
fig, ax = plt.subplots(figsize=(7.5, 6.2))
ax.scatter(M[:, 0], M[:, 1], s=7, color=CINZA, alpha=0.6)
val, vec = np.linalg.eigh(S)
ang = np.degrees(np.arctan2(vec[1, 1], vec[0, 1]))
ax.add_patch(Ellipse((0, 0), 2 * np.sqrt(lim * val[1]), 2 * np.sqrt(lim * val[0]), angle=ang,
                     fill=False, color=VERDE, lw=2, label="Mahalanobis: limite do qui-quadrado 97,5%"))
circ = plt.Circle((0, 0), np.sqrt(2), fill=False, color=AMBAR, lw=1.8, ls="--",
                  label="euclidiana: mesma distância de A e B")
ax.add_patch(circ)
for P_, nome, d in [(A, "A", dA), (B, "B", dB)]:
    ax.scatter(*P_, s=120, color=VERMELHO if d > lim else AZUL, marker="X", zorder=6)
    ax.annotate(f"{nome}: D²={d:.2f}", P_, xytext=(P_[0] + 0.25, P_[1] + 0.25), fontsize=10.5,
                fontweight="bold")
ax.set_aspect("equal"); ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
ax.legend(loc="lower right", fontsize=8.5)
ax.set_title("A e B estão à mesma distância euclidiana do centro,\nmas só B viola a correlação dos dados",
             fontsize=11)
salva(fig, DESTINO / "mahalanobis-vs-euclidiana.png")
print(f"Mahalanobis: D2(A)={dA:.3f}, D2(B)={dB:.3f}, limite={lim:.3f}")

# ---------------------------------------------------------------------------
# 4. Isolation Forest: isolar um ponto normal exige muito mais cortes
# ---------------------------------------------------------------------------
pts = np.vstack([rng.normal(0, 1, (120, 2)), [[3.6, 3.2]]])
alvo_normal, alvo_anomalo = 0, len(pts) - 1
pts[alvo_normal] = [0.1, -0.2]

def isola(pontos, alvo, semente):
    g = np.random.default_rng(semente)
    idx = np.arange(len(pontos))
    caixa = [pontos[:, 0].min(), pontos[:, 0].max(), pontos[:, 1].min(), pontos[:, 1].max()]
    cortes = []
    while len(idx) > 1:
        f = g.integers(0, 2)
        lo, hi = pontos[idx, f].min(), pontos[idx, f].max()
        c = g.uniform(lo, hi)
        cortes.append((f, c, caixa.copy()))
        lado = pontos[idx, f] < c
        mantem = lado if pontos[alvo, f] < c else ~lado
        if f == 0:
            caixa[1 if pontos[alvo, 0] < c else 0] = c
        else:
            caixa[3 if pontos[alvo, 1] < c else 2] = c
        idx = idx[mantem]
    return cortes

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for ax, alvo, nome, cor in [(axes[0], alvo_normal, "ponto normal", AZUL),
                            (axes[1], alvo_anomalo, "anomalia", VERMELHO)]:
    cortes = isola(pts, alvo, semente=3)
    ax.scatter(pts[:, 0], pts[:, 1], s=10, color=CINZA)
    for f, c, cx in cortes:
        if f == 0:
            ax.plot([c, c], [cx[2], cx[3]], color=AMBAR, lw=1)
        else:
            ax.plot([cx[0], cx[1]], [c, c], color=AMBAR, lw=1)
    ax.scatter(*pts[alvo], s=130, color=cor, marker="X", zorder=5)
    ax.set_title(f"isolar o {nome}: {len(cortes)} cortes", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
comprimentos_n = [len(isola(pts, alvo_normal, s)) for s in range(300)]
comprimentos_a = [len(isola(pts, alvo_anomalo, s)) for s in range(300)]
axes[2].hist(comprimentos_n, bins=range(0, 30), color=AZUL, alpha=0.7, label="ponto normal")
axes[2].hist(comprimentos_a, bins=range(0, 30), color=VERMELHO, alpha=0.7, label="anomalia")
axes[2].set_xlabel("nº de cortes até isolar (300 árvores aleatórias)")
axes[2].set_title(f"média: {np.mean(comprimentos_n):.1f} cortes vs. {np.mean(comprimentos_a):.1f} cortes",
                  fontsize=11)
axes[2].legend()
fig.suptitle("Isolation Forest: cortes aleatórios isolam anomalias muito mais cedo", fontsize=11.5,
             fontweight="bold", y=1.02)
salva(fig, DESTINO / "isolation-forest-cortes.png")
print(f"IF cortes: normal={np.mean(comprimentos_n):.2f}, anomalia={np.mean(comprimentos_a):.2f}")

# ---------------------------------------------------------------------------
# 5. LOF: densidade local vs. distância global
# ---------------------------------------------------------------------------
denso = rng.normal([0, 0], 0.25, (200, 2))
esparso = rng.normal([5, 0], 1.6, (200, 2))
suspeito = np.array([[1.3, 0.9]])        # perto do grupo denso, mas fora dele
D = np.vstack([denso, esparso, suspeito])
knn = NearestNeighbors(n_neighbors=11).fit(D)
dist_k = knn.kneighbors(D)[0][:, -1]
lof = LocalOutlierFactor(n_neighbors=20).fit(D)
escore_lof = -lof.negative_outlier_factor_
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
for ax, esc, tit in [(axes[0], dist_k, "distância ao 10º vizinho (global)"),
                     (axes[1], escore_lof, "LOF (densidade relativa aos vizinhos)")]:
    top = np.argsort(esc)[-8:]
    ax.scatter(D[:, 0], D[:, 1], s=9, color=CINZA)
    ax.scatter(D[top, 0], D[top, 1], s=110, facecolor="none", edgecolor=VERMELHO, lw=2,
               label="8 maiores escores")
    ax.scatter(*suspeito.T, s=90, marker="X", color=ROXO, zorder=6, label="ponto suspeito")
    rank = int((esc > esc[-1]).sum()) + 1
    ax.set_title(f"{tit}\nposição do suspeito no ranking: {rank}º de {len(D)}", fontsize=11)
    ax.set_aspect("equal"); ax.legend(fontsize=8.5, loc="upper right")
salva(fig, DESTINO / "lof-densidade-local.png")
print(f"LOF do suspeito={escore_lof[-1]:.2f}; rank dist_k={(dist_k > dist_k[-1]).sum() + 1}, "
      f"rank LOF={(escore_lof > escore_lof[-1]).sum() + 1}")

# ---------------------------------------------------------------------------
# 6. Orçamento de alertas: precisão e recall em função de quantos casos se investiga
# ---------------------------------------------------------------------------
n, n_anom = 20000, 60
Xn = rng.normal(0, 1, (n - n_anom, 6))
Xa = rng.normal(0, 1, (n_anom, 6)) * 2.2 + rng.choice([-1, 1], (n_anom, 6)) * 1.2
X = np.vstack([Xn, Xa])
y = np.r_[np.zeros(n - n_anom), np.ones(n_anom)]
esc = -IsolationForest(n_estimators=300, random_state=0).fit(X).score_samples(X)
ordem = np.argsort(esc)[::-1]
ks = np.arange(10, 1001, 10)
prec = np.array([y[ordem[:k]].mean() for k in ks])
rec = np.array([y[ordem[:k]].sum() / n_anom for k in ks])
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(ks, prec, color=AZUL, lw=2.2, label="precisão@k (fração dos alertas que é anomalia)")
ax.plot(ks, rec, color=VERMELHO, lw=2.2, label="recall@k (fração das anomalias encontradas)")
ax.axvline(100, color=CINZA, ls=":", lw=1.5)
ax.annotate(f"orçamento de 100 alertas:\nprecisão {prec[9]:.0%}, recall {rec[9]:.0%}", (100, 0.5),
            xytext=(260, 0.55), arrowprops=dict(arrowstyle="->", color=CINZA), fontsize=9.5)
ax.set_xlabel("k = número de casos investigados (maiores escores primeiro)")
ax.set_ylim(0, 1.02); ax.legend(fontsize=9)
ax.set_title("Com 60 anomalias em 20 mil registros, cada alerta a mais investigado\n"
             "compra recall ao custo de precisão", fontsize=11)
salva(fig, DESTINO / "orcamento-de-alertas.png")
print(f"orçamento 100: precisão={prec[9]:.3f}, recall={rec[9]:.3f}; k=300: prec={prec[29]:.3f} rec={rec[29]:.3f}")
print("done")
