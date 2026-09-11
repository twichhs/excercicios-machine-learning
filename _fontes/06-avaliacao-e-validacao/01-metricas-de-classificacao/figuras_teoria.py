"""Gera as figuras do teoria.pdf do módulo 01-metricas-de-classificacao."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ / "_ferramentas"))
from estilo_figuras import AZUL, VERDE, VERMELHO, ROXO, AMBAR, CINZA, salva  # noqa: E402

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

rng = np.random.default_rng(61)
MODULO_REL = Path(__file__).resolve().parent.relative_to(RAIZ / "_fontes")
DESTINO = RAIZ / MODULO_REL / "figuras"

# ---------------------------------------------------------------------------
# 1. Matriz de confusão anotada (exemplo do texto)
# ---------------------------------------------------------------------------
TP, FN, FP, TN = 80, 20, 40, 860
fig, ax = plt.subplots(figsize=(9.5, 5.8))
ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
celulas = [(2, 3.5, f"VP = {TP}", "verdadeiro positivo", VERDE),
           (5, 3.5, f"FN = {FN}", "falso negativo", VERMELHO),
           (2, 0.8, f"FP = {FP}", "falso positivo", VERMELHO),
           (5, 0.8, f"VN = {TN}", "verdadeiro negativo", VERDE)]
for x, y, txt, sub, cor in celulas:
    ax.add_patch(FancyBboxPatch((x, y), 2.8, 2.4, boxstyle="round,pad=0.02", fc=cor, alpha=0.15,
                                ec=cor, lw=2))
    ax.text(x + 1.4, y + 1.45, txt, ha="center", va="center", fontsize=15, fontweight="bold")
    ax.text(x + 1.4, y + 0.75, sub, ha="center", va="center", fontsize=9.5, color=CINZA)
ax.text(3.4, 6.35, "previsto: doente", ha="center", fontsize=11, fontweight="bold")
ax.text(6.4, 6.35, "previsto: saudável", ha="center", fontsize=11, fontweight="bold")
ax.text(1.8, 4.7, "real:\ndoente\n(100)", ha="right", va="center", fontsize=11, fontweight="bold")
ax.text(1.8, 2.0, "real:\nsaudável\n(900)", ha="right", va="center", fontsize=11, fontweight="bold")
ax.text(8.2, 4.7, f"recall = {TP}/{TP + FN} = {TP / (TP + FN):.0%}", va="center", fontsize=10.5, color=AZUL)
ax.text(8.2, 2.0, f"especificidade =\n{TN}/{TN + FP} = {TN / (TN + FP):.1%}", va="center", fontsize=10.5,
        color=AZUL)
ax.text(3.4, 0.25, f"precisão = {TP}/{TP + FP} = {TP / (TP + FP):.1%}", ha="center", fontsize=10.5,
        color=ROXO)
ax.set_title("Matriz de confusão do exemplo: 1.000 pacientes, 100 doentes", fontsize=12)
salva(fig, DESTINO / "matriz-confusao-anotada.png")

# ---------------------------------------------------------------------------
# Um modelo de referência em dados desbalanceados (10% de positivos)
# ---------------------------------------------------------------------------
def gera(n, prev, semente):
    g = np.random.default_rng(semente)
    y = (g.random(n) < prev).astype(int)
    X = g.normal(0, 1, (n, 4)) + y[:, None] * np.array([1.0, 0.7, 0.5, 0.0])
    return X, y

X, y = gera(20000, 0.10, 1)
Xt, yt = gera(20000, 0.10, 2)
modelo = LogisticRegression().fit(X, y)
p = modelo.predict_proba(Xt)[:, 1]

# ---------------------------------------------------------------------------
# 2. Métricas em função do limiar
# ---------------------------------------------------------------------------
limiares = np.linspace(0.01, 0.95, 200)
acc, prec, rec, f1 = [], [], [], []
for t in limiares:
    yp = p >= t
    tp, fp = (yp & (yt == 1)).sum(), (yp & (yt == 0)).sum()
    fn = (~yp & (yt == 1)).sum()
    acc.append((yp == yt).mean())
    prec.append(tp / max(tp + fp, 1))
    rec.append(tp / (tp + fn))
    f1.append(2 * tp / (2 * tp + fp + fn))
fig, ax = plt.subplots(figsize=(9, 4.8))
for serie, nome, cor in [(acc, "acurácia", CINZA), (prec, "precisão", ROXO), (rec, "recall", AZUL),
                         (f1, "F1", VERDE)]:
    ax.plot(limiares, serie, lw=2.2, color=cor, label=nome)
t_f1 = limiares[int(np.argmax(f1))]
ax.axvline(0.5, color="black", ls=":", lw=1.2)
ax.axvline(t_f1, color=VERDE, ls="--", lw=1.2)
ax.annotate(f"F1 máximo em {t_f1:.2f}", (t_f1, max(f1)), xytext=(t_f1 + 0.12, max(f1) + 0.08),
            arrowprops=dict(arrowstyle="->", color=VERDE), fontsize=9.5)
ax.set_xlabel("limiar de decisão"); ax.set_ylim(0, 1.02); ax.legend(loc="center right")
ax.set_title("O mesmo modelo, métricas diferentes a cada limiar: 0,5 não é especial\n"
             "(10% de positivos — a acurácia quase não se mexe)", fontsize=11)
salva(fig, DESTINO / "metricas-vs-limiar.png")
print(f"limiar de F1 máximo: {t_f1:.3f}; F1 max={max(f1):.3f}; F1 em 0.5={f1[int(np.argmin(abs(limiares - 0.5)))]:.3f}")

# ---------------------------------------------------------------------------
# 3. ROC vs. PR sob prevalências diferentes
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
for prev, cor in [(0.5, AZUL), (0.1, VERDE), (0.01, VERMELHO)]:
    Xp, yp_ = gera(60000, prev, 7)
    s = modelo.decision_function(Xp)
    fpr, tpr, _ = roc_curve(yp_, s)
    pr, rc, _ = precision_recall_curve(yp_, s)
    auc, ap = roc_auc_score(yp_, s), average_precision_score(yp_, s)
    ax1.plot(fpr, tpr, color=cor, lw=2, label=f"prevalência {prev:.0%}: AUC={auc:.3f}")
    ax2.plot(rc, pr, color=cor, lw=2, label=f"prevalência {prev:.0%}: AP={ap:.3f}")
    ax2.axhline(prev, color=cor, ls=":", lw=1)
    print(f"prev={prev}: ROC-AUC={auc:.3f}, AP={ap:.3f}")
ax1.plot([0, 1], [0, 1], color=CINZA, ls="--", lw=1)
ax1.set_xlabel("taxa de falso positivo"); ax1.set_ylabel("taxa de verdadeiro positivo (recall)")
ax1.set_title("ROC: praticamente a mesma curva nas três prevalências")
ax1.legend(fontsize=9, loc="lower right")
ax2.set_xlabel("recall"); ax2.set_ylabel("precisão")
ax2.set_title("Precisão-recall: a curva desaba com a prevalência")
ax2.legend(fontsize=9, loc="upper right")
fig.tight_layout()
salva(fig, DESTINO / "roc-vs-pr-prevalencia.png")

# ---------------------------------------------------------------------------
# 4. AUC como probabilidade de ordenar certo um par
# ---------------------------------------------------------------------------
s_pos, s_neg = modelo.decision_function(Xt[yt == 1]), modelo.decision_function(Xt[yt == 0])
pares = rng.integers(0, [len(s_pos), len(s_neg)], (200000, 2))
prob_par = np.mean(s_pos[pares[:, 0]] > s_neg[pares[:, 1]])
fig, ax = plt.subplots(figsize=(9, 4.6))
bins = np.linspace(-6, 4, 70)
ax.hist(s_neg, bins=bins, density=True, alpha=0.55, color=AZUL, label="negativos")
ax.hist(s_pos, bins=bins, density=True, alpha=0.55, color=VERMELHO, label="positivos")
ax.set_xlabel("escore do modelo"); ax.set_ylabel("densidade"); ax.legend()
ax.set_title(f"AUC = P(escore de um positivo > escore de um negativo)\n"
             f"sorteando 200 mil pares: {prob_par:.3f}  |  roc_auc_score: {roc_auc_score(yt, p):.3f}",
             fontsize=11)
salva(fig, DESTINO / "auc-como-probabilidade.png")
print(f"AUC por pares={prob_par:.4f} vs sklearn={roc_auc_score(yt, p):.4f}")

# ---------------------------------------------------------------------------
# 5. Curva de ganho acumulado e lift
# ---------------------------------------------------------------------------
ordem = np.argsort(p)[::-1]
frac = np.arange(1, len(p) + 1) / len(p)
ganho = np.cumsum(yt[ordem]) / yt.sum()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
ax1.plot(frac, ganho, color=AZUL, lw=2.2, label="modelo")
ax1.plot([0, 1], [0, 1], color=CINZA, ls="--", label="aleatório")
ax1.plot([0, yt.mean(), 1], [0, 1, 1], color=VERDE, ls=":", label="perfeito")
g20 = ganho[int(0.2 * len(p)) - 1]
ax1.annotate(f"top 20% da lista contém\n{g20:.0%} dos positivos", (0.2, g20), xytext=(0.35, 0.45),
             arrowprops=dict(arrowstyle="->", color=CINZA), fontsize=9.5)
ax1.set_xlabel("fração da base contatada (maiores escores primeiro)")
ax1.set_ylabel("fração dos positivos encontrados"); ax1.legend(loc="lower right")
ax1.set_title("Curva de ganho acumulado")
decis = np.array_split(ordem, 10)
lift = [yt[d].mean() / yt.mean() for d in decis]
ax2.bar(range(1, 11), lift, color=AZUL)
ax2.axhline(1, color=CINZA, ls="--")
ax2.set_xlabel("decil (1 = maiores escores)"); ax2.set_ylabel("lift")
ax2.set_title(f"Lift por decil: o 1º decil tem {lift[0]:.1f}x a taxa média")
fig.tight_layout()
salva(fig, DESTINO / "ganho-e-lift.png")
print(f"ganho top20={g20:.3f}; lift decil1={lift[0]:.2f}")

# ---------------------------------------------------------------------------
# 6. Custo esperado em função do limiar
# ---------------------------------------------------------------------------
C_FP, C_FN = 10, 200
custos = [(C_FP * ((p >= t) & (yt == 0)).sum() + C_FN * ((p < t) & (yt == 1)).sum()) / len(p)
          for t in limiares]
t_teo = C_FP / (C_FP + C_FN)
t_emp = limiares[int(np.argmin(custos))]
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(limiares, custos, color=VERMELHO, lw=2.2)
ax.axvline(t_teo, color=VERDE, ls="--", lw=1.5, label=f"teórico C_FP/(C_FP+C_FN) = {t_teo:.3f}")
ax.axvline(0.5, color="black", ls=":", lw=1.2, label="0,5")
ax.set_xlabel("limiar de decisão"); ax.set_ylabel("custo médio por caso (R$)")
ax.set_title(f"Falso positivo custa R$ {C_FP}, falso negativo R$ {C_FN}: o limiar ótimo fica\n"
             f"muito abaixo de 0,5 (empírico: {t_emp:.3f})", fontsize=11)
ax.legend()
salva(fig, DESTINO / "custo-vs-limiar.png")
c05 = custos[int(np.argmin(abs(limiares - 0.5)))]
print(f"custo: limiar teórico={t_teo:.3f} empírico={t_emp:.3f}; custo min={min(custos):.2f}, "
      f"custo em 0.5={c05:.2f}")
print("done")
