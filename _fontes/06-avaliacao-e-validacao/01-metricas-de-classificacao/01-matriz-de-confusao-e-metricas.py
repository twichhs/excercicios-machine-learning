# %% [markdown]
# # Matriz de confusão e métricas
#
# **Tema:** Avaliação e Validação de Modelos › Métricas de Classificação
#
# Este notebook calcula todas as métricas do `teoria.pdf` do zero a partir
# da matriz de confusão e confere contra o `scikit-learn`, mostra a
# precisão mudando com a prevalência (Bayes), compara F1 e MCC quando se
# troca a classe positiva, resume um problema multiclasse de três formas,
# coloca um intervalo de confiança por bootstrap numa métrica e compara
# dois modelos com o teste de McNemar.

# %%
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
                             balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
                             confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(611)
print("pronto")

# %% [markdown]
# ## 1. O exemplo do `teoria.pdf`, do zero
#
# 1.000 pacientes, 100 doentes: VP=80, FN=20, FP=40, VN=860. Montamos
# vetores de rótulo real e previsto que produzem exatamente essa matriz.

# %%
y_real = np.r_[np.ones(100), np.zeros(900)].astype(int)
y_prev = np.r_[np.ones(80), np.zeros(20), np.ones(40), np.zeros(860)].astype(int)

def metricas_do_zero(y, yp):
    vp = np.sum((y == 1) & (yp == 1)); fn = np.sum((y == 1) & (yp == 0))
    fp = np.sum((y == 0) & (yp == 1)); vn = np.sum((y == 0) & (yp == 0))
    n = vp + fn + fp + vn
    prec, rec, esp = vp / (vp + fp), vp / (vp + fn), vn / (vn + fp)
    po = (vp + vn) / n
    pe = ((vp + fp) / n) * ((vp + fn) / n) + ((vn + fn) / n) * ((vn + fp) / n)
    return {"acurácia": po, "precisão": prec, "recall": rec, "especificidade": esp,
            "F1": 2 * vp / (2 * vp + fp + fn),
            "F2": 5 * prec * rec / (4 * prec + rec),
            "acurácia balanceada": (rec + esp) / 2,
            "MCC": (vp * vn - fp * fn) / np.sqrt(float((vp + fp) * (vp + fn) * (vn + fp) * (vn + fn))),
            "kappa": (po - pe) / (1 - pe)}

nossas = metricas_do_zero(y_real, y_prev)
sk = {"acurácia": accuracy_score(y_real, y_prev), "precisão": precision_score(y_real, y_prev),
      "recall": recall_score(y_real, y_prev),
      "especificidade": recall_score(y_real, y_prev, pos_label=0),
      "F1": f1_score(y_real, y_prev), "F2": fbeta_score(y_real, y_prev, beta=2),
      "acurácia balanceada": balanced_accuracy_score(y_real, y_prev),
      "MCC": matthews_corrcoef(y_real, y_prev), "kappa": cohen_kappa_score(y_real, y_prev)}
print(pd.DataFrame({"do zero": nossas, "sklearn": sk}).round(4).to_string())
print("\nmatriz (linhas = real 0/1, colunas = previsto 0/1):\n", confusion_matrix(y_real, y_prev))

# %% [markdown]
# ## 2. O classificador preguiçoso
#
# "Sempre saudável": quais métricas percebem que ele é inútil?

# %%
preguicoso = np.zeros_like(y_real)
with np.errstate(invalid="ignore", divide="ignore"):
    print({k: round(float(v), 3) for k, v in metricas_do_zero(y_real, preguicoso).items()
           if k in ["acurácia", "recall", "acurácia balanceada", "kappa"]})
print(f"MCC (sklearn trata 0/0 como 0): {matthews_corrcoef(y_real, preguicoso):.3f}")

# %% [markdown]
# ## 3. A precisão depende da prevalência
#
# O mesmo "exame" (recall 80%, especificidade 95,6%) aplicado a populações
# com prevalências diferentes — simulado e pela fórmula de Bayes.

# %%
recall_exame, esp_exame = 0.80, 860 / 900
linhas = []
for prev in [0.5, 0.2, 0.1, 0.05, 0.01, 0.001]:
    n = 200_000
    doente = rng.random(n) < prev
    alarme = np.where(doente, rng.random(n) < recall_exame, rng.random(n) > esp_exame)
    simulada = (doente & alarme).sum() / alarme.sum()
    bayes = recall_exame * prev / (recall_exame * prev + (1 - esp_exame) * (1 - prev))
    linhas.append({"prevalência": prev, "precisão simulada": simulada, "precisão (Bayes)": bayes,
                   "recall simulado": (doente & alarme).sum() / doente.sum()})
print(pd.DataFrame(linhas).round(3).to_string(index=False))

# %% [markdown]
# O recall fica em 0,80 em todas as linhas; a precisão vai de 95% a menos
# de 2%. O modelo não mudou — a população mudou.

# %% [markdown]
# ## 4. Trocar a classe positiva: F1 muda, MCC não

# %%
y_inv, yp_inv = 1 - y_real, 1 - y_prev
print(f"F1  com 'doente' positivo: {f1_score(y_real, y_prev):.3f} | "
      f"com 'saudável' positivo: {f1_score(y_inv, yp_inv):.3f}")
print(f"MCC com 'doente' positivo: {matthews_corrcoef(y_real, y_prev):.3f} | "
      f"com 'saudável' positivo: {matthews_corrcoef(y_inv, yp_inv):.3f}")

# %% [markdown]
# ## 5. Multiclasse: macro, micro e ponderada
#
# Um classificador de tickets com três classes muito desbalanceadas.

# %%
n = 6000
classes = np.array(["dúvida", "reclamação", "cancelamento"])
y_mc = rng.choice(3, n, p=[0.80, 0.15, 0.05])
X_mc = rng.normal(0, 1, (n, 5)) + np.array([[0, 0, 0, 0, 0], [1.2, 0.8, 0, 0, 0],
                                            [0.6, 0.2, 0.9, 0, 0]])[y_mc]
X_tr, X_te, y_tr, y_te = train_test_split(X_mc, y_mc, test_size=0.4, random_state=0, stratify=y_mc)
pred_mc = LogisticRegression(max_iter=2000).fit(X_tr, y_tr).predict(X_te)
print(classification_report(y_te, pred_mc, target_names=classes, digits=3, zero_division=0))
for media in ["macro", "weighted", "micro"]:
    print(f"F1 {media:>8s}: {f1_score(y_te, pred_mc, average=media, zero_division=0):.3f}")
print(f"acurácia   : {accuracy_score(y_te, pred_mc):.3f}  (= F1 micro)")

# %% [markdown]
# A média ponderada e a micro ficam altas porque "dúvida" domina; a macro
# expõe que "cancelamento" — a classe que mais importa para retenção — é
# mal classificada.

# %% [markdown]
# ## 6. Intervalo de confiança por bootstrap para o F1

# %%
X, y = X_mc[:, :3], (y_mc == 2).astype(int)            # binário: cancelamento vs. resto
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
p_lr = LogisticRegression(class_weight="balanced").fit(X_tr, y_tr).predict(X_te)
f1_boot = []
for b in range(2000):
    idx = rng.integers(0, len(y_te), len(y_te))
    f1_boot.append(f1_score(y_te[idx], p_lr[idx], zero_division=0))
lo, hi = np.percentile(f1_boot, [2.5, 97.5])
print(f"F1 = {f1_score(y_te, p_lr):.3f}, IC 95% por bootstrap = [{lo:.3f}; {hi:.3f}] "
      f"com {y_te.sum()} positivos no teste")

# %% [markdown]
# Com poucas dezenas de positivos no teste, o intervalo do F1 tem vários
# pontos percentuais de largura — diferenças de 0,01 entre modelos não
# significam nada aqui.

# %% [markdown]
# ## 7. McNemar: comparar dois modelos nos mesmos exemplos

# %%
p_rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=5,
                              random_state=0).fit(X_tr, y_tr).predict(X_te)
acerto_lr, acerto_rf = p_lr == y_te, p_rf == y_te
b = np.sum(acerto_lr & ~acerto_rf)     # logística acerta, floresta erra
c = np.sum(~acerto_lr & acerto_rf)     # o contrário
estat = (abs(b - c) - 1) ** 2 / (b + c)
print(f"acurácia logística={acerto_lr.mean():.3f} | floresta={acerto_rf.mean():.3f}")
print(f"discordâncias: b={b}, c={c} -> qui-quadrado={estat:.2f}, p-valor={chi2.sf(estat, 1):.4f}")
print(f"exemplo do teoria.pdf (b=40, c=20): qui-quadrado={(abs(40 - 20) - 1) ** 2 / 60:.2f}, "
      f"p-valor={chi2.sf((abs(40 - 20) - 1) ** 2 / 60, 1):.4f}")

# %% [markdown]
# ## O que levar deste notebook
#
# - Todas as métricas saem da matriz de confusão; cada uma ignora algo.
# - Acurácia balanceada, MCC e kappa percebem o classificador preguiçoso;
#   acurácia não.
# - Precisão muda com a prevalência — recall e especificidade, não.
# - F1 depende de qual classe é a positiva; MCC é simétrico.
# - Métricas têm incerteza, e modelos avaliados nos mesmos exemplos se
#   comparam com testes pareados.
#
# → Próximo: **Curvas ROC e Precisão-Recall**.
