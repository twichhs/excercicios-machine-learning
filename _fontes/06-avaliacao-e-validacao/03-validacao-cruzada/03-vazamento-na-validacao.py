# %% [markdown]
# # Vazamento na validação
#
# **Tema:** Avaliação e Validação de Modelos › Validação Cruzada
#
# Um catálogo de vazamentos, cada um reproduzido duas vezes: com a
# validação feita do jeito errado e do jeito certo. No fim, uma tabela com
# o tamanho da inflação de cada um. Os dados são construídos para que a
# resposta honesta seja conhecida — em vários casos, o rótulo é ruído puro
# e a acurácia honesta é 0,5.

# %%
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(633)
CV = StratifiedKFold(5, shuffle=True, random_state=0)
tabela = []
print("pronto")

def registra(nome, errado, certo, verdade):
    tabela.append({"vazamento": nome, "CV errada": errado, "CV certa": certo, "verdade": verdade})
    print(f"{nome}: CV errada = {errado:.3f} | CV certa = {certo:.3f} | verdade ≈ {verdade}")

# %% [markdown]
# ## 1. Escalonamento e imputação fora da CV
#
# O vazamento clássico do tema 3. Com dados bem comportados, é pequeno —
# vale conhecê-lo para não gastar energia com ele e, principalmente, para
# não confundir com os grandes.

# %%
n = 1000
X = rng.normal(0, 1, (n, 10)) * rng.uniform(1, 100, 10)
y = (rng.random(n) < 1 / (1 + np.exp(-(X[:, 0] / 30 - X[:, 1] / 50)))).astype(int)
X[rng.random(X.shape) < 0.15] = np.nan
X_global = StandardScaler().fit_transform(SimpleImputer().fit_transform(X))      # ERRADO
errado = cross_val_score(LogisticRegression(max_iter=2000), X_global, y, cv=CV, scoring="roc_auc").mean()
certo = cross_val_score(make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=2000)),
                        X, y, cv=CV, scoring="roc_auc").mean()
registra("escala e imputação globais (AUC)", errado, certo, "igual à CV certa")

# %% [markdown]
# ## 2. Seleção de atributos fora da CV (Ambroise & McLachlan)

# %%
Xr = rng.normal(0, 1, (60, 3000))
yr = np.r_[np.zeros(30), np.ones(30)].astype(int)
sel = SelectKBest(f_classif, k=30).fit(Xr, yr)                                  # ERRADO
errado = cross_val_score(LogisticRegression(max_iter=2000), sel.transform(Xr), yr, cv=CV).mean()
certo = cross_val_score(make_pipeline(SelectKBest(f_classif, k=30), LogisticRegression(max_iter=2000)),
                        Xr, yr, cv=CV).mean()
registra("seleção de atributos global (acurácia)", errado, certo, 0.5)

# %% [markdown]
# ## 3. Target encoding fora da CV
#
# Uma categórica de alta cardinalidade (400 lojas, ~5 vendas por loja) e um
# rótulo que **não depende** da loja. Codificar cada loja pela média do
# alvo, calculada com a base inteira, coloca o rótulo de cada linha dentro
# da sua própria feature.

# %%
n = 2000
loja = rng.integers(0, 400, n)
ruido_num = rng.normal(0, 1, (n, 3))
yt = rng.integers(0, 2, n)
media_loja = pd.Series(yt).groupby(loja).mean()
X_te_global = np.column_stack([media_loja.loc[loja].to_numpy(), ruido_num])     # ERRADO
errado = cross_val_score(LogisticRegression(), X_te_global, yt, cv=CV, scoring="roc_auc").mean()

class TargetEncoderSimples(BaseEstimator, TransformerMixin):
    """Codifica a coluna 0 (categoria) pela média do alvo no TREINO, com suavização."""
    def __init__(self, m=10):
        self.m = m
    def fit(self, X, y):
        cat, y = X[:, 0].astype(int), np.asarray(y)
        self.global_ = y.mean()
        est = pd.DataFrame({"c": cat, "y": y}).groupby("c")["y"].agg(["sum", "count"])
        self.mapa_ = ((est["sum"] + self.m * self.global_) / (est["count"] + self.m)).to_dict()
        return self
    def transform(self, X):
        cod = np.array([self.mapa_.get(c, self.global_) for c in X[:, 0].astype(int)])
        return np.column_stack([cod, X[:, 1:]])

X_cru = np.column_stack([loja, ruido_num])
certo = cross_val_score(make_pipeline(TargetEncoderSimples(), LogisticRegression()), X_cru, yt,
                        cv=CV, scoring="roc_auc").mean()
registra("target encoding global (AUC)", errado, certo, 0.5)

# %% [markdown]
# O `sklearn.preprocessing.TargetEncoder` faz ainda melhor: no `fit`, ele
# usa uma CV **interna** para codificar os próprios dados de treino, o que
# evita até o vazamento dentro do treino.

# %% [markdown]
# ## 4. Oversampling antes da CV
#
# Dados desbalanceados (10% positivos) com sinal moderado. Duplicar os
# positivos **antes** de dividir coloca cópias do mesmo exemplo em treino e
# validação — e um modelo que decora (floresta com folhas pequenas) passa a
# "reconhecer" as cópias.

# %%
n = 2000
Xo = rng.normal(0, 1, (n, 8))
yo = (rng.random(n) < 1 / (1 + np.exp(-(-2.6 + 0.8 * Xo[:, 0])))).astype(int)
pos = np.where(yo == 1)[0]
extra = rng.choice(pos, 6 * len(pos))
Xo_dup, yo_dup = np.vstack([Xo, Xo[extra]]), np.r_[yo, yo[extra]]              # ERRADO
floresta = RandomForestClassifier(n_estimators=300, min_samples_leaf=1, random_state=0)
errado = cross_val_score(floresta, Xo_dup, yo_dup, cv=CV, scoring="roc_auc").mean()
aucs = []
for tr, te in CV.split(Xo, yo):                                                 # CERTO: duplica só no treino
    pos_tr = tr[yo[tr] == 1]
    tr_dup = np.r_[tr, rng.choice(pos_tr, 6 * len(pos_tr))]
    m = floresta.fit(Xo[tr_dup], yo[tr_dup])
    aucs.append(roc_auc_score(yo[te], m.predict_proba(Xo[te])[:, 1]))
registra("oversampling antes da CV (AUC)", errado, np.mean(aucs), "igual à CV certa")

# %% [markdown]
# ## 5. Duplicatas na base
#
# Muitas bases reais têm registros repetidos (cadastros duplicados, eventos
# gravados duas vezes). Com o rótulo sendo ruído puro, um k-NN com $k=1$
# acerta os duplicados que caem em folds diferentes.

# %%
base = rng.normal(0, 1, (800, 5))
y_base = rng.integers(0, 2, 800)
rep = rng.choice(800, 800)                                    # cada registro reaparece ~1 vez
Xd, yd = np.vstack([base, base[rep]]), np.r_[y_base, y_base[rep]]
grupo_d = np.r_[np.arange(800), rep]                          # identidade do registro original
knn = KNeighborsClassifier(1)
errado = cross_val_score(knn, Xd, yd, cv=CV).mean()
certo = cross_val_score(knn, Xd, yd, cv=GroupKFold(5), groups=grupo_d).mean()
registra("duplicatas entre folds (acurácia)", errado, certo, 0.5)

# %% [markdown]
# ## 6. Feature do futuro
#
# Um modelo de inadimplência com a feature `contatos_cobranca_30d` — o
# número de ligações da cobrança nos 30 dias **seguintes** à concessão. Ela
# é quase o próprio rótulo: a cobrança só liga para quem atrasou. Nenhum
# esquema de CV detecta isso, porque o vazamento está na **definição** da
# feature, não na divisão.

# %%
n = 5000
Xc = rng.normal(0, 1, (n, 4))
yc = (rng.random(n) < 1 / (1 + np.exp(-(-1.8 + 0.7 * Xc[:, 0] - 0.5 * Xc[:, 1])))).astype(int)
contatos = np.where(yc == 1, rng.poisson(3, n), rng.poisson(0.1, n))           # só existe DEPOIS
X_com = np.column_stack([Xc, contatos])
cv_com = cross_val_score(LogisticRegression(), X_com, yc, cv=CV, scoring="roc_auc").mean()
cv_sem = cross_val_score(LogisticRegression(), Xc, yc, cv=CV, scoring="roc_auc").mean()
# em produção, no momento da concessão, a feature vale 0 para todo mundo
modelo_com = LogisticRegression().fit(X_com, yc)
Xn = rng.normal(0, 1, (n, 4))
yn = (rng.random(n) < 1 / (1 + np.exp(-(-1.8 + 0.7 * Xn[:, 0] - 0.5 * Xn[:, 1])))).astype(int)
prod = roc_auc_score(yn, modelo_com.predict_proba(np.column_stack([Xn, np.zeros(n)]))[:, 1])
registra("feature do futuro (AUC)", cv_com, cv_sem, f"{prod:.3f} em produção com a feature zerada")

# %% [markdown]
# Com a feature, a CV mostra AUC quase perfeita; em produção, o valor da
# feature ainda não existe (vale 0), e o modelo — que aprendeu a confiar
# nela — perde desempenho em relação ao modelo honesto. A única defesa é a
# auditoria: para cada feature, "no instante da previsão, eu teria esse
# valor?".

# %% [markdown]
# ## Tabela-resumo

# %%
resumo = pd.DataFrame(tabela).set_index("vazamento")
resumo["inflação"] = pd.to_numeric(resumo["CV errada"]) - pd.to_numeric(resumo["CV certa"])
print(resumo.round(3).to_string())

# %% [markdown]
# **Leitura esperada:** escalonamento e imputação globais quase não mudam
# nada nestes dados. Seleção de atributos, target encoding, oversampling
# e duplicatas transformam ruído (ou sinal fraco) em desempenho alto. A
# feature do futuro engana a CV qualquer que seja o esquema. Vazamentos não
# são todos iguais: os perigosos são os que usam o **rótulo** (seleção,
# encoding) ou a **identidade** do exemplo (duplicatas, reamostragem,
# grupos) — e os que trazem informação do **futuro**.

# %% [markdown]
# ## O que levar deste notebook
#
# - Tudo que olha o rótulo (seleção, encoding) ou multiplica exemplos
#   (reamostragem) vai **dentro** do `Pipeline`.
# - Duplicatas e grupos se tratam com deduplicação e `GroupKFold`.
# - Feature do futuro não se resolve com CV — se resolve auditando a
#   definição de cada feature.
#
# → Próximo: **Exercícios** do módulo.
