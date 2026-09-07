# %% [markdown]
# # Exercícios — Decomposições: SVD, Autovalores e PCA
#
# **Tema:** Álgebra Linear e Otimização › Decomposições: SVD, Autovalores e PCA
#
# Resolva antes de olhar o gabarito. A estrutura é sempre a mesma: enunciado,
# célula de resposta, gabarito comentado.
#
# **Dificuldade:** 🟢 base · 🟡 aplicação · 🔴 síntese

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

rng = np.random.default_rng(99)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
AZUL, VERDE, VERMELHO, ROXO, AMBAR = "#1F5C8B", "#1E6B4F", "#9C2B2B", "#5B3E86", "#8A6100"
np.set_printoptions(precision=4, suppress=True)
print("ambiente pronto")

# %% [markdown]
# ---
# ## Exercício 1 🟢 — Verificando a definição na mão
#
# Para a matriz `A` definida abaixo, calcule os autovalores e autovetores com
# `np.linalg.eig`. Depois verifique manualmente, para cada autovetor, que
# $Av = \lambda v$ (a diferença deve ser ~0).

# %%
A = np.array([[4.0, 1.0],
              [2.0, 3.0]])

# --- sua resposta ---


# %% [markdown]
# ### Gabarito 1

# %%
autoval, autovec = np.linalg.eig(A)
print("autovalores:", autoval)
for i in range(2):
    v = autovec[:, i]
    diff = A @ v - autoval[i] * v
    print(f"autovetor {i + 1} = {v}   |A v - lambda v| = {np.linalg.norm(diff):.2e}")

# %% [markdown]
# **Ponto de atenção:** `A` aqui **não é simétrica** ($A \neq A^\top$), então o
# teorema espectral não se aplica — não há garantia de autovetores ortogonais.
# Verifique: `autovec[:,0] @ autovec[:,1]` não é zero em geral.

# %%
print(f"produto interno entre os dois autovetores: "
      f"{np.dot(autovec[:, 0], autovec[:, 1]):.4f}   (não-ortogonais, como esperado)")

# %% [markdown]
# ---
# ## Exercício 2 🟢 — `eig` vs `eigh` em uma matriz simétrica
#
# Construa uma matriz de covariância 3x3 a partir de dados gerados com
# `rng.multivariate_normal` (3 variáveis, com alguma correlação de sua escolha).
# Calcule os autovalores com `eig` e com `eigh` e compare: os valores batem? A
# ordem é a mesma? Os autovetores são ortogonais em cada caso?

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 2

# %%
cov_alvo = np.array([[3.0, 1.2, -0.5],
                      [1.2, 2.0, 0.3],
                      [-0.5, 0.3, 1.5]])
dados = rng.multivariate_normal(mean=[0, 0, 0], cov=cov_alvo, size=5000)
C = np.cov(dados, rowvar=False)

autoval_eig, autovec_eig = np.linalg.eig(C)
autoval_eigh, autovec_eigh = np.linalg.eigh(C)

print("autovalores (eig) :", np.sort(autoval_eig.real)[::-1])
print("autovalores (eigh):", np.sort(autoval_eigh)[::-1])
print(f"\northogonalidade eig  : {np.allclose(autovec_eig.T @ autovec_eig, np.eye(3))}")
print(f"ortogonalidade eigh : {np.allclose(autovec_eigh.T @ autovec_eigh, np.eye(3))}")

# %% [markdown]
# **Resposta:** os *valores* batem (a menos de ordenação e ruído numérico), mas
# `eigh` **garante** ortogonalidade explorando a simetria; `eig` genérico pode
# devolver autovetores não perfeitamente ortogonais por causa do algoritmo
# numérico usado, mesmo em matrizes simétricas. Use sempre `eigh` quando souber
# que a matriz é simétrica — é mais rápido e mais garantido.

# %% [markdown]
# ---
# ## Exercício 3 🟡 — Reconstrução parcial e erro de Eckart-Young
#
# Gere uma matriz aleatória $30 \times 20$. Calcule a SVD completa. Para
# $k = 1, 2, 5, 10, 15, 20$, calcule a aproximação de posto $k$ e o erro (norma
# de Frobenius) entre ela e a matriz original. Compare esse erro com a fórmula
# fechada $\sqrt{\sum_{i>k}\sigma_i^2}$ — eles devem coincidir.

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 3

# %%
M = rng.normal(size=(30, 20))
U, s, Vt = np.linalg.svd(M, full_matrices=False)

print(f"{'k':>4s} {'erro medido (Frobenius)':>26s} {'erro pela fórmula':>20s}")
print("-" * 54)
for k in [1, 2, 5, 10, 15, 20]:
    M_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    erro_medido = np.linalg.norm(M - M_k, "fro")
    erro_formula = np.sqrt(np.sum(s[k:] ** 2))
    print(f"{k:>4d} {erro_medido:>26.6f} {erro_formula:>20.6f}")

# %% [markdown]
# Os dois métodos coincidem até a precisão numérica — essa é a demonstração
# empírica do teorema de Eckart-Young: o erro da melhor aproximação de posto
# $k$ é exatamente a energia deixada nos valores singulares descartados.

# %% [markdown]
# ---
# ## Exercício 4 🟡 — PCA no dataset de vinhos: o efeito da padronização
#
# Carregue `load_wine()` (dataset local do sklearn, 13 features em escalas bem
# diferentes: teor alcoólico, magnésio em mg, prolina em centenas de mg...).
# Aplique PCA com e sem `StandardScaler` antes. Para cada caso, imprima a
# variância explicada pelos 2 primeiros componentes e a feature de maior
# variância bruta. Explique a diferença.

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 4

# %%
vinho = load_wine()
X_vinho = vinho.data
nomes_features = vinho.feature_names

variancias_brutas = X_vinho.var(axis=0, ddof=1)
feature_dominante = nomes_features[np.argmax(variancias_brutas)]
print(f"feature de maior variância bruta: '{feature_dominante}' "
      f"(variância = {variancias_brutas.max():.1f})")

pca_bruto = PCA(n_components=2).fit(X_vinho)
X_padronizado = StandardScaler().fit_transform(X_vinho)
pca_padronizado = PCA(n_components=2).fit(X_padronizado)

print(f"\nvariância explicada (SEM padronizar), 2 componentes: "
      f"{pca_bruto.explained_variance_ratio_}")
print(f"soma: {pca_bruto.explained_variance_ratio_.sum():.1%}")
print(f"\nvariância explicada (COM padronizar), 2 componentes: "
      f"{pca_padronizado.explained_variance_ratio_}")
print(f"soma: {pca_padronizado.explained_variance_ratio_.sum():.1%}")

peso_dominante_bruto = np.abs(pca_bruto.components_[0]).argmax()
print(f"\nno PCA SEM padronizar, a feature com maior peso no PC1 é: "
      f"'{nomes_features[peso_dominante_bruto]}'")

# %% [markdown]
# **Explicação:** sem padronizar, o PC1 do PCA bruto é dominado pela feature de
# maior variância *numérica* (tipicamente `proline`, que varia na casa das
# centenas/milhares, enquanto outras variam entre 0 e 5). A "variância
# explicada" alta não significa estrutura interessante — significa que uma
# única unidade de medida está dominando a conta. Depois de padronizar, o PC1
# passa a refletir uma combinação real de várias variáveis correlacionadas.

# %% [markdown]
# ---
# ## Exercício 5 🔴 — Quantos componentes preservam a separação entre classes?
#
# Usando `load_digits()`, aplique PCA padronizado e reduza para $k \in \{2, 5,
# 10, 20\}$ componentes. Para cada $k$, treine um `KNeighborsClassifier` simples
# (`n_neighbors=5`) com validação cruzada de 5 folds (`cross_val_score`) e
# reporte a acurácia média. Compare com a acurácia usando os 64 pixels
# originais (sem PCA).
#
# **Pergunta:** existe um $k$ muito menor que 64 que preserva quase toda a
# acurácia? O que isso sugere sobre a dimensão "real" da informação nos dados?

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 5

# %%
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

digitos = load_digits()
X_dig, y_dig = digitos.data, digitos.target
X_dig_padronizado = StandardScaler().fit_transform(X_dig)

resultados = {}
acc_completa = cross_val_score(
    KNeighborsClassifier(n_neighbors=5), X_dig_padronizado, y_dig, cv=5).mean()
resultados["64 (sem PCA)"] = acc_completa

for k in [2, 5, 10, 20]:
    X_reduzido = PCA(n_components=k).fit_transform(X_dig_padronizado)
    acc = cross_val_score(
        KNeighborsClassifier(n_neighbors=5), X_reduzido, y_dig, cv=5).mean()
    resultados[f"{k}"] = acc

print(f"{'nº de componentes':>20s} {'acurácia média (5-fold)':>26s}")
print("-" * 48)
for nome, acc in resultados.items():
    print(f"{nome:>20s} {acc:>26.4f}")

# %% [markdown]
# **Leitura esperada:** com $k=10$ ou $k=20$ componentes (de 64 originais), a
# acurácia costuma ficar muito próxima da versão completa — às vezes até
# ligeiramente melhor, porque o PCA descarta parte do ruído junto com a
# variância residual. Isso sugere que a informação "real" que separa os
# dígitos vive num subespaço de dimensão bem menor que 64: os pixels têm
# redundância espacial enorme (pixels vizinhos são correlacionados), e a SVD
# captura exatamente esse tipo de estrutura.

# %% [markdown]
# ---
# ## Fechamento
#
# - Verificar a definição de autovalor/autovetor na mão tira o mistério da
#   fórmula — é sempre $Av = \lambda v$, nada mais.
# - `eigh` não é só mais rápido: é a escolha **correta** para matrizes
#   simétricas, com garantias que `eig` não oferece.
# - Eckart-Young não é uma heurística — é uma igualdade exata, verificável.
# - Padronização antes de PCA muda o resultado qualitativamente, não só a
#   escala dos números.
# - PCA como pré-processamento pode preservar (e às vezes até melhorar) o
#   desempenho preditivo com uma fração das dimensões originais.
#
# → Próximo módulo: **Cálculo e Gradiente Descendente**, onde a mesma
# geometria de autovalores explica por que alguns problemas de otimização são
# fáceis e outros são teimosos.
