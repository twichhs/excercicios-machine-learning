# %% [markdown]
# # Exercícios — Detecção de Anomalias
#
# **Tema:** Aprendizado Não Supervisionado › Detecção de Anomalias
#
# Resolva antes de olhar o gabarito. **Dificuldade:** 🟢 base · 🟡 aplicação ·
# 🔴 síntese

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(535)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
AZUL, VERDE, VERMELHO, ROXO, AMBAR = "#1F5C8B", "#1E6B4F", "#9C2B2B", "#5B3E86", "#8A6100"
print("ambiente pronto")

# %% [markdown]
# ---
# ## Dataset de trabalho: métricas de servidores de um datacenter
#
# 4.000 janelas de 5 minutos de dois tipos de servidor: **web** (carga
# estável, latência baixa, muitas requisições) e **batch** (carga alta e
# variável, latência alta, poucas requisições). Existem 60 incidentes
# rotulados (`incidente = 1`), de três tipos — a coluna `tipo_incidente`
# serve só para avaliar, e nenhum método deve usá-la:
#
# - **vazamento de memória** — memória perto de 100% e latência altíssima;
# - **web degradado** — um servidor web com métricas que seriam normais
#   para um batch;
# - **correlação quebrada** — muitas requisições com CPU baixa demais para
#   elas (tráfego que não está sendo processado).

# %%
n_web, n_batch = 2400, 1600
web = pd.DataFrame({"cpu_pct": rng.normal(35, 5, n_web),
                    "requisicoes_min": rng.normal(1200, 150, n_web),
                    "latencia_ms": rng.lognormal(np.log(80), 0.12, n_web),
                    "memoria_pct": rng.normal(55, 6, n_web),
                    "taxa_erro_pct": rng.gamma(2, 0.1, n_web)})
web["cpu_pct"] += (web["requisicoes_min"] - 1200) / 150 * 4.5   # CPU acompanha as requisições
batch = pd.DataFrame({"cpu_pct": rng.normal(70, 12, n_batch),
                      "requisicoes_min": rng.normal(100, 35, n_batch).clip(5),
                      "latencia_ms": rng.lognormal(np.log(300), 0.3, n_batch),
                      "memoria_pct": rng.normal(65, 10, n_batch),
                      "taxa_erro_pct": rng.gamma(2, 0.5, n_batch)})
web["tipo_servidor"], batch["tipo_servidor"] = "web", "batch"

vazamento = batch.sample(20, random_state=1).assign(
    memoria_pct=lambda d: rng.uniform(96, 100, len(d)),
    latencia_ms=lambda d: rng.uniform(1500, 4000, len(d)), tipo_servidor="batch")
degradado = web.sample(25, random_state=2).assign(
    cpu_pct=lambda d: rng.normal(62, 5, len(d)),
    latencia_ms=lambda d: rng.normal(240, 30, len(d)),
    requisicoes_min=lambda d: rng.normal(300, 60, len(d)), tipo_servidor="web")
quebrada = web.sample(15, random_state=3).assign(
    requisicoes_min=lambda d: rng.normal(1500, 60, len(d)),
    cpu_pct=lambda d: rng.normal(22, 3, len(d)), tipo_servidor="web")

servidores = pd.concat([web.assign(tipo_incidente="normal"), batch.assign(tipo_incidente="normal"),
                        vazamento.assign(tipo_incidente="vazamento de memória"),
                        degradado.assign(tipo_incidente="web degradado"),
                        quebrada.assign(tipo_incidente="correlação quebrada")], ignore_index=True)
servidores = servidores.sample(frac=1, random_state=0).reset_index(drop=True)
servidores["incidente"] = (servidores["tipo_incidente"] != "normal").astype(int)
metricas = ["cpu_pct", "requisicoes_min", "latencia_ms", "memoria_pct", "taxa_erro_pct"]
print(servidores.groupby("tipo_servidor")[metricas].median().round(1).to_string())
print(f"\n{len(servidores)} janelas | {servidores['incidente'].sum()} incidentes")

# %% [markdown]
# ---
# ## Exercício 1 🟢 — Mascaramento na mão
#
# Dez medições de latência de um servidor web, em ms:
# `[80, 82, 79, 85, 81, 250, 260, 78, 83, 80]`. Calcule o z-score clássico
# e o z robusto (mediana e 1,4826 × MAD) da medição de 250 ms. Qual deles
# dispara o alerta com o limiar $|z| > 3$? Por quê?

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 1

# %%
lat = np.array([80, 82, 79, 85, 81, 250, 260, 78, 83, 80], dtype=float)
media, dp = lat.mean(), lat.std(ddof=1)
mediana = np.median(lat)
mad = np.median(np.abs(lat - mediana))
print(f"média={media:.1f}, dp={dp:.2f}  -> z(250) = {(250 - media) / dp:.2f}")
print(f"mediana={mediana:.1f}, MAD={mad:.1f}, 1,4826*MAD={1.4826 * mad:.3f}  "
      f"-> z_rob(250) = {(250 - mediana) / (1.4826 * mad):.1f}")

# %% [markdown]
# **Por quê:** as duas medições lentas puxam a média para 115,8 ms e inflam
# o desvio-padrão para cerca de 73 ms — maior que a diferença entre as
# medições normais e as lentas. O z clássico de 250 ms fica abaixo de 2: sem
# alerta. A mediana (81,5 ms) e o MAD (2 ms) praticamente ignoram as duas
# lentas, e o z robusto passa de 50. Com 20% de contaminação numa amostra
# pequena, o mascaramento é total.

# %% [markdown]
# ---
# ## Exercício 2 🟢 — Mahalanobis na mão
#
# Nos servidores web, CPU e requisições (padronizadas) têm correlação 0,9.
# Calcule $D^2$ para os pontos $P = (2, 2)$ e $Q = (1, -1)$ e compare com o
# limiar $\chi^2_{2;\,0{,}975}$. Qual é mais distante do centro pela distância
# euclidiana? E qual é anômalo?

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 2

# %%
S = np.array([[1.0, 0.9], [0.9, 1.0]])
S_inv = np.linalg.inv(S)
lim = chi2.ppf(0.975, 2)
for nome, p in [("P", np.array([2.0, 2.0])), ("Q", np.array([1.0, -1.0]))]:
    print(f"{nome}={p}: euclidiana={np.linalg.norm(p):.2f} | D²={p @ S_inv @ p:.2f} | "
          f"anômalo? {p @ S_inv @ p > lim}")
print(f"limiar: {lim:.2f}   |   inversa = {1 / (1 - 0.81):.3f} x [[1, -0.9], [-0.9, 1]]")

# %% [markdown]
# **Por quê:** $\Sigma^{-1} = \frac{1}{1-0{,}81}$ vezes a matriz com 1 na
# diagonal e $-0{,}9$ fora, isto é, $5{,}263$ vezes ela. Para $P$:
# $5{,}263 \times (4 - 7{,}2 + 4) \approx 4{,}2$; para $Q$:
# $5{,}263 \times (1 + 1{,}8 + 1) = 20{,}0$. $P$ está duas vezes mais longe
# do centro em linha reta, mas segue a correlação (muitas requisições, muita
# CPU); $Q$ viola a correlação (CPU acima da média com requisições abaixo) e
# passa com folga do limiar de 7,38. É o padrão do incidente "correlação
# quebrada" deste dataset.

# %% [markdown]
# ---
# ## Exercício 3 🟡 — Isolation Forest e o orçamento
#
# (a) Calcule $c(128)$ e o escore $s$ de um ponto isolado, em média, em 3
# cortes, com `max_samples=128`.
# (b) Ajuste um `IsolationForest(max_samples=128)` nas cinco métricas e
# calcule precisão e recall nos 50 maiores escores (o time de SRE investiga
# 50 alertas). Qual tipo de incidente ele pega melhor, e qual pior?
# (c) Mostre que mudar `contamination` não muda o ranking.

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 3

# %%
def c(n):
    return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n

print(f"(a) c(128) = {c(128):.3f}  -> s(E[h]=3) = {2 ** (-3 / c(128)):.3f}")

X = servidores[metricas].to_numpy()
y = servidores["incidente"].to_numpy()
tipo = servidores["tipo_incidente"].to_numpy()

def no_orcamento(escore, k=50):
    top = np.argsort(escore)[::-1][:k]
    linha = {"precisão@k": y[top].mean(), "recall@k": y[top].sum() / y.sum(),
             "PR-AUC": average_precision_score(y, escore)}
    for t in ["vazamento de memória", "web degradado", "correlação quebrada"]:
        linha[t] = np.isin(np.where(tipo == t)[0], top).mean()
    return linha

esc_if = -IsolationForest(n_estimators=300, max_samples=128, random_state=0).fit(X).score_samples(X)
print("\n(b)", pd.Series(no_orcamento(esc_if)).round(3).to_dict())

rankings = [np.argsort(-IsolationForest(n_estimators=300, max_samples=128, contamination=cc,
                                        random_state=0).fit(X).score_samples(X))[:50]
            for cc in ["auto", 0.01, 0.05]]
print("(c) mesmo top-50 com contamination auto/0.01/0.05?",
      all(np.array_equal(rankings[0], r) for r in rankings[1:]))

# %% [markdown]
# **Por quê:** (a) $H(127) \approx \ln 127 + 0{,}5772 \approx 5{,}42$, então
# $c(128) \approx 2 \times 5{,}42 - 2 \times 127/128 \approx 8{,}86$ e
# $s = 2^{-3/8{,}86} \approx 0{,}79$ — bem acima de 0,5, anômalo. (b) O
# vazamento de memória é extremo em duas métricas e é isolado em poucos
# cortes: recall de 100%. Os outros dois tipos passam **em branco**: o web
# degradado tem métricas normais para um servidor batch, e a correlação
# quebrada tem cada métrica dentro da faixa normal — só a combinação é
# estranha, e cortes em uma feature por vez não a enxergam. Com uma noção
# global de anomalia, o Isolation Forest gasta o resto do orçamento com os
# batch mais extremos, que são legítimos. (c) `contamination` só define o
# limiar do `predict`; o escore e o ranking são idênticos.

# %% [markdown]
# ---
# ## Exercício 4 🟡 — Anomalia local: LOF e condicionar pelo contexto
#
# O incidente "web degradado" é uma anomalia **contextual**: as métricas são
# comuns para um batch, e anômalas para um web. Compare três abordagens no
# mesmo orçamento de 50 alertas:
#
# (a) LOF (`n_neighbors=30`) nas métricas padronizadas;
# (b) um Isolation Forest **por tipo de servidor** (o contexto é
# conhecido: use-o), com os escores dos dois modelos convertidos em
# percentil dentro de cada tipo antes de juntar;
# (c) o Isolation Forest global do exercício 3.

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 4

# %%
Xs = StandardScaler().fit_transform(X)
esc_lof = -LocalOutlierFactor(n_neighbors=30).fit(Xs).negative_outlier_factor_

esc_por_tipo = np.zeros(len(servidores))
for t in ["web", "batch"]:
    m = (servidores["tipo_servidor"] == t).to_numpy()
    s = -IsolationForest(n_estimators=300, max_samples=128, random_state=0).fit(X[m]).score_samples(X[m])
    esc_por_tipo[m] = pd.Series(s).rank(pct=True).to_numpy()   # percentil dentro do tipo

comparacao = pd.DataFrame({"LOF": no_orcamento(esc_lof),
                           "IF por tipo de servidor": no_orcamento(esc_por_tipo),
                           "IF global": no_orcamento(esc_if)}).T
print(comparacao.round(3).to_string())

# %% [markdown]
# **Por quê:** (a) o LOF não pega **nenhum** web degradado: no espaço das
# métricas, eles caem na borda da nuvem dos servidores batch, com densidade
# parecida com a de seus vizinhos batch — localmente, parecem normais. Em
# compensação, pega metade das correlações quebradas, que ficam numa região
# rarefeita ao lado dos webs. "Local" não é o mesmo que "contextual": o LOF
# usa a vizinhança no espaço das features, não o contexto de negócio.
# (b) Quando o contexto é **conhecido** (o tipo de servidor), modelar cada
# contexto separadamente transforma a anomalia contextual em pontual — um
# web com latência de 240 ms é extremo **entre os webs** — e o recall de web
# degradado vai de 0 a 100%. Converter para percentil dentro de cada tipo
# evita que um dos modelos domine a fila só porque seus escores têm outra
# escala. É o mesmo raciocínio da temperatura mensal do notebook
# `01-metodos-estatisticos`. Mas o IF por tipo continua cego para a
# correlação quebrada, pelo motivo do exercício 3.

# %% [markdown]
# ---
# ## Exercício 5 🔴 — Desenhe o sistema de alertas
#
# O time de SRE investiga 50 alertas. Monte o detector que você colocaria em
# produção, combinando o que achar melhor entre: z robusto por métrica (por
# tipo de servidor), Mahalanobis robusto por tipo de servidor, Isolation
# Forest por tipo e LOF. Reporte precisão@50, recall@50 e o recall de cada
# tipo de incidente, e justifique a escolha — inclusive o que você **não**
# usaria em produção e por quê.

# %%
# --- sua resposta ---


# %% [markdown]
# ### Gabarito 5

# %%
from sklearn.covariance import MinCovDet

def z_rob_max(Xm):
    """Maior |z robusto| entre as métricas (log nas de cauda longa já aplicado)."""
    med = np.median(Xm, axis=0)
    mad = np.median(np.abs(Xm - med), axis=0)
    return (np.abs(Xm - med) / (1.4826 * mad)).max(axis=1)

Xlog = X.copy()
Xlog[:, [2, 4]] = np.log1p(Xlog[:, [2, 4]])          # latência e taxa de erro têm cauda longa
componentes = {"z robusto": np.zeros(len(X)), "Mahalanobis (MCD)": np.zeros(len(X)),
               "IF": np.zeros(len(X))}
for t in ["web", "batch"]:
    m = (servidores["tipo_servidor"] == t).to_numpy()
    pct = lambda s: pd.Series(s).rank(pct=True).to_numpy()
    componentes["z robusto"][m] = pct(z_rob_max(Xlog[m]))
    componentes["Mahalanobis (MCD)"][m] = pct(MinCovDet(random_state=0).fit(Xlog[m]).mahalanobis(Xlog[m]))
    componentes["IF"][m] = pct(-IsolationForest(n_estimators=300, max_samples=128, random_state=0)
                               .fit(X[m]).score_samples(X[m]))

candidatos = dict(componentes)
candidatos["máximo dos três"] = np.max(np.column_stack(list(componentes.values())), axis=1) \
    + 1e-6 * componentes["Mahalanobis (MCD)"]     # desempate
print(pd.DataFrame({k: no_orcamento(v) for k, v in candidatos.items()}).T.round(3).to_string())

# %% [markdown]
# **Como avaliar sua resposta:** o ponto central é condicionar pelo tipo de
# servidor — todas as abordagens por tipo superam as globais dos exercícios
# anteriores. E o vencedor é o método mais simples bem especificado: o
# **Mahalanobis robusto por tipo**, sozinho, acerta 49 dos 50 alertas
# (com 60 incidentes e 50 alertas, o recall máximo possível é 0,83 — ele
# chega a 0,82). É o único que enxerga a "correlação quebrada", porque
# modela a covariância entre CPU e requisições; os outros olham, direta ou
# indiretamente, uma métrica de cada vez.
#
# Combinar pelo **máximo** dos percentis, aqui, **piora**: o z robusto e o
# IF não trazem nenhum incidente que o Mahalanobis já não pegue, e só
# adicionam à fila os seus próprios falsos alarmes. Combinar detectores vale
# quando cada um enxerga algo que os outros não enxergam — como o
# supervisionado e o Isolation Forest no notebook de fraude, com as duas
# filas. Quem decide é a medição no orçamento, não a intuição de que "mais
# modelos é melhor".
#
# O que não usar em produção, e por quê: o LOF exige buscar vizinhos numa
# base que cresce a cada 5 minutos (custo $O(n^2)$ no pior caso) e, com o
# contexto conhecido, não traz nada que o Mahalanobis por tipo não traga.
# Qualquer que seja a escolha, o limiar deve vir do orçamento (50 alertas),
# não de `contamination`. E um bom sistema real mostraria ao time **por
# que** cada alerta disparou — por exemplo, a contribuição de cada métrica
# para o $D^2$, o que o Mahalanobis permite e o Isolation Forest não.

# %% [markdown]
# ---
# ## Fechamento
#
# - Média e desvio-padrão se contaminam; mediana e MAD não — até um ponto.
# - Mahalanobis enxerga combinações; o limiar vem do qui-quadrado.
# - Isolation Forest é rápido e global; `contamination` só move o limiar.
# - Anomalia contextual com contexto conhecido: um modelo por contexto.
# - O método mais simples bem especificado (aqui, Mahalanobis robusto por
#   contexto) muitas vezes vence; combine detectores só quando cada um
#   enxerga algo diferente, e meça tudo no orçamento de investigação.
#
# → Este é o fim do tema **Aprendizado Não Supervisionado**. O próximo tema,
# **Avaliação e Validação de Modelos**, volta aos modelos supervisionados
# para responder a pergunta que atravessa todo o curso: o número que você
# reportou é real?
