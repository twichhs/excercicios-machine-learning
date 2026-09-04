# %% [markdown]
# # Medidas-resumo na prática
#
# **Tema:** Estatística › Fundamentos e Estatística Descritiva
#
# Este notebook acompanha o capítulo *Medidas de posição* e *Medidas de
# dispersão* do `teoria.pdf`. A ideia não é decorar fórmulas, e sim **ver com os
# próprios olhos** o que cada medida faz e o que ela ignora.
#
# > **Analogia que vale para o notebook inteiro:** imagine que você precisa
# > descrever uma turma de 500 alunos para alguém que nunca a viu, usando no
# > máximo dois números. Qualquer par de números que você escolher vai apagar
# > alguma coisa. O trabalho do estatístico é escolher *o que apagar de
# > propósito*, em vez de descobrir depois o que foi apagado por acidente.
#
# ## O que vamos fazer
#
# 1. Construir um conjunto de dados realista de e-commerce (com a assimetria que
#    dados de receita sempre têm).
# 2. Comparar média, mediana e média aparada — e ver a média "quebrar".
# 3. Entender na prática por que a variância amostral divide por `n-1`.
# 4. Usar média geométrica e harmônica nos casos em que a aritmética dá a
#    resposta errada.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Semente fixa: todo resultado deste notebook é reprodutível.
# Em trabalho real, fixar a semente é o que separa "meu modelo deu 0.87" de
# "meu modelo dá 0.87 sempre que qualquer pessoa rodar".
rng = np.random.default_rng(42)

plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
pd.set_option("display.float_format", lambda v: f"{v:,.2f}")

print("Ambiente pronto.")
print("numpy", np.__version__, "| pandas", pd.__version__)

# %% [markdown]
# ## 1. Um conjunto de dados que se parece com a realidade
#
# Dados de receita, tempo de resposta, valor de sinistro e renda **nunca** são
# simétricos. Eles têm cauda longa à direita: a maioria dos valores é pequena e
# uns poucos são enormes.
#
# A distribuição **log-normal** é o modelo natural para isso. A intuição:
# se um valor é o resultado de vários fatores que se **multiplicam** (tráfego ×
# taxa de conversão × ticket), o logaritmo vira uma **soma** de fatores — e somas
# de muitos fatores tendem à normal (Teorema Central do Limite). Logo o valor
# original é a exponencial de algo aproximadamente normal: log-normal.
#
# Vamos simular 5.000 pedidos de um e-commerce.

# %%
n_pedidos = 5_000

# sigma=1.1 produz a cauda longa típica de ticket de e-commerce.
# exp(mu) ≈ 120 é a MEDIANA da distribuição (não a média — veja adiante).
ticket = rng.lognormal(mean=np.log(120), sigma=1.1, size=n_pedidos)

# Alguns pedidos corporativos: raros, mas ordens de magnitude maiores.
# É exatamente esse tipo de cliente que aparece em todo negócio real.
n_corporativos = 25
indices_corp = rng.choice(n_pedidos, size=n_corporativos, replace=False)
ticket[indices_corp] *= rng.uniform(40, 90, size=n_corporativos)

pedidos = pd.DataFrame({
    "ticket": ticket,
    "corporativo": np.isin(np.arange(n_pedidos), indices_corp),
})

print(f"{n_pedidos:,} pedidos, dos quais {n_corporativos} são corporativos "
      f"({n_corporativos / n_pedidos:.2%} da base).")
pedidos.head()

# %% [markdown]
# ## 2. A média, a mediana e a distância entre elas
#
# Vamos calcular as três medidas de posição centrais e comparar.

# %%
x = pedidos["ticket"].to_numpy()

media = x.mean()
mediana = np.median(x)
# Média aparada 10%: descarta os 10% menores E os 10% maiores antes de promediar.
# É o mesmo princípio das notas de ginástica olímpica: fora o juiz mais generoso
# e o mais rigoroso, promedia o resto.
media_aparada = stats.trim_mean(x, proportiontocut=0.10)
moda_aprox = stats.mode(np.round(x, -1), keepdims=False).mode  # arredondada à dezena

resumo = pd.Series({
    "Média": media,
    "Mediana": mediana,
    "Média aparada 10%": media_aparada,
    "Moda (aprox.)": moda_aprox,
})
print(resumo.to_string())
print()
print(f"A média é {media / mediana:.2f}x a mediana.")
print("Ordenação observada: moda < mediana < média  ->  assimetria POSITIVA.")

# %% [markdown]
# Repare no que acabou de acontecer. **25 pedidos em 5.000** — meio por cento da
# base — deslocaram a média para muito acima da mediana.
#
# Se você reportar "nosso ticket médio é R$ X" para a área de marketing, e ela
# desenhar uma campanha para o "cliente médio", a campanha vai mirar um cliente
# que **quase não existe**: a maioria esmagadora compra bem abaixo desse valor.
#
# Vamos ver isso graficamente.

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))

# Painel da esquerda: escala original. A cauda é tão longa que esmaga o corpo.
ax1.hist(x, bins=120, color="#1F5C8B", edgecolor="white", linewidth=0.3)
ax1.axvline(media, color="#9C2B2B", lw=2, label=f"Média = {media:,.0f}")
ax1.axvline(mediana, color="#1E6B4F", lw=2, label=f"Mediana = {mediana:,.0f}")
ax1.set_title("Escala original: a cauda domina o eixo")
ax1.set_xlabel("Ticket (R$)")
ax1.legend()

# Painel da direita: escala log. Agora o CORPO da distribuição fica visível.
# Truque prático: sempre que um histograma parecer "uma barra e nada mais",
# tente o eixo logarítmico antes de concluir que os dados estão errados.
ax2.hist(np.log10(x), bins=80, color="#5B3E86", edgecolor="white", linewidth=0.3)
ax2.axvline(np.log10(media), color="#9C2B2B", lw=2, label="Média")
ax2.axvline(np.log10(mediana), color="#1E6B4F", lw=2, label="Mediana")
ax2.set_title("Escala log$_{10}$: o corpo reaparece, quase simétrico")
ax2.set_xlabel("log$_{10}$(Ticket)")
ax2.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# O painel da direita é a assinatura visual de uma log-normal: **simétrica na
# escala logarítmica**. Essa é uma checagem que vale incorporar ao seu reflexo de
# EDA.
#
# ## 3. Quanto uma única observação pode mover cada medida?
#
# Este é o experimento que fixa o conceito de **ponto de ruptura**. Vamos pegar
# a amostra, trocar **um único** pedido por um valor cada vez maior, e observar
# o que acontece com cada estatística.

# %%
valores_intrusos = np.logspace(3, 9, 40)  # de mil a um bilhão de reais
trajetorias = {"Média": [], "Mediana": [], "Média aparada 10%": []}

for intruso in valores_intrusos:
    contaminada = x.copy()
    contaminada[0] = intruso           # UM ponto alterado, só isso
    trajetorias["Média"].append(contaminada.mean())
    trajetorias["Mediana"].append(np.median(contaminada))
    trajetorias["Média aparada 10%"].append(
        stats.trim_mean(contaminada, 0.10))

fig, ax = plt.subplots(figsize=(10, 4.2))
cores = {"Média": "#9C2B2B", "Mediana": "#1E6B4F", "Média aparada 10%": "#1F5C8B"}
for nome, serie in trajetorias.items():
    ax.plot(valores_intrusos, serie, lw=2, label=nome, color=cores[nome])
ax.set_xscale("log")
ax.set_xlabel("Valor do único pedido contaminado (R$, escala log)")
ax.set_ylabel("Estatística resultante (R$)")
ax.set_title("Ponto de ruptura: o efeito de UMA observação em 5.000")
ax.legend()
plt.tight_layout()
plt.show()

print("Com o intruso valendo R$ 1 bilhão:")
print(f"  Média              -> R$ {trajetorias['Média'][-1]:>14,.2f}")
print(f"  Média aparada 10%  -> R$ {trajetorias['Média aparada 10%'][-1]:>14,.2f}")
print(f"  Mediana            -> R$ {trajetorias['Mediana'][-1]:>14,.2f}")

# %% [markdown]
# A média é uma linha reta subindo sem limite: **ponto de ruptura 0%**. Mediana e
# média aparada são linhas horizontais — elas nem tomam conhecimento do intruso.
#
# > **Por que isso importa no trabalho:** um erro de digitação (vírgula no lugar
# > errado), um teste de carga que gravou latência de 10 minutos, um registro de
# > estorno com sinal invertido. Qualquer um desses, sozinho, envenena uma média
# > que vai para um dashboard executivo.
#
# ## 4. Dispersão: variância, desvio-padrão, IQR e MAD

# %%
def descreve(v: np.ndarray, rotulo: str) -> pd.Series:
    """Resumo que reporta as medidas clássicas E as robustas lado a lado.

    ddof=1 é explícito de propósito: o default do numpy é ddof=0 (divide por n)
    e o do pandas é ddof=1 (divide por n-1). Nunca confie no default.
    """
    q1, q3 = np.percentile(v, [25, 75])
    mad = np.median(np.abs(v - np.median(v)))
    return pd.Series({
        "n": len(v),
        "média": v.mean(),
        "mediana": np.median(v),
        "desvio-padrão (ddof=1)": v.std(ddof=1),
        "IQR": q3 - q1,
        "MAD": mad,
        "MAD x 1.4826": 1.4826 * mad,   # escala comparável a um desvio-padrão
        "CV = s / média": v.std(ddof=1) / v.mean(),
        "assimetria": stats.skew(v),
        "curtose excedente": stats.kurtosis(v),  # scipy já devolve a EXCEDENTE
    }, name=rotulo)


comparacao = pd.concat([
    descreve(x, "com corporativos"),
    descreve(x[~pedidos["corporativo"].to_numpy()], "sem corporativos"),
], axis=1)
comparacao

# %% [markdown]
# Leia essa tabela com atenção — ela é um resumo de tudo neste notebook:
#
# * O **desvio-padrão** despenca ao remover 0,5% dos dados. Medida frágil.
# * O **IQR** e o **MAD** quase não se mexem. Medidas robustas.
# * A **curtose excedente** com os corporativos é gigantesca: o alarme de "há
#   eventos extremos aqui que uma normal jamais preveria".
# * O `MAD x 1.4826` é bem menor que o desvio-padrão na coluna contaminada.
#   Essa **discrepância entre as duas estimativas de escala** é, ela própria, um
#   ótimo detector automático de cauda pesada.
#
# ## 5. Por que dividir por `n-1`? (o experimento que resolve a dúvida)
#
# A explicação verbal é: a soma de quadrados é calculada em torno da média
# **amostral**, que está "puxada" para o centro dos próprios dados, então os
# desvios saem pequenos demais.
#
# A explicação convincente é uma simulação. Vamos criar uma população com
# variância que **conhecemos** e ver qual divisor acerta na média.

# %%
VARIANCIA_VERDADEIRA = 25.0     # população N(10, 5²) — nós definimos, então sabemos
TAMANHO_AMOSTRA = 5             # amostras pequenas: é aí que a diferença aparece
N_REPETICOES = 200_000

amostras = rng.normal(loc=10, scale=np.sqrt(VARIANCIA_VERDADEIRA),
                      size=(N_REPETICOES, TAMANHO_AMOSTRA))

var_n = amostras.var(axis=1, ddof=0)    # divide por n
var_n1 = amostras.var(axis=1, ddof=1)   # divide por n-1 (correção de Bessel)

print(f"Variância verdadeira da população .......... {VARIANCIA_VERDADEIRA:.4f}")
print(f"Média de {N_REPETICOES:,} estimativas com ddof=0 ... {var_n.mean():.4f}"
      f"   (viés = {var_n.mean() - VARIANCIA_VERDADEIRA:+.4f})")
print(f"Média de {N_REPETICOES:,} estimativas com ddof=1 ... {var_n1.mean():.4f}"
      f"   (viés = {var_n1.mean() - VARIANCIA_VERDADEIRA:+.4f})")
print()
print(f"Razão observada var_n / var_n1 ............. {var_n.mean()/var_n1.mean():.4f}")
print(f"Razão teórica (n-1)/n ...................... "
      f"{(TAMANHO_AMOSTRA-1)/TAMANHO_AMOSTRA:.4f}")

# %% [markdown]
# O estimador com `ddof=0` erra **sistematicamente para baixo**, por um fator
# exato de $(n-1)/n$ — que com $n=5$ significa subestimar em 20%. Não é ruído:
# mesmo com 200 mil repetições o viés não some, porque viés não é aleatório.
#
# > **Onde isso te pega na prática:** validação cruzada com 5 folds. Você tem
# > 5 números de acurácia e quer reportar "0,84 ± incerteza". Se calcular o
# > desvio-padrão com `ddof=0`, sua barra de erro é 10% menor do que deveria — e
# > você declara vitória sobre um modelo concorrente que na verdade empatou.
#
# ## 6. Médias que não são a aritmética
#
# ### 6.1 Média geométrica: retornos e crescimento composto

# %%
# Um fundo que rende +50% e depois -50%. Quanto rendeu no total?
retornos = np.array([0.50, -0.50])
fatores = 1 + retornos

media_aritmetica = retornos.mean()
fator_final = np.prod(fatores)
media_geometrica = stats.gmean(fatores) - 1

print(f"Média aritmética dos retornos ...... {media_aritmetica:+.2%}  <- ERRADO")
print(f"Fator acumulado real .............. {fator_final:.4f} "
      f"({fator_final - 1:+.2%})")
print(f"Média geométrica dos fatores ...... {media_geometrica:+.2%}  <- CORRETO")
print()
print("Verificação: R$ 1.000 investidos viram "
      f"R$ {1000 * fator_final:,.2f} — perda real de "
      f"R$ {1000 * (1 - fator_final):,.2f}.")

# %% [markdown]
# > **Analogia:** a média aritmética responde "quanto eu somaria por período". A
# > média geométrica responde "quanto eu **multiplicaria** por período". Retorno
# > financeiro, crescimento de base de usuários e taxa de infecção se
# > multiplicam — nunca se somam. Usar a média errada aqui é como somar
# > velocidades quando deveria multiplicar fatores de escala.
#
# ### 6.2 Média harmônica: taxas com denominador fixo

# %%
# Um lote de 10.000 requisições processado em 3 etapas de pipeline.
# Cada etapa tem uma vazão diferente (requisições por segundo).
vazoes = np.array([500.0, 100.0, 250.0])   # req/s em cada etapa
n_req = 10_000

tempo_por_etapa = n_req / vazoes
tempo_total = tempo_por_etapa.sum()
vazao_real = (n_req * len(vazoes)) / tempo_total   # 3 lotes de n_req

print("Tempo em cada etapa (s):", np.round(tempo_por_etapa, 1))
print(f"Tempo total ................................ {tempo_total:.1f} s")
print()
print(f"Média ARITMÉTICA das vazões ................ {vazoes.mean():.1f} req/s  <- ERRADO")
print(f"Média HARMÔNICA das vazões ................. {stats.hmean(vazoes):.1f} req/s  <- CORRETO")
print(f"Vazão efetiva medida ....................... {vazao_real:.1f} req/s")
print()
print("Vale sempre:  harmônica <= geométrica <= aritmética")
print(f"  {stats.hmean(vazoes):.2f}  <=  {stats.gmean(vazoes):.2f}  <=  {vazoes.mean():.2f}")

# %% [markdown]
# A média harmônica é sempre puxada para o **menor** valor — e é isso que a torna
# correta aqui: um pipeline é tão rápido quanto seu gargalo. A média aritmética
# esconde o gargalo; a harmônica o denuncia.
#
# > **O F1-score é exatamente isso.** F1 é a média harmônica entre precisão e
# > recall. Um modelo com precisão 0,99 e recall 0,01 tem média aritmética 0,50
# > (parece razoável!) e F1 de 0,0198 (é péssimo, e o F1 conta a verdade).

# %%
precisao, recall = 0.99, 0.01
print(f"Média aritmética : {(precisao + recall) / 2:.4f}")
print(f"F1 (harmônica)   : {2 * precisao * recall / (precisao + recall):.4f}")

# %% [markdown]
# ## 7. Exercícios
#
# 1. Aumente `n_corporativos` de 25 para 250 e refaça a seção 2. A partir de que
#    ponto a **média aparada de 10%** também começa a quebrar? Relacione com o
#    ponto de ruptura da tabela do PDF.
# 2. Troque a log-normal por `rng.pareto(a=1.5, size=n) * 100`. Calcule a média
#    acumulada conforme `n` cresce (`np.cumsum(x) / np.arange(1, n+1)`) e mostre
#    que ela **não converge**. Por que a Lei dos Grandes Números não se aplica?
#    (Dica: para a Pareto com $a \le 1$ a média teórica é infinita.)
# 3. Simule notas de NPS de 1 a 5 com uma distribuição bimodal (muita nota 1 e
#    muita nota 5, poucas no meio). Calcule a média. Ela descreve **algum**
#    cliente real? Qual visualização revelaria o problema em dois segundos?
# 4. Refaça o experimento da seção 5 com `TAMANHO_AMOSTRA = 100`. O viés do
#    `ddof=0` ainda é relevante? Isso justifica ignorá-lo em amostras grandes?
#
# ## Próximo passo
#
# `02-robustez-e-outliers.ipynb` — como detectar contaminação de forma
# sistemática, e o que fazer quando a cauda pesada **não** é erro, e sim o
# fenômeno em si.
