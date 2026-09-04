<!-- tema: Estatística > Estatística Bayesiana -->
<!-- subtitulo: Atualizar crenças com dados, e finalmente poder dizer "95% de chance de o efeito estar aqui" -->
<!-- resumo: A inferência bayesiana troca a pergunta. Em vez de "com que frequência eu veria estes dados se não houvesse efeito?", ela pergunta "o que eu devo acreditar sobre o efeito, dado o que observei?". Este material constrói o teorema de Bayes desde a intuição, apresenta as famílias conjugadas que dão solução fechada, implementa MCMC do zero, e mostra por que a indústria migrou para o pensamento bayesiano em A/B testing, previsão de demanda e sistemas de decisão. -->
<!-- nivel: Intermediário -->
<!-- prerequisitos: Módulos 01 a 04 (Descritiva, Distribuições, Inferência, Testes) -->
<!-- duracao: 10 a 12 horas (leitura + 4 notebooks) -->
<!-- notebooks: 01-teorema-de-bayes · 02-conjugadas-e-posteriori · 03-mcmc-do-zero · 99-exercicios -->
<!-- autor: Material do curso de ML & DL -->
<!-- versao: 1.0 -->

# Estatística Bayesiana

## A troca de pergunta

Todo o módulo anterior girou em torno de uma quantidade: $P(\text{dados} \mid H_0)$.
O p-valor responde "se a hipótese nula fosse verdadeira, com que frequência eu
veria algo tão extremo?". É uma resposta honesta a uma pergunta que quase
ninguém faz.

A pergunta que as pessoas realmente fazem é a inversa:

> *Dado o que eu observei, o que devo acreditar sobre o parâmetro?*

Isto é, $P(\theta \mid \text{dados})$. Trocar uma condicional pela outra não é
detalhe de notação — é uma mudança de objeto. E ela tem preço: para inverter a
condicional você precisa declarar o que acreditava **antes** de ver os dados.

> [!ANALOGIA] Pense em um diagnóstico médico. O exame tem 99% de sensibilidade —
> essa é a verossimilhança, $P(\text{positivo} \mid \text{doente})$. Mas o
> paciente quer saber $P(\text{doente} \mid \text{positivo})$, e essa depende
> criticamente de quão comum a doença é na população. Sem a prevalência, o
> resultado do exame é ininterpretável. A **prevalência é a priori**, e ela não
> é opcional: ignorá-la é assumir uma, geralmente uma absurda.

### O teorema, e o que cada termo faz

$$P(\theta \mid D) = \frac{P(D \mid \theta) \, P(\theta)}{P(D)}$$

| Termo | Nome | O que representa |
|---|---|---|
| $P(\theta)$ | **priori** | o que você acreditava antes de ver os dados |
| $P(D \mid \theta)$ | **verossimilhança** | quão bem cada valor de $\theta$ explica os dados |
| $P(\theta \mid D)$ | **posteriori** | a crença atualizada — o objeto de interesse |
| $P(D)$ | **evidência** | constante de normalização; garante que a posteriori integre 1 |

Na prática quase sempre se trabalha com a versão proporcional, porque $P(D)$ não
depende de $\theta$:

$$P(\theta \mid D) \;\propto\; P(D \mid \theta) \times P(\theta)$$

Em palavras: **posteriori $\propto$ verossimilhança $\times$ priori**.

> [!FORMULA] **A leitura operacional:** a posteriori é a priori *reponderada*
> pela verossimilhança. Onde os dados são informativos, a verossimilhança é
> pontuda e domina; onde os dados são escassos, a priori sobrevive. O bayesiano
> não escolhe entre teoria e dados — ele especifica a taxa de câmbio entre as
> duas.

## O exemplo que resolve a intuição: o teste médico

Uma doença atinge 1 em cada 1.000 pessoas. Existe um exame com 99% de
sensibilidade e 99% de especificidade. Você testa positivo. Qual a chance de
estar doente?

A resposta intuitiva — 99% — está errada por quase duas ordens de grandeza.

Em 100.000 pessoas: 100 têm a doença, e o exame acerta 99 delas. Das 99.900
saudáveis, o exame erra em 1%, produzindo **999 falsos positivos**. Entre os
1.098 positivos, apenas 99 estão doentes:

$$P(\text{doente} \mid +) = \frac{99}{99 + 999} \approx 9{,}0\%$$

> [!ARMADILHA] Esse é o **erro da taxa-base**, e ele reaparece disfarçado em
> todo modelo de classificação de evento raro. Um detector de fraude com 99% de
> acurácia aplicado a uma base com 0,1% de fraude gera dez alarmes falsos para
> cada fraude verdadeira. É a mesma conta, e a razão pela qual precisão e recall
> existem: a acurácia esconde a taxa-base.

## Prioris: o que são e como escolher

A objeção clássica à inferência bayesiana é que a priori é subjetiva. A resposta
madura é que **toda análise tem uma priori** — a frequentista simplesmente não a
declara. Máxima verossimilhança é matematicamente idêntica a inferência bayesiana
com priori uniforme, o que é uma escolha, e frequentemente uma escolha ruim
(uniforme em $\theta$ não é uniforme em $\log \theta$).

| Tipo de priori | Quando usar | Exemplo |
|---|---|---|
| **Informativa** | há conhecimento sólido de domínio ou histórico | conversão histórica do site é 3% ± 0,5% |
| **Fracamente informativa** | você sabe a ordem de grandeza, não o valor | "o lift está entre −50% e +50%" |
| **Não informativa / de referência** | quer deixar os dados falarem | Jeffreys, uniforme no espaço adequado |
| **Regularizadora** | quer evitar estimativas absurdas com pouco dado | Beta(2,2) evita $\hat{p} = 0$ ou $1$ |

> [!MERCADO] Em A/B testing de e-commerce, usar a taxa de conversão histórica
> como priori Beta é o que impede um resultado de "0 conversões em 40 visitas"
> de virar uma estimativa de conversão zero. A priori informativa não é um viés
> escondido — é o histórico da empresa entrando na conta explicitamente, e
> auditável.

### O que a priori faz com pouco e com muito dado

A propriedade que resolve a maior parte da ansiedade sobre subjetividade:
conforme $n$ cresce, a verossimilhança domina e prioris razoavelmente diferentes
convergem para a mesma posteriori. Com $n = 10$, a priori manda; com $n = 10.000$,
ela é irrelevante. A escolha da priori importa exatamente onde você tem pouco
dado — que é onde declarar suas suposições importa mais.

## Famílias conjugadas: quando existe solução fechada

Uma priori é **conjugada** para uma verossimilhança quando a posteriori pertence
à mesma família da priori. Isso transforma inferência em aritmética.

| Verossimilhança | Priori conjugada | Posteriori |
|---|---|---|
| Bernoulli / Binomial($p$) | Beta($\alpha, \beta$) | Beta($\alpha + k$, $\beta + n - k$) |
| Poisson($\lambda$) | Gama($\alpha, \beta$) | Gama($\alpha + \sum x_i$, $\beta + n$) |
| Normal($\mu$), $\sigma$ conhecido | Normal($\mu_0, \tau_0^2$) | Normal (média ponderada por precisão) |
| Exponencial($\lambda$) | Gama($\alpha, \beta$) | Gama($\alpha + n$, $\beta + \sum x_i$) |
| Multinomial | Dirichlet($\boldsymbol{\alpha}$) | Dirichlet($\boldsymbol{\alpha} + \mathbf{k}$) |

> [!FORMULA] **Beta-Binomial, o cavalo de batalha:**
>
> priori Beta($\alpha$, $\beta$) + $k$ sucessos em $n$ ensaios
> $\Rightarrow$ posteriori Beta($\alpha + k$, $\beta + n - k$).
>
> Média posterior $= \dfrac{\alpha + k}{\alpha + \beta + n}$.
>
> Interpretação de $\alpha$ e $\beta$: **pseudo-contagens**. Beta(3, 97) é
> literalmente "eu já vi 3 sucessos em 100 tentativas".

A conjugação Normal-Normal revela algo ainda mais bonito. Com priori
$\mathcal{N}(\mu_0, \tau_0^2)$ e $n$ observações de média $\bar{x}$ e variância
conhecida $\sigma^2$, a média posterior é

$$\mu_{\text{post}} = \frac{\frac{1}{\tau_0^2}\mu_0 + \frac{n}{\sigma^2}\bar{x}}{\frac{1}{\tau_0^2} + \frac{n}{\sigma^2}}$$

— uma média ponderada entre a priori e os dados, com pesos iguais às
**precisões** (inverso das variâncias). Quem tem menos incerteza pesa mais. É
exatamente a fórmula do filtro de Kalman, e a mesma ideia por trás de
regularização Ridge: a penalidade L2 **é** uma priori normal sobre os
coeficientes.

## Intervalo de credibilidade × intervalo de confiança

Esta é a diferença que mais importa na comunicação de resultados.

| | Intervalo de confiança (95%) | Intervalo de credibilidade (95%) |
|---|---|---|
| Objeto aleatório | o intervalo | o parâmetro |
| Interpretação | 95% dos intervalos assim construídos conteriam $\theta$ | há 95% de probabilidade de $\theta$ estar aqui |
| Depende de priori | não | sim |
| Frase legítima | "o procedimento acerta 95% das vezes" | "95% de chance de o efeito estar entre A e B" |

O intervalo de credibilidade diz **exatamente** o que todo mundo já interpreta
erradamente do intervalo de confiança. Existem duas construções comuns: o
**intervalo de caudas iguais** (percentis 2,5 e 97,5 da posteriori) e o **HDI**
(*highest density interval*), a região mais estreita que contém 95% da massa. Em
posterioris assimétricas ou bimodais, o HDI é o resumo honesto.

## MCMC: o que fazer quando não há conjugada

Fora dos casos conjugados, a posteriori não tem forma fechada — o denominador
$P(D)$ é uma integral intratável em dimensão alta. A saída não é calcular a
posteriori, e sim **amostrar** dela.

O algoritmo de **Metropolis-Hastings** é surpreendentemente simples:

1. Comece em um valor qualquer $\theta_t$.
2. Proponha $\theta^* \sim q(\cdot \mid \theta_t)$ (por exemplo, um passo gaussiano).
3. Calcule a razão $r = \dfrac{P(D \mid \theta^*)P(\theta^*)}{P(D \mid \theta_t)P(\theta_t)}$.
4. Aceite $\theta^*$ com probabilidade $\min(1, r)$; senão, fique onde está.
5. Repita.

O denominador $P(D)$ **cancela na razão** — é por isso que o método funciona sem
nunca calcular a integral. Após um período de aquecimento (*burn-in*), as
amostras vêm da posteriori.

> [!ARMADILHA] MCMC não avisa quando falha. Cadeia presa em uma moda, taxa de
> aceitação de 2% ou de 95%, autocorrelação altíssima — em todos esses casos o
> algoritmo devolve números com toda a confiança. **Sempre diagnostique:**
> trace plot (deve parecer ruído branco, não um passeio), $\hat{R} < 1{,}01$
> entre múltiplas cadeias, e tamanho efetivo de amostra (ESS) na casa das
> centenas no mínimo. Taxa de aceitação-alvo: ~23% para passeio aleatório
> multivariado, ~44% em uma dimensão.

Ferramentas modernas (Stan, PyMC, NumPyro) usam **HMC/NUTS**, que aproveita o
gradiente da log-posteriori para propor saltos longos e bem direcionados em vez
de tatear. A diferença de eficiência em dimensão alta é de ordens de grandeza —
mas a lógica de aceitar/rejeitar continua sendo a de Metropolis.

## Por que a indústria migrou

> [!MERCADO] **A/B testing bayesiano.** A pergunta do time de produto nunca foi
> "rejeitamos $H_0$?", e sim "qual a chance de B ser melhor que A, e quanto eu
> perco se escolher errado?". A posteriori responde as duas diretamente:
> $P(p_B > p_A)$ é uma média sobre amostras, e a **perda esperada** de escolher
> a variante errada é calculável. Além disso, não há problema de *peeking*: a
> posteriori é válida a qualquer momento, porque não há taxa de erro de longo
> prazo sendo controlada.

Outros lugares onde o pensamento bayesiano venceu:

- **Multi-armed bandits** (Thompson Sampling): alocar tráfego proporcionalmente à
  probabilidade de cada variante ser a melhor. Ganha dinheiro durante o
  experimento, em vez de esperar o fim.
- **Modelos hierárquicos**: estimar conversão por loja, região ou usuário com
  *shrinkage* automático — lojas com pouco dado são puxadas para a média global.
  É a solução principiada para o problema de "essa loja converteu 100%, com 2
  visitas".
- **Previsão de demanda e séries temporais** (Prophet, modelos de espaço de
  estados): incerteza propagada corretamente até o horizonte de previsão.
- **Deep learning**: dropout como aproximação bayesiana, ensembles profundos e
  quantificação de incerteza em modelos de decisão crítica.

## O custo do método

Ser justo com as objeções:

- **Escolher priori dá trabalho** e exige defender a escolha. Análise de
  sensibilidade (rodar com 2 ou 3 prioris) deixa de ser opcional.
- **Custo computacional.** MCMC em modelo grande leva minutos a horas, contra
  milissegundos de um `fit` frequentista.
- **Diagnóstico é obrigatório** e é uma competência a mais.
- **Comunicação**: parte do público espera p-valor, e "probabilidade de o efeito
  ser positivo" precisa ser explicada.

Nenhuma dessas é razão para não usar. São razões para usar com disciplina.

> [!FORMULA] **Resumo operacional**
>
> Posteriori $\propto$ verossimilhança $\times$ priori.
>
> **Beta-Binomial:** Beta($\alpha + k$, $\beta + n - k$).
> **Gama-Poisson:** Gama($\alpha + \sum x_i$, $\beta + n$).
> **Normal-Normal:** média ponderada pelas precisões $1/\tau_0^2$ e $n/\sigma^2$.
>
> **Metropolis-Hastings:** aceite com $\min\left(1, \frac{\mathcal{L}(\theta^*)\pi(\theta^*)}{\mathcal{L}(\theta_t)\pi(\theta_t)}\right)$.
>
> **Diagnóstico mínimo:** $\hat{R} < 1{,}01$, ESS > 400, aceitação 20–50%.

## Erros que custam caro — checklist

- Usar priori informativa forte sem declarar e sem análise de sensibilidade.
- Reportar o intervalo de credibilidade como se fosse de confiança (ou vice-versa).
- Rodar MCMC sem olhar trace plot nem $\hat{R}$.
- Interpretar $P(p_B > p_A) = 0{,}93$ como "93% de lift" — é probabilidade de
  ser melhor, não magnitude do efeito.
- Esquecer a taxa-base ao interpretar um classificador de evento raro.
- Usar priori uniforme achando que é "neutra" — ela não é invariante a
  reparametrização.
- Escolher a priori depois de ver os dados.

## Para ir além

- McElreath, *Statistical Rethinking* — o melhor livro-texto de inferência
  bayesiana aplicada que existe, com código.
- Gelman et al., *Bayesian Data Analysis* (BDA3) — a referência canônica.
- Kruschke, *Doing Bayesian Data Analysis* — didático, forte em HDI e ROPE.
- Documentação do Stan sobre *prior choice recommendations* — o guia prático
  mais usado para escolher prioris fracamente informativas.
