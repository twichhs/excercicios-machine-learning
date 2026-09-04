<!-- tema: Estatística > Distribuições de Probabilidade -->
<!-- subtitulo: O catálogo de modelos de incerteza que todo cientista de dados precisa reconhecer de olhos fechados -->
<!-- resumo: Uma distribuição não é uma curva bonita: é uma hipótese sobre o mecanismo que gerou seus dados. Este material apresenta cada distribuição a partir da história física que a produz — e mostra como reconhecer essa história olhando para os dados. -->
<!-- nivel: Intermediário — a probabilidade é construída do zero -->
<!-- prerequisitos: Módulo 01 (Fundamentos e Estatística Descritiva) -->
<!-- duracao: 8 a 10 horas (leitura + 4 notebooks) -->
<!-- notebooks: 01-discretas · 02-continuas · 03-tlc-e-lei-dos-grandes-numeros · 04-ajuste-a-dados-reais -->
<!-- autor: Material do curso de ML & DL -->
<!-- versao: 1.0 -->

# Distribuições de Probabilidade

## A ideia que organiza o capítulo

Estudantes costumam decorar distribuições como quem decora tabela periódica:
nome, fórmula, média, variância. É a forma mais rápida de esquecer tudo e a mais
lenta de aprender.

A abordagem deste capítulo é outra. **Cada distribuição é a resposta matemática a
uma história sobre como os dados nasceram.** Se você conhece a história, a
fórmula deixa de ser arbitrária, e — o que importa muito mais na prática — você
consegue fazer o caminho inverso: olhar para um problema de negócio e reconhecer
qual história está acontecendo ali.

> [!ANALOGIA] Pense em distribuições como **ferramentas de oficina**. Ninguém
> decora o catálogo de chaves de boca. Você olha a porca, vê o formato, e pega a
> chave certa. Uma contagem de eventos raros num intervalo é uma "porca de
> Poisson"; um tempo até a falha é uma "porca exponencial". O trabalho é
> aprender a olhar a porca.

### Duas grandes famílias

| | Discretas | Contínuas |
|---|---|---|
| O que descrevem | Contagens, categorias, sucessos | Medidas, tempos, grandezas |
| Função que as define | **PMF**: $P(X = x)$ | **PDF**: $f(x)$, com $P(X = x) = 0$ |
| Como se soma | $\sum_x P(X=x) = 1$ | $\int f(x)\,dx = 1$ |
| Exemplo de negócio | Nº de cliques, nº de sinistros | Latência, receita, temperatura |

Um ponto que confunde quase todo iniciante: para variáveis contínuas,
$P(X = 3{,}14) = 0$. Não é uma esquisitice técnica — é consequência de haver
infinitos valores possíveis. Só faz sentido perguntar por
$P(a \le X \le b) = \int_a^b f(x)\,dx$. A densidade $f(x)$ pode inclusive ser
maior que 1; o que precisa somar 1 é a **área**, não a altura.

---

## Ferramentas comuns a todas as distribuições

Antes do catálogo, cinco objetos que se aplicam a qualquer distribuição.

**Função de distribuição acumulada (CDF)** — a mais útil na prática:

$$
F(x) = P(X \leq x)
$$

**Função quantil** — a inversa da CDF, que responde "qual valor deixa 95% abaixo
de si?". É o que produz um p95 e o que define um VaR.

$$
Q(p) = F^{-1}(p)
$$

**Esperança e variância**:

$$
\mathbb{E}[X] = \sum_x x \, P(X=x) \quad \text{ou} \quad \int x f(x)\,dx
$$
$$
\text{Var}(X) = \mathbb{E}[X^2] - \left( \mathbb{E}[X] \right)^2
$$

**Função geradora de momentos**, $M_X(t) = \mathbb{E}[e^{tX}]$, cujas derivadas
em $t=0$ produzem os momentos. Ela é a ferramenta que prova, em três linhas, que
a soma de normais independentes é normal e que a soma de Poissons é Poisson.

**Propriedades de transformação** — as duas que mais se usam no dia a dia:

$$
\mathbb{E}[aX + b] = a\,\mathbb{E}[X] + b
\qquad
\text{Var}(aX + b) = a^2 \text{Var}(X)
$$

Note que $b$ some da variância (deslocar não muda espalhamento) e que $a$ entra
ao quadrado. Isso explica por que o erro-padrão da média é $\sigma/\sqrt{n}$ e
não $\sigma/n$ — assunto do módulo de inferência.

---

## Distribuições discretas

### Bernoulli — o átomo

**História:** um único experimento com dois resultados. Converteu ou não
converteu. Clicou ou não clicou. Pagou ou deu calote.

$$
P(X = x) = p^x (1-p)^{1-x}, \qquad x \in \{0, 1\}
$$
$$
\mathbb{E}[X] = p \qquad \text{Var}(X) = p(1-p)
$$

Detalhe com consequência prática: a variância é máxima em $p = 0{,}5$ (vale
$0{,}25$) e vai a zero nos extremos. É por isso que **testes A/B com taxas de
conversão muito baixas precisam de amostras enormes** — não porque a variância
seja alta, mas porque o *efeito relativo* que você quer detectar fica pequeno
perto do ruído.

### Binomial — somando Bernoullis

**História:** $n$ experimentos de Bernoulli **independentes** e com o **mesmo**
$p$; conte quantos sucessos.

$$
P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}
$$
$$
\mathbb{E}[X] = np \qquad \text{Var}(X) = np(1-p)
$$

> [!ARMADILHA] As duas hipóteses da binomial — independência e $p$ constante —
> são violadas o tempo todo em dados de produto. Usuários que vieram da mesma
> campanha não são independentes; a taxa de conversão às 3 da manhã não é a
> mesma do horário de pico. Quando isso acontece, a variância real é **maior**
> que $np(1-p)$: é a **superdispersão**, e ela faz seus intervalos de confiança
> ficarem estreitos demais. Você declara significância que não existe.

### Poisson — eventos raros num intervalo

**História:** eventos ocorrem de forma independente, a uma taxa média constante
$\lambda$ por intervalo. Conte quantos ocorreram.

$$
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}
$$
$$
\mathbb{E}[X] = \lambda \qquad \text{Var}(X) = \lambda
$$

A propriedade mais característica — e o melhor teste diagnóstico de campo — é
**média igual a variância**. Se você suspeita de Poisson, calcule a razão
$\text{Var}/\text{média}$ nos dados. Se der muito acima de 1, não é Poisson: é
superdispersão, e o modelo correto é a **binomial negativa**.

A Poisson é o limite da binomial quando $n \to \infty$ e $p \to 0$ com
$np = \lambda$ fixo. Traduzindo: **muitas oportunidades, cada uma improvável**. É
por isso que ela modela tão bem chamadas em call center, sinistros de seguro,
falhas de servidor, acessos por segundo e defeitos por lote.

> [!MERCADO] Precificação de seguros é essencialmente um modelo de frequência ×
> severidade. A **frequência** de sinistros é Poisson (ou binomial negativa) e a
> **severidade** de cada sinistro é uma distribuição de cauda pesada (log-normal,
> gama ou Pareto). O prêmio é aproximadamente
> $\mathbb{E}[\text{frequência}] \times \mathbb{E}[\text{severidade}]$ mais uma
> margem de risco derivada da variância. Toda seguradora do mundo roda alguma
> versão disso.

### Binomial negativa — a Poisson com superdispersão

**História:** conte fracassos até obter $r$ sucessos. Mas o uso moderno é outro,
e mais importante: ela é uma **mistura de Poissons** em que o próprio $\lambda$
varia entre indivíduos segundo uma distribuição gama.

Essa segunda leitura é a que importa. Se cada cliente tem sua própria taxa de
compra e essas taxas variam na população, a contagem agregada é binomial
negativa, não Poisson. A variância vira

$$
\text{Var}(X) = \mu + \frac{\mu^2}{r}
$$

sempre maior que a média — exatamente o que se observa em quase todo dado de
contagem do mundo real.

### Geométrica — quanto tempo até o primeiro sucesso

$$
P(X = k) = (1-p)^{k-1} p, \qquad \mathbb{E}[X] = \frac{1}{p}
$$

Tem a propriedade de **falta de memória**: se você já tentou 10 vezes sem
sucesso, a distribuição do número de tentativas restantes é idêntica à do
começo. É a versão discreta da exponencial, e a base da modelagem de churn em
tempo discreto.

---

## Distribuições contínuas

### Normal — e por que ela está em todo lugar

$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(x-\mu)^2}{2\sigma^2} \right)
$$

A normal não é onipresente por acaso, nem porque "a natureza gosta dela". Há três
razões técnicas:

1. **Teorema Central do Limite.** Somas e médias de muitas contribuições
   independentes tendem à normal, praticamente sem importar a distribuição de
   origem.
2. **Máxima entropia.** Entre todas as distribuições com uma dada média e
   variância, a normal é a que assume *menos* além disso. É a escolha
   honestamente mais ignorante.
3. **Conveniência algébrica.** Somas de normais são normais; combinações lineares
   de normais são normais; a normal é conjugada consigo mesma. Isso torna a
   matemática tratável.

A regra empírica 68–95–99,7 (dentro de 1, 2 e 3 desvios-padrão) merece ser
decorada, com uma ressalva: ela **só vale se os dados forem realmente normais**,
e a maior parte dos dados de negócio não é.

> [!ARMADILHA] "Assumir normalidade" quase nunca significa assumir que *os dados*
> são normais. Em regressão linear, a suposição é sobre os **resíduos**, não
> sobre $X$ nem sobre $y$. Em testes de média, o TLC cuida da distribuição da
> **média amostral**, não das observações. Confundir esses três níveis é o erro
> conceitual mais comum de quem aprendeu estatística por receita.

### Log-normal — quando os efeitos se multiplicam

Se $\ln X \sim \mathcal{N}(\mu, \sigma^2)$, então $X$ é log-normal. Surge sempre
que o resultado é o **produto** de muitos fatores independentes (o TLC atua sobre
o log).

$$
\mathbb{E}[X] = \exp\left(\mu + \frac{\sigma^2}{2}\right)
\qquad
\text{mediana}(X) = e^{\mu}
$$

Repare: a média é sempre **maior** que a mediana, e a diferença cresce
exponencialmente com $\sigma$. É a assinatura matemática de "ticket médio muito
acima do ticket típico". Renda, receita por cliente, tempo de resposta, tamanho
de arquivo e preço de imóvel são todos aproximadamente log-normais.

### Exponencial — tempo até o próximo evento

$$
f(x) = \lambda e^{-\lambda x}, \quad x \geq 0
\qquad
\mathbb{E}[X] = \frac{1}{\lambda}
\qquad
\text{Var}(X) = \frac{1}{\lambda^2}
$$

É a irmã contínua da Poisson: **se as contagens por intervalo são Poisson($\lambda$),
os tempos entre eventos são Exponencial($\lambda$)**. Essas são duas descrições do
mesmo processo.

Também é **sem memória**: $P(X > s+t \mid X > s) = P(X > t)$. Um componente que
já durou 1.000 horas tem a mesma distribuição de vida restante que um novo — o
que é uma hipótese frequentemente *falsa* na engenharia real, e é exatamente por
isso que existe a Weibull.

### Weibull — falha com desgaste

$$
f(x) = \frac{k}{\lambda} \left( \frac{x}{\lambda} \right)^{k-1} e^{-(x/\lambda)^k}
$$

O parâmetro de forma $k$ é o que importa:

| $k$ | Taxa de falha | Fase do produto |
|---|---|---|
| $k < 1$ | Decrescente | Mortalidade infantil (defeitos de fábrica) |
| $k = 1$ | Constante | Falhas aleatórias (= exponencial) |
| $k > 1$ | Crescente | Desgaste, envelhecimento |

Juntas, essas três fases formam a **curva da banheira** da confiabilidade. A
Weibull é o modelo padrão em manutenção preditiva e análise de sobrevivência
industrial.

### Uniforme, Beta, Gama e Qui-quadrado

**Uniforme** $U(a,b)$ — ignorância total dentro de um intervalo. É a base de todo
gerador de números aleatórios: qualquer distribuição pode ser amostrada aplicando
$F^{-1}(U)$ a uma uniforme (método da **transformada inversa**).

**Beta** $\text{Beta}(\alpha, \beta)$ — definida em $[0,1]$, é a distribuição
natural para modelar uma **probabilidade desconhecida**. É a priori conjugada da
binomial, e por isso o motor de testes A/B bayesianos e de algoritmos de
*multi-armed bandit* (Thompson sampling). Interpretação memorável: $\alpha - 1$
sucessos e $\beta - 1$ fracassos observados.

**Gama** — soma de $k$ exponenciais independentes. Modela tempo até a $k$-ésima
falha, e é a distribuição de severidade preferida em seguros.

**Qui-quadrado** $\chi^2_k$ — soma de $k$ normais padrão ao quadrado. Você
raramente a "vê" nos dados; ela aparece como distribuição de **estatísticas de
teste** (teste qui-quadrado de independência, teste da razão de verossimilhança).

### t de Student — a normal com incerteza sobre a escala

Quando você estima $\sigma$ a partir da amostra em vez de conhecê-lo, a
estatística padronizada deixa de ser normal e passa a seguir uma $t$ com $n-1$
graus de liberdade. Ela tem **caudas mais pesadas** que a normal — exatamente
para pagar o preço de não conhecer a variância. Com $df > 30$ as duas são
praticamente indistinguíveis.

---

## Como escolher: um guia de decisão

| A pergunta do negócio | Distribuição | Sinal nos dados |
|---|---|---|
| Converteu ou não? | Bernoulli | Só 0 e 1 |
| Quantos sucessos em $n$ tentativas? | Binomial | Inteiro limitado por $n$ |
| Quantos eventos por hora/dia? | Poisson | Inteiro $\geq 0$; Var $\approx$ média |
| Quantos eventos, com clientes heterogêneos? | Binomial negativa | Inteiro $\geq 0$; Var $\gg$ média |
| Quanto tempo até o próximo evento? | Exponencial | Contínuo $> 0$; sem memória |
| Quanto tempo até falhar, com desgaste? | Weibull | Contínuo $> 0$; taxa de falha varia |
| Receita, renda, latência | Log-normal | Contínuo $> 0$; simétrico em escala log |
| Média de muitas contribuições | Normal | Simétrico; QQ-plot na diagonal |
| Uma proporção desconhecida | Beta | Contínuo em $[0,1]$ |
| Contagem com excesso de zeros | Poisson/BN inflada em zero | Pico enorme em 0 |

> [!MERCADO] O erro mais caro em modelagem de contagens é usar **regressão
> linear** para prever número de eventos. Ela permite previsões negativas (nº de
> pedidos = $-3$), assume variância constante (falso: contagens têm variância
> proporcional à média) e assume resíduos simétricos. O correto é uma **GLM**
> com família Poisson ou binomial negativa e função de ligação log. É uma troca
> de uma linha de código que muda o resultado por completo.

---

## Os dois teoremas que sustentam tudo

### Lei dos Grandes Números

A média amostral converge para a média populacional conforme $n$ cresce:

$$
\bar{X}_n \to \mu \quad \text{quando} \quad n \to \infty
$$

Ela justifica que estimar funciona — mas **exige que $\mu$ exista**. Para
distribuições de cauda muito pesada (Pareto com $\alpha \leq 1$, Cauchy) a média
teórica é infinita ou indefinida e a média amostral **nunca converge**, por mais
dados que você colete. Vimos isso empiricamente no módulo anterior.

### Teorema Central do Limite

$$
\frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \to \mathcal{N}(0, 1)
$$

A distribuição da **média amostral** tende à normal, quase independentemente da
distribuição de origem — desde que a variância seja finita.

Três consequências que se usam todos os dias:

1. O erro-padrão da média cai com $\sqrt{n}$, não com $n$. **Para reduzir o erro
   pela metade é preciso quadruplicar a amostra.** É a economia fundamental de
   todo teste A/B.
2. Testes t e intervalos de confiança funcionam mesmo com dados não-normais,
   desde que $n$ seja suficiente.
3. "$n \geq 30$" é uma regra de bolso, não um teorema. Para dados muito
   assimétricos, pode ser necessário $n$ de centenas ou milhares. Para dados já
   quase simétricos, $n = 10$ basta. A pergunta certa é sempre sobre a
   assimetria, não sobre um número mágico.

> [!ARMADILHA] O TLC fala sobre a distribuição da **média**, não sobre a
> distribuição dos **dados**. Coletar mais dados não torna sua variável mais
> normal — ela continua exatamente tão assimétrica quanto era. O que fica normal
> é a distribuição amostral da estatística.

---

## Formulário do capítulo

> [!FORMULA] **Bernoulli($p$)**: $\mathbb{E} = p$, $\text{Var} = p(1-p)$.
>
> **Binomial($n,p$)**: $P(X=k) = \binom{n}{k}p^k(1-p)^{n-k}$; $\mathbb{E} = np$,
> $\text{Var} = np(1-p)$.
>
> **Poisson($\lambda$)**: $P(X=k) = \lambda^k e^{-\lambda}/k!$;
> $\mathbb{E} = \text{Var} = \lambda$.
>
> **Binomial negativa**: $\text{Var} = \mu + \mu^2/r > \mu$.
>
> **Normal($\mu,\sigma^2$)**: $f(x) = (2\pi\sigma^2)^{-1/2}
> \exp(-(x-\mu)^2 / 2\sigma^2)$.
>
> **Log-normal**: $\mathbb{E}[X] = \exp(\mu + \sigma^2/2)$,
> $\text{mediana} = e^{\mu}$.
>
> **Exponencial($\lambda$)**: $f(x) = \lambda e^{-\lambda x}$;
> $\mathbb{E} = 1/\lambda$, $\text{Var} = 1/\lambda^2$.
>
> **Weibull($k,\lambda$)**: $k<1$ mortalidade infantil, $k=1$ exponencial,
> $k>1$ desgaste.
>
> **TLC**: $(\bar{X}_n - \mu) / (\sigma/\sqrt{n}) \to \mathcal{N}(0,1)$.

## Erros que custam caro — checklist

- Assumir Poisson sem checar se $\text{Var} \approx \text{média}$.
- Usar regressão linear para contagens.
- Aplicar a regra 68–95–99,7 a dados assimétricos.
- Confundir normalidade dos dados com normalidade dos resíduos.
- Usar o TLC com $n=30$ em dados de cauda pesada.
- Ignorar excesso de zeros (a maioria dos dados de contagem em produto tem).
- Modelar tempo até falha com exponencial quando há desgaste evidente.

## Para ir além

- Blitzstein & Hwang, *Introduction to Probability* — o melhor livro-texto atual,
  disponível gratuitamente com vídeos das aulas de Harvard.
- Wasserman, *All of Statistics*, capítulos 2 e 3 — denso e direto ao ponto.
- Hilbe, *Modeling Count Data* — a referência prática sobre superdispersão.
