<!-- tema: Estatística > Fundamentos e Estatística Descritiva -->
<!-- subtitulo: Como resumir dados sem mentir — e por que quase todo resumo mente um pouco -->
<!-- resumo: Descrever dados é o primeiro ato de modelagem, não uma etapa preliminar. Toda medida-resumo é uma compressão com perda: este material mostra exatamente o que cada uma joga fora, quando isso é aceitável e quando destrói a decisão de negócio que depende dela. -->
<!-- nivel: Intermediário — requer Python; a estatística é construída do zero -->
<!-- prerequisitos: NumPy e pandas básicos -->
<!-- duracao: 6 a 8 horas (leitura + 3 notebooks) -->
<!-- notebooks: 01-medidas-resumo · 02-robustez-e-outliers · 03-visualizacao-distribuicoes -->
<!-- autor: Material do curso de ML & DL -->
<!-- versao: 1.0 -->

# Fundamentos e Estatística Descritiva

## Por que este capítulo existe

Existe uma crença confortável de que estatística descritiva é a parte fácil — a
que se resolve com `df.describe()` antes de começar o trabalho "de verdade". Essa
crença custa caro. Praticamente todo desastre de modelagem que eu já vi
autopsiar tinha, na origem, um resumo mal interpretado: uma média calculada sobre
uma distribuição bimodal, um desvio-padrão sobre uma cauda pesada, uma correlação
sobre uma relação em forma de U.

Vale fixar a ideia central antes de qualquer fórmula:

> [!DEFINICAO] Uma medida-resumo é uma função que colapsa $n$ números em um só.
> Como $n > 1$, a função **não é injetora**: infinitas amostras diferentes
> produzem o mesmo resumo. Escolher um resumo é escolher **qual informação
> descartar**.

A pergunta profissional nunca é "qual é a média?". É "o que eu perco ao
substituir estes 4 milhões de transações pela média delas, e essa perda é
tolerável para a decisão que vou tomar?".

> [!ANALOGIA] Um resumo estatístico é como a foto de uma cidade tirada de um
> avião a dez mil metros. A média diz onde fica o centro geográfico. O
> desvio-padrão diz o quanto a cidade se espalha. Nenhum dos dois diz que existe
> um rio cortando a cidade ao meio — e é o rio que determina onde você consegue
> construir. Histograma e quantis são o voo baixo.

### O que você vai conseguir fazer ao final

- Escolher entre média, mediana e média aparada com um argumento técnico, não por
  hábito.
- Explicar por que a variância amostral divide por $n-1$ e o que acontece se você
  esquecer disso.
- Ler assimetria e curtose como diagnóstico, e não como números decorativos.
- Definir SLAs em percentis e entender por que p99 é uma métrica de engenharia
  fundamentalmente diferente da média.
- Detectar, antes de modelar, os três padrões que mais quebram modelos:
  bimodalidade, cauda pesada e mistura de populações.

---

## Tipos de variável: a decisão que vem antes de tudo

Antes de calcular qualquer coisa, é preciso saber que tipo de número está na sua
mão. A taxonomia clássica de Stevens organiza as variáveis em quatro **escalas de
medida**, e cada escala autoriza um conjunto diferente de operações.

| Escala | Exemplo | O que faz sentido | O que **não** faz sentido |
|---|---|---|---|
| Nominal | UF, canal de aquisição | Contagem, moda, frequência | Média, ordenação |
| Ordinal | Nota NPS 1–5, rating de crédito AAA–D | Mediana, quantis, ordenação | Média (com ressalva), diferenças |
| Intervalar | Temperatura em °C, data | Média, diferença, desvio-padrão | Razão ("30 °C é o dobro de 15 °C" é falso) |
| Razão | Receita, latência, contagem | Tudo, inclusive razão e média geométrica | — |

O erro mais comum e mais caro dessa tabela é tratar **ordinal como razão**. Uma
nota de satisfação de 1 a 5 tem ordem, mas não tem distância definida: a
diferença de experiência entre 1 e 2 não é a mesma entre 4 e 5. Calcular "NPS
médio = 3,8" empilha uma suposição de equidistância que ninguém verificou.

> [!ARMADILHA] Variáveis categóricas codificadas como inteiros (`estado = 1, 2,
> 3, ...`) passam despercebidas por qualquer `describe()`: o pandas calcula
> média, desvio-padrão e quantis sem reclamar. O output é sintaticamente válido e
> semanticamente lixo. Antes de qualquer análise, verifique o **dtype** e a
> **cardinalidade** de cada coluna.

Uma distinção adicional, ausente em Stevens mas decisiva na prática: variáveis
**contadas** (inteiros não-negativos, com massa em zero) versus variáveis
**medidas** (contínuas). Contagens quase nunca são simétricas e quase nunca são
bem descritas por média ± desvio-padrão. Elas pedem Poisson, binomial negativa ou
modelos com inflação de zeros — assunto do módulo de distribuições.

---

## Medidas de posição

### A média aritmética e o que ela realmente é

$$
\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i
$$

A média tem uma caracterização variacional que explica todo o seu comportamento —
e que vale memorizar, porque ela reaparece em regressão, em redes neurais e em
praticamente todo lugar onde se minimiza erro quadrático:

$$
\bar{x} = \arg\min_{c} \sum_{i=1}^{n} (x_i - c)^2
$$

Ou seja: **a média é o ponto que minimiza a soma dos erros quadráticos**. Como o
erro é elevado ao quadrado, um ponto três vezes mais distante pesa nove vezes
mais. Isso não é um defeito acidental; é a definição. A média é sensível a
valores extremos *por construção*.

A mediana tem a caracterização análoga com erro absoluto:

$$
\mathrm{mediana}(x) = \arg\min_{c} \sum_{i=1}^{n} |x_i - c|
$$

> [!NOTA] Essa dualidade é a mesma que separa a função de perda MSE da MAE em
> regressão. Um modelo treinado com MSE prevê a **média** condicional; treinado
> com MAE, prevê a **mediana** condicional. Quando alguém reclama que "o modelo
> superestima na maioria dos casos mas o erro médio está ótimo", quase sempre a
> resposta está aqui.

### Médias que não são a aritmética

Três médias aparecem com frequência no mercado e são sistematicamente usadas
erradas.

**Média geométrica** — para taxas de crescimento compostas:

$$
G = \left( \prod_{i=1}^{n} x_i \right)^{1/n} = \exp\left( \frac{1}{n} \sum_{i=1}^{n} \ln x_i \right)
$$

Se uma carteira rende $+50\%$ num ano e $-50\%$ no seguinte, a média aritmética
dos retornos é $0\%$ — e você teria concluído, erradamente, que ficou no zero a
zero. Na verdade $1{,}5 \times 0{,}5 = 0{,}75$: você perdeu 25%. A média
geométrica dos fatores $(1{,}5 \times 0{,}5)^{1/2} \approx 0{,}866$ acerta a
resposta.

**Média harmônica** — para taxas por unidade fixa (velocidade, throughput):

$$
H = \frac{n}{\sum_{i=1}^{n} 1/x_i}
$$

Vale sempre a desigualdade $H \le G \le \bar{x}$, com igualdade apenas se todos
os valores forem idênticos. O F1-score, aliás, é exatamente a média harmônica
entre precisão e recall — e é harmônica justamente para penalizar o desequilíbrio
entre as duas.

**Média aparada** (*trimmed mean*) — descarta os $\alpha\%$ extremos de cada cauda
antes de promediar. É o meio-termo controlado entre média e mediana, e é o que o
Comitê Olímpico usa nas notas de ginástica pelo mesmo motivo que um cientista de
dados deveria usar: neutralizar juízes extremos sem jogar fora toda a informação.

> [!MERCADO] O Banco Central do Brasil publica o **IPCA com médias aparadas**
> como medida de núcleo de inflação, precisamente porque a média cheia é
> sequestrada por choques pontuais (uma geada que triplica o preço do tomate não
> é inflação estrutural). Esse é o mesmo raciocínio que você deve aplicar a
> métricas de produto: o pico de um bug não é o comportamento do sistema.

---

## Medidas de dispersão

### Variância, desvio-padrão e a correção de Bessel

$$
s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2
\qquad\qquad
\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2
$$

A pergunta que todo aluno faz — *por que $n-1$?* — tem uma resposta precisa. Se
usássemos $n$, o estimador seria **viesado para baixo**. A intuição: a soma de
quadrados é calculada em torno de $\bar{x}$, que é ela própria estimada dos
mesmos dados e portanto está "puxada" para o centro da amostra. Os desvios em
torno de $\bar{x}$ são sistematicamente menores do que seriam em torno do $\mu$
verdadeiro, que você não conhece.

Formalmente, $\mathbb{E}\left[\sum (x_i - \bar{x})^2\right] = (n-1)\sigma^2$, e
dividir por $n-1$ corrige exatamente esse fator. Diz-se que a amostra tem $n-1$
**graus de liberdade**: uma vez fixada a média, o último desvio fica determinado
pelos outros $n-1$, porque $\sum (x_i - \bar{x}) = 0$.

> [!ARMADILHA] `numpy.var()` usa `ddof=0` (divide por $n$) e `pandas.Series.var()`
> usa `ddof=1` (divide por $n-1$). As duas bibliotecas mais usadas do ecossistema
> discordam no default. Em $n$ grande a diferença é irrelevante; em amostras
> pequenas — validação cruzada com poucos folds, testes A/B em segmentos
> pequenos, calibração por estrato — ela muda a conclusão. Sempre passe `ddof`
> explicitamente.

### Alternativas robustas

O desvio-padrão herda a fragilidade da média, e piora: os desvios são elevados ao
quadrado. Duas alternativas:

**Amplitude interquartil (IQR)**, $\mathrm{IQR} = Q_3 - Q_1$, cobre os 50%
centrais dos dados e ignora completamente as caudas.

**Desvio absoluto mediano (MAD)**:

$$
\mathrm{MAD} = \mathrm{mediana}\left( \left| x_i - \mathrm{mediana}(x) \right| \right)
$$

Para dados aproximadamente normais, $1{,}4826 \times \mathrm{MAD}$ estima o mesmo
que $\sigma$ — a constante existe só para tornar as duas escalas comparáveis. A
diferença é a robustez, que se mede pelo **ponto de ruptura** (*breakdown
point*): a fração de observações que podem ser corrompidas arbitrariamente sem
que o estimador vá para o infinito.

| Estimador | Ponto de ruptura | Leitura prática |
|---|---|---|
| Média | $0\%$ | **Um único** ponto contaminado destrói a estimativa |
| Média aparada 10% | $10\%$ | Aguenta 10% de lixo em cada cauda |
| Mediana | $50\%$ | Aguenta metade dos dados corrompidos |
| Desvio-padrão | $0\%$ | Idem à média, e mais sensível ainda |
| MAD / IQR | $50\%$ | Referência para detecção de outliers |

### Coeficiente de variação

$$
\mathrm{CV} = \frac{s}{\bar{x}}
$$

Adimensional, permite comparar dispersão entre grandezas de escalas diferentes
("a variabilidade do ticket médio é maior que a do tempo de sessão?"). Só faz
sentido para variáveis de razão com $\bar{x} > 0$ — aplicar CV a temperatura em
Celsius ou a uma variável que troca de sinal produz um número sem significado.

---

## Momentos: assimetria e curtose

Os momentos centrais padronizados descrevem o **formato** da distribuição, indo
além de centro e espalhamento.

$$
\tilde{\mu}_k = \mathbb{E}\left[ \left( \frac{X - \mu}{\sigma} \right)^{k} \right]
$$

**Assimetria** (*skewness*), $k = 3$: mede o desequilíbrio entre as caudas.
Positiva significa cauda longa à direita — o caso típico de receita, renda, tempo
de resposta e valor de sinistro, em que a maioria é pequena e alguns poucos são
enormes. Quando a assimetria é positiva, vale a ordenação
$\mathrm{moda} < \mathrm{mediana} < \mathrm{média}$, e é exatamente por isso que
"a maioria das pessoas ganha menos que a média".

**Curtose** (*kurtosis*), $k = 4$: mede o peso das caudas. A normal tem curtose
$3$; por isso se reporta a **curtose excedente** $\tilde{\mu}_4 - 3$, que é zero
para a normal. Excedente positivo indica caudas mais pesadas que a normal — mais
eventos extremos do que o modelo gaussiano prevê.

> [!MERCADO] A crise de 2008 é, em boa medida, um erro de curtose. Modelos de
> risco calibravam VaR supondo retornos normais. Sob normalidade, uma queda de 5
> desvios-padrão acontece uma vez a cada milhão de dias úteis — ou seja,
> "nunca". Nos dados reais de mercado, com curtose excedente alta, esse evento
> ocorre a cada poucos anos. O modelo não estava errado por pouco: estava errado
> por várias ordens de magnitude, e só na cauda — justamente onde o dinheiro
> estava.

> [!ARMADILHA] Assimetria e curtose amostrais são estimadores de **alta
> variância**: dependem de potências terceira e quarta, dominadas pelos pontos
> extremos. Com $n < 300$ eles são ruidosos a ponto de serem quase inúteis
> isoladamente. Use-os como sinal de alerta acompanhado de histograma e QQ-plot,
> nunca como veredito.

---

## Quantis, percentis e o mundo real de SLA

O quantil de ordem $q \in (0,1)$ é o valor $x_q$ tal que uma fração $q$ dos dados
fica abaixo dele. Formalmente, via função de distribuição acumulada:

$$
x_q = F^{-1}(q) = \inf \{ x : F(x) \ge q \}
$$

Em dados discretos, esse ínfimo raramente cai exatamente sobre uma observação, e
existem **nove definições distintas de quantil amostral** na literatura
(implementadas em `numpy.quantile(..., method=...)`). Para $n$ grande a escolha é
irrelevante; para $n$ pequeno, dois times podem reportar p95 diferentes a partir
dos mesmos dados e passar uma tarde discutindo.

### Por que engenharia mede latência em p99, e não em média

Um serviço com latência média de 100 ms parece saudável. Mas se a distribuição
tem cauda longa, o p99 pode ser 2 000 ms. Isso significa que **1 em cada 100
requisições** demora dois segundos.

O ponto crítico, e frequentemente ignorado: uma única tela de aplicativo costuma
disparar dezenas de chamadas. Se uma página faz 50 requisições independentes, a
probabilidade de **pelo menos uma** cair na cauda do p99 é

$$
1 - (1 - 0{,}01)^{50} \approx 39{,}5\%
$$

Quase 40% dos carregamentos de página sofrem o pior caso do p99. A média nunca
revelaria isso — e é por essa razão que SLAs sérios são escritos em percentis
altos, nunca em média.

> [!ANALOGIA] Planejar capacidade pela média é como comprar um colete
> salva-vidas para a profundidade média de um rio. A profundidade média é de 1,20
> m; você se afoga no trecho de 3 m.

---

## Relação entre variáveis

### Covariância e correlação de Pearson

$$
\mathrm{Cov}(X,Y) = \mathbb{E}\left[ (X - \mu_X)(Y - \mu_Y) \right]
\qquad
\rho = \frac{\mathrm{Cov}(X,Y)}{\sigma_X \sigma_Y}
$$

A covariância tem unidade (reais × minutos, por exemplo), o que a torna
incomparável entre pares de variáveis. Pearson normaliza para $[-1, 1]$.

Duas limitações precisam ficar absolutamente claras:

1. **Pearson mede apenas relação linear.** Se $Y = X^2$ com $X$ simétrico em
   torno de zero, então $\rho = 0$ — e a dependência é total. Correlação zero
   **não** é independência.
2. **Pearson é frágil a outliers**, pela mesma razão que a média é: entram
   produtos de desvios.

### Spearman e Kendall

**Spearman** é o Pearson calculado sobre os **postos** (*ranks*). Captura
qualquer relação monotônica, não só linear, e é robusto a outliers, porque o
maior valor vira apenas "posto $n$", não importa quão distante esteja.

**Kendall tau** conta pares concordantes e discordantes:

$$
\tau = \frac{C - D}{\binom{n}{2}}
$$

É mais interpretável (é uma probabilidade de concordância) e mais estável em
amostras pequenas, ao custo de ser mais caro computacionalmente.

| Situação | Use |
|---|---|
| Relação linear, sem outliers, dados contínuos | Pearson |
| Relação monotônica não-linear, ou outliers presentes | Spearman |
| Amostra pequena, ou muitos empates | Kendall |
| Relação não-monotônica (U, ciclo) | Nenhum dos três — use *mutual information* ou gráfico |

> [!ARMADILHA] A matriz de correlação é a ferramenta de EDA mais usada e mais mal
> usada do mercado. Um `heatmap` de Pearson não vê relações em U, não vê
> interações, é distorcido por outliers e diz *nada* sobre causalidade. Trate-a
> como um índice para decidir quais pares merecem um *scatter plot* — nunca como
> conclusão.

### O quarteto de Anscombe

Quatro conjuntos de dados com **médias, variâncias, correlação e reta de
regressão idênticas até a segunda casa decimal** — e com formatos completamente
diferentes: um linear, um quadrático, um linear com um outlier, e um que é uma
coluna vertical mais um ponto isolado. É a demonstração definitiva de que
estatísticas-resumo não substituem o gráfico. O notebook 03 reproduz o quarteto e
sua versão moderna, o *Datasaurus Dozen*.

---

## Padronização e escala

O **z-score** recentra e reescala:

$$
z_i = \frac{x_i - \bar{x}}{s}
$$

Após a transformação, $\bar{z} = 0$ e $s_z = 1$. É requisito para todo algoritmo
sensível a distância ou a magnitude de coeficientes: k-NN, k-means, SVM, PCA,
regressão regularizada e redes neurais. Árvores e ensembles de árvores são
invariantes a transformações monotônicas e não precisam.

> [!ARMADILHA] O erro de vazamento mais comum do mercado: calcular média e
> desvio-padrão **no conjunto completo** e só depois separar treino e teste. A
> estatística de padronização carrega informação do teste para dentro do treino,
> e a performance reportada fica otimista. O ajuste (`fit`) pertence
> exclusivamente ao treino. Este é o tema central do módulo *Encoding, Escala e
> Vazamento de Dados*.

---

## Um protocolo de descrição que funciona

Sequência que eu recomendo aplicar a toda variável nova, nesta ordem:

1. **Tipo e cardinalidade.** Qual é o dtype? Quantos valores únicos? Uma coluna
   "numérica" com 4 valores distintos é categórica disfarçada.
2. **Completude.** Qual o percentual de ausentes? Há um valor sentinela
   disfarçado (`-999`, `0`, `1900-01-01`) fazendo o papel de nulo?
3. **Posição e dispersão robustas juntas.** Reporte média **e** mediana. Se
   divergem muito, há assimetria ou contaminação — investigue antes de seguir.
4. **Quantis, não só o desvio-padrão.** Mínimo, p1, p25, p50, p75, p99, máximo. O
   par (p1, p99) revela cauda que o desvio-padrão esconde.
5. **Formato.** Histograma com número de *bins* variado, mais um *boxplot* ou
   *violin*. Bimodalidade quase sempre significa duas populações misturadas — e
   duas populações misturadas quase sempre pedem dois modelos.
6. **Plausibilidade de domínio.** Idade negativa, receita de R$ 0,00 em pedido
   fechado, timestamp no futuro. Nenhuma estatística substitui olhar 30 linhas
   cruas.

> [!MERCADO] Em produção, esse mesmo protocolo vira **monitoramento**. Perfis
> descritivos (média, quantis, taxa de nulos, cardinalidade) de cada feature são
> calculados por janela e comparados com a janela de treino. Quando o p99 de uma
> feature dobra, o modelo já está degradando — e você descobre isso *antes* do
> impacto aparecer na métrica de negócio, que costuma ter semanas de atraso.
> Retomamos isso no tema *MLOps e Produção*.

---

## Formulário do capítulo

> [!FORMULA] **Posição** — média $\bar{x} = \frac{1}{n}\sum x_i$; mediana =
> minimiza $\sum |x_i - c|$; geométrica $G = (\prod x_i)^{1/n}$; harmônica
> $H = n / \sum (1/x_i)$; vale $H \le G \le \bar{x}$.
>
> **Dispersão** — variância amostral $s^2 = \frac{1}{n-1}\sum (x_i - \bar{x})^2$;
> $\mathrm{IQR} = Q_3 - Q_1$; $\mathrm{MAD} = \mathrm{mediana}(|x_i -
> \mathrm{mediana}(x)|)$; $\mathrm{CV} = s / \bar{x}$.
>
> **Formato** — assimetria $\tilde{\mu}_3$; curtose excedente $\tilde{\mu}_4 -
> 3$; para assimetria positiva, $\mathrm{moda} < \mathrm{mediana} < \bar{x}$.
>
> **Associação** — $\rho = \mathrm{Cov}(X,Y) / (\sigma_X \sigma_Y)$; Spearman =
> Pearson sobre postos; $\tau = (C - D) / \binom{n}{2}$.
>
> **Padronização** — $z_i = (x_i - \bar{x}) / s$; ajuste **só** no treino.

## Erros que custam caro — checklist

- Reportar média sem mediana em variável assimétrica.
- Calcular média de variável ordinal sem justificar a equidistância.
- Usar `ddof` default sem verificar qual biblioteca está sendo chamada.
- Concluir independência a partir de correlação zero.
- Definir SLA em média quando o usuário sente a cauda.
- Padronizar antes de separar treino e teste.
- Interpretar assimetria/curtose amostrais com $n$ pequeno.
- Confiar na matriz de correlação sem olhar um único *scatter plot*.

## Para ir além

- Tukey, *Exploratory Data Analysis* (1977) — a origem do boxplot e da filosofia
  de olhar antes de testar.
- Huber & Ronchetti, *Robust Statistics* — formalização de ponto de ruptura.
- Wilke, *Fundamentals of Data Visualization* — disponível gratuitamente online.
- Taleb, *Statistical Consequences of Fat Tails* — denso, e o melhor tratamento
  de por que a intuição gaussiana falha.
