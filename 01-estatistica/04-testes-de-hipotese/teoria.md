<!-- tema: Estatística > Testes de Hipótese -->
<!-- subtitulo: Decidir sob incerteza com uma régua explícita para o erro que você aceita cometer -->
<!-- resumo: Um teste de hipótese não descobre a verdade: ele controla a taxa de um tipo específico de erro. Este material constrói a lógica do teste desde o princípio, apresenta o catálogo de testes clássicos com seus pressupostos, trata poder e tamanho de amostra como problema de projeto, e enfrenta o problema das comparações múltiplas — a razão silenciosa pela qual tanta descoberta não se reproduz. -->
<!-- nivel: Intermediário -->
<!-- prerequisitos: Módulos 01 a 03 (Descritiva, Distribuições, Inferência) -->
<!-- duracao: 10 a 12 horas (leitura + 4 notebooks) -->
<!-- notebooks: 01-logica-do-teste · 02-testes-classicos · 03-poder-e-tamanho-de-amostra · 04-comparacoes-multiplas -->
<!-- autor: Material do curso de ML & DL -->
<!-- versao: 1.0 -->

# Testes de Hipótese

## O que um teste de hipótese realmente faz

Comece por descartar a intuição errada. Um teste de hipótese **não** responde
"minha hipótese é verdadeira?" nem "qual a probabilidade de eu estar certo?".

Ele responde a uma pergunta muito mais estreita e muito mais estranha:

> *Se não houvesse efeito nenhum, com que frequência eu veria um resultado tão
> extremo quanto o que vi?*

Se essa frequência for suficientemente baixa, você declara que a hipótese de
"efeito nenhum" é uma explicação pouco confortável para o que observou — e a
rejeita. É um argumento por contradição probabilística, não uma prova.

> [!ANALOGIA] O teste de hipótese é um **júri**, não um detector de verdade. A
> hipótese nula é a presunção de inocência. Você não "prova a inocência"; você
> apenas decide se há evidência **além da dúvida razoável** para condenar. O
> nível $\alpha$ é a definição operacional de "dúvida razoável" — e é você quem
> escolhe onde colocá-la, com base no custo de condenar um inocente.

### As duas hipóteses

- **Hipótese nula ($H_0$)**: o estado de coisas que se assume por padrão.
  Normalmente "não há efeito", "as médias são iguais", "o coeficiente é zero".
- **Hipótese alternativa ($H_1$)**: o que você suspeita.

$H_0$ é sempre a hipótese **específica** — a que permite calcular a distribuição
da estatística de teste. É por isso que você nunca "aceita $H_0$": não encontrar
evidência contra não é o mesmo que ter evidência a favor.

### Os dois erros

| | $H_0$ é verdadeira | $H_0$ é falsa |
|---|---|---|
| **Rejeita $H_0$** | **Erro Tipo I** ($\alpha$) — falso positivo | Decisão correta (poder $= 1-\beta$) |
| **Não rejeita $H_0$** | Decisão correta | **Erro Tipo II** ($\beta$) — falso negativo |

O ponto que define o método: **você escolhe $\alpha$ antes de ver os dados**. O
0,05 não tem nada de sagrado — foi uma conveniência de Fisher em 1925 que virou
tradição. Em física de partículas usa-se $5\sigma$ ($\alpha \approx 3 \times
10^{-7}$); em triagem médica exploratória usa-se 0,10.

> [!ARMADILHA] $\alpha$ e $\beta$ trocam entre si. Reduzir $\alpha$ (menos falsos
> positivos) aumenta $\beta$ (mais falsos negativos), com $n$ fixo. A única
> forma de melhorar os dois ao mesmo tempo é **aumentar $n$** ou **reduzir a
> variância**. Toda discussão sobre "qual $\alpha$ usar" é, no fundo, uma
> discussão sobre qual dos dois erros custa mais caro no seu contexto.

---

## O p-valor: definição precisa e usos errados

$$
p = P\left( \text{estatística tão ou mais extrema que a observada} \mid H_0 \right)
$$

Cada palavra importa. O p-valor é uma probabilidade **condicional a $H_0$ ser
verdadeira**, calculada sobre **dados hipotéticos** que você não observou.

### O que o p-valor NÃO é

| Interpretação errada | Por que está errada |
|---|---|
| "Probabilidade de $H_0$ ser verdadeira" | Isso é $P(H_0 \mid \text{dados})$; o p-valor é $P(\text{dados} \mid H_0)$ |
| "Probabilidade de o resultado ser sorte" | Confunde as duas condicionais acima |
| "$p = 0{,}04$ é o dobro de forte que $p = 0{,}08$" | O p-valor não é uma medida de tamanho de efeito |
| "$p > 0{,}05$ prova que não há efeito" | Ausência de evidência não é evidência de ausência |
| "$p < 0{,}05$ significa que o efeito é importante" | Significância estatística $\neq$ relevância prática |

A última linha é a mais cara de todas no mercado.

> [!MERCADO] Com $n = 10$ milhões de usuários, uma diferença de 0,003 p.p. na
> taxa de clique sai com $p < 0{,}001$. É estatisticamente significativa e
> comercialmente irrelevante — o efeito não paga nem o custo de manter o código
> da variante. **Sempre reporte o tamanho do efeito com seu intervalo de
> confiança**, e defina o *efeito mínimo relevante* antes do experimento.

### A relação com intervalos de confiança

Um teste bilateral ao nível $\alpha$ rejeita $H_0: \theta = \theta_0$ **se e
somente se** o intervalo de confiança de $(1-\alpha)$ não contém $\theta_0$. São
duas faces da mesma moeda — mas o intervalo é estritamente mais informativo,
porque mostra a magnitude e a precisão, não só a decisão.

---

## O catálogo de testes

### Comparação de médias

| Teste | Situação | Pressupostos-chave |
|---|---|---|
| $t$ de uma amostra | Média vs. valor de referência | Normalidade da média (TLC) |
| $t$ de Welch | Duas amostras independentes | **Não** exige variâncias iguais |
| $t$ pareado | Duas medidas do mesmo sujeito | Normalidade das diferenças |
| ANOVA | Três ou mais grupos | Normalidade, homocedasticidade |
| Mann-Whitney U | Duas amostras, não-paramétrico | Testa deslocamento de distribuição |
| Wilcoxon | Pareado, não-paramétrico | Simetria das diferenças |
| Kruskal-Wallis | 3+ grupos, não-paramétrico | — |

> [!NOTA] **Use Welch por padrão**, não o teste t de Student clássico. O teste de
> Student exige variâncias iguais; o de Welch não, e perde quase nada de poder
> quando elas são iguais. A prática de "primeiro testar se as variâncias são
> iguais e depois escolher o teste" é pior do que simplesmente usar Welch sempre
> — o pré-teste distorce a taxa de erro do teste principal.

### Proporções e tabelas de contingência

- **Teste z de duas proporções** — o padrão de A/B testing.
- **Qui-quadrado de independência** — para tabelas $r \times c$. Requer
  frequências esperadas $\geq 5$ em quase todas as células.
- **Exato de Fisher** — para tabelas pequenas, onde o qui-quadrado falha.

### Aderência e normalidade

- **Kolmogorov-Smirnov**, **Anderson-Darling**, **Shapiro-Wilk**.

> [!ARMADILHA] Todo teste de normalidade tem poder crescente com $n$. Com $n$
> pequeno ele aceita tudo (inclusive dados claramente não-normais); com $n$
> grande ele rejeita tudo (porque nenhum dado real é exatamente normal). Ele
> responde "é *exatamente* normal?" quando a pergunta útil é "o desvio importa
> para a minha análise?". **Use QQ-plot.**

---

## Poder estatístico

O **poder** é a probabilidade de detectar um efeito que realmente existe:

$$
\text{Poder} = 1 - \beta = P\left( \text{rejeitar } H_0 \mid H_1 \text{ verdadeira} \right)
$$

Ele depende de quatro quantidades, e fixadas três, a quarta fica determinada:

1. **$\alpha$** — nível de significância.
2. **$n$** — tamanho da amostra.
3. **Tamanho do efeito** — a magnitude que você quer detectar.
4. **Variabilidade** — o ruído dos dados.

Para comparação de duas médias, o tamanho de efeito padronizado é o **d de
Cohen**:

$$
d = \frac{\mu_1 - \mu_0}{\sigma}
$$

e o $n$ por grupo, para $\alpha$ e poder dados, é aproximadamente

$$
n \approx \frac{2 \left( z_{1-\alpha/2} + z_{1-\beta} \right)^2}{d^2}
$$

Com $\alpha = 0{,}05$ e poder de 80%, o numerador vale $\approx 15{,}7$, então
$n \approx 15{,}7 / d^2$ por grupo. Um efeito pequeno ($d = 0{,}2$) exige
$\approx 393$ por grupo; um efeito muito pequeno ($d = 0{,}05$) exige
$\approx 6.300$.

> [!ARMADILHA] **Análise de poder post-hoc é inútil.** Calcular o poder *depois*
> do experimento usando o efeito *observado* não acrescenta informação nenhuma:
> ele é uma função determinística do p-valor obtido. Se deu não-significativo, o
> poder observado será baixo — sempre, por construção. A análise de poder tem de
> ser feita **antes**, com o efeito mínimo relevante definido pelo negócio.

### A maldição do estudo subdimensionado

Um experimento com poder baixo tem um problema pior do que "não detectar nada":
quando ele *detecta* algo, a estimativa é **exagerada**. Só efeitos anormalmente
grandes conseguem cruzar o limiar de significância com pouca amostra. Isso se
chama **erro de tipo M** (magnitude), e é uma das causas centrais da crise de
replicação nas ciências sociais e biomédicas.

---

## Comparações múltiplas

Se você testa 20 hipóteses independentes ao nível $\alpha = 0{,}05$ e nenhuma é
verdadeira, a chance de pelo menos um falso positivo é

$$
1 - (1 - 0{,}05)^{20} \approx 64\%
$$

Com 100 testes, sobe para 99,4%. Testar muitas coisas **garante** achar algo.

> [!ANALOGIA] É o atirador texano: dispara na parede do celeiro e depois desenha
> o alvo em volta do maior aglomerado de furos. A regra de ouro é declarar o alvo
> **antes** de atirar.

### Correções

| Método | Controla | Comportamento |
|---|---|---|
| **Bonferroni** — use $\alpha/m$ | FWER | Simples, muito conservador |
| **Holm** | FWER | Uniformemente melhor que Bonferroni; use este |
| **Benjamini-Hochberg** | FDR | Menos conservador; padrão em exploração |

A distinção entre **FWER** (probabilidade de *qualquer* falso positivo) e **FDR**
(proporção esperada de falsos positivos entre as descobertas) é a decisão de
projeto principal. Para uma decisão confirmatória única, controle FWER. Para
triagem exploratória de milhares de hipóteses (genômica, seleção de features,
varredura de segmentos), controle FDR.

> [!MERCADO] O caso clássico em produto: rodar um teste A/B e, ao não achar
> efeito na métrica principal, procurar em 30 segmentos até um deles dar
> significativo ("funcionou para mulheres de 25-34 no Android!"). Isso é
> *p-hacking* por *slicing*, e a taxa real de falso positivo é próxima de 80%.
> A análise por segmento precisa ser **pré-registrada** e **corrigida**.

### Outras formas de p-hacking

- **Espiar continuamente** e parar quando fica significativo. Isso infla o erro
  tipo I para bem acima de 50% com espiadas frequentes. A solução são métodos
  sequenciais próprios (limites de O'Brien-Fleming, testes sempre válidos).
- **Testar várias métricas** e reportar a que funcionou.
- **Escolher a transformação, a exclusão de outliers ou os covariáveis depois de
  ver o resultado** — os "graus de liberdade do pesquisador".

---

## Formulário do capítulo

> [!FORMULA] **p-valor** $= P(\text{estatística tão extrema} \mid H_0)$.
>
> **$t$ de Welch**:
> $t = (\bar{x}_1 - \bar{x}_2) / \sqrt{s_1^2/n_1 + s_2^2/n_2}$.
>
> **$z$ de duas proporções**:
> $z = (\hat{p}_1 - \hat{p}_2) / \sqrt{\hat{p}(1-\hat{p})(1/n_1 + 1/n_2)}$.
>
> **d de Cohen** $= (\mu_1 - \mu_0)/\sigma$; convenção: 0,2 pequeno, 0,5 médio,
> 0,8 grande.
>
> **Tamanho de amostra** (2 médias, 80% de poder, $\alpha=0{,}05$):
> $n \approx 15{,}7/d^2$ por grupo.
>
> **FWER com $m$ testes** $= 1 - (1-\alpha)^m$;
> **Bonferroni**: use $\alpha/m$.

## Erros que custam caro — checklist

- Interpretar o p-valor como probabilidade de $H_0$.
- Reportar significância sem tamanho de efeito e intervalo.
- "Aceitar $H_0$" a partir de $p > 0{,}05$.
- Escolher o teste depois de ver os dados.
- Espiar o experimento e parar ao atingir significância.
- Analisar dezenas de segmentos sem correção.
- Fazer análise de poder post-hoc.
- Usar teste de normalidade com $n$ grande como critério de decisão.

## Para ir além

- Wasserstein & Lazar (2016), *ASA Statement on p-Values* — a declaração oficial
  da American Statistical Association sobre os usos indevidos.
- Gelman & Carlin (2014), *Beyond Power Calculations* — erros de tipo S e M.
- Ioannidis (2005), *Why Most Published Research Findings Are False*.
- Kohavi, Tang & Xu, *Trustworthy Online Controlled Experiments* — a referência
  prática de A/B testing em escala industrial.
