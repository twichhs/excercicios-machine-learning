<!-- tema: Estatística > Inferência e Estimação -->
<!-- subtitulo: De uma amostra para a população — e o preço honesto dessa extrapolação -->
<!-- resumo: Inferência é o ato de afirmar algo sobre o que você não viu, a partir do pouco que viu. Este material constrói o aparato inteiro: estimadores e suas propriedades, máxima verossimilhança, erro-padrão, intervalos de confiança e bootstrap — com atenção especial ao que um intervalo de confiança realmente promete (e ao que ele decididamente não promete). -->
<!-- nivel: Intermediário — todo o cálculo necessário é construído no texto -->
<!-- prerequisitos: Módulos 01 e 02 (Descritiva e Distribuições) -->
<!-- duracao: 8 a 10 horas (leitura + 3 notebooks) -->
<!-- notebooks: 01-estimadores-e-propriedades · 02-maxima-verossimilhanca · 03-intervalos-e-bootstrap -->
<!-- autor: Material do curso de ML & DL -->
<!-- versao: 1.0 -->

# Inferência e Estimação

## O problema, formulado com precisão

Você tem 3.000 usuários de um total de 4 milhões. Mediu a taxa de conversão
desses 3.000 e obteve 4,2%. A pergunta de negócio é sobre os 4 milhões, não sobre
os 3.000.

Formalmente: existe uma **população** governada por um parâmetro desconhecido
$\theta$ (uma taxa, uma média, um coeficiente). Você observa uma **amostra**
$X_1, \dots, X_n$ e calcula uma **estatística** $\hat{\theta}$ — uma função dos
dados. A inferência é o conjunto de técnicas que responde: *o que $\hat{\theta}$
autoriza dizer sobre $\theta$?*

> [!DEFINICAO] Um **estimador** é uma função da amostra:
> $\hat{\theta} = g(X_1, \dots, X_n)$. Como a amostra é aleatória, o estimador é
> **ele próprio uma variável aleatória** — ele tem uma distribuição, chamada
> **distribuição amostral**. Compreender inferência é compreender essa
> distribuição.

> [!ANALOGIA] Estimar é como avaliar a temperatura de uma piscina inteira
> enfiando um dedo em um ponto. O valor que você sente é uma estimativa. Ela tem
> dois defeitos possíveis: um **viés** (se seu dedo está sempre frio, você
> subestima sempre) e uma **variância** (se a piscina tem correntes, cada ponto
> dá um número diferente). Inferência é quantificar esses dois defeitos.

---

## Propriedades de um bom estimador

### Viés

$$
\text{Viés}(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta
$$

Um estimador é **não-viesado** se acerta *na média* ao longo de infinitas
amostras. É a propriedade que motiva o $n-1$ da variância amostral, vista no
módulo 01.

Um alerta importante: **não-viesado não significa bom**. Um estimador pode ser
não-viesado e ter variância tão alta que cada estimativa individual é inútil.

### Variância e erro quadrático médio

$$
\text{EQM}(\hat{\theta}) = \mathbb{E}\left[ (\hat{\theta} - \theta)^2 \right]
= \text{Var}(\hat{\theta}) + \text{Viés}(\hat{\theta})^2
$$

Essa decomposição é uma das identidades mais importantes de toda a estatística
aplicada — e é literalmente a mesma decomposição viés-variância que reaparece em
machine learning. Ela diz que **aceitar um pouco de viés em troca de muito menos
variância pode reduzir o erro total**.

> [!MERCADO] Regressão Ridge é exatamente esse trade. O estimador de mínimos
> quadrados é não-viesado; a Ridge é viesada de propósito. Em problemas com
> muitas variáveis correlacionadas, a Ridge tem EQM *menor* — ela erra menos, na
> prática, apesar de ser viesada. O mesmo raciocínio justifica *shrinkage* em
> estimativas por segmento (empréstimo de força entre grupos pequenos).

### Consistência e eficiência

**Consistência**: $\hat{\theta} \to \theta$ conforme $n \to \infty$. É o requisito
mínimo — um estimador inconsistente não melhora com mais dados.

**Eficiência**: entre estimadores não-viesados, o de menor variância. Existe um
piso teórico para essa variância, o **limite de Cramér-Rao**:

$$
\text{Var}(\hat{\theta}) \geq \frac{1}{n \, I(\theta)}
$$

onde $I(\theta)$ é a **informação de Fisher**, que mede quanto uma observação
carrega de informação sobre $\theta$:

$$
I(\theta) = \mathbb{E}\left[ \left( \frac{\partial}{\partial \theta}
\log f(X \mid \theta) \right)^2 \right]
$$

A leitura intuitiva: a informação de Fisher mede quão *pontuda* é a
verossimilhança. Verossimilhança pontuda significa que os dados distinguem bem
valores próximos de $\theta$ — muita informação, pouca variância possível.

---

## Erro-padrão: a medida da incerteza

O **erro-padrão** é o desvio-padrão da distribuição amostral do estimador. Para a
média:

$$
\text{EP}(\bar{X}) = \frac{\sigma}{\sqrt{n}}
\qquad\qquad
\widehat{\text{EP}}(\bar{X}) = \frac{s}{\sqrt{n}}
$$

> [!ARMADILHA] Desvio-padrão e erro-padrão são coisas diferentes e a confusão
> entre os dois é epidêmica. O **desvio-padrão** $s$ descreve o espalhamento dos
> **dados** e não diminui com $n$ — coletar mais dados não torna as pessoas mais
> parecidas. O **erro-padrão** $s/\sqrt{n}$ descreve a precisão da **estimativa**
> e tende a zero. Um gráfico com barras de erro precisa dizer qual dos dois está
> mostrando; barras de erro-padrão são ~$\sqrt{n}$ vezes menores e fazem qualquer
> diferença parecer dramática.

Para proporções:

$$
\widehat{\text{EP}}(\hat{p}) = \sqrt{\frac{\hat{p}(1 - \hat{p})}{n}}
$$

---

## Máxima verossimilhança

É o método de estimação mais importante que existe. Regressão logística, GLMs,
modelos de sobrevivência, redes neurais treinadas com entropia cruzada — todos
são máxima verossimilhança por baixo do capô.

### A ideia

A **função de verossimilhança** é a probabilidade dos dados observados, vista
como função do parâmetro:

$$
L(\theta) = \prod_{i=1}^{n} f(x_i \mid \theta)
$$

O estimador de máxima verossimilhança (MLE) é o $\theta$ que torna os dados
observados os mais prováveis possíveis:

$$
\hat{\theta}_{\text{MLE}} = \underset{\theta}{\arg\max} \; L(\theta)
$$

Na prática, maximiza-se a **log-verossimilhança**, porque produtos viram somas
(numericamente estáveis e algebricamente tratáveis):

$$
\ell(\theta) = \sum_{i=1}^{n} \log f(x_i \mid \theta)
$$

> [!ANALOGIA] Você encontra uma pegada na areia. A verossimilhança pergunta: para
> cada animal possível, qual a chance de ele deixar *exatamente* esta pegada? O
> MLE é o animal que torna a pegada menos surpreendente. Note a inversão: não
> perguntamos "qual animal é mais provável?" (isso seria bayesiano, e exigiria
> saber quais animais existem na região) — perguntamos "sob qual animal esta
> pegada seria mais esperada?".

### Por que o MLE é o método padrão

Sob condições de regularidade, o MLE tem três propriedades assintóticas:

1. **Consistente**: $\hat{\theta} \to \theta$.
2. **Assintoticamente normal**:
   $\hat{\theta} \approx \mathcal{N}\left( \theta, \frac{1}{n I(\theta)} \right)$.
3. **Assintoticamente eficiente**: atinge o limite de Cramér-Rao.

A propriedade 2 é o motor prático de tudo: é dela que saem os erros-padrão e os
p-valores reportados por `statsmodels`, por `sklearn` e por qualquer software
estatístico.

**Invariância**: se $\hat{\theta}$ é o MLE de $\theta$, então $g(\hat{\theta})$ é
o MLE de $g(\theta)$. Isso permite estimar odds ratio a partir do MLE dos
coeficientes, sem refazer nada.

> [!ARMADILHA] O MLE é **viesado** em amostras finitas. O exemplo canônico: o MLE
> da variância divide por $n$, não por $n-1$ — é exatamente o estimador viesado
> do módulo 01. As boas propriedades do MLE são *assintóticas*; com $n$ pequeno
> elas podem não valer.

---

## Intervalos de confiança

### A definição, com a interpretação correta

Um intervalo de confiança de 95% para $\theta$ é um intervalo aleatório
$[\hat{L}, \hat{U}]$ construído de modo que

$$
P\left( \hat{L} \leq \theta \leq \hat{U} \right) = 0{,}95
$$

onde a probabilidade é sobre **amostras repetidas**, não sobre $\theta$.

> [!DEFINICAO] A interpretação correta: *se eu repetisse este experimento
> infinitas vezes e construísse um intervalo a cada vez, 95% desses intervalos
> conteriam o valor verdadeiro.*
>
> A interpretação **incorreta**, e quase universal: "há 95% de chance de $\theta$
> estar neste intervalo". No paradigma frequentista, $\theta$ é uma constante
> fixa — ela está ou não está dentro do seu intervalo específico, com
> probabilidade 0 ou 1. O que tem 95% de probabilidade é o **procedimento**, não
> o intervalo particular que você calculou.
>
> Se a segunda interpretação é a que você quer (e geralmente é), a ferramenta
> correta é o **intervalo de credibilidade bayesiano** — módulo 05.

### Construção para a média

$$
\bar{x} \pm t_{\alpha/2, \, n-1} \cdot \frac{s}{\sqrt{n}}
$$

Usa-se a distribuição $t$ porque $\sigma$ foi estimado a partir dos dados. Com
$n > 30$ a diferença para a normal é desprezível.

### Para proporções: por que a fórmula do livro é ruim

O intervalo de Wald,
$\hat{p} \pm z\sqrt{\hat{p}(1-\hat{p})/n}$, é o que todo mundo aprende e tem
comportamento **ruim** quando $p$ é próximo de 0 ou 1, ou quando $n$ é pequeno.
Ele pode produzir limites fora de $[0,1]$ e sua cobertura real fica bem abaixo
dos 95% nominais.

| Método | Quando usar |
|---|---|
| Wald | Praticamente nunca; só didático |
| **Wilson** | Padrão recomendado — bom em todo regime |
| Agresti-Coull | Simples: some 2 sucessos e 2 fracassos e use Wald |
| Clopper-Pearson | Exato e conservador; auditoria e regulação |

> [!MERCADO] Em testes A/B com conversões raras (0,5% ou menos), o intervalo de
> Wald sistematicamente sobrestima a certeza. Times de produto declaram vitórias
> que não se sustentam. A troca para Wilson custa uma linha de código
> (`statsmodels.stats.proportion.proportion_confint(..., method="wilson")`) e
> corrige o problema.

---

## Bootstrap: inferência sem fórmula

Nem toda estatística tem fórmula fechada para o erro-padrão. Qual é o
erro-padrão da **mediana**? E do **p95**? E do coeficiente de Gini? E da
diferença entre as medianas de dois grupos?

O bootstrap responde a todas essas perguntas com o mesmo algoritmo.

### A ideia central

Você não pode reamostrar da população — só tem uma amostra. Mas se a amostra é
representativa, ela é uma **maquete da população**. Então reamostre *dela*, com
reposição, e observe a variabilidade que aparece.

O algoritmo cabe em quatro linhas:

1. Sorteie $n$ observações **com reposição** da sua amostra de tamanho $n$.
2. Calcule a estatística nessa reamostra, obtendo $\hat{\theta}^{*}$.
3. Repita $B$ vezes (tipicamente $B = 2.000$ a $10.000$).
4. Use a distribuição de $\hat{\theta}^{*(1)}, \dots, \hat{\theta}^{*(B)}$ como
   se fosse a distribuição amostral de $\hat{\theta}$.

O desvio-padrão dessas $B$ réplicas estima o erro-padrão. Os percentis 2,5 e 97,5
formam o intervalo.

> [!ANALOGIA] Bootstrap é o Barão de Münchhausen se puxando pelos próprios
> cabelos para sair do pântano — daí o nome. Parece trapaça, mas funciona porque
> a variabilidade *entre reamostras da amostra* imita a variabilidade *entre
> amostras da população*.

### Variantes

| Variante | Como funciona | Quando usar |
|---|---|---|
| Percentil | Percentis das réplicas | Padrão simples; estatística quase não-viesada |
| BCa | Corrige viés e assimetria | Recomendado por padrão; mais caro |
| Bootstrap-t | Estudentiza cada réplica | Melhor cobertura; exige EP por réplica |
| Por blocos | Reamostra blocos contíguos | **Séries temporais** (preserva autocorrelação) |
| Por grupos | Reamostra grupos inteiros | Dados agrupados (usuários com várias sessões) |

> [!ARMADILHA] O bootstrap padrão assume observações **independentes e
> identicamente distribuídas**. Aplicá-lo a séries temporais, a dados agrupados
> (várias linhas por usuário) ou a dados espaciais produz intervalos estreitos
> demais, porque destrói a estrutura de dependência. Use bootstrap por blocos ou
> por grupos. Este é um erro extremamente comum em análises de produto.
>
> O bootstrap também falha para estatísticas de **extremo** (máximo, mínimo) e
> quando $n$ é muito pequeno (abaixo de ~20 a 30).

---

## Tamanho de amostra: planejando antes de coletar

Invertendo a fórmula da margem de erro $E = z \cdot \sigma/\sqrt{n}$:

$$
n = \left( \frac{z \, \sigma}{E} \right)^2
$$

Para proporções, o pior caso ($p = 0{,}5$, variância máxima) dá:

$$
n = \frac{z^2 \cdot 0{,}25}{E^2}
$$

Com $z = 1{,}96$ e $E = 0{,}03$, isso dá $n \approx 1.067$ — o famoso "mil
entrevistados" das pesquisas eleitorais, com margem de 3 pontos.

Repare no expoente: **a amostra cresce com o quadrado da precisão desejada**. É a
economia do $\sqrt{n}$ vista pelo avesso.

---

## Formulário do capítulo

> [!FORMULA] **Viés** $= \mathbb{E}[\hat{\theta}] - \theta$;
> **EQM** $= \text{Var}(\hat{\theta}) + \text{Viés}^2$.
>
> **Erro-padrão da média** $= \sigma/\sqrt{n}$;
> **de proporção** $= \sqrt{\hat{p}(1-\hat{p})/n}$.
>
> **Log-verossimilhança** $\ell(\theta) = \sum_i \log f(x_i \mid \theta)$;
> $\hat{\theta}_{\text{MLE}} = \arg\max_\theta \ell(\theta)$.
>
> **Normalidade assintótica do MLE**:
> $\hat{\theta} \approx \mathcal{N}(\theta, 1/(n I(\theta)))$.
>
> **IC para a média**: $\bar{x} \pm t_{\alpha/2,n-1} \, s/\sqrt{n}$.
>
> **Tamanho de amostra**: $n = (z\sigma/E)^2$.

## Erros que custam caro — checklist

- Dizer "há 95% de chance de $\theta$ estar no intervalo".
- Trocar desvio-padrão por erro-padrão em gráficos (ou não dizer qual é).
- Usar Wald para proporções raras.
- Aplicar bootstrap i.i.d. a séries temporais ou dados agrupados.
- Confiar nas propriedades assintóticas do MLE com $n$ pequeno.
- Calcular tamanho de amostra sem definir antes o efeito mínimo relevante.
- Reportar estimativa pontual sem nenhuma medida de incerteza.

## Para ir além

- Efron & Tibshirani, *An Introduction to the Bootstrap* — a fonte original.
- Wasserman, *All of Statistics*, capítulos 6 a 9.
- Brown, Cai & DasGupta (2001), *Interval Estimation for a Binomial Proportion* —
  o artigo que documenta o mau comportamento do intervalo de Wald.
