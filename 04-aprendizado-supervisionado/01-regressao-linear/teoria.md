<!-- tema: Aprendizado Supervisionado > Regressão Linear -->
<!-- subtitulo: O modelo mais simples que ainda é o baseline mais honesto do mercado -->
<!-- resumo: Regressão linear não é "o modelo básico que se aprende antes dos de verdade" — é o modelo cuja matemática já foi construída nos temas 2 e 3 (projeção ortogonal, gradiente descendente) e cuja interpretabilidade nenhum modelo mais complexo supera. Este material formaliza o modelo, deriva o diagnóstico de resíduos que decide se ele é adequado, e aplica tudo a um caso real de precificação. -->
<!-- nivel: Intermediário — requer Álgebra Linear e Otimização (tema 2) e Preparação de Dados (tema 3) -->
<!-- prerequisitos: Vetores, Matrizes e Projeções; Cálculo e Gradiente Descendente; todos os módulos do tema 3 -->
<!-- duracao: 8 a 10 horas (leitura + 3 notebooks) -->
<!-- notebooks: 01-regressao-do-zero · 02-pressupostos-e-diagnostico · 03-caso-real-precificacao · 99-exercicios -->
<!-- autor: Material do curso de ML & DL -->
<!-- versao: 1.0 -->

# Regressão Linear

## Por que este módulo existe

Regressão linear é o único modelo deste tema cuja matemática você já construiu
inteira, em dois temas anteriores: o tema 2 mostrou que ajustar uma reta é uma
projeção ortogonal, e que $\hat\beta = (X^\top X)^{-1}X^\top y$ nasce da
exigência de que o resíduo seja perpendicular às features. O tema 3 encheu essa
formulação de cuidados práticos — sem eles, o "modelo mais simples" produz os
erros mais silenciosos. Este módulo não repete a derivação geométrica — ele
formaliza o modelo estatístico por trás dela, o diagnóstico que diz se ele é
adequado, e o caso de uso mais comum em produção: precificação.

> [!ANALOGIA] Se os modelos deste tema fossem ferramentas de uma oficina, a
> regressão linear seria a chave de fenda: simples, previsível, e a primeira
> coisa que um profissional experiente tenta antes de pegar a furadeira
> elétrica. Não porque seja fraca — porque, quando o parafuso é o problema
> certo, nada é mais rápido de usar nem mais fácil de explicar por que
> funcionou.

### O que você vai conseguir fazer ao final

- Escrever o modelo de regressão linear com os pressupostos estatísticos
  explícitos, não só a fórmula do ajuste.
- Diagnosticar, com gráficos e testes, quando os pressupostos são violados e o
  que fazer a respeito.
- Interpretar coeficientes, intervalos de confiança e $R^2$ com o rigor que o
  tema 1 e o tema 2 já construíram.
- Aplicar regressão linear a um problema real de precificação, incluindo a
  preparação de dados que o tema 3 ensinou.

---

## O modelo estatístico

$$y_i = \beta_0 + \beta_1 x_{i1} + \dots + \beta_p x_{ip} + \varepsilon_i$$

O tema 2 já mostrou **como** encontrar $\hat\beta$ (projeção ortogonal). O que
falta é a parte estatística: $\varepsilon_i$ é modelado como uma variável
aleatória, e é sobre ela que recaem os pressupostos que tornam válidas as
inferências (intervalos de confiança, testes de hipótese sobre coeficientes).

> [!DEFINICAO] Os quatro pressupostos clássicos (memorizáveis pelo acrônimo
> **LINE**): **L**inearidade (a relação entre $X$ e $E[y]$ é de fato linear
> nos parâmetros), **I**ndependência dos erros, **N**ormalidade dos erros, e
> **E**quivariância — variância constante dos erros (homocedasticidade).
> Nenhum desses pressupostos é necessário para *calcular* $\hat\beta$ — o
> tema 2 fez isso com álgebra pura. Eles são necessários para que
> erros-padrão, intervalos de confiança e p-valores dos coeficientes
> **signifiquem o que dizem significar**.

> [!ARMADILHA] Um erro comum: achar que "linear" significa "só retas". A
> regressão é linear **nos parâmetros** — $y = \beta_0 + \beta_1 x + \beta_2
> x^2$ é uma regressão linear (é linear em $\beta_0, \beta_1, \beta_2$), ainda
> que a curva ajustada seja uma parábola. É por isso que polinômios, splines e
> até certas interações cabem no mesmo arcabouço matemático do tema 2 — só
> muda a matriz $X$ que se constrói antes de projetar.

## Interpretação de coeficientes

Em $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \varepsilon$, o coeficiente
$\beta_1$ é: **o quanto $y$ muda, em média, para um aumento de uma unidade em
$x_1$, mantendo $x_2$ constante.**

> [!NOTA] "Mantendo $x_2$ constante" não é retórica — é o teorema de
> Frisch-Waugh-Lovell do tema 2: $\beta_1$ é literalmente o coeficiente da
> parte de $x_1$ que é **ortogonal** a $x_2$. Quando $x_1$ e $x_2$ são
> correlacionadas, o "efeito isolado" de $x_1$ só existe depois de remover
> dela o que $x_2$ já explica.

> [!ARMADILHA] Coeficientes de features em escalas muito diferentes não são
> diretamente comparáveis em magnitude — um coeficiente de 50.000 numa
> feature em milhões e um coeficiente de 0,003 numa proporção podem
> representar efeitos de importância parecida. Para comparar magnitude de
> efeito entre features, padronize-as antes (tema 3) e compare os
> coeficientes padronizados.

### Erro-padrão, intervalo de confiança e p-valor de um coeficiente

Sob os pressupostos LINE, $\hat\beta_j$ tem uma distribuição amostral com
erro-padrão calculável a partir de $\hat\sigma^2 (X^\top X)^{-1}$ (a diagonal
dessa matriz, mais precisamente). Isso permite testar $H_0: \beta_j = 0$ (tema
1) e construir intervalos de confiança — a mesma lógica de inferência
construída para médias, agora aplicada a coeficientes de regressão.

> [!MERCADO] Um coeficiente estatisticamente significativo ($p < 0{,}05$) não
> é automaticamente um coeficiente **relevante** para o negócio — com $n$
> grande, até efeitos minúsculos ficam significativos. E um coeficiente não
> significativo não prova ausência de efeito — pode ser apenas falta de
> poder estatístico (tema 1, módulo 4). Sempre reporte a magnitude do
> coeficiente (e seu intervalo de confiança) junto do p-valor.

## $R^2$: quanto da variância o modelo explica

$$R^2 = \frac{\text{SQ explicada}}{\text{SQ total}} = 1 - \frac{\text{SQ residual}}{\text{SQ total}}$$

Já visto no tema 2 como $\cos^2$ do ângulo entre $y$ e $\hat y$ centrados.

> [!ARMADILHA] $R^2$ **nunca diminui** quando se adiciona uma nova feature ao
> modelo — mesmo uma feature de ruído puro. Isso torna $R^2$ enganoso para
> comparar modelos com números diferentes de features. O **$R^2$ ajustado**
> penaliza pelo número de parâmetros:
>
> $$R^2_{aj} = 1 - (1 - R^2)\frac{n-1}{n-p-1}$$
>
> e só aumenta se a nova feature reduzir o erro mais do que o esperado por
> puro acaso.

## Diagnóstico de resíduos: como saber se o modelo serve

O diagnóstico é a parte prática que decide se os pressupostos LINE são
razoáveis para os dados em mãos.

| Gráfico | O que revela |
|---|---|
| Resíduo vs. valor previsto | Padrão curvo = não-linearidade; funil = heterocedasticidade |
| Q-Q plot dos resíduos | Desvio da reta = não-normalidade dos erros |
| Resíduo vs. cada feature | Padrão restante = a feature precisa de transformação (ex.: log, polinômio) |
| Alavancagem (leverage) vs. resíduo | Pontos influentes que distorcem o ajuste sozinhos |

> [!FORMULA] A **distância de Cook** combina alavancagem e magnitude do
> resíduo num único número por observação, medindo o quanto os coeficientes
> mudariam se aquele ponto fosse removido. Pontos com distância de Cook alta
> merecem investigação — não remoção automática (tema 3, módulo 2: outlier
> não é sinônimo de erro).

> [!ARMADILHA] Um padrão em forma de funil no gráfico resíduo-vs-previsto
> (heterocedasticidade — variância do erro cresce com o valor previsto) não
> invalida os coeficientes estimados (eles continuam não-viesados), mas
> invalida os erros-padrão calculados da forma clássica — os intervalos de
> confiança ficam errados. A correção mais simples costuma ser transformar o
> alvo (log, se ele for positivo e a variância crescer com a média — o mesmo
> padrão de dados assimétricos do tema 1).

## Multicolinearidade, de novo

O tema 2 já mostrou que colinearidade **infla variância, não enviesa**
coeficientes. Aqui, a consequência prática: coeficientes instáveis (mudam
muito com pequenas mudanças nos dados), sinais que trocam de forma
contraintuitiva, e intervalos de confiança largos demais para serem úteis.

> [!MERCADO] VIF acima de 5-10 (tema 2) é o sinal de alerta. As saídas
> práticas: remover uma das features colineares, combiná-las, ou usar
> regularização (Ridge, próximo módulo) — que foi desenhada precisamente para
> estabilizar $X^\top X$ mal condicionada.

## Regressão linear como baseline

> [!MERCADO] Antes de qualquer modelo mais sofisticado, treinar uma regressão
> linear (ou logística, para classificação) é o primeiro passo profissional
> em qualquer projeto novo — não por ingenuidade, mas porque ela estabelece um
> **piso de comparação interpretável**. Se um Random Forest bate a regressão
> linear por uma margem pequena, isso é informação: talvez o problema seja
> majoritariamente linear, e a complexidade extra não compense o custo de
> interpretabilidade perdido.

## Erros que custam caro — checklist

- Reportar coeficientes sem checar resíduos — um modelo com pressupostos
  violados pode ter coeficientes tecnicamente calculados e estatisticamente
  sem sentido.
- Comparar magnitude de coeficientes de features em escalas diferentes sem
  padronizar.
- Usar $R^2$ para decidir se vale a pena adicionar uma feature — use $R^2$
  ajustado ou validação cruzada (tema 6).
- Ignorar heterocedasticidade e reportar intervalos de confiança calculados
  da forma clássica mesmo assim.
- Tratar colinearidade removendo features às cegas, sem entender por que
  estão correlacionadas.
- Pular direto para um modelo complexo sem primeiro estabelecer o baseline
  linear.

## Para ir além

- Kutner et al., *Applied Linear Statistical Models* — o tratado de
  referência sobre diagnóstico de regressão.
- Gelman & Hill, *Data Analysis Using Regression and Multilevel Models* — a
  ponte entre a teoria clássica e a prática aplicada moderna.
- Cook & Weisberg, *Residuals and Influence in Regression* — o texto que
  formaliza a distância de Cook e outras medidas de influência.
