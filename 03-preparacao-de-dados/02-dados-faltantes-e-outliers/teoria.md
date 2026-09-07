<!-- tema: Preparação de Dados > Dados Faltantes e Outliers -->
<!-- subtitulo: O buraco no dado e o ponto fora da curva — e por que nenhum dos dois tem uma resposta universal -->
<!-- resumo: "Preencher com a média" e "remover outliers" são os dois reflexos mais automáticos — e mais perigosos — da preparação de dados. Este material constrói o vocabulário certo para diagnosticar por que um dado falta (MCAR, MAR, MNAR) antes de decidir como preenchê-lo, e mostra que um outlier pode ser um erro de digitação ou o cliente mais importante da base — e só a investigação, não uma regra fixa, separa os dois casos. -->
<!-- nivel: Intermediário — requer o módulo de EDA deste tema -->
<!-- prerequisitos: Análise Exploratória (EDA); estatística descritiva (tema 1) -->
<!-- duracao: 8 a 10 horas (leitura + 3 notebooks) -->
<!-- notebooks: 01-mecanismos-de-ausencia · 02-estrategias-de-imputacao · 03-outliers · 99-exercicios -->
<!-- autor: Material do curso de ML & DL -->
<!-- versao: 1.0 -->

# Dados Faltantes e Outliers

## Por que este módulo existe

`df.fillna(df.mean())` é uma linha de código que qualquer pessoa escreve no
primeiro dia. É também uma decisão estatística que pode destruir a variância
dos dados, introduzir viés sistemático, e — no pior caso — vazar informação do
futuro para o passado. O mesmo vale para outliers: `df[df.z_score.abs() < 3]`
parece higiene de dados, mas às vezes está removendo exatamente os casos que o
negócio mais precisa entender (fraude, clientes VIP, falhas de equipamento).

> [!ANALOGIA] Um dado faltante é uma pergunta não respondida numa pesquisa. A
> forma certa de lidar com ela depende de **por que** ela não foi respondida:
> a pessoa esqueceu (aleatório), a pergunta não se aplicava a ela (estrutural),
> ou ela evitou responder de propósito porque a resposta era constrangedora
> (e aí a ausência **é**, ela mesma, uma informação). Tratar os três casos da
> mesma forma é jogar fora exatamente a informação que mais importa no
> terceiro caso.

### O que você vai conseguir fazer ao final

- Classificar um padrão de dado faltante em MCAR, MAR ou MNAR, e explicar por
  que essa classificação muda a estratégia de tratamento.
- Escolher entre imputação simples, por KNN e múltipla (MICE), com
  justificativa, não por hábito.
- Detectar outliers com métodos robustos e não-robustos, e saber quando cada um
  falha.
- Decidir — com um processo, não um reflexo — se um outlier deve ser removido,
  capado, transformado ou preservado.

---

## Os três mecanismos de ausência

A literatura de Rubin (1976) formalizou três mecanismos, e a diferença entre
eles é a coisa mais importante deste módulo.

> [!DEFINICAO] **MCAR (Missing Completely At Random):** a probabilidade de
> faltar não depende de nada — nem do próprio valor, nem de outras variáveis.
> Exemplo: um sensor que falha aleatoriamente por instabilidade de rede.
>
> **MAR (Missing At Random):** a probabilidade de faltar depende de **outras
> variáveis observadas**, mas não do próprio valor faltante (depois de
> controlar por essas outras variáveis). Exemplo: clientes mais velhos têm mais
> chance de não preencher o campo "e-mail" — a ausência depende da idade
> (observada), não do valor do e-mail em si.
>
> **MNAR (Missing Not At Random):** a probabilidade de faltar depende do
> **próprio valor que falta**. Exemplo: pessoas com renda muito alta ou muito
> baixa têm mais chance de não responder "qual sua renda?" numa pesquisa — a
> ausência carrega informação sobre o valor ausente.

> [!ARMADILHA] MNAR é o caso mais comum em dados de negócio e o mais ignorado.
> "Cliente não avaliou o produto" é frequentemente MNAR: clientes com
> experiências medianas avaliam menos que os muito satisfeitos ou muito
> insatisfeitos. Imputar essa ausência com a média das avaliações existentes
> produz um viés sistemático — porque quem respondeu não é uma amostra
> aleatória de quem não respondeu.

### Como diagnosticar (na prática, sem certeza absoluta)

Não existe um teste que prove MNAR — a própria natureza do problema (o valor
que falta é desconhecido) impede verificação direta. O que se pode fazer:

1. **Testar se a ausência de uma variável se associa com outras variáveis
   observadas** (uma regressão logística de "é nulo?" contra as demais
   colunas). Associação forte sugere MAR (ou MNAR, se a variável associada for
   um proxy do próprio valor ausente).
2. **Buscar padrões estruturais**: nulo que ocorre exatamente quando outra
   coluna tem um valor específico costuma ser ausência **estrutural**, não uma
   das três categorias de Rubin — é o caso de "data de cancelamento" ser nula
   para quem nunca cancelou, visto no módulo anterior.
3. **Considerar o processo de negócio**: entender como o dado é coletado
   costuma revelar o mecanismo mais rápido do que qualquer teste estatístico.

## Estratégias de imputação

### Simples: média, mediana, moda

Rápidas e frequentemente um baseline razoável — mas com um efeito colateral
sistemático: **reduzem artificialmente a variância** da coluna (todo valor
imputado é idêntico, então a dispersão em torno dele desaparece) e podem
**atenuar correlações** com outras variáveis.

> [!FORMULA] Se $p\%$ dos valores de uma coluna são substituídos pela média, a
> variância observada da coluna cai para aproximadamente $(1-p)$ vezes a
> variância real — a fração de dados imputados vira "achatamento" artificial.
> Modelos que dependem de variância (e quase todos dependem) recebem um sinal
> distorcido exatamente na proporção de dados faltantes.

> [!MERCADO] Uma prática que mitiga (não elimina) o problema: além de
> imputar, criar uma coluna binária indicadora `_era_nulo`. Isso preserva, como
> uma feature explícita, a informação "esse valor foi observado ou inferido" —
> útil sobretudo quando a ausência é MAR ou MNAR e carrega sinal preditivo por
> si só.

### KNN: usar vizinhos parecidos

`KNNImputer` preenche um valor faltante com a média (ponderada pela distância)
dos $k$ vizinhos mais próximos, usando as demais colunas para medir
similaridade. Captura relações locais que a média global ignora, mas herda
todos os problemas de distância vistos no tema 2: **exige padronizar antes**,
e é sensível à maldição da dimensionalidade em datasets com muitas colunas.

### Múltipla (MICE): a resposta ao problema da incerteza escondida

Toda imputação simples comete um erro conceitual: trata o valor imputado como
se fosse **conhecido com certeza**, quando na verdade é uma estimativa. A
**imputação múltipla por equações encadeadas** (MICE — *Multiple Imputation by
Chained Equations*, implementada em `sklearn` como `IterativeImputer`) modela
cada coluna com nulos como uma regressão nas demais colunas, de forma
iterativa, e — na formulação completa — gera **várias** versões imputadas do
dataset, cuja variação entre si estima a incerteza da imputação.

> [!NOTA] O `IterativeImputer` do `sklearn`, usado de forma direta (uma
> imputação só, não múltiplas), já é uma melhoria substancial sobre a média:
> modela relações entre colunas em vez de ignorá-las. A imputação **múltipla**
> completa (combinar resultados de vários datasets imputados, com regras de
> Rubin para agregar) é mais rigorosa estatisticamente e mais rara na prática
> de mercado do que deveria ser — o custo de implementação correto costuma
> perder para "roda rápido e destrava o projeto".

> [!ARMADILHA] Ajustar qualquer imputador (média, KNN, MICE) usando o dataset
> **inteiro**, antes de separar treino e teste, vaza informação do conjunto de
> teste para o de treino — o imputador "viu" estatísticas de dados que deveria
> ignorar. Sempre ajuste (`fit`) o imputador **só no treino**, e aplique
> (`transform`) da mesma forma no teste. Isso é o mesmo princípio de vazamento
> que o módulo 4 formaliza com `Pipeline`.

## Outliers: o problema da definição

Não existe uma definição única e objetiva de outlier — existe um conjunto de
regras que sinalizam candidatos, e sempre uma decisão que exige contexto.

### Métodos de detecção

| Método | Fórmula / regra | Robustez |
|---|---|---|
| Z-score | $\operatorname{abs}(x - \bar x) / s > 3$ | Baixa — média e desvio-padrão são sensíveis a outliers |
| Z-score modificado | $0.6745 \operatorname{abs}(x - \text{mediana}) / \text{MAD} > 3.5$ | Alta — usa mediana e MAD (tema 1) |
| Regra do IQR | fora de $[Q1 - 1.5\,IQR,\; Q3 + 1.5\,IQR]$ | Alta — baseada em quantis |
| Visual | boxplot, scatter, histograma | Depende do olho, mas pega padrões que regras fixas não pegam |

> [!ARMADILHA] O z-score clássico tem um problema circular: ele usa a média e
> o desvio-padrão **calculados com os próprios outliers incluídos** — e ambos
> são sensíveis a outliers (tema 1). Um outlier extremo o suficiente pode
> inflar tanto o desvio-padrão que ele mesmo deixa de parecer extremo pelo
> critério. O z-score modificado, baseado em mediana e MAD, evita esse efeito
> porque ambos têm ponto de ruptura de 50%.

### O que fazer depois de detectar

Detectar não é decidir. As opções, em ordem de quão "destrutivas" são:

1. **Investigar a origem.** É erro de digitação (idade = 999), erro de unidade
   (valor em centavos lido como reais), ou um valor real e raro?
2. **Manter e usar um modelo robusto.** Árvores de decisão e gradient boosting
   são naturalmente resistentes a outliers em features (o particionamento não
   depende de distância); modelos baseados em distância ou em soma de
   quadrados (regressão linear, k-means, PCA) são mais vulneráveis.
3. **Transformar.** Uma transformação log ou Box-Cox comprime a cauda e reduz
   o efeito de valores extremos sem descartá-los (o mesmo raciocínio do
   histograma em escala log do tema 1).
4. **Capar (winsorizar).** Substituir valores acima/abaixo de um percentil
   pelo próprio limite, preservando a existência do ponto sem deixá-lo dominar
   a escala.
5. **Remover.** A opção mais forte, e a única realmente irreversível — usar só
   depois de descartar erro de coleta como causa e confirmar que o ponto não
   representa um segmento de negócio real.

> [!MERCADO] Em detecção de fraude, os "outliers" **são o problema que o
> modelo existe para resolver** — removê-los da base de treino equivale a
> ensinar o modelo a nunca reconhecer fraude. Em manutenção preditiva, o
> outlier no sinal de um sensor pode ser o primeiro indício de uma falha
> mecânica real. A pergunta certa nunca é "isso é estatisticamente raro?" —
> é "isso é raro **e também um erro**, ou é raro **e informativo**?"

## Erros que custam caro — checklist

- Imputar com a média sem checar o mecanismo de ausência (MCAR/MAR/MNAR).
- Ajustar imputadores no dataset completo antes do split treino/teste.
- Usar z-score clássico para detectar outliers em dados que já têm outliers
  (problema circular) — prefira a versão modificada, baseada em mediana/MAD.
- Remover outliers automaticamente sem investigar se são erro ou sinal.
- Ignorar que uma coluna indicadora `_era_nulo` pode carregar sinal preditivo
  por si só, mesmo depois de imputar o valor.
- Tratar ausência estrutural (não pode existir) como se fosse MCAR/MAR/MNAR —
  são categorias diferentes com soluções diferentes.

## Para ir além

- Rubin, *Inference and Missing Data* (1976) — o artigo que define
  MCAR/MAR/MNAR.
- van Buuren, *Flexible Imputation of Missing Data* — o tratamento de
  referência sobre MICE, com o autor do pacote `mice` original em R.
- Aggarwal, *Outlier Analysis* — cobertura ampla de métodos de detecção,
  estatísticos e baseados em modelo.
- Rousseeuw & Hubert, sobre robustez estatística — a base teórica do z-score
  modificado e de outras medidas robustas usadas neste módulo.
