<!-- tema: Preparação de Dados > Análise Exploratória (EDA) -->
<!-- subtitulo: O protocolo que decide se o resto do projeto vale a pena -->
<!-- resumo: Análise exploratória não é a etapa que se apressa para "chegar logo no modelo" — é onde se descobre se o dado sustenta a pergunta de negócio, onde vazamentos e armadilhas são pegos antes de custarem caro, e onde nascem as hipóteses que o modelo vai testar. Este material constrói um protocolo disciplinado de EDA e as ferramentas de correlação e associação que sustentam suas conclusões. -->
<!-- nivel: Introdutório a Intermediário -->
<!-- prerequisitos: Estatística descritiva (tema 1); noções de pandas -->
<!-- duracao: 6 a 8 horas (leitura + 2 notebooks) -->
<!-- notebooks: 01-protocolo-de-eda · 02-relacoes-e-correlacao · 99-exercicios -->
<!-- autor: Material do curso de ML & DL -->
<!-- versao: 1.0 -->

# Análise Exploratória (EDA)

## Por que este módulo existe

A maioria dos projetos de dados que falha, falha silenciosamente muito antes do
modelo: uma coluna de data importada como texto, um identificador de cliente
tratado como feature numérica, uma variável que na verdade é o próprio alvo
disfarçado, um grupo de outliers que representa um segmento de negócio inteiro
sendo descartado como "ruído". Nenhum desses problemas aparece na acurácia do
modelo — eles aparecem, meses depois, como "por que o modelo em produção está
errando tanto".

> [!ANALOGIA] EDA é a inspeção do terreno antes de construir. Um engenheiro não
> começa a erguer paredes sem saber se o solo é rocha ou areia movediça. Um
> cientista de dados que pula a EDA está apostando que o "solo" dos dados é
> firme — às vezes é, e o projeto sai barato dessa aposta; frequentemente não é.

### O que você vai conseguir fazer ao final

- Executar um protocolo de EDA reproduzível, sem depender de "olhar e ver o que
  aparece".
- Escolher a medida de correlação/associação certa para cada combinação de
  tipos de variável (numérica-numérica, categórica-categórica,
  categórica-numérica).
- Reconhecer quando correlação linear esconde uma relação real, e quando uma
  correlação alta esconde uma relação espúria.
- Detectar, antes de modelar, os sinais mais comuns de vazamento de dados e de
  paradoxo de agregação (Simpson).

---

## O protocolo: da estrutura ao conteúdo

EDA disciplinada segue uma ordem: primeiro entender a **forma** dos dados,
depois o **conteúdo** de cada variável isolada, depois as **relações** entre
variáveis. Pular etapas custa caro — é comum alguém já estar plotando
correlações quando ainda nem sabe que 30% de uma coluna são nulos.

### Etapa 1 — Forma e proveniência

- Quantas linhas, quantas colunas, qual o período coberto.
- **Cada linha é o quê?** Uma transação? Um cliente-mês? Essa pergunta parece
  óbvia e é a que mais gente erra — confundir a granularidade da linha é a
  origem de metade dos bugs de agregação.
- Duplicatas exatas e **quase-duplicatas** (mesmo cliente, registros quase
  idênticos por erro de captura).
- Tipos de dado (`dtype`) — e se eles batem com o tipo **real** da variável
  (revisite a taxonomia do tema 1: nominal, ordinal, intervalar, razão).

> [!ARMADILHA] Um identificador (`id_cliente`, `cpf`, `cep`) armazenado como
> `int64` passa despercebido por qualquer `.describe()` e o pandas calcula
> média e desvio-padrão dele sem reclamar. O resultado é sintaticamente válido
> e completamente sem sentido. Sempre audite: toda coluna numérica **é**
> mesmo uma quantidade, ou é um rótulo disfarçado de número?

### Etapa 2 — Perfilamento univariado

Para cada variável, isoladamente:

- **Numéricas:** distribuição (histograma), medidas de posição e dispersão
  robustas e não-robustas (tema 1), fração de nulos, fração de zeros
  (frequentemente um valor "especial" disfarçado de número comum).
- **Categóricas:** cardinalidade (quantas categorias distintas), frequência de
  cada categoria, categorias raras (que podem precisar de agrupamento antes de
  qualquer encoding — assunto do módulo 4).
- **Datas:** intervalo coberto, buracos no tempo, granularidade real
  (timestamps ou só dia?).

> [!MERCADO] Cardinalidade alta em uma coluna categórica (`id_produto` com
> 50.000 valores únicos) não é um problema de EDA — é um aviso antecipado de
> um problema de **encoding**: one-hot nessa coluna criaria 50.000 novas
> colunas. Detectar isso agora, e não na hora de treinar o modelo, economiza
> retrabalho.

### Etapa 3 — Relações entre variáveis

Aqui entra a maior parte deste módulo: como medir associação entre pares de
variáveis de tipos diferentes, e como não se enganar com o resultado. Isso é
desenvolvido em detalhe nas próximas seções.

### Etapa 4 — Hipóteses e checagem de vazamento

EDA termina em **perguntas testáveis**, não em gráficos bonitos:

- Que variáveis parecem separar bem a classe/alvo de interesse?
- Existe alguma variável **suspeitosamente** preditiva? (Um sinal clássico de
  vazamento: uma feature que só existe **depois** do evento que se quer
  prever, ou que é uma proxy quase perfeita do alvo.)
- As relações fazem sentido de negócio, ou parecem artefato de coleta?

> [!ARMADILHA] "Essa feature tem correlação de 0,98 com o alvo" quase nunca é
> boa notícia antes de investigada — é mais frequentemente vazamento
> (`data_de_cancelamento` presente nas features de um modelo de previsão de
> cancelamento) do que um sinal genuinamente forte. Correlação alta demais deve
> aumentar a suspeita, não a confiança.

---

## Correlação e associação: escolhendo a ferramenta certa

A tabela a seguir resume o problema central desta seção: **não existe uma
única "correlação"** — existe uma família de medidas, cada uma adequada a um
tipo de par de variáveis e a um tipo de relação.

| Par de variáveis | Relação linear | Relação monotônica (não-linear) | Relação qualquer |
|---|---|---|---|
| Numérica × Numérica | Pearson | Spearman | Informação mútua |
| Categórica × Categórica | — | Qui-quadrado / Cramér's V | Informação mútua |
| Categórica × Numérica | ANOVA (F) / eta² | — | Informação mútua |

### Pearson: mede só a parte linear

$$r = \frac{\sum_i (x_i - \bar x)(y_i - \bar y)}{\sqrt{\sum_i (x_i-\bar x)^2}\sqrt{\sum_i(y_i-\bar y)^2}}$$

Já visto no tema 2 como cosseno de vetores centrados. $r$ mede **apenas** o
grau de relação linear — uma relação perfeita, porém curva, pode ter $r$
próximo de zero.

### Spearman: mede a parte monotônica

Spearman calcula Pearson sobre os **postos** (ranks) dos dados em vez dos
valores brutos. O resultado: captura qualquer relação **monotônica**
(sempre crescente ou sempre decrescente), mesmo que não seja uma reta.

> [!DEFINICAO] $\rho_{Spearman} = r_{Pearson}(\text{posto}(x), \text{posto}(y))$.
> Uma relação exponencial $y = e^x$ tem Spearman igual a 1 (é estritamente
> crescente) mas Pearson bem menor que 1 (a curva se afasta da reta).

> [!MERCADO] Spearman é mais robusto a outliers que Pearson, pelo mesmo motivo
> que a mediana é mais robusta que a média: postos ignoram *quão* extremo um
> valor é, só a sua ordem. É a escolha padrão quando a relação é suspeita de
> não ser linear, ou quando há outliers que distorceriam Pearson.

### Qui-quadrado e Cramér's V: associação entre categóricas

Para duas variáveis categóricas, organiza-se uma **tabela de contingência**
(contagens cruzadas) e compara-se com o que se esperaria se as variáveis
fossem independentes. O teste qui-quadrado de independência mede o quanto essa
diferença é estatisticamente significativa; **Cramér's V** normaliza a
estatística para um número entre 0 e 1, comparável entre tabelas de tamanhos
diferentes:

$$V = \sqrt{\frac{\chi^2/n}{\min(k-1, r-1)}}$$

onde $r$ e $k$ são o número de linhas e colunas da tabela de contingência.

> [!NOTA] $\chi^2$ (qui-quadrado) sozinho cresce com o tamanho da amostra — não
> é comparável entre datasets de tamanhos diferentes. Cramér's V corrige isso e
> é a medida que de fato se reporta como "força de associação" entre duas
> categóricas.

### ANOVA e eta²: associação entre categórica e numérica

Quando uma variável é categórica (grupo) e a outra numérica, a pergunta é: a
média da numérica muda entre grupos mais do que se esperaria por acaso? O
teste F da ANOVA responde a parte de significância; **eta² (η²)** mede a
**força** — a fração da variância total da variável numérica que é explicada
pela pertença ao grupo:

$$\eta^2 = \frac{\text{SQ entre grupos}}{\text{SQ total}}$$

> [!NOTA] Essa é literalmente a mesma decomposição de soma de quadrados
> (Pitágoras no espaço das observações) construída no módulo de projeção
> ortogonal do tema 2 — ANOVA é um caso particular de regressão linear com uma
> feature categórica.

### Informação mútua: quando nem "monotônico" é garantido

A **informação mútua** mede quanta incerteza sobre $Y$ é reduzida ao conhecer
$X$, sem assumir nenhuma forma funcional específica — linear, monotônica, ou
qualquer outra. Captura, por exemplo, uma relação em U (onde $Y$ é grande tanto
para $X$ pequeno quanto para $X$ grande), que Pearson e Spearman veem como
"sem relação".

> [!ARMADILHA] Informação mútua não tem um teto natural comparável entre pares
> de variáveis (ao contrário de Pearson/Spearman, limitados a $[-1,1]$, ou
> Cramér's V, limitado a $[0,1]$), e é mais sensível ao tamanho da amostra e ao
> método de estimação. Use-a para **descobrir** relações que as medidas
> clássicas não veem — não para comparar "força" de relações diferentes de
> forma direta.

## O paradoxo de Simpson: quando agregação mente

Uma correlação pode **inverter de sinal** quando os dados são desagregados por
um grupo escondido. O exemplo clássico: uma universidade parece favorecer
homens nas admissões gerais, mas, dentro de **cada departamento**, favorece
mulheres — porque mulheres se candidatam desproporcionalmente aos
departamentos mais concorridos.

> [!ARMADILHA] Toda vez que uma relação parece forte demais para fazer
> sentido, ou contrária à intuição de negócio, pergunte: **existe uma variável
> de agrupamento que, se eu controlar por ela, muda a conclusão?** Isso não é
> paranoia — é o motivo pelo qual EDA sempre deve incluir a relação
> **dentro de subgrupos relevantes**, não só agregada.

## Erros que custam caro — checklist

- Calcular `.describe()` em colunas que são identificadores, não quantidades.
- Reportar Pearson como "a" correlação sem checar se a relação é não-linear.
- Ignorar cardinalidade alta em categóricas até a hora de treinar o modelo.
- Aceitar uma correlação altíssima com o alvo sem investigar vazamento.
- Analisar dados agregados sem checar se a conclusão se sustenta dentro de
  subgrupos (Simpson).
- Confundir "sem correlação de Pearson" com "sem relação" — Anscombe e o
  Datasaurus (tema 1) já provaram que isso é falso.
- Pular a etapa de "o que é uma linha" e descobrir tarde que o dataset tem
  granularidade diferente da esperada.

## Para ir além

- Tukey, *Exploratory Data Analysis* — o texto fundador da área, ainda
  relevante.
- Wickham & Grolemund, *R for Data Science*, capítulos de EDA — o protocolo
  mais citado da prática moderna (as ideias são independentes de linguagem).
- Anscombe (1973) e Matejka & Fitzmaurice (2017, *Datasaurus*) — os dois
  artigos que provam visualmente por que resumos numéricos não bastam.
