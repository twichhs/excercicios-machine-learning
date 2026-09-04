<!-- tema: Estatística > A/B Testing e Desenho Experimental -->
<!-- subtitulo: A engenharia de fazer o experimento valer, antes de qualquer análise -->
<!-- resumo: A maior parte dos experimentos em produção não falha na análise — falha no desenho. Este material trata A/B testing como problema de engenharia: unidade de aleatorização, escolha de métricas, dimensionamento, os testes de sanidade que detectam um experimento inválido, técnicas de redução de variância como CUPED e estratificação, e o catálogo de armadilhas (SRM, peeking, efeito novidade, interferência entre unidades) que invalidam resultados aparentemente limpos. -->
<!-- nivel: Intermediário -->
<!-- prerequisitos: Módulos 01 a 05 (Descritiva, Distribuições, Inferência, Testes, Bayesiana) -->
<!-- duracao: 10 a 12 horas (leitura + 4 notebooks) -->
<!-- notebooks: 01-desenho-e-aleatorizacao · 02-analise-de-experimento · 03-cuped-e-variancia · 99-exercicios -->
<!-- autor: Material do curso de ML & DL -->
<!-- versao: 1.0 -->

# A/B Testing e Desenho Experimental

## Por que experimentar

Toda análise observacional sofre do mesmo problema: quem escolheu usar o
recurso novo é diferente de quem não escolheu, em dezenas de dimensões que você
não mediu. Comparar os dois grupos mede a diferença entre as **pessoas**, não o
efeito do **recurso**.

A aleatorização resolve isso de um jeito que nenhum controle estatístico
consegue: ao sortear quem recebe o tratamento, você garante que os dois grupos
são equivalentes em **tudo** — inclusive nas variáveis que você nem sabe que
existem. É a única técnica que controla confundidores não observados.

> [!ANALOGIA] Aleatorizar é embaralhar o baralho antes de distribuir. Não
> importa o quanto as cartas estavam ordenadas antes; depois do embaralhamento,
> qualquer diferença sistemática entre as mãos foi eliminada por construção. Se
> uma mão vencer consistentemente, a causa está no que você fez com ela depois —
> não em como as cartas chegaram.

O preço é que experimentar custa tempo, tráfego e disciplina. E o retorno só
existe se o desenho estiver certo — uma análise impecável de um experimento mal
desenhado produz um número preciso e errado.

## A unidade de aleatorização

A primeira decisão, e a que mais gente erra.

| Unidade | Quando usar | Risco |
|---|---|---|
| **Usuário** (mais comum) | mudanças de UI, features, preço | precisa de identidade estável entre sessões |
| **Sessão** | mudanças que não persistem | mesma pessoa vê versões diferentes — inconsistência visível |
| **Cluster** (loja, cidade, empresa) | efeitos de rede, marketplaces | perde muito poder; $n$ efetivo é o número de clusters |
| **Tempo** (switchback) | logística, precificação dinâmica | contaminação entre períodos |

A regra: aleatorize na **menor unidade que não gere interferência** entre
tratamento e controle. Se o efeito de tratar um usuário vaza para outro — redes
sociais, marketplaces, sistemas de entrega — a suposição de independência quebra
e a estimativa fica viesada.

> [!ARMADILHA] **Interferência (SUTVA violada).** Em um marketplace, dar
> desconto ao grupo B faz o grupo A perder as ofertas que B comprou. O efeito
> medido não é "o que acontece se todos receberem o desconto" — é "o que
> acontece a quem recebe, às custas de quem não recebe". Testes de preço,
> algoritmos de matching e mudanças de ranking sofrem disso quase sempre, e a
> saída é aleatorizar por região, por mercado ou por período.

## Métricas: a hierarquia que evita autoengano

Um experimento tem uma métrica primária, algumas secundárias e um conjunto de
guardrails. Escolher tudo isso **antes** de ver os dados é o que separa um
experimento de uma pescaria.

- **Métrica primária** (uma só): a que decide o lançamento. Deve ser sensível ao
  tratamento, mensurável no horizonte do teste, e alinhada ao objetivo real.
- **Métricas secundárias**: ajudam a entender o mecanismo. Não decidem nada
  sozinhas.
- **Guardrails**: métricas que não podem piorar, mesmo que a primária melhore —
  tempo de carregamento, taxa de erro, receita, cancelamentos, reclamações.

> [!MERCADO] A tensão clássica: uma mudança que aumenta cliques em 8% e derruba
> a receita por sessão em 2%. Sem guardrail de receita, o experimento é declarado
> vencedor e a empresa perde dinheiro por trimestres até alguém notar. Times
> maduros bloqueiam o lançamento automaticamente quando um guardrail degrada
> além de um limite pré-definido.

### Métricas proxy e o horizonte do teste

O que interessa (retenção em 12 meses, LTV) não cabe em um experimento de duas
semanas. A saída é usar **proxies** — mas proxies precisam ser validados contra
o resultado de longo prazo em experimentos anteriores, não escolhidos por
conveniência. Uma proxy não validada é uma hipótese disfarçada de métrica.

## Dimensionamento: decidir antes, não depois

O tamanho de amostra sai de quatro números: efeito mínimo relevante (MDE),
variância da métrica, $\alpha$ e poder desejado.

> [!FORMULA] Para comparar duas médias com variância $\sigma^2$:
>
> $n \approx \dfrac{2\sigma^2 (z_{1-\alpha/2} + z_{1-\beta})^2}{\Delta^2}$ por grupo.
>
> Com $\alpha = 0{,}05$ e poder de 80%, o numerador constante vale $\approx 15{,}7$,
> então $n \approx 15{,}7\,\sigma^2/\Delta^2$.
>
> Para proporções, $\sigma^2 = p(1-p)$.

O **MDE não é o efeito que você espera** — é o menor efeito que mudaria a
decisão de negócio. Definir MDE de 0,5% quando a empresa só implantaria uma
mudança que rendesse 3% é queimar tráfego para responder uma pergunta que
ninguém fez.

> [!ARMADILHA] **Análise de poder post-hoc.** Calcular "o poder que tivemos" a
> partir do efeito observado é uma reescrita do p-valor, não informação nova. O
> poder é uma propriedade do **desenho**, e só existe antes do experimento.

## Os testes de sanidade que salvam a análise

Antes de olhar a métrica primária, verifique se o experimento é válido.

### Sample Ratio Mismatch (SRM)

Se a alocação era 50/50 e você observa 50,4% / 49,6% com 200 mil usuários, o
experimento está quebrado. Um qui-quadrado de aderência responde em uma linha.

SRM é o sintoma, não a doença. As causas típicas: bots atribuídos a um grupo só,
falha no SDK que descarta eventos de uma variante, redirecionamento que perde
usuários, filtro aplicado depois da aleatorização.

> [!ARMADILHA] **Um experimento com SRM não pode ser analisado, ponto.** Não
> existe correção estatística — a aleatorização falhou, e você não sabe *quem*
> foi perdido. A única ação é achar a causa e refazer.

### Teste A/A

Rode a mesma versão contra ela mesma. Você deveria observar p-valores uniformes
e ~5% de "significância" espúria. Se der significativo com frequência maior, há
algo errado na infraestrutura — atribuição, logging ou a própria métrica.

### Checagem de pré-experimento

Compare os grupos em métricas **anteriores** ao tratamento. Elas devem ser
indistinguíveis. Diferença pré-existente significa aleatorização quebrada ou
contaminação.

## Redução de variância: mais poder sem mais tráfego

Como $n \propto \sigma^2$, cortar a variância pela metade tem o mesmo efeito que
dobrar a amostra — de graça.

### CUPED (Controlled-experiment Using Pre-Experiment Data)

A ideia: use uma covariável $X$ medida **antes** do experimento (tipicamente a
mesma métrica no período anterior) para remover a variabilidade que já existia.

> [!FORMULA] $Y_{\text{cuped}} = Y - \theta(X - \bar{X})$, com
> $\theta = \dfrac{\mathrm{Cov}(Y, X)}{\mathrm{Var}(X)}$.
>
> A variância cai por um fator $(1 - \rho^2)$, onde $\rho$ é a correlação entre
> $Y$ e $X$. Com $\rho = 0{,}7$, a variância cai 51% — equivalente a dobrar o
> tráfego.

CUPED é não-viesado porque $X$ é anterior ao tratamento e, portanto, não pode ter
sido afetado por ele. Essa é a condição inegociável: **qualquer covariável
medida depois da aleatorização pode ser um colisor e enviesar o resultado**.

### Outras técnicas

- **Estratificação / aleatorização por blocos**: garante balanceamento em
  variáveis conhecidas (plataforma, país, segmento) em vez de torcer por ele.
- **Winsorização** de métricas de cauda pesada (receita, tempo na página): apara
  o 0,1% superior antes de analisar. Reduz variância enormemente, mas muda o
  estimando — declare que você está estimando uma média winsorizada.
- **Regressão com covariáveis** (CUPAC): generalização do CUPED usando um modelo
  preditivo treinado em dados pré-experimento.

## O catálogo de armadilhas

| Armadilha | O que acontece | Defesa |
|---|---|---|
| **Peeking** | parar ao ver significância infla o erro tipo I para ~20% | duração fixa, ou teste sequencial (O'Brien-Fleming, always-valid) |
| **SRM** | alocação desbalanceada denuncia perda de dados | qui-quadrado antes de tudo; abortar se falhar |
| **Efeito novidade** | usuários reagem à mudança, não ao valor dela | rodar tempo suficiente; olhar o efeito por semana |
| **Comparações múltiplas** | 20 segmentos ⇒ 64% de chance de falso positivo | métrica primária declarada; FDR nos exploratórios |
| **Interferência** | tratamento vaza para o controle | aleatorizar por cluster/região/tempo |
| **Simpson** | efeito positivo em cada segmento, negativo no total | checar composição dos grupos; usar médias ponderadas |
| **Métrica de razão** | usuário com 1.000 eventos domina a média | delta method ou bootstrap por usuário |
| **Sobrevivência** | analisar só quem completou o funil | analisar por intenção de tratar (ITT) |

> [!MERCADO] A regra de ouro da indústria (Kohavi, na Microsoft e Airbnb): a
> **maioria** das ideias testadas não melhora a métrica. Na Bing, cerca de um
> terço dos experimentos dá resultado positivo, um terço neutro e um terço
> negativo. Um time cujo A/B test "sempre dá positivo" não tem boas ideias — tem
> um problema de método.

## Intenção de tratar × efeito no tratado

Se 30% dos usuários alocados ao tratamento nunca chegaram a ver a mudança,
comparar "quem viu" com todo o controle é comparar grupos não equivalentes —
quem viu é quem chegou mais fundo no funil.

A análise correta por padrão é a de **intenção de tratar (ITT)**: compare os
grupos **como alocados**, independentemente do que aconteceu depois. O ITT
subestima o efeito em quem realmente foi exposto, mas é a única estimativa não
viesada. Para recuperar o efeito no tratado, existe o estimador **CACE/LATE**,
que divide o efeito ITT pela taxa de adesão — com suas próprias suposições.

> [!FORMULA] $\text{CACE} = \dfrac{\text{efeito ITT}}{\text{taxa de adesão}}$,
> válido sob exclusão (a alocação só afeta o resultado via o tratamento) e
> ausência de *defiers*.

## Erros que custam caro — checklist

- Analisar antes de checar SRM.
- Parar o teste ao ver significância.
- Escolher a métrica primária depois de ver os resultados.
- Definir MDE pelo efeito esperado em vez do efeito relevante.
- Ignorar guardrails porque a primária subiu.
- Usar covariável pós-tratamento em CUPED ou em regressão de ajuste.
- Analisar só quem completou o funil, em vez de ITT.
- Fatiar por segmento até achar um resultado positivo.
- Rodar uma semana e generalizar para o ano (sazonalidade, efeito novidade).
- Extrapolar um efeito de 2 semanas para LTV sem proxy validada.

## Para ir além

- Kohavi, Tang & Xu, *Trustworthy Online Controlled Experiments* — a referência
  prática definitiva, escrita por quem rodou dezenas de milhares de experimentos.
- Deng et al. (2013), *Improving the Sensitivity of Online Controlled Experiments
  by Utilizing Pre-Experiment Data* — o artigo original do CUPED.
- Fabijan et al., *Diagnosing Sample Ratio Mismatch in Online Controlled
  Experiments* — o catálogo de causas de SRM.
- Johari et al., *Always Valid Inference* — a base dos testes sequenciais
  modernos que permitem espiar sem inflar erro.
