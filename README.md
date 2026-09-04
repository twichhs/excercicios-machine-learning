# Machine Learning & Deep Learning — Material do Curso

Material completo para formação de cientistas de dados, escrito para quem tem **Python intermediário** e **estatística superficial**. Cada conceito estatístico é construído do zero antes de ser usado.

**13 temas · 54 módulos · 142 notebooks-guia · 54 notebooks de exercícios**

## Como o material está organizado

```
<tema>/
  <módulo>/
    teoria.md      ← fonte do material teórico
    teoria.pdf     ← PDF denso: conceitos, fórmulas, aplicações reais
    NN-*.ipynb     ← notebooks-guia executáveis, muito comentados
    99-exercicios.ipynb  ← exercícios do módulo, com gabarito comentado
```

## Índice

### [Estatística](01-estatistica/README.md)

_A base sobre a qual todo o resto se apoia: descrever, modelar a incerteza, estimar e decidir a partir de dados._

- **[Fundamentos e Estatística Descritiva](01-estatistica/01-fundamentos-e-estatistica-descritiva/teoria.pdf)** — Tipos de variável, medidas de posição e dispersão, momentos, robustez e as armadilhas de resumir dados em um número.
- **[Distribuições de Probabilidade](01-estatistica/02-distribuicoes-de-probabilidade/teoria.pdf)** — O catálogo de modelos de incerteza: Bernoulli a Weibull, quando cada uma aparece e como reconhecê-la nos dados.
- **[Inferência e Estimação](01-estatistica/03-inferencia-e-estimacao/teoria.pdf)** — De amostra para população: máxima verossimilhança, viés, erro-padrão, intervalos de confiança e bootstrap.
- **[Testes de Hipótese](01-estatistica/04-testes-de-hipotese/teoria.pdf)** — p-valor, poder, erros tipo I e II, testes paramétricos e não-paramétricos, e o problema das comparações múltiplas.
- **[Estatística Bayesiana](01-estatistica/05-estatistica-bayesiana/teoria.pdf)** — Priori, verossimilhança e posteriori; conjugadas, MCMC artesanal e por que o mercado migrou para o pensamento bayesiano.
- **[A/B Testing e Desenho Experimental](01-estatistica/06-ab-testing-e-desenho-experimental/teoria.pdf)** — Aleatorização, métricas-guia, testes sequenciais, CUPED e os erros que invalidam experimentos em produção.

### [Álgebra Linear e Otimização](02-algebra-linear-e-otimizacao/README.md)

_A maquinaria matemática que faz os modelos funcionarem: espaços vetoriais, decomposições e descida de gradiente._

- **[Vetores, Matrizes e Projeções](02-algebra-linear-e-otimizacao/01-vetores-matrizes-e-projecoes/teoria.pdf)** — Produto interno, norma, independência linear, posto e a projeção ortogonal que está por trás dos mínimos quadrados.
- **[Decomposições: SVD, Autovalores e PCA](02-algebra-linear-e-otimizacao/02-decomposicoes-svd-e-autovalores/teoria.pdf)** — Autovalores, SVD, posto baixo e compressão — a mesma ideia sustenta PCA, sistemas de recomendação e embeddings.
- **[Cálculo e Gradiente Descendente](02-algebra-linear-e-otimizacao/03-calculo-e-gradiente-descendente/teoria.pdf)** — Derivadas, gradiente, Hessiana, convexidade e as variantes de descida de gradiente que treinam todo modelo moderno.

### [Preparação de Dados](03-preparacao-de-dados/README.md)

_Onde um cientista de dados sênior gasta a maior parte do tempo — e onde a maioria dos projetos falha silenciosamente._

- **[Análise Exploratória (EDA)](03-preparacao-de-dados/01-analise-exploratoria/teoria.pdf)** — Um protocolo disciplinado de EDA: perfilamento, relações, hipóteses e o que procurar antes de qualquer modelo.
- **[Dados Faltantes e Outliers](03-preparacao-de-dados/02-dados-faltantes-e-outliers/teoria.pdf)** — MCAR, MAR e MNAR; imputação simples, múltipla e por modelo; detecção e tratamento honesto de valores extremos.
- **[Feature Engineering](03-preparacao-de-dados/03-feature-engineering/teoria.pdf)** — Transformações, interações, agregações temporais e features de domínio — o que ainda separa modelos bons de medianos.
- **[Encoding, Escala e Vazamento de Dados](03-preparacao-de-dados/04-encoding-escala-e-vazamento/teoria.pdf)** — One-hot, ordinal, target encoding e escalonamento — sempre dentro de um Pipeline, para não vazar informação do futuro.
- **[Dados Desbalanceados](03-preparacao-de-dados/05-dados-desbalanceados/teoria.pdf)** — Reamostragem, SMOTE, pesos de classe e ajuste de limiar — e por que acurácia é inútil quando a classe rara é o que importa.

### [Aprendizado Supervisionado](04-aprendizado-supervisionado/README.md)

_Os modelos que preveem um rótulo conhecido — do mais interpretável ao campeão de competições._

- **[Regressão Linear](04-aprendizado-supervisionado/01-regressao-linear/teoria.pdf)** — O modelo fundador: formulação matricial, pressupostos, diagnóstico de resíduos e interpretação de coeficientes.
- **[Regularização: Ridge, Lasso e Elastic Net](04-aprendizado-supervisionado/02-regularizacao/teoria.pdf)** — Penalizar para generalizar: a geometria de L1 e L2, seleção automática de variáveis e o trade-off viés-variância na prática.
- **[Regressão Logística](04-aprendizado-supervisionado/03-regressao-logistica/teoria.pdf)** — Classificação probabilística, odds ratio, entropia cruzada e por que ela continua sendo o baseline de crédito e risco.
- **[k-NN e Naive Bayes](04-aprendizado-supervisionado/04-knn-e-naive-bayes/teoria.pdf)** — Dois extremos didáticos: o modelo que não aprende nada e o que assume independência total — ambos ainda úteis.
- **[Máquinas de Vetores de Suporte](04-aprendizado-supervisionado/05-maquinas-de-vetores-de-suporte/teoria.pdf)** — Margem máxima, o truque do kernel e a dualidade — a ideia geométrica mais elegante do aprendizado supervisionado.
- **[Árvores de Decisão](04-aprendizado-supervisionado/06-arvores-de-decisao/teoria.pdf)** — Impureza de Gini, entropia, ganho de informação, poda e as razões pelas quais uma árvore sozinha quase sempre sobreajusta.
- **[Bagging e Random Forest](04-aprendizado-supervisionado/07-bagging-e-random-forest/teoria.pdf)** — Bootstrap agregado, descorrelação por amostragem de features, erro OOB e importância de variáveis feita direito.
- **[Boosting: Gradient Boosting, XGBoost e LightGBM](04-aprendizado-supervisionado/08-boosting/teoria.pdf)** — Aprendizado sequencial sobre resíduos, o gradiente funcional e o estado da arte em dados tabulares.

### [Aprendizado Não Supervisionado](05-aprendizado-nao-supervisionado/README.md)

_Encontrar estrutura sem rótulo: agrupar, comprimir e detectar o que foge do padrão._

- **[Clustering](05-aprendizado-nao-supervisionado/01-clustering/teoria.pdf)** — k-means, hierárquico, DBSCAN e misturas gaussianas — como escolher k e como saber se o agrupamento significa algo.
- **[Redução de Dimensionalidade](05-aprendizado-nao-supervisionado/02-reducao-de-dimensionalidade/teoria.pdf)** — PCA, t-SNE, UMAP e autoencoders: comprimir preservando o que importa, e como não se enganar com visualizações 2D.
- **[Detecção de Anomalias](05-aprendizado-nao-supervisionado/03-deteccao-de-anomalias/teoria.pdf)** — Métodos estatísticos, Isolation Forest, LOF e autoencoders para fraude, falha de equipamento e segurança.

### [Avaliação e Validação de Modelos](06-avaliacao-e-validacao/README.md)

_A disciplina que separa o cientista de dados sênior do júnior: saber se o número que você reportou é real._

- **[Métricas de Classificação](06-avaliacao-e-validacao/01-metricas-de-classificacao/teoria.pdf)** — Matriz de confusão, precisão, recall, F1, ROC-AUC, PR-AUC e como escolher a métrica a partir do custo do erro.
- **[Métricas de Regressão](06-avaliacao-e-validacao/02-metricas-de-regressao/teoria.pdf)** — MAE, RMSE, MAPE, R² e quantis — cada uma otimiza um comportamento diferente do modelo.
- **[Validação Cruzada](06-avaliacao-e-validacao/03-validacao-cruzada/teoria.pdf)** — k-fold, estratificado, por grupo e temporal — e o vazamento sutil que infla o resultado de quase todo notebook.
- **[Viés, Variância e Curvas de Aprendizado](06-avaliacao-e-validacao/04-vies-variancia/teoria.pdf)** — A decomposição do erro, curvas de aprendizado e de complexidade como ferramenta de diagnóstico.
- **[Calibração de Probabilidades](06-avaliacao-e-validacao/05-calibracao-de-probabilidades/teoria.pdf)** — Quando 0,8 precisa mesmo significar 80%: Platt, isotônica, Brier score e diagramas de confiabilidade.
- **[Otimização de Hiperparâmetros](06-avaliacao-e-validacao/06-otimizacao-de-hiperparametros/teoria.pdf)** — Grid, random, otimização bayesiana e Hyperband — com o orçamento computacional como restrição de projeto.

### [Séries Temporais](07-series-temporais/README.md)

_Dados com memória: onde a suposição de independência quebra e a validação precisa respeitar o tempo._

- **[Fundamentos e Estacionariedade](07-series-temporais/01-fundamentos-e-estacionariedade/teoria.pdf)** — Tendência, sazonalidade, autocorrelação, decomposição e os testes de raiz unitária.
- **[ARIMA, SARIMA e Suavização Exponencial](07-series-temporais/02-modelos-classicos/teoria.pdf)** — A família Box-Jenkins e os modelos de espaço de estados que ainda vencem baselines em produção.
- **[Machine Learning para Séries Temporais](07-series-temporais/03-ml-para-series-temporais/teoria.pdf)** — Transformar previsão em problema supervisionado: janelas, features de defasagem e validação com origem móvel.

### [Deep Learning](08-deep-learning/README.md)

_Redes neurais do neurônio artificial ao Transformer, sempre implementadas antes de serem usadas por bibliotecas._

- **[Fundamentos de Redes Neurais](08-deep-learning/01-fundamentos-de-redes-neurais/teoria.pdf)** — Perceptron, funções de ativação, a rede densa como composição de funções e o teorema da aproximação universal.
- **[Backpropagation](08-deep-learning/02-backpropagation/teoria.pdf)** — A regra da cadeia como grafo computacional: derivar, implementar e depurar o algoritmo que treina tudo.
- **[Treinamento, Otimizadores e Regularização](08-deep-learning/03-treinamento-e-regularizacao/teoria.pdf)** — SGD, momentum, Adam, agendadores de taxa, dropout, batch norm, early stopping e o diagnóstico de um treino que trava.
- **[Redes Convolucionais](08-deep-learning/04-redes-convolucionais/teoria.pdf)** — Convolução, campo receptivo, pooling e as arquiteturas que resolveram visão computacional.
- **[Modelos Sequenciais: RNN, LSTM e GRU](08-deep-learning/05-modelos-sequenciais/teoria.pdf)** — Memória, gradientes que desaparecem e as portas que resolveram o problema — com aplicação em séries e texto.
- **[Atenção e Transformers](08-deep-learning/06-atencao-e-transformers/teoria.pdf)** — Self-attention, multi-head, codificação posicional e a arquitetura que redefiniu a área inteira.
- **[Embeddings e Transfer Learning](08-deep-learning/07-embeddings-e-transferencia/teoria.pdf)** — Representações densas, similaridade vetorial, fine-tuning e a economia de reaproveitar modelos pré-treinados.

### [NLP e Modelos de Linguagem](09-nlp-e-llms/README.md)

_Do saco de palavras aos LLMs: representar, classificar e avaliar sistemas que lidam com linguagem._

- **[Representação de Texto](09-nlp-e-llms/01-representacao-de-texto/teoria.pdf)** — Tokenização, normalização, bag-of-words, TF-IDF, n-gramas e embeddings estáticos.
- **[Classificação de Texto e Tópicos](09-nlp-e-llms/02-classificacao-e-topicos/teoria.pdf)** — Pipelines de classificação, análise de sentimento e modelagem de tópicos com NMF e LDA.
- **[LLMs, RAG e Avaliação](09-nlp-e-llms/03-llms-rag-e-avaliacao/teoria.pdf)** — Como um LLM gera texto, o que é RAG, engenharia de contexto e como avaliar sistemas generativos sem se enganar.

### [Inferência Causal](10-inferencia-causal/README.md)

_A pergunta que o negócio realmente faz — 'o que acontece se eu agir?' — e que correlação nunca responde._

- **[Fundamentos de Causalidade](10-inferencia-causal/01-fundamentos-de-causalidade/teoria.pdf)** — Resultados potenciais, confundimento, DAGs, o critério de porta dos fundos e o paradoxo de Simpson.
- **[Métodos Quase-Experimentais](10-inferencia-causal/02-metodos-quase-experimentais/teoria.pdf)** — Pareamento, escore de propensão, diferenças-em-diferenças e variáveis instrumentais quando o A/B é impossível.
- **[Uplift Modeling](10-inferencia-causal/03-uplift-modeling/teoria.pdf)** — Prever o efeito incremental do tratamento por indivíduo — a diferença entre prever churn e evitar churn.

### [Interpretabilidade e Fairness](11-interpretabilidade-e-fairness/README.md)

_Explicar decisões automatizadas e auditar seus danos — hoje uma exigência regulatória, não um diferencial._

- **[Interpretabilidade de Modelos](11-interpretabilidade-e-fairness/01-interpretabilidade/teoria.pdf)** — Importância por permutação, PDP, ICE, LIME e SHAP — o que cada método realmente responde.
- **[Fairness e Viés Algorítmico](11-interpretabilidade-e-fairness/02-fairness-e-vies/teoria.pdf)** — Definições concorrentes de justiça, sua incompatibilidade matemática e técnicas de mitigação.

### [Sistemas de Recomendação](12-sistemas-de-recomendacao/README.md)

_O produto de machine learning que mais gera receita direta — e um problema de avaliação notoriamente traiçoeiro._

- **[Filtragem Colaborativa](12-sistemas-de-recomendacao/01-filtragem-colaborativa/teoria.pdf)** — Vizinhança usuário-item, fatoração de matrizes, ALS e o problema da partida a frio.
- **[Avaliação e Modelos Híbridos](12-sistemas-de-recomendacao/02-avaliacao-e-hibridos/teoria.pdf)** — Métricas de ranking, viés de popularidade, diversidade e arquiteturas híbridas de duas torres.

### [MLOps e Produção](13-mlops-e-producao/README.md)

_O modelo só vale quando roda, é observável e pode ser revertido — a engenharia que transforma notebook em sistema._

- **[Pipelines e Reprodutibilidade](13-mlops-e-producao/01-pipelines-e-reprodutibilidade/teoria.pdf)** — Pipelines do scikit-learn, versionamento de dados e modelos, seeds e o registro de experimentos.
- **[Deploy e Monitoramento](13-mlops-e-producao/02-deploy-e-monitoramento/teoria.pdf)** — Batch vs online, feature store, contrato de dados, latência e o que instrumentar antes de ir para produção.
- **[Data Drift e Retreino](13-mlops-e-producao/03-drift-e-retreino/teoria.pdf)** — Drift de covariáveis, de conceito e de rótulo; testes estatísticos de detecção e políticas de retreino.

## Preparando o ambiente

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name curso-ml \
    --display-name "Python (curso ML)"
jupyter lab
```

## Reconstruindo o material

```bash
python _ferramentas/build.py tudo        # estrutura + PDFs + notebooks
python _ferramentas/build.py pdfs 01-estatistica   # só um tema
python _ferramentas/build.py status      # o que falta
```

Os PDFs são gerados por uma toolchain própria (`_ferramentas/md2pdf.py`) que usa ReportLab para o layout e o motor `mathtext` do Matplotlib para renderizar as fórmulas — sem depender de LaTeX instalado.

Os notebooks são escritos como scripts em `_fontes/` (formato *percent*) e só viram `.ipynb` **depois de executarem sem erro**, com as saídas embutidas. Nenhum notebook do curso chega ao aluno quebrado.
