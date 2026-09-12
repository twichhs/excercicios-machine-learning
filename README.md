# 📚 Machine Learning & Deep Learning

> Uma trilha de estudos completa para virar cientista de dados — da primeira média aritmética até um modelo rodando em produção.

### 🧭 13 temas · 54 módulos · 142 notebooks-guia · 54 notebooks de exercícios

Escrito para quem tem **Python intermediário** e **estatística superficial**. Nenhum conceito estatístico aparece sem ser construído do zero antes — se você não sabe o que é um desvio-padrão, comece pelo tema 1 e siga a ordem.

---

## 🚀 Comece por aqui

### 1️⃣ Prepare o ambiente (uma vez só)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name curso-ml \
    --display-name "Python (curso ML)"
jupyter lab
```

### 2️⃣ Abra o primeiro módulo

```
01-estatistica/01-fundamentos-e-estatistica-descritiva/
```

### 3️⃣ Siga sempre o mesmo ciclo

```
   ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
   │  📕  LEIA       │     │  💻  RODE       │     │  ✏️  RESOLVA    │
   │                 │ ──▶ │                 │ ──▶ │                 │
   │   teoria.pdf    │     │ notebooks-guia  │     │  99-exercicios  │
   └─────────────────┘     └─────────────────┘     └─────────────────┘
     entenda a ideia         mexa nos números        agora sem apoio
```

---

## 🗂️ O que tem dentro de cada módulo

| Arquivo | O que é | Como usar |
| :-- | :-- | :-- |
| 📕 `teoria.pdf` *(módulos mais antigos)* | O material denso: conceitos, fórmulas, analogias e aplicações reais de mercado. | Leia um capítulo por vez, sem pressa. |
| 💻 `01-*.ipynb`, `02-*.ipynb`… | Notebooks-guia — já trazem a teoria embutida em células markdown (analogias, contexto de mercado, fórmulas), o código e um exemplo visual com parâmetros expostos para você alterar. Vêm com as saídas prontas. | Rode, mude os parâmetros no topo da célula, rode de novo, veja o que muda. |
| ✏️ `99-exercicios.ipynb` | Exercícios do módulo, com gabarito comentado logo abaixo de cada um. | Resolva **antes** de olhar a resposta. |
| 📝 `teoria.md` *(módulos mais antigos)* | A fonte de onde o PDF é gerado. | Só interessa se você for editar o material. |


> 💡 A partir de **Viés e Variância** (tema 6), os módulos novos não têm mais `teoria.pdf`: a teoria mora dentro dos próprios notebooks-guia, ao lado do código que a demonstra. Os módulos mais antigos continuam com os dois formatos, sem mudança.

> 💡 **Como saber se entendeu?** Se você consegue resolver o `99-exercicios` sem olhar o gabarito, entendeu. Se não consegue, volte à teoria (PDF ou notebook-guia) — não adianta seguir em frente, porque o próximo módulo assume este.

> 🟢 🟡 🔴 Os exercícios são marcados por dificuldade: **base**, **aplicação** e **síntese**. Se o tempo estiver curto, faça os 🟢 e 🟡 de todos os módulos antes de voltar aos 🔴.

---

## 🗺️ A trilha completa

### 📊 1. Estatística

> A base sobre a qual todo o resto se apoia: descrever, modelar a incerteza, estimar e decidir a partir de dados.

| Módulo | O que você vai aprender |
| :-- | :-- |
| **[Fundamentos e Estatística Descritiva](01-estatistica/01-fundamentos-e-estatistica-descritiva/teoria.pdf)** | Tipos de variável, medidas de posição e dispersão, momentos, robustez e as armadilhas de resumir dados em um número. |
| **[Distribuições de Probabilidade](01-estatistica/02-distribuicoes-de-probabilidade/teoria.pdf)** | O catálogo de modelos de incerteza: Bernoulli a Weibull, quando cada uma aparece e como reconhecê-la nos dados. |
| **[Inferência e Estimação](01-estatistica/03-inferencia-e-estimacao/teoria.pdf)** | De amostra para população: máxima verossimilhança, viés, erro-padrão, intervalos de confiança e bootstrap. |
| **[Testes de Hipótese](01-estatistica/04-testes-de-hipotese/teoria.pdf)** | p-valor, poder, erros tipo I e II, testes paramétricos e não-paramétricos, e o problema das comparações múltiplas. |
| **[Estatística Bayesiana](01-estatistica/05-estatistica-bayesiana/teoria.pdf)** | Priori, verossimilhança e posteriori; conjugadas, MCMC artesanal e por que o mercado migrou para o pensamento bayesiano. |
| **[A/B Testing e Desenho Experimental](01-estatistica/06-ab-testing-e-desenho-experimental/teoria.pdf)** | Aleatorização, métricas-guia, testes sequenciais, CUPED e os erros que invalidam experimentos em produção. |

📂 **[Abrir o tema completo](01-estatistica/README.md)**

---

### 🔢 2. Álgebra Linear e Otimização

> A maquinaria matemática que faz os modelos funcionarem: espaços vetoriais, decomposições e descida de gradiente.

| Módulo | O que você vai aprender |
| :-- | :-- |
| **[Vetores, Matrizes e Projeções](02-algebra-linear-e-otimizacao/01-vetores-matrizes-e-projecoes/teoria.pdf)** | Produto interno, norma, independência linear, posto e a projeção ortogonal que está por trás dos mínimos quadrados. |
| **[Decomposições: SVD, Autovalores e PCA](02-algebra-linear-e-otimizacao/02-decomposicoes-svd-e-autovalores/teoria.pdf)** | Autovalores, SVD, posto baixo e compressão — a mesma ideia sustenta PCA, sistemas de recomendação e embeddings. |
| **[Cálculo e Gradiente Descendente](02-algebra-linear-e-otimizacao/03-calculo-e-gradiente-descendente/teoria.pdf)** | Derivadas, gradiente, Hessiana, convexidade e as variantes de descida de gradiente que treinam todo modelo moderno. |

📂 **[Abrir o tema completo](02-algebra-linear-e-otimizacao/README.md)**

---

### 🧹 3. Preparação de Dados

> Onde um cientista de dados sênior gasta a maior parte do tempo — e onde a maioria dos projetos falha silenciosamente.

| Módulo | O que você vai aprender |
| :-- | :-- |
| **[Análise Exploratória (EDA)](03-preparacao-de-dados/01-analise-exploratoria/teoria.pdf)** | Um protocolo disciplinado de EDA: perfilamento, relações, hipóteses e o que procurar antes de qualquer modelo. |
| **[Dados Faltantes e Outliers](03-preparacao-de-dados/02-dados-faltantes-e-outliers/teoria.pdf)** | MCAR, MAR e MNAR; imputação simples, múltipla e por modelo; detecção e tratamento honesto de valores extremos. |
| **[Feature Engineering](03-preparacao-de-dados/03-feature-engineering/teoria.pdf)** | Transformações, interações, agregações temporais e features de domínio — o que ainda separa modelos bons de medianos. |
| **[Encoding, Escala e Vazamento de Dados](03-preparacao-de-dados/04-encoding-escala-e-vazamento/teoria.pdf)** | One-hot, ordinal, target encoding e escalonamento — sempre dentro de um Pipeline, para não vazar informação do futuro. |
| **[Dados Desbalanceados](03-preparacao-de-dados/05-dados-desbalanceados/teoria.pdf)** | Reamostragem, SMOTE, pesos de classe e ajuste de limiar — e por que acurácia é inútil quando a classe rara é o que importa. |

📂 **[Abrir o tema completo](03-preparacao-de-dados/README.md)**

---

### 🎯 4. Aprendizado Supervisionado

> Os modelos que preveem um rótulo conhecido — do mais interpretável ao campeão de competições.

| Módulo | O que você vai aprender |
| :-- | :-- |
| **[Regressão Linear](04-aprendizado-supervisionado/01-regressao-linear/teoria.pdf)** | O modelo fundador: formulação matricial, pressupostos, diagnóstico de resíduos e interpretação de coeficientes. |
| **[Regularização: Ridge, Lasso e Elastic Net](04-aprendizado-supervisionado/02-regularizacao/teoria.pdf)** | Penalizar para generalizar: a geometria de L1 e L2, seleção automática de variáveis e o trade-off viés-variância na prática. |
| **[Regressão Logística](04-aprendizado-supervisionado/03-regressao-logistica/teoria.pdf)** | Classificação probabilística, odds ratio, entropia cruzada e por que ela continua sendo o baseline de crédito e risco. |
| **[k-NN e Naive Bayes](04-aprendizado-supervisionado/04-knn-e-naive-bayes/teoria.pdf)** | Dois extremos didáticos: o modelo que não aprende nada e o que assume independência total — ambos ainda úteis. |
| **[Máquinas de Vetores de Suporte](04-aprendizado-supervisionado/05-maquinas-de-vetores-de-suporte/teoria.pdf)** | Margem máxima, o truque do kernel e a dualidade — a ideia geométrica mais elegante do aprendizado supervisionado. |
| **[Árvores de Decisão](04-aprendizado-supervisionado/06-arvores-de-decisao/teoria.pdf)** | Impureza de Gini, entropia, ganho de informação, poda e as razões pelas quais uma árvore sozinha quase sempre sobreajusta. |
| **[Bagging e Random Forest](04-aprendizado-supervisionado/07-bagging-e-random-forest/teoria.pdf)** | Bootstrap agregado, descorrelação por amostragem de features, erro OOB e importância de variáveis feita direito. |
| **[Boosting: Gradient Boosting, XGBoost e LightGBM](04-aprendizado-supervisionado/08-boosting/teoria.pdf)** | Aprendizado sequencial sobre resíduos, o gradiente funcional e o estado da arte em dados tabulares. |

📂 **[Abrir o tema completo](04-aprendizado-supervisionado/README.md)**

---

### 🔍 5. Aprendizado Não Supervisionado

> Encontrar estrutura sem rótulo: agrupar, comprimir e detectar o que foge do padrão.

| Módulo | O que você vai aprender |
| :-- | :-- |
| **[Clustering](05-aprendizado-nao-supervisionado/01-clustering/teoria.pdf)** | k-means, hierárquico, DBSCAN e misturas gaussianas — como escolher k e como saber se o agrupamento significa algo. |
| **[Redução de Dimensionalidade](05-aprendizado-nao-supervisionado/02-reducao-de-dimensionalidade/teoria.pdf)** | PCA, t-SNE, UMAP e autoencoders: comprimir preservando o que importa, e como não se enganar com visualizações 2D. |
| **[Detecção de Anomalias](05-aprendizado-nao-supervisionado/03-deteccao-de-anomalias/teoria.pdf)** | Métodos estatísticos, Isolation Forest, LOF e autoencoders para fraude, falha de equipamento e segurança. |

📂 **[Abrir o tema completo](05-aprendizado-nao-supervisionado/README.md)**

---

### ⚖️ 6. Avaliação e Validação de Modelos

> A disciplina que separa o cientista de dados sênior do júnior: saber se o número que você reportou é real.

| Módulo | O que você vai aprender |
| :-- | :-- |
| **[Métricas de Classificação](06-avaliacao-e-validacao/01-metricas-de-classificacao/teoria.pdf)** | Matriz de confusão, precisão, recall, F1, ROC-AUC, PR-AUC e como escolher a métrica a partir do custo do erro. |
| **[Métricas de Regressão](06-avaliacao-e-validacao/02-metricas-de-regressao/teoria.pdf)** | MAE, RMSE, MAPE, R² e quantis — cada uma otimiza um comportamento diferente do modelo. |
| **[Validação Cruzada](06-avaliacao-e-validacao/03-validacao-cruzada/teoria.pdf)** | k-fold, estratificado, por grupo e temporal — e o vazamento sutil que infla o resultado de quase todo notebook. |
| **[Viés, Variância e Curvas de Aprendizado](06-avaliacao-e-validacao/04-vies-variancia/01-decomposicao-do-erro.ipynb)** | A decomposição do erro, curvas de aprendizado e de complexidade como ferramenta de diagnóstico. |
| **[Calibração de Probabilidades](06-avaliacao-e-validacao/05-calibracao-de-probabilidades/01-por-que-calibrar.ipynb)** | Quando 0,8 precisa mesmo significar 80%: Platt, isotônica, Brier score e diagramas de confiabilidade. |
| **[Otimização de Hiperparâmetros](06-avaliacao-e-validacao/06-otimizacao-de-hiperparametros/01-grid-e-random-search.ipynb)** | Grid, random, otimização bayesiana e Hyperband — com o orçamento computacional como restrição de projeto. |

📂 **[Abrir o tema completo](06-avaliacao-e-validacao/README.md)**

---

### 📈 7. Séries Temporais

> Dados com memória: onde a suposição de independência quebra e a validação precisa respeitar o tempo.

| Módulo | O que você vai aprender |
| :-- | :-- |
| **[Fundamentos e Estacionariedade](07-series-temporais/01-fundamentos-e-estacionariedade/01-decomposicao-e-acf.ipynb)** | Tendência, sazonalidade, autocorrelação, decomposição e os testes de raiz unitária. |
| **[ARIMA, SARIMA e Suavização Exponencial](07-series-temporais/02-modelos-classicos/01-arima.ipynb)** | A família Box-Jenkins e os modelos de espaço de estados que ainda vencem baselines em produção. |
| **[Machine Learning para Séries Temporais](07-series-temporais/03-ml-para-series-temporais/01-features-de-defasagem.ipynb)** | Transformar previsão em problema supervisionado: janelas, features de defasagem e validação com origem móvel. |

📂 **[Abrir o tema completo](07-series-temporais/README.md)**

---

### 🧠 8. Deep Learning

> Redes neurais do neurônio artificial ao Transformer, sempre implementadas antes de serem usadas por bibliotecas.

| Módulo | O que você vai aprender |
| :-- | :-- |
| **[Fundamentos de Redes Neurais](08-deep-learning/01-fundamentos-de-redes-neurais/01-perceptron.ipynb)** | Perceptron, funções de ativação, a rede densa como composição de funções e o teorema da aproximação universal. |
| **[Backpropagation](08-deep-learning/02-backpropagation/01-regra-da-cadeia-e-grafo.ipynb)** | A regra da cadeia como grafo computacional: derivar, implementar e depurar o algoritmo que treina tudo. |
| **[Treinamento, Otimizadores e Regularização](08-deep-learning/03-treinamento-e-regularizacao/01-otimizadores.ipynb)** | SGD, momentum, Adam, agendadores de taxa, dropout, batch norm, early stopping e o diagnóstico de um treino que trava. |
| **[Redes Convolucionais](08-deep-learning/04-redes-convolucionais/01-convolucao-na-mao.ipynb)** | Convolução, campo receptivo, pooling e as arquiteturas que resolveram visão computacional. |
| **[Modelos Sequenciais: RNN, LSTM e GRU](08-deep-learning/05-modelos-sequenciais/01-rnn-do-zero.ipynb)** | Memória, gradientes que desaparecem e as portas que resolveram o problema — com aplicação em séries e texto. |
| **[Atenção e Transformers](08-deep-learning/06-atencao-e-transformers/01-atencao-do-zero.ipynb)** | Self-attention, multi-head, codificação posicional e a arquitetura que redefiniu a área inteira. |
| **[Embeddings e Transfer Learning](08-deep-learning/07-embeddings-e-transferencia/01-embeddings.ipynb)** | Representações densas, similaridade vetorial, fine-tuning e a economia de reaproveitar modelos pré-treinados. |

📂 **[Abrir o tema completo](08-deep-learning/README.md)**

---

### 💬 9. NLP e Modelos de Linguagem

> Do saco de palavras aos LLMs: representar, classificar e avaliar sistemas que lidam com linguagem.

| Módulo | O que você vai aprender |
| :-- | :-- |
| **[Representação de Texto](09-nlp-e-llms/01-representacao-de-texto/01-tokenizacao-e-bow.ipynb)** | Tokenização, normalização, bag-of-words, TF-IDF, n-gramas e embeddings estáticos. |
| **[Classificação de Texto e Tópicos](09-nlp-e-llms/02-classificacao-e-topicos/01-classificacao-de-texto.ipynb)** | Pipelines de classificação, análise de sentimento e modelagem de tópicos com NMF e LDA. |
| **[LLMs, RAG e Avaliação](09-nlp-e-llms/03-llms-rag-e-avaliacao/01-como-um-llm-gera-texto.ipynb)** | Como um LLM gera texto, o que é RAG, engenharia de contexto e como avaliar sistemas generativos sem se enganar. |

📂 **[Abrir o tema completo](09-nlp-e-llms/README.md)**

---

### 🔗 10. Inferência Causal

> A pergunta que o negócio realmente faz — 'o que acontece se eu agir?' — e que correlação nunca responde.

| Módulo | O que você vai aprender |
| :-- | :-- |
| **[Fundamentos de Causalidade](10-inferencia-causal/01-fundamentos-de-causalidade/01-resultados-potenciais.ipynb)** | Resultados potenciais, confundimento, DAGs, o critério de porta dos fundos e o paradoxo de Simpson. |
| **[Métodos Quase-Experimentais](10-inferencia-causal/02-metodos-quase-experimentais/01-propensity-score.ipynb)** | Pareamento, escore de propensão, diferenças-em-diferenças e variáveis instrumentais quando o A/B é impossível. |
| **[Uplift Modeling](10-inferencia-causal/03-uplift-modeling/01-efeito-heterogeneo.ipynb)** | Prever o efeito incremental do tratamento por indivíduo — a diferença entre prever churn e evitar churn. |

📂 **[Abrir o tema completo](10-inferencia-causal/README.md)**

---

### 🔬 11. Interpretabilidade e Fairness

> Explicar decisões automatizadas e auditar seus danos — hoje uma exigência regulatória, não um diferencial.

| Módulo | O que você vai aprender |
| :-- | :-- |
| **[Interpretabilidade de Modelos](11-interpretabilidade-e-fairness/01-interpretabilidade/01-importancia-e-pdp.ipynb)** | Importância por permutação, PDP, ICE, LIME e SHAP — o que cada método realmente responde. |
| **[Fairness e Viés Algorítmico](11-interpretabilidade-e-fairness/02-fairness-e-vies/01-metricas-de-fairness.ipynb)** | Definições concorrentes de justiça, sua incompatibilidade matemática e técnicas de mitigação. |

📂 **[Abrir o tema completo](11-interpretabilidade-e-fairness/README.md)**

---

### 🛒 12. Sistemas de Recomendação

> O produto de machine learning que mais gera receita direta — e um problema de avaliação notoriamente traiçoeiro.

| Módulo | O que você vai aprender |
| :-- | :-- |
| **[Filtragem Colaborativa](12-sistemas-de-recomendacao/01-filtragem-colaborativa/01-vizinhanca.ipynb)** | Vizinhança usuário-item, fatoração de matrizes, ALS e o problema da partida a frio. |
| **[Avaliação e Modelos Híbridos](12-sistemas-de-recomendacao/02-avaliacao-e-hibridos/01-metricas-de-ranking.ipynb)** | Métricas de ranking, viés de popularidade, diversidade e arquiteturas híbridas de duas torres. |

📂 **[Abrir o tema completo](12-sistemas-de-recomendacao/README.md)**

---

### 🚀 13. MLOps e Produção

> O modelo só vale quando roda, é observável e pode ser revertido — a engenharia que transforma notebook em sistema.

| Módulo | O que você vai aprender |
| :-- | :-- |
| **[Pipelines e Reprodutibilidade](13-mlops-e-producao/01-pipelines-e-reprodutibilidade/01-pipelines-robustos.ipynb)** | Pipelines do scikit-learn, versionamento de dados e modelos, seeds e o registro de experimentos. |
| **[Deploy e Monitoramento](13-mlops-e-producao/02-deploy-e-monitoramento/01-servindo-um-modelo.ipynb)** | Batch vs online, feature store, contrato de dados, latência e o que instrumentar antes de ir para produção. |
| **[Data Drift e Retreino](13-mlops-e-producao/03-drift-e-retreino/01-deteccao-de-drift.ipynb)** | Drift de covariáveis, de conceito e de rótulo; testes estatísticos de detecção e políticas de retreino. |

📂 **[Abrir o tema completo](13-mlops-e-producao/README.md)**

---

## 🛠️ Reconstruindo o material

Só é necessário se você for **editar** o conteúdo. Para estudar, basta abrir os arquivos.

```bash
python _ferramentas/build.py status      # o que já existe e o que falta
python _ferramentas/build.py tudo        # estrutura + PDFs + notebooks
python _ferramentas/build.py pdfs 01-estatistica   # só um tema
```

### ⚙️ Como o material é construído

| Etapa | Ferramenta | Detalhe |
| :-- | :-- | :-- |
| Estrutura e índices | `_ferramentas/curriculo.py` | Fonte única de verdade: temas, módulos e notebooks. |
| PDFs | `_ferramentas/md2pdf.py` | ReportLab para o layout e o motor `mathtext` do Matplotlib para as fórmulas — **sem depender de LaTeX**. |
| Notebooks | `_ferramentas/py2nb.py` | Os notebooks são escritos como scripts em `_fontes/` e só viram `.ipynb` **depois de executarem sem erro**. Nenhum notebook chega quebrado. |

---

## ❓ Dúvidas frequentes

**Preciso saber matemática avançada?**  
Não. Álgebra do ensino médio basta. O tema 2 constrói a álgebra linear necessária a partir da geometria.

**Posso pular temas?**  
Os temas 1 a 3 são pré-requisito de tudo. A partir do tema 4, dá para escolher — mas Deep Learning (8) assume Otimização (2), e NLP (9) assume Deep Learning.

**Os notebooks precisam ser executados?**  
Não para ler: as saídas já vêm embutidas. Sim para aprender: o material foi feito para ser alterado.

**Quanto tempo leva?**  
Cada módulo tem entre 8 e 12 horas de leitura mais prática. São 54 módulos — trate como uma maratona de meses, não um fim de semana.
