"""
curriculo.py — Fonte unica de verdade da estrutura do curso.

Cada TEMA vira uma pasta na raiz; cada MODULO vira uma subpasta do tema; cada
subpasta abriga um `teoria.md` (compilado para `teoria.pdf`) e os notebooks.

Os campos de cada modulo:
    slug       nome da subpasta
    titulo     titulo humano, usado na capa do PDF e nos indices
    resumo     uma frase que aparece na capa do PDF e no README do tema
    notebooks  lista de (slug_do_notebook, titulo_do_notebook)
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Modulo:
    slug: str
    titulo: str
    resumo: str
    notebooks: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class Tema:
    slug: str
    titulo: str
    resumo: str
    modulos: list[Modulo]


CURRICULO: list[Tema] = [
    Tema("01-estatistica", "Estatística",
         "A base sobre a qual todo o resto se apoia: descrever, modelar a "
         "incerteza, estimar e decidir a partir de dados.",
         [
             Modulo("01-fundamentos-e-estatistica-descritiva",
                    "Fundamentos e Estatística Descritiva",
                    "Tipos de variável, medidas de posição e dispersão, momentos, "
                    "robustez e as armadilhas de resumir dados em um número.",
                    [("01-medidas-resumo", "Medidas-resumo na prática"),
                     ("02-robustez-e-outliers", "Robustez: quando a média mente"),
                     ("03-visualizacao-distribuicoes", "Enxergando a distribuição")]),
             Modulo("02-distribuicoes-de-probabilidade",
                    "Distribuições de Probabilidade",
                    "O catálogo de modelos de incerteza: Bernoulli a Weibull, "
                    "quando cada uma aparece e como reconhecê-la nos dados.",
                    [("01-discretas", "Distribuições discretas"),
                     ("02-continuas", "Distribuições contínuas"),
                     ("03-tlc-e-lei-dos-grandes-numeros", "TLC e Lei dos Grandes Números"),
                     ("04-ajuste-a-dados-reais", "Ajustando distribuições a dados reais")]),
             Modulo("03-inferencia-e-estimacao",
                    "Inferência e Estimação",
                    "De amostra para população: máxima verossimilhança, viés, "
                    "erro-padrão, intervalos de confiança e bootstrap.",
                    [("01-estimadores-e-propriedades", "Estimadores e suas propriedades"),
                     ("02-maxima-verossimilhanca", "Máxima verossimilhança do zero"),
                     ("03-intervalos-e-bootstrap", "Intervalos de confiança e bootstrap")]),
             Modulo("04-testes-de-hipotese",
                    "Testes de Hipótese",
                    "p-valor, poder, erros tipo I e II, testes paramétricos e "
                    "não-paramétricos, e o problema das comparações múltiplas.",
                    [("01-logica-do-teste", "A lógica do teste de hipótese"),
                     ("02-testes-classicos", "Catálogo de testes clássicos"),
                     ("03-poder-e-tamanho-de-amostra", "Poder e tamanho de amostra"),
                     ("04-comparacoes-multiplas", "Comparações múltiplas")]),
             Modulo("05-estatistica-bayesiana",
                    "Estatística Bayesiana",
                    "Priori, verossimilhança e posteriori; conjugadas, MCMC "
                    "artesanal e por que o mercado migrou para o pensamento bayesiano.",
                    [("01-teorema-de-bayes", "Teorema de Bayes na prática"),
                     ("02-conjugadas-e-posteriori", "Conjugadas e atualização sequencial"),
                     ("03-mcmc-do-zero", "MCMC do zero: Metropolis-Hastings")]),
             Modulo("06-ab-testing-e-desenho-experimental",
                    "A/B Testing e Desenho Experimental",
                    "Aleatorização, métricas-guia, testes sequenciais, CUPED e "
                    "os erros que invalidam experimentos em produção.",
                    [("01-desenho-e-aleatorizacao", "Desenho e aleatorização"),
                     ("02-analise-de-experimento", "Análise ponta a ponta de um A/B"),
                     ("03-cuped-e-variancia", "CUPED e redução de variância")]),
         ]),

    Tema("02-algebra-linear-e-otimizacao", "Álgebra Linear e Otimização",
         "A maquinaria matemática que faz os modelos funcionarem: espaços "
         "vetoriais, decomposições e descida de gradiente.",
         [
             Modulo("01-vetores-matrizes-e-projecoes",
                    "Vetores, Matrizes e Projeções",
                    "Produto interno, norma, independência linear, posto e a "
                    "projeção ortogonal que está por trás dos mínimos quadrados.",
                    [("01-algebra-com-numpy", "Álgebra linear com NumPy"),
                     ("02-projecao-e-minimos-quadrados", "Projeção e mínimos quadrados")]),
             Modulo("02-decomposicoes-svd-e-autovalores",
                    "Decomposições: SVD, Autovalores e PCA",
                    "Autovalores, SVD, posto baixo e compressão — a mesma ideia "
                    "sustenta PCA, sistemas de recomendação e embeddings.",
                    [("01-autovalores-e-autovetores", "Autovalores e autovetores"),
                     ("02-svd-e-posto-baixo", "SVD e aproximação de posto baixo"),
                     ("03-pca-do-zero", "PCA implementado do zero")]),
             Modulo("03-calculo-e-gradiente-descendente",
                    "Cálculo e Gradiente Descendente",
                    "Derivadas, gradiente, Hessiana, convexidade e as variantes "
                    "de descida de gradiente que treinam todo modelo moderno.",
                    [("01-derivadas-e-gradientes", "Derivadas e gradientes numéricos"),
                     ("02-gradiente-descendente", "Gradiente descendente do zero"),
                     ("03-convexidade-e-condicionamento", "Convexidade e condicionamento")]),
         ]),

    Tema("03-preparacao-de-dados", "Preparação de Dados",
         "Onde um cientista de dados sênior gasta a maior parte do tempo — e "
         "onde a maioria dos projetos falha silenciosamente.",
         [
             Modulo("01-analise-exploratoria",
                    "Análise Exploratória (EDA)",
                    "Um protocolo disciplinado de EDA: perfilamento, relações, "
                    "hipóteses e o que procurar antes de qualquer modelo.",
                    [("01-protocolo-de-eda", "Protocolo de EDA ponta a ponta"),
                     ("02-relacoes-e-correlacao", "Relações, correlação e associação")]),
             Modulo("02-dados-faltantes-e-outliers",
                    "Dados Faltantes e Outliers",
                    "MCAR, MAR e MNAR; imputação simples, múltipla e por modelo; "
                    "detecção e tratamento honesto de valores extremos.",
                    [("01-mecanismos-de-ausencia", "Mecanismos de ausência"),
                     ("02-estrategias-de-imputacao", "Estratégias de imputação"),
                     ("03-outliers", "Detecção e tratamento de outliers")]),
             Modulo("03-feature-engineering",
                    "Feature Engineering",
                    "Transformações, interações, agregações temporais e features "
                    "de domínio — o que ainda separa modelos bons de medianos.",
                    [("01-transformacoes-numericas", "Transformações numéricas"),
                     ("02-features-temporais-e-agregacoes", "Features temporais e agregações"),
                     ("03-selecao-de-features", "Seleção de features")]),
             Modulo("04-encoding-escala-e-vazamento",
                    "Encoding, Escala e Vazamento de Dados",
                    "One-hot, ordinal, target encoding e escalonamento — sempre "
                    "dentro de um Pipeline, para não vazar informação do futuro.",
                    [("01-encoding-de-categoricas", "Encoding de variáveis categóricas"),
                     ("02-escalonamento", "Escalonamento e normalização"),
                     ("03-vazamento-de-dados", "Vazamento de dados: anatomia de um desastre")]),
             Modulo("05-dados-desbalanceados",
                    "Dados Desbalanceados",
                    "Reamostragem, SMOTE, pesos de classe e ajuste de limiar — e "
                    "por que acurácia é inútil quando a classe rara é o que importa.",
                    [("01-o-problema-do-desbalanceamento", "O problema do desbalanceamento"),
                     ("02-tecnicas-de-reamostragem", "Reamostragem e pesos de classe"),
                     ("03-limiar-e-custo", "Ajuste de limiar por custo de negócio")]),
         ]),

    Tema("04-aprendizado-supervisionado", "Aprendizado Supervisionado",
         "Os modelos que preveem um rótulo conhecido — do mais interpretável "
         "ao campeão de competições.",
         [
             Modulo("01-regressao-linear", "Regressão Linear",
                    "O modelo fundador: formulação matricial, pressupostos, "
                    "diagnóstico de resíduos e interpretação de coeficientes.",
                    [("01-regressao-do-zero", "Regressão linear do zero"),
                     ("02-pressupostos-e-diagnostico", "Pressupostos e diagnóstico"),
                     ("03-caso-real-precificacao", "Caso real: precificação")]),
             Modulo("02-regularizacao", "Regularização: Ridge, Lasso e Elastic Net",
                    "Penalizar para generalizar: a geometria de L1 e L2, seleção "
                    "automática de variáveis e o trade-off viés-variância na prática.",
                    [("01-ridge-e-lasso", "Ridge e Lasso na prática"),
                     ("02-geometria-da-regularizacao", "A geometria de L1 vs L2"),
                     ("03-elasticnet-e-tuning", "Elastic Net e ajuste de alpha")]),
             Modulo("03-regressao-logistica", "Regressão Logística",
                    "Classificação probabilística, odds ratio, entropia cruzada "
                    "e por que ela continua sendo o baseline de crédito e risco.",
                    [("01-logistica-do-zero", "Regressão logística do zero"),
                     ("02-interpretacao-odds-ratio", "Interpretando odds ratio"),
                     ("03-caso-real-credito", "Caso real: scoring de crédito")]),
             Modulo("04-knn-e-naive-bayes", "k-NN e Naive Bayes",
                    "Dois extremos didáticos: o modelo que não aprende nada e o "
                    "que assume independência total — ambos ainda úteis.",
                    [("01-knn", "k-NN e a maldição da dimensionalidade"),
                     ("02-naive-bayes", "Naive Bayes e classificação de texto")]),
             Modulo("05-maquinas-de-vetores-de-suporte", "Máquinas de Vetores de Suporte",
                    "Margem máxima, o truque do kernel e a dualidade — a ideia "
                    "geométrica mais elegante do aprendizado supervisionado.",
                    [("01-margem-e-svm-linear", "Margem máxima e SVM linear"),
                     ("02-kernels", "O truque do kernel")]),
             Modulo("06-arvores-de-decisao", "Árvores de Decisão",
                    "Impureza de Gini, entropia, ganho de informação, poda e as "
                    "razões pelas quais uma árvore sozinha quase sempre sobreajusta.",
                    [("01-arvore-do-zero", "Árvore de decisão do zero"),
                     ("02-poda-e-hiperparametros", "Poda e hiperparâmetros")]),
             Modulo("07-bagging-e-random-forest", "Bagging e Random Forest",
                    "Bootstrap agregado, descorrelação por amostragem de features, "
                    "erro OOB e importância de variáveis feita direito.",
                    [("01-bagging-e-variancia", "Bagging e redução de variância"),
                     ("02-random-forest", "Random Forest na prática"),
                     ("03-importancia-de-features", "Importância de features sem enganação")]),
             Modulo("08-boosting", "Boosting: Gradient Boosting, XGBoost e LightGBM",
                    "Aprendizado sequencial sobre resíduos, o gradiente funcional "
                    "e o estado da arte em dados tabulares.",
                    [("01-gradient-boosting-do-zero", "Gradient Boosting do zero"),
                     ("02-xgboost-e-lightgbm", "XGBoost e LightGBM"),
                     ("03-tuning-e-early-stopping", "Tuning e early stopping")]),
         ]),

    Tema("05-aprendizado-nao-supervisionado", "Aprendizado Não Supervisionado",
         "Encontrar estrutura sem rótulo: agrupar, comprimir e detectar o que "
         "foge do padrão.",
         [
             Modulo("01-clustering", "Clustering",
                    "k-means, hierárquico, DBSCAN e misturas gaussianas — como "
                    "escolher k e como saber se o agrupamento significa algo.",
                    [("01-kmeans", "k-means e escolha de k"),
                     ("02-hierarquico-e-dbscan", "Hierárquico e DBSCAN"),
                     ("03-gmm-e-avaliacao", "Misturas gaussianas e avaliação"),
                     ("04-caso-real-segmentacao", "Caso real: segmentação de clientes")]),
             Modulo("02-reducao-de-dimensionalidade", "Redução de Dimensionalidade",
                    "PCA, t-SNE, UMAP e autoencoders: comprimir preservando o que "
                    "importa, e como não se enganar com visualizações 2D.",
                    [("01-pca-aplicado", "PCA aplicado"),
                     ("02-tsne-e-manifolds", "t-SNE e métodos de manifold")]),
             Modulo("03-deteccao-de-anomalias", "Detecção de Anomalias",
                    "Métodos estatísticos, Isolation Forest, LOF e autoencoders "
                    "para fraude, falha de equipamento e segurança.",
                    [("01-metodos-estatisticos", "Métodos estatísticos"),
                     ("02-isolation-forest-e-lof", "Isolation Forest e LOF"),
                     ("03-caso-real-fraude", "Caso real: detecção de fraude")]),
         ]),

    Tema("06-avaliacao-e-validacao", "Avaliação e Validação de Modelos",
         "A disciplina que separa o cientista de dados sênior do júnior: saber "
         "se o número que você reportou é real.",
         [
             Modulo("01-metricas-de-classificacao", "Métricas de Classificação",
                    "Matriz de confusão, precisão, recall, F1, ROC-AUC, PR-AUC e "
                    "como escolher a métrica a partir do custo do erro.",
                    [("01-matriz-de-confusao-e-metricas", "Matriz de confusão e métricas"),
                     ("02-roc-e-pr", "Curvas ROC e Precisão-Recall"),
                     ("03-metrica-guiada-por-custo", "Escolhendo a métrica pelo custo")]),
             Modulo("02-metricas-de-regressao", "Métricas de Regressão",
                    "MAE, RMSE, MAPE, R² e quantis — cada uma otimiza um "
                    "comportamento diferente do modelo.",
                    [("01-metricas-e-o-que-elas-punem", "O que cada métrica pune"),
                     ("02-erro-assimetrico-e-quantis", "Erro assimétrico e regressão quantílica")]),
             Modulo("03-validacao-cruzada", "Validação Cruzada",
                    "k-fold, estratificado, por grupo e temporal — e o vazamento "
                    "sutil que infla o resultado de quase todo notebook.",
                    [("01-esquemas-de-validacao", "Esquemas de validação"),
                     ("02-validacao-aninhada", "Validação aninhada"),
                     ("03-vazamento-na-validacao", "Vazamento na validação")]),
             Modulo("04-vies-variancia", "Viés, Variância e Curvas de Aprendizado",
                    "A decomposição do erro, curvas de aprendizado e de "
                    "complexidade como ferramenta de diagnóstico.",
                    [("01-decomposicao-do-erro", "Decomposição viés-variância"),
                     ("02-curvas-de-aprendizado", "Curvas de aprendizado e diagnóstico")]),
             Modulo("05-calibracao-de-probabilidades", "Calibração de Probabilidades",
                    "Quando 0,8 precisa mesmo significar 80%: Platt, isotônica, "
                    "Brier score e diagramas de confiabilidade.",
                    [("01-por-que-calibrar", "Por que calibrar"),
                     ("02-metodos-de-calibracao", "Platt scaling e regressão isotônica")]),
             Modulo("06-otimizacao-de-hiperparametros", "Otimização de Hiperparâmetros",
                    "Grid, random, otimização bayesiana e Hyperband — com o "
                    "orçamento computacional como restrição de projeto.",
                    [("01-grid-e-random-search", "Grid e Random Search"),
                     ("02-otimizacao-bayesiana", "Otimização bayesiana do zero")]),
         ]),

    Tema("07-series-temporais", "Séries Temporais",
         "Dados com memória: onde a suposição de independência quebra e a "
         "validação precisa respeitar o tempo.",
         [
             Modulo("01-fundamentos-e-estacionariedade", "Fundamentos e Estacionariedade",
                    "Tendência, sazonalidade, autocorrelação, decomposição e os "
                    "testes de raiz unitária.",
                    [("01-decomposicao-e-acf", "Decomposição, ACF e PACF"),
                     ("02-estacionariedade", "Estacionariedade e diferenciação")]),
             Modulo("02-modelos-classicos", "ARIMA, SARIMA e Suavização Exponencial",
                    "A família Box-Jenkins e os modelos de espaço de estados que "
                    "ainda vencem baselines em produção.",
                    [("01-arima", "ARIMA passo a passo"),
                     ("02-sazonalidade-e-sarimax", "SARIMAX com variáveis exógenas"),
                     ("03-suavizacao-exponencial", "Suavização exponencial e ETS")]),
             Modulo("03-ml-para-series-temporais", "Machine Learning para Séries Temporais",
                    "Transformar previsão em problema supervisionado: janelas, "
                    "features de defasagem e validação com origem móvel.",
                    [("01-features-de-defasagem", "Features de defasagem e janelas"),
                     ("02-validacao-temporal", "Validação com origem móvel"),
                     ("03-caso-real-demanda", "Caso real: previsão de demanda")]),
         ]),

    Tema("08-deep-learning", "Deep Learning",
         "Redes neurais do neurônio artificial ao Transformer, sempre "
         "implementadas antes de serem usadas por bibliotecas.",
         [
             Modulo("01-fundamentos-de-redes-neurais", "Fundamentos de Redes Neurais",
                    "Perceptron, funções de ativação, a rede densa como "
                    "composição de funções e o teorema da aproximação universal.",
                    [("01-perceptron", "Do perceptron ao neurônio moderno"),
                     ("02-rede-densa-em-numpy", "Rede densa em NumPy puro"),
                     ("03-ativacoes", "Funções de ativação comparadas")]),
             Modulo("02-backpropagation", "Backpropagation",
                    "A regra da cadeia como grafo computacional: derivar, "
                    "implementar e depurar o algoritmo que treina tudo.",
                    [("01-regra-da-cadeia-e-grafo", "Regra da cadeia e grafo computacional"),
                     ("02-autograd-do-zero", "Um autograd minimalista do zero"),
                     ("03-backprop-em-pytorch", "Do NumPy ao PyTorch")]),
             Modulo("03-treinamento-e-regularizacao", "Treinamento, Otimizadores e Regularização",
                    "SGD, momentum, Adam, agendadores de taxa, dropout, batch "
                    "norm, early stopping e o diagnóstico de um treino que trava.",
                    [("01-otimizadores", "Otimizadores comparados"),
                     ("02-regularizacao-em-redes", "Dropout, weight decay e batch norm"),
                     ("03-diagnostico-de-treino", "Diagnóstico de um treino que não converge")]),
             Modulo("04-redes-convolucionais", "Redes Convolucionais",
                    "Convolução, campo receptivo, pooling e as arquiteturas que "
                    "resolveram visão computacional.",
                    [("01-convolucao-na-mao", "Convolução implementada na mão"),
                     ("02-cnn-em-pytorch", "Uma CNN completa em PyTorch"),
                     ("03-aumento-de-dados", "Aumento de dados e regularização visual")]),
             Modulo("05-modelos-sequenciais", "Modelos Sequenciais: RNN, LSTM e GRU",
                    "Memória, gradientes que desaparecem e as portas que "
                    "resolveram o problema — com aplicação em séries e texto.",
                    [("01-rnn-do-zero", "RNN do zero"),
                     ("02-lstm-e-gru", "LSTM e GRU na prática")]),
             Modulo("06-atencao-e-transformers", "Atenção e Transformers",
                    "Self-attention, multi-head, codificação posicional e a "
                    "arquitetura que redefiniu a área inteira.",
                    [("01-atencao-do-zero", "Atenção implementada do zero"),
                     ("02-transformer-minimo", "Um Transformer mínimo em PyTorch"),
                     ("03-escala-e-custo", "Escala, custo computacional e KV cache")]),
             Modulo("07-embeddings-e-transferencia", "Embeddings e Transfer Learning",
                    "Representações densas, similaridade vetorial, fine-tuning e "
                    "a economia de reaproveitar modelos pré-treinados.",
                    [("01-embeddings", "Embeddings e espaço vetorial"),
                     ("02-transfer-learning", "Transfer learning e fine-tuning")]),
         ]),

    Tema("09-nlp-e-llms", "NLP e Modelos de Linguagem",
         "Do saco de palavras aos LLMs: representar, classificar e avaliar "
         "sistemas que lidam com linguagem.",
         [
             Modulo("01-representacao-de-texto", "Representação de Texto",
                    "Tokenização, normalização, bag-of-words, TF-IDF, n-gramas e "
                    "embeddings estáticos.",
                    [("01-tokenizacao-e-bow", "Tokenização e bag-of-words"),
                     ("02-tfidf-e-similaridade", "TF-IDF e similaridade de documentos")]),
             Modulo("02-classificacao-e-topicos", "Classificação de Texto e Tópicos",
                    "Pipelines de classificação, análise de sentimento e "
                    "modelagem de tópicos com NMF e LDA.",
                    [("01-classificacao-de-texto", "Classificação de texto ponta a ponta"),
                     ("02-modelagem-de-topicos", "Modelagem de tópicos")]),
             Modulo("03-llms-rag-e-avaliacao", "LLMs, RAG e Avaliação",
                    "Como um LLM gera texto, o que é RAG, engenharia de contexto "
                    "e como avaliar sistemas generativos sem se enganar.",
                    [("01-como-um-llm-gera-texto", "Como um LLM gera texto: amostragem"),
                     ("02-rag-do-zero", "RAG do zero com busca vetorial"),
                     ("03-avaliacao-de-sistemas-generativos", "Avaliação de sistemas generativos")]),
         ]),

    Tema("10-inferencia-causal", "Inferência Causal",
         "A pergunta que o negócio realmente faz — 'o que acontece se eu "
         "agir?' — e que correlação nunca responde.",
         [
             Modulo("01-fundamentos-de-causalidade", "Fundamentos de Causalidade",
                    "Resultados potenciais, confundimento, DAGs, o critério de "
                    "porta dos fundos e o paradoxo de Simpson.",
                    [("01-resultados-potenciais", "Resultados potenciais e confundimento"),
                     ("02-dags-e-vies-de-colisor", "DAGs, portas dos fundos e colisores")]),
             Modulo("02-metodos-quase-experimentais", "Métodos Quase-Experimentais",
                    "Pareamento, escore de propensão, diferenças-em-diferenças e "
                    "variáveis instrumentais quando o A/B é impossível.",
                    [("01-propensity-score", "Escore de propensão e pareamento"),
                     ("02-diferencas-em-diferencas", "Diferenças-em-diferenças"),
                     ("03-variaveis-instrumentais", "Variáveis instrumentais")]),
             Modulo("03-uplift-modeling", "Uplift Modeling",
                    "Prever o efeito incremental do tratamento por indivíduo — a "
                    "diferença entre prever churn e evitar churn.",
                    [("01-efeito-heterogeneo", "Efeito heterogêneo de tratamento"),
                     ("02-modelos-de-uplift", "Meta-learners e avaliação de uplift")]),
         ]),

    Tema("11-interpretabilidade-e-fairness", "Interpretabilidade e Fairness",
         "Explicar decisões automatizadas e auditar seus danos — hoje uma "
         "exigência regulatória, não um diferencial.",
         [
             Modulo("01-interpretabilidade", "Interpretabilidade de Modelos",
                    "Importância por permutação, PDP, ICE, LIME e SHAP — o que "
                    "cada método realmente responde.",
                    [("01-importancia-e-pdp", "Permutação, PDP e ICE"),
                     ("02-shap", "Valores de Shapley e SHAP"),
                     ("03-lime-e-explicacoes-locais", "LIME e explicações locais")]),
             Modulo("02-fairness-e-vies", "Fairness e Viés Algorítmico",
                    "Definições concorrentes de justiça, sua incompatibilidade "
                    "matemática e técnicas de mitigação.",
                    [("01-metricas-de-fairness", "Métricas de fairness"),
                     ("02-mitigacao-de-vies", "Mitigação de viés")]),
         ]),

    Tema("12-sistemas-de-recomendacao", "Sistemas de Recomendação",
         "O produto de machine learning que mais gera receita direta — e um "
         "problema de avaliação notoriamente traiçoeiro.",
         [
             Modulo("01-filtragem-colaborativa", "Filtragem Colaborativa",
                    "Vizinhança usuário-item, fatoração de matrizes, ALS e o "
                    "problema da partida a frio.",
                    [("01-vizinhanca", "Filtragem por vizinhança"),
                     ("02-fatoracao-de-matrizes", "Fatoração de matrizes do zero")]),
             Modulo("02-avaliacao-e-hibridos", "Avaliação e Modelos Híbridos",
                    "Métricas de ranking, viés de popularidade, diversidade e "
                    "arquiteturas híbridas de duas torres.",
                    [("01-metricas-de-ranking", "Métricas de ranking"),
                     ("02-hibridos-e-conteudo", "Modelos baseados em conteúdo e híbridos")]),
         ]),

    Tema("13-mlops-e-producao", "MLOps e Produção",
         "O modelo só vale quando roda, é observável e pode ser revertido — a "
         "engenharia que transforma notebook em sistema.",
         [
             Modulo("01-pipelines-e-reprodutibilidade", "Pipelines e Reprodutibilidade",
                    "Pipelines do scikit-learn, versionamento de dados e modelos, "
                    "seeds e o registro de experimentos.",
                    [("01-pipelines-robustos", "Pipelines robustos ponta a ponta"),
                     ("02-reprodutibilidade", "Reprodutibilidade e registro de experimentos")]),
             Modulo("02-deploy-e-monitoramento", "Deploy e Monitoramento",
                    "Batch vs online, feature store, contrato de dados, latência "
                    "e o que instrumentar antes de ir para produção.",
                    [("01-servindo-um-modelo", "Servindo um modelo"),
                     ("02-monitoramento", "O que monitorar em produção")]),
             Modulo("03-drift-e-retreino", "Data Drift e Retreino",
                    "Drift de covariáveis, de conceito e de rótulo; testes "
                    "estatísticos de detecção e políticas de retreino.",
                    [("01-deteccao-de-drift", "Detecção de drift"),
                     ("02-politicas-de-retreino", "Políticas de retreino")]),
         ]),
]


def todos_os_modulos():
    for tema in CURRICULO:
        for modulo in tema.modulos:
            yield tema, modulo


if __name__ == "__main__":
    n_mod = sum(len(t.modulos) for t in CURRICULO)
    n_nb = sum(len(m.notebooks) for _, m in todos_os_modulos())
    print(f"{len(CURRICULO)} temas | {n_mod} módulos | {n_nb} notebooks")
    for t in CURRICULO:
        print(f"\n{t.slug}  ({len(t.modulos)} módulos)")
        for m in t.modulos:
            print(f"   {m.slug}  [{len(m.notebooks)} nb]")
