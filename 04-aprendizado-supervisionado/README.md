# Aprendizado Supervisionado

Os modelos que preveem um rótulo conhecido — do mais interpretável ao campeão de competições.

| Módulo | Conteúdo | Teoria | Notebooks-guia | Prática |
|---|---|---|---|---|
| **Regressão Linear** | O modelo fundador: formulação matricial, pressupostos, diagnóstico de resíduos e interpretação de coeficientes. | [PDF](01-regressao-linear/teoria.pdf) | [1](01-regressao-linear/01-regressao-do-zero.ipynb), [2](01-regressao-linear/02-pressupostos-e-diagnostico.ipynb), [3](01-regressao-linear/03-caso-real-precificacao.ipynb) | [exercícios](01-regressao-linear/99-exercicios.ipynb) |
| **Regularização: Ridge, Lasso e Elastic Net** | Penalizar para generalizar: a geometria de L1 e L2, seleção automática de variáveis e o trade-off viés-variância na prática. | [PDF](02-regularizacao/teoria.pdf) | [1](02-regularizacao/01-ridge-e-lasso.ipynb), [2](02-regularizacao/02-geometria-da-regularizacao.ipynb), [3](02-regularizacao/03-elasticnet-e-tuning.ipynb) | [exercícios](02-regularizacao/99-exercicios.ipynb) |
| **Regressão Logística** | Classificação probabilística, odds ratio, entropia cruzada e por que ela continua sendo o baseline de crédito e risco. | [PDF](03-regressao-logistica/teoria.pdf) | [1](03-regressao-logistica/01-logistica-do-zero.ipynb), [2](03-regressao-logistica/02-interpretacao-odds-ratio.ipynb), [3](03-regressao-logistica/03-caso-real-credito.ipynb) | [exercícios](03-regressao-logistica/99-exercicios.ipynb) |
| **k-NN e Naive Bayes** | Dois extremos didáticos: o modelo que não aprende nada e o que assume independência total — ambos ainda úteis. | [PDF](04-knn-e-naive-bayes/teoria.pdf) | [1](04-knn-e-naive-bayes/01-knn.ipynb), [2](04-knn-e-naive-bayes/02-naive-bayes.ipynb) | [exercícios](04-knn-e-naive-bayes/99-exercicios.ipynb) |
| **Máquinas de Vetores de Suporte** | Margem máxima, o truque do kernel e a dualidade — a ideia geométrica mais elegante do aprendizado supervisionado. | [PDF](05-maquinas-de-vetores-de-suporte/teoria.pdf) | [1](05-maquinas-de-vetores-de-suporte/01-margem-e-svm-linear.ipynb), [2](05-maquinas-de-vetores-de-suporte/02-kernels.ipynb) | [exercícios](05-maquinas-de-vetores-de-suporte/99-exercicios.ipynb) |
| **Árvores de Decisão** | Impureza de Gini, entropia, ganho de informação, poda e as razões pelas quais uma árvore sozinha quase sempre sobreajusta. | [PDF](06-arvores-de-decisao/teoria.pdf) | [1](06-arvores-de-decisao/01-arvore-do-zero.ipynb), [2](06-arvores-de-decisao/02-poda-e-hiperparametros.ipynb) | [exercícios](06-arvores-de-decisao/99-exercicios.ipynb) |
| **Bagging e Random Forest** | Bootstrap agregado, descorrelação por amostragem de features, erro OOB e importância de variáveis feita direito. | [PDF](07-bagging-e-random-forest/teoria.pdf) | [1](07-bagging-e-random-forest/01-bagging-e-variancia.ipynb), [2](07-bagging-e-random-forest/02-random-forest.ipynb), [3](07-bagging-e-random-forest/03-importancia-de-features.ipynb) | [exercícios](07-bagging-e-random-forest/99-exercicios.ipynb) |
| **Boosting: Gradient Boosting, XGBoost e LightGBM** | Aprendizado sequencial sobre resíduos, o gradiente funcional e o estado da arte em dados tabulares. | [PDF](08-boosting/teoria.pdf) | [1](08-boosting/01-gradient-boosting-do-zero.ipynb), [2](08-boosting/02-xgboost-e-lightgbm.ipynb), [3](08-boosting/03-tuning-e-early-stopping.ipynb) | [exercícios](08-boosting/99-exercicios.ipynb) |

---

Cada módulo tem um `teoria.pdf` (denso, com fórmulas e aplicações de mercado), notebooks-guia executáveis e um notebook de **exercícios**. Sugestão de uso: leia o PDF até o fim de um capítulo, rode o notebook-guia correspondente mexendo nos parâmetros, e só então abra os exercícios — eles são o único lugar onde o código é seu.

[← voltar ao índice geral](../README.md)
