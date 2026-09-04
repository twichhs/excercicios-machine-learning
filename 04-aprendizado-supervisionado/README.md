# 🎯 Aprendizado Supervisionado

> Os modelos que preveem um rótulo conhecido — do mais interpretável ao campeão de competições.

**8 módulos** · leia de cima para baixo — cada um assume o anterior.

---

## 📚 Módulos deste tema

| # | Módulo | O que você vai aprender |
| :-: | :-- | :-- |
| **1** | **[Regressão Linear](01-regressao-linear/teoria.pdf)** | O modelo fundador: formulação matricial, pressupostos, diagnóstico de resíduos e interpretação de coeficientes. |
| **2** | **[Regularização: Ridge, Lasso e Elastic Net](02-regularizacao/teoria.pdf)** | Penalizar para generalizar: a geometria de L1 e L2, seleção automática de variáveis e o trade-off viés-variância na prática. |
| **3** | **[Regressão Logística](03-regressao-logistica/teoria.pdf)** | Classificação probabilística, odds ratio, entropia cruzada e por que ela continua sendo o baseline de crédito e risco. |
| **4** | **[k-NN e Naive Bayes](04-knn-e-naive-bayes/teoria.pdf)** | Dois extremos didáticos: o modelo que não aprende nada e o que assume independência total — ambos ainda úteis. |
| **5** | **[Máquinas de Vetores de Suporte](05-maquinas-de-vetores-de-suporte/teoria.pdf)** | Margem máxima, o truque do kernel e a dualidade — a ideia geométrica mais elegante do aprendizado supervisionado. |
| **6** | **[Árvores de Decisão](06-arvores-de-decisao/teoria.pdf)** | Impureza de Gini, entropia, ganho de informação, poda e as razões pelas quais uma árvore sozinha quase sempre sobreajusta. |
| **7** | **[Bagging e Random Forest](07-bagging-e-random-forest/teoria.pdf)** | Bootstrap agregado, descorrelação por amostragem de features, erro OOB e importância de variáveis feita direito. |
| **8** | **[Boosting: Gradient Boosting, XGBoost e LightGBM](08-boosting/teoria.pdf)** | Aprendizado sequencial sobre resíduos, o gradiente funcional e o estado da arte em dados tabulares. |

---

## 🗂️ Arquivos de cada módulo

| Módulo | 📕 Teoria | 💻 Notebooks-guia | ✏️ Prática |
| :-- | :-: | :-- | :-: |
| Regressão Linear | [PDF](01-regressao-linear/teoria.pdf) | [1](01-regressao-linear/01-regressao-do-zero.ipynb) · [2](01-regressao-linear/02-pressupostos-e-diagnostico.ipynb) · [3](01-regressao-linear/03-caso-real-precificacao.ipynb) | [abrir](01-regressao-linear/99-exercicios.ipynb) |
| Regularização: Ridge, Lasso e Elastic Net | [PDF](02-regularizacao/teoria.pdf) | [1](02-regularizacao/01-ridge-e-lasso.ipynb) · [2](02-regularizacao/02-geometria-da-regularizacao.ipynb) · [3](02-regularizacao/03-elasticnet-e-tuning.ipynb) | [abrir](02-regularizacao/99-exercicios.ipynb) |
| Regressão Logística | [PDF](03-regressao-logistica/teoria.pdf) | [1](03-regressao-logistica/01-logistica-do-zero.ipynb) · [2](03-regressao-logistica/02-interpretacao-odds-ratio.ipynb) · [3](03-regressao-logistica/03-caso-real-credito.ipynb) | [abrir](03-regressao-logistica/99-exercicios.ipynb) |
| k-NN e Naive Bayes | [PDF](04-knn-e-naive-bayes/teoria.pdf) | [1](04-knn-e-naive-bayes/01-knn.ipynb) · [2](04-knn-e-naive-bayes/02-naive-bayes.ipynb) | [abrir](04-knn-e-naive-bayes/99-exercicios.ipynb) |
| Máquinas de Vetores de Suporte | [PDF](05-maquinas-de-vetores-de-suporte/teoria.pdf) | [1](05-maquinas-de-vetores-de-suporte/01-margem-e-svm-linear.ipynb) · [2](05-maquinas-de-vetores-de-suporte/02-kernels.ipynb) | [abrir](05-maquinas-de-vetores-de-suporte/99-exercicios.ipynb) |
| Árvores de Decisão | [PDF](06-arvores-de-decisao/teoria.pdf) | [1](06-arvores-de-decisao/01-arvore-do-zero.ipynb) · [2](06-arvores-de-decisao/02-poda-e-hiperparametros.ipynb) | [abrir](06-arvores-de-decisao/99-exercicios.ipynb) |
| Bagging e Random Forest | [PDF](07-bagging-e-random-forest/teoria.pdf) | [1](07-bagging-e-random-forest/01-bagging-e-variancia.ipynb) · [2](07-bagging-e-random-forest/02-random-forest.ipynb) · [3](07-bagging-e-random-forest/03-importancia-de-features.ipynb) | [abrir](07-bagging-e-random-forest/99-exercicios.ipynb) |
| Boosting: Gradient Boosting, XGBoost e LightGBM | [PDF](08-boosting/teoria.pdf) | [1](08-boosting/01-gradient-boosting-do-zero.ipynb) · [2](08-boosting/02-xgboost-e-lightgbm.ipynb) · [3](08-boosting/03-tuning-e-early-stopping.ipynb) | [abrir](08-boosting/99-exercicios.ipynb) |

---

## 🔄 Como estudar cada módulo

```
   ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
   │  📕  LEIA       │     │  💻  RODE       │     │  ✏️  RESOLVA    │
   │                 │ ──▶ │                 │ ──▶ │                 │
   │   teoria.pdf    │     │ notebooks-guia  │     │  99-exercicios  │
   └─────────────────┘     └─────────────────┘     └─────────────────┘
     entenda a ideia         mexa nos números        agora sem apoio
```

> 💡 Os notebooks-guia **já vêm com as saídas prontas** — dá para ler sem rodar nada. Mas o material foi escrito para ser alterado: mude um parâmetro, rode de novo, veja o que quebra.

> ✏️ O `99-exercicios` é o único lugar onde o código é seu. Cada exercício tem o gabarito logo abaixo — resolva **antes** de rolar a página, porque ler a solução dá a sensação de entender sem o entendimento.

---

[⬅️ Voltar para o índice geral](../README.md)
