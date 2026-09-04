# Avaliação e Validação de Modelos

A disciplina que separa o cientista de dados sênior do júnior: saber se o número que você reportou é real.

| Módulo | Conteúdo | Teoria | Notebooks |
|---|---|---|---|
| **Métricas de Classificação** | Matriz de confusão, precisão, recall, F1, ROC-AUC, PR-AUC e como escolher a métrica a partir do custo do erro. | [PDF](01-metricas-de-classificacao/teoria.pdf) | [1](01-metricas-de-classificacao/01-matriz-de-confusao-e-metricas.ipynb), [2](01-metricas-de-classificacao/02-roc-e-pr.ipynb), [3](01-metricas-de-classificacao/03-metrica-guiada-por-custo.ipynb) |
| **Métricas de Regressão** | MAE, RMSE, MAPE, R² e quantis — cada uma otimiza um comportamento diferente do modelo. | [PDF](02-metricas-de-regressao/teoria.pdf) | [1](02-metricas-de-regressao/01-metricas-e-o-que-elas-punem.ipynb), [2](02-metricas-de-regressao/02-erro-assimetrico-e-quantis.ipynb) |
| **Validação Cruzada** | k-fold, estratificado, por grupo e temporal — e o vazamento sutil que infla o resultado de quase todo notebook. | [PDF](03-validacao-cruzada/teoria.pdf) | [1](03-validacao-cruzada/01-esquemas-de-validacao.ipynb), [2](03-validacao-cruzada/02-validacao-aninhada.ipynb), [3](03-validacao-cruzada/03-vazamento-na-validacao.ipynb) |
| **Viés, Variância e Curvas de Aprendizado** | A decomposição do erro, curvas de aprendizado e de complexidade como ferramenta de diagnóstico. | [PDF](04-vies-variancia/teoria.pdf) | [1](04-vies-variancia/01-decomposicao-do-erro.ipynb), [2](04-vies-variancia/02-curvas-de-aprendizado.ipynb) |
| **Calibração de Probabilidades** | Quando 0,8 precisa mesmo significar 80%: Platt, isotônica, Brier score e diagramas de confiabilidade. | [PDF](05-calibracao-de-probabilidades/teoria.pdf) | [1](05-calibracao-de-probabilidades/01-por-que-calibrar.ipynb), [2](05-calibracao-de-probabilidades/02-metodos-de-calibracao.ipynb) |
| **Otimização de Hiperparâmetros** | Grid, random, otimização bayesiana e Hyperband — com o orçamento computacional como restrição de projeto. | [PDF](06-otimizacao-de-hiperparametros/teoria.pdf) | [1](06-otimizacao-de-hiperparametros/01-grid-e-random-search.ipynb), [2](06-otimizacao-de-hiperparametros/02-otimizacao-bayesiana.ipynb) |

---

Cada módulo tem um `teoria.pdf` (denso, com fórmulas e aplicações de mercado) e notebooks executáveis. Sugestão de uso: leia o PDF até o fim de um capítulo, depois rode o notebook correspondente mexendo nos parâmetros — o material foi escrito para ser alterado, não só lido.

[← voltar ao índice geral](../README.md)
