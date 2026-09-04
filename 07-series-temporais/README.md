# Séries Temporais

Dados com memória: onde a suposição de independência quebra e a validação precisa respeitar o tempo.

| Módulo | Conteúdo | Teoria | Notebooks-guia | Prática |
|---|---|---|---|---|
| **Fundamentos e Estacionariedade** | Tendência, sazonalidade, autocorrelação, decomposição e os testes de raiz unitária. | [PDF](01-fundamentos-e-estacionariedade/teoria.pdf) | [1](01-fundamentos-e-estacionariedade/01-decomposicao-e-acf.ipynb), [2](01-fundamentos-e-estacionariedade/02-estacionariedade.ipynb) | [exercícios](01-fundamentos-e-estacionariedade/99-exercicios.ipynb) |
| **ARIMA, SARIMA e Suavização Exponencial** | A família Box-Jenkins e os modelos de espaço de estados que ainda vencem baselines em produção. | [PDF](02-modelos-classicos/teoria.pdf) | [1](02-modelos-classicos/01-arima.ipynb), [2](02-modelos-classicos/02-sazonalidade-e-sarimax.ipynb), [3](02-modelos-classicos/03-suavizacao-exponencial.ipynb) | [exercícios](02-modelos-classicos/99-exercicios.ipynb) |
| **Machine Learning para Séries Temporais** | Transformar previsão em problema supervisionado: janelas, features de defasagem e validação com origem móvel. | [PDF](03-ml-para-series-temporais/teoria.pdf) | [1](03-ml-para-series-temporais/01-features-de-defasagem.ipynb), [2](03-ml-para-series-temporais/02-validacao-temporal.ipynb), [3](03-ml-para-series-temporais/03-caso-real-demanda.ipynb) | [exercícios](03-ml-para-series-temporais/99-exercicios.ipynb) |

---

Cada módulo tem um `teoria.pdf` (denso, com fórmulas e aplicações de mercado), notebooks-guia executáveis e um notebook de **exercícios**. Sugestão de uso: leia o PDF até o fim de um capítulo, rode o notebook-guia correspondente mexendo nos parâmetros, e só então abra os exercícios — eles são o único lugar onde o código é seu.

[← voltar ao índice geral](../README.md)
