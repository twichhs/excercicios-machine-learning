# Estatística

A base sobre a qual todo o resto se apoia: descrever, modelar a incerteza, estimar e decidir a partir de dados.

| Módulo | Conteúdo | Teoria | Notebooks-guia | Prática |
|---|---|---|---|---|
| **Fundamentos e Estatística Descritiva** | Tipos de variável, medidas de posição e dispersão, momentos, robustez e as armadilhas de resumir dados em um número. | [PDF](01-fundamentos-e-estatistica-descritiva/teoria.pdf) | [1](01-fundamentos-e-estatistica-descritiva/01-medidas-resumo.ipynb), [2](01-fundamentos-e-estatistica-descritiva/02-robustez-e-outliers.ipynb), [3](01-fundamentos-e-estatistica-descritiva/03-visualizacao-distribuicoes.ipynb) | [exercícios](01-fundamentos-e-estatistica-descritiva/99-exercicios.ipynb) |
| **Distribuições de Probabilidade** | O catálogo de modelos de incerteza: Bernoulli a Weibull, quando cada uma aparece e como reconhecê-la nos dados. | [PDF](02-distribuicoes-de-probabilidade/teoria.pdf) | [1](02-distribuicoes-de-probabilidade/01-discretas.ipynb), [2](02-distribuicoes-de-probabilidade/02-continuas.ipynb), [3](02-distribuicoes-de-probabilidade/03-tlc-e-lei-dos-grandes-numeros.ipynb), [4](02-distribuicoes-de-probabilidade/04-ajuste-a-dados-reais.ipynb) | [exercícios](02-distribuicoes-de-probabilidade/99-exercicios.ipynb) |
| **Inferência e Estimação** | De amostra para população: máxima verossimilhança, viés, erro-padrão, intervalos de confiança e bootstrap. | [PDF](03-inferencia-e-estimacao/teoria.pdf) | [1](03-inferencia-e-estimacao/01-estimadores-e-propriedades.ipynb), [2](03-inferencia-e-estimacao/02-maxima-verossimilhanca.ipynb), [3](03-inferencia-e-estimacao/03-intervalos-e-bootstrap.ipynb) | [exercícios](03-inferencia-e-estimacao/99-exercicios.ipynb) |
| **Testes de Hipótese** | p-valor, poder, erros tipo I e II, testes paramétricos e não-paramétricos, e o problema das comparações múltiplas. | [PDF](04-testes-de-hipotese/teoria.pdf) | [1](04-testes-de-hipotese/01-logica-do-teste.ipynb), [2](04-testes-de-hipotese/02-testes-classicos.ipynb), [3](04-testes-de-hipotese/03-poder-e-tamanho-de-amostra.ipynb), [4](04-testes-de-hipotese/04-comparacoes-multiplas.ipynb) | [exercícios](04-testes-de-hipotese/99-exercicios.ipynb) |
| **Estatística Bayesiana** | Priori, verossimilhança e posteriori; conjugadas, MCMC artesanal e por que o mercado migrou para o pensamento bayesiano. | [PDF](05-estatistica-bayesiana/teoria.pdf) | [1](05-estatistica-bayesiana/01-teorema-de-bayes.ipynb), [2](05-estatistica-bayesiana/02-conjugadas-e-posteriori.ipynb), [3](05-estatistica-bayesiana/03-mcmc-do-zero.ipynb) | [exercícios](05-estatistica-bayesiana/99-exercicios.ipynb) |
| **A/B Testing e Desenho Experimental** | Aleatorização, métricas-guia, testes sequenciais, CUPED e os erros que invalidam experimentos em produção. | [PDF](06-ab-testing-e-desenho-experimental/teoria.pdf) | [1](06-ab-testing-e-desenho-experimental/01-desenho-e-aleatorizacao.ipynb), [2](06-ab-testing-e-desenho-experimental/02-analise-de-experimento.ipynb), [3](06-ab-testing-e-desenho-experimental/03-cuped-e-variancia.ipynb) | [exercícios](06-ab-testing-e-desenho-experimental/99-exercicios.ipynb) |

---

Cada módulo tem um `teoria.pdf` (denso, com fórmulas e aplicações de mercado), notebooks-guia executáveis e um notebook de **exercícios**. Sugestão de uso: leia o PDF até o fim de um capítulo, rode o notebook-guia correspondente mexendo nos parâmetros, e só então abra os exercícios — eles são o único lugar onde o código é seu.

[← voltar ao índice geral](../README.md)
