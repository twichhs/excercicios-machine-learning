# Inferência Causal

A pergunta que o negócio realmente faz — 'o que acontece se eu agir?' — e que correlação nunca responde.

| Módulo | Conteúdo | Teoria | Notebooks-guia | Prática |
|---|---|---|---|---|
| **Fundamentos de Causalidade** | Resultados potenciais, confundimento, DAGs, o critério de porta dos fundos e o paradoxo de Simpson. | [PDF](01-fundamentos-de-causalidade/teoria.pdf) | [1](01-fundamentos-de-causalidade/01-resultados-potenciais.ipynb), [2](01-fundamentos-de-causalidade/02-dags-e-vies-de-colisor.ipynb) | [exercícios](01-fundamentos-de-causalidade/99-exercicios.ipynb) |
| **Métodos Quase-Experimentais** | Pareamento, escore de propensão, diferenças-em-diferenças e variáveis instrumentais quando o A/B é impossível. | [PDF](02-metodos-quase-experimentais/teoria.pdf) | [1](02-metodos-quase-experimentais/01-propensity-score.ipynb), [2](02-metodos-quase-experimentais/02-diferencas-em-diferencas.ipynb), [3](02-metodos-quase-experimentais/03-variaveis-instrumentais.ipynb) | [exercícios](02-metodos-quase-experimentais/99-exercicios.ipynb) |
| **Uplift Modeling** | Prever o efeito incremental do tratamento por indivíduo — a diferença entre prever churn e evitar churn. | [PDF](03-uplift-modeling/teoria.pdf) | [1](03-uplift-modeling/01-efeito-heterogeneo.ipynb), [2](03-uplift-modeling/02-modelos-de-uplift.ipynb) | [exercícios](03-uplift-modeling/99-exercicios.ipynb) |

---

Cada módulo tem um `teoria.pdf` (denso, com fórmulas e aplicações de mercado), notebooks-guia executáveis e um notebook de **exercícios**. Sugestão de uso: leia o PDF até o fim de um capítulo, rode o notebook-guia correspondente mexendo nos parâmetros, e só então abra os exercícios — eles são o único lugar onde o código é seu.

[← voltar ao índice geral](../README.md)
