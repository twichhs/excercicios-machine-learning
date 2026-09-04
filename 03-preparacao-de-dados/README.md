# Preparação de Dados

Onde um cientista de dados sênior gasta a maior parte do tempo — e onde a maioria dos projetos falha silenciosamente.

| Módulo | Conteúdo | Teoria | Notebooks-guia | Prática |
|---|---|---|---|---|
| **Análise Exploratória (EDA)** | Um protocolo disciplinado de EDA: perfilamento, relações, hipóteses e o que procurar antes de qualquer modelo. | [PDF](01-analise-exploratoria/teoria.pdf) | [1](01-analise-exploratoria/01-protocolo-de-eda.ipynb), [2](01-analise-exploratoria/02-relacoes-e-correlacao.ipynb) | [exercícios](01-analise-exploratoria/99-exercicios.ipynb) |
| **Dados Faltantes e Outliers** | MCAR, MAR e MNAR; imputação simples, múltipla e por modelo; detecção e tratamento honesto de valores extremos. | [PDF](02-dados-faltantes-e-outliers/teoria.pdf) | [1](02-dados-faltantes-e-outliers/01-mecanismos-de-ausencia.ipynb), [2](02-dados-faltantes-e-outliers/02-estrategias-de-imputacao.ipynb), [3](02-dados-faltantes-e-outliers/03-outliers.ipynb) | [exercícios](02-dados-faltantes-e-outliers/99-exercicios.ipynb) |
| **Feature Engineering** | Transformações, interações, agregações temporais e features de domínio — o que ainda separa modelos bons de medianos. | [PDF](03-feature-engineering/teoria.pdf) | [1](03-feature-engineering/01-transformacoes-numericas.ipynb), [2](03-feature-engineering/02-features-temporais-e-agregacoes.ipynb), [3](03-feature-engineering/03-selecao-de-features.ipynb) | [exercícios](03-feature-engineering/99-exercicios.ipynb) |
| **Encoding, Escala e Vazamento de Dados** | One-hot, ordinal, target encoding e escalonamento — sempre dentro de um Pipeline, para não vazar informação do futuro. | [PDF](04-encoding-escala-e-vazamento/teoria.pdf) | [1](04-encoding-escala-e-vazamento/01-encoding-de-categoricas.ipynb), [2](04-encoding-escala-e-vazamento/02-escalonamento.ipynb), [3](04-encoding-escala-e-vazamento/03-vazamento-de-dados.ipynb) | [exercícios](04-encoding-escala-e-vazamento/99-exercicios.ipynb) |
| **Dados Desbalanceados** | Reamostragem, SMOTE, pesos de classe e ajuste de limiar — e por que acurácia é inútil quando a classe rara é o que importa. | [PDF](05-dados-desbalanceados/teoria.pdf) | [1](05-dados-desbalanceados/01-o-problema-do-desbalanceamento.ipynb), [2](05-dados-desbalanceados/02-tecnicas-de-reamostragem.ipynb), [3](05-dados-desbalanceados/03-limiar-e-custo.ipynb) | [exercícios](05-dados-desbalanceados/99-exercicios.ipynb) |

---

Cada módulo tem um `teoria.pdf` (denso, com fórmulas e aplicações de mercado), notebooks-guia executáveis e um notebook de **exercícios**. Sugestão de uso: leia o PDF até o fim de um capítulo, rode o notebook-guia correspondente mexendo nos parâmetros, e só então abra os exercícios — eles são o único lugar onde o código é seu.

[← voltar ao índice geral](../README.md)
