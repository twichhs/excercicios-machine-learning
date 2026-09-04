# Aprendizado Não Supervisionado

Encontrar estrutura sem rótulo: agrupar, comprimir e detectar o que foge do padrão.

| Módulo | Conteúdo | Teoria | Notebooks-guia | Prática |
|---|---|---|---|---|
| **Clustering** | k-means, hierárquico, DBSCAN e misturas gaussianas — como escolher k e como saber se o agrupamento significa algo. | [PDF](01-clustering/teoria.pdf) | [1](01-clustering/01-kmeans.ipynb), [2](01-clustering/02-hierarquico-e-dbscan.ipynb), [3](01-clustering/03-gmm-e-avaliacao.ipynb), [4](01-clustering/04-caso-real-segmentacao.ipynb) | [exercícios](01-clustering/99-exercicios.ipynb) |
| **Redução de Dimensionalidade** | PCA, t-SNE, UMAP e autoencoders: comprimir preservando o que importa, e como não se enganar com visualizações 2D. | [PDF](02-reducao-de-dimensionalidade/teoria.pdf) | [1](02-reducao-de-dimensionalidade/01-pca-aplicado.ipynb), [2](02-reducao-de-dimensionalidade/02-tsne-e-manifolds.ipynb) | [exercícios](02-reducao-de-dimensionalidade/99-exercicios.ipynb) |
| **Detecção de Anomalias** | Métodos estatísticos, Isolation Forest, LOF e autoencoders para fraude, falha de equipamento e segurança. | [PDF](03-deteccao-de-anomalias/teoria.pdf) | [1](03-deteccao-de-anomalias/01-metodos-estatisticos.ipynb), [2](03-deteccao-de-anomalias/02-isolation-forest-e-lof.ipynb), [3](03-deteccao-de-anomalias/03-caso-real-fraude.ipynb) | [exercícios](03-deteccao-de-anomalias/99-exercicios.ipynb) |

---

Cada módulo tem um `teoria.pdf` (denso, com fórmulas e aplicações de mercado), notebooks-guia executáveis e um notebook de **exercícios**. Sugestão de uso: leia o PDF até o fim de um capítulo, rode o notebook-guia correspondente mexendo nos parâmetros, e só então abra os exercícios — eles são o único lugar onde o código é seu.

[← voltar ao índice geral](../README.md)
