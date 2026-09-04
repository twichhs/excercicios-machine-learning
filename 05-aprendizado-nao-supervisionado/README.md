# 🔍 Aprendizado Não Supervisionado

> Encontrar estrutura sem rótulo: agrupar, comprimir e detectar o que foge do padrão.

**3 módulos** · leia de cima para baixo — cada um assume o anterior.

---

## 📚 Módulos deste tema

| # | Módulo | O que você vai aprender |
| :-: | :-- | :-- |
| **1** | **[Clustering](01-clustering/teoria.pdf)** | k-means, hierárquico, DBSCAN e misturas gaussianas — como escolher k e como saber se o agrupamento significa algo. |
| **2** | **[Redução de Dimensionalidade](02-reducao-de-dimensionalidade/teoria.pdf)** | PCA, t-SNE, UMAP e autoencoders: comprimir preservando o que importa, e como não se enganar com visualizações 2D. |
| **3** | **[Detecção de Anomalias](03-deteccao-de-anomalias/teoria.pdf)** | Métodos estatísticos, Isolation Forest, LOF e autoencoders para fraude, falha de equipamento e segurança. |

---

## 🗂️ Arquivos de cada módulo

| Módulo | 📕 Teoria | 💻 Notebooks-guia | ✏️ Prática |
| :-- | :-: | :-- | :-: |
| Clustering | [PDF](01-clustering/teoria.pdf) | [1](01-clustering/01-kmeans.ipynb) · [2](01-clustering/02-hierarquico-e-dbscan.ipynb) · [3](01-clustering/03-gmm-e-avaliacao.ipynb) · [4](01-clustering/04-caso-real-segmentacao.ipynb) | [abrir](01-clustering/99-exercicios.ipynb) |
| Redução de Dimensionalidade | [PDF](02-reducao-de-dimensionalidade/teoria.pdf) | [1](02-reducao-de-dimensionalidade/01-pca-aplicado.ipynb) · [2](02-reducao-de-dimensionalidade/02-tsne-e-manifolds.ipynb) | [abrir](02-reducao-de-dimensionalidade/99-exercicios.ipynb) |
| Detecção de Anomalias | [PDF](03-deteccao-de-anomalias/teoria.pdf) | [1](03-deteccao-de-anomalias/01-metodos-estatisticos.ipynb) · [2](03-deteccao-de-anomalias/02-isolation-forest-e-lof.ipynb) · [3](03-deteccao-de-anomalias/03-caso-real-fraude.ipynb) | [abrir](03-deteccao-de-anomalias/99-exercicios.ipynb) |

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
