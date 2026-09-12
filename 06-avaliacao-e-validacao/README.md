# ⚖️ Avaliação e Validação de Modelos

> A disciplina que separa o cientista de dados sênior do júnior: saber se o número que você reportou é real.

**6 módulos** · leia de cima para baixo — cada um assume o anterior.

---

## 📚 Módulos deste tema

| # | Módulo | O que você vai aprender |
| :-: | :-- | :-- |
| **1** | **[Métricas de Classificação](01-metricas-de-classificacao/teoria.pdf)** | Matriz de confusão, precisão, recall, F1, ROC-AUC, PR-AUC e como escolher a métrica a partir do custo do erro. |
| **2** | **[Métricas de Regressão](02-metricas-de-regressao/teoria.pdf)** | MAE, RMSE, MAPE, R² e quantis — cada uma otimiza um comportamento diferente do modelo. |
| **3** | **[Validação Cruzada](03-validacao-cruzada/teoria.pdf)** | k-fold, estratificado, por grupo e temporal — e o vazamento sutil que infla o resultado de quase todo notebook. |
| **4** | **[Viés, Variância e Curvas de Aprendizado](04-vies-variancia/01-decomposicao-do-erro.ipynb)** | A decomposição do erro, curvas de aprendizado e de complexidade como ferramenta de diagnóstico. |
| **5** | **[Calibração de Probabilidades](05-calibracao-de-probabilidades/01-por-que-calibrar.ipynb)** | Quando 0,8 precisa mesmo significar 80%: Platt, isotônica, Brier score e diagramas de confiabilidade. |
| **6** | **[Otimização de Hiperparâmetros](06-otimizacao-de-hiperparametros/01-grid-e-random-search.ipynb)** | Grid, random, otimização bayesiana e Hyperband — com o orçamento computacional como restrição de projeto. |

---

## 🗂️ Arquivos de cada módulo

| Módulo | 📕 Teoria | 💻 Notebooks-guia | ✏️ Prática |
| :-- | :-: | :-- | :-: |
| Métricas de Classificação | [PDF](01-metricas-de-classificacao/teoria.pdf) | [1](01-metricas-de-classificacao/01-matriz-de-confusao-e-metricas.ipynb) · [2](01-metricas-de-classificacao/02-roc-e-pr.ipynb) · [3](01-metricas-de-classificacao/03-metrica-guiada-por-custo.ipynb) | [abrir](01-metricas-de-classificacao/99-exercicios.ipynb) |
| Métricas de Regressão | [PDF](02-metricas-de-regressao/teoria.pdf) | [1](02-metricas-de-regressao/01-metricas-e-o-que-elas-punem.ipynb) · [2](02-metricas-de-regressao/02-erro-assimetrico-e-quantis.ipynb) | [abrir](02-metricas-de-regressao/99-exercicios.ipynb) |
| Validação Cruzada | [PDF](03-validacao-cruzada/teoria.pdf) | [1](03-validacao-cruzada/01-esquemas-de-validacao.ipynb) · [2](03-validacao-cruzada/02-validacao-aninhada.ipynb) · [3](03-validacao-cruzada/03-vazamento-na-validacao.ipynb) | [abrir](03-validacao-cruzada/99-exercicios.ipynb) |
| Viés, Variância e Curvas de Aprendizado | embutida nos notebooks | [1](04-vies-variancia/01-decomposicao-do-erro.ipynb) · [2](04-vies-variancia/02-curvas-de-aprendizado.ipynb) | [abrir](04-vies-variancia/99-exercicios.ipynb) |
| Calibração de Probabilidades | embutida nos notebooks | [1](05-calibracao-de-probabilidades/01-por-que-calibrar.ipynb) · [2](05-calibracao-de-probabilidades/02-metodos-de-calibracao.ipynb) | [abrir](05-calibracao-de-probabilidades/99-exercicios.ipynb) |
| Otimização de Hiperparâmetros | embutida nos notebooks | [1](06-otimizacao-de-hiperparametros/01-grid-e-random-search.ipynb) · [2](06-otimizacao-de-hiperparametros/02-otimizacao-bayesiana.ipynb) | [abrir](06-otimizacao-de-hiperparametros/99-exercicios.ipynb) |

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
