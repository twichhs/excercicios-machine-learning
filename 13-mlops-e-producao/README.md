# MLOps e Produção

O modelo só vale quando roda, é observável e pode ser revertido — a engenharia que transforma notebook em sistema.

| Módulo | Conteúdo | Teoria | Notebooks-guia | Prática |
|---|---|---|---|---|
| **Pipelines e Reprodutibilidade** | Pipelines do scikit-learn, versionamento de dados e modelos, seeds e o registro de experimentos. | [PDF](01-pipelines-e-reprodutibilidade/teoria.pdf) | [1](01-pipelines-e-reprodutibilidade/01-pipelines-robustos.ipynb), [2](01-pipelines-e-reprodutibilidade/02-reprodutibilidade.ipynb) | [exercícios](01-pipelines-e-reprodutibilidade/99-exercicios.ipynb) |
| **Deploy e Monitoramento** | Batch vs online, feature store, contrato de dados, latência e o que instrumentar antes de ir para produção. | [PDF](02-deploy-e-monitoramento/teoria.pdf) | [1](02-deploy-e-monitoramento/01-servindo-um-modelo.ipynb), [2](02-deploy-e-monitoramento/02-monitoramento.ipynb) | [exercícios](02-deploy-e-monitoramento/99-exercicios.ipynb) |
| **Data Drift e Retreino** | Drift de covariáveis, de conceito e de rótulo; testes estatísticos de detecção e políticas de retreino. | [PDF](03-drift-e-retreino/teoria.pdf) | [1](03-drift-e-retreino/01-deteccao-de-drift.ipynb), [2](03-drift-e-retreino/02-politicas-de-retreino.ipynb) | [exercícios](03-drift-e-retreino/99-exercicios.ipynb) |

---

Cada módulo tem um `teoria.pdf` (denso, com fórmulas e aplicações de mercado), notebooks-guia executáveis e um notebook de **exercícios**. Sugestão de uso: leia o PDF até o fim de um capítulo, rode o notebook-guia correspondente mexendo nos parâmetros, e só então abra os exercícios — eles são o único lugar onde o código é seu.

[← voltar ao índice geral](../README.md)
