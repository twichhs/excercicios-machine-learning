# MLOps e Produção

O modelo só vale quando roda, é observável e pode ser revertido — a engenharia que transforma notebook em sistema.

| Módulo | Conteúdo | Teoria | Notebooks |
|---|---|---|---|
| **Pipelines e Reprodutibilidade** | Pipelines do scikit-learn, versionamento de dados e modelos, seeds e o registro de experimentos. | [PDF](01-pipelines-e-reprodutibilidade/teoria.pdf) | [1](01-pipelines-e-reprodutibilidade/01-pipelines-robustos.ipynb), [2](01-pipelines-e-reprodutibilidade/02-reprodutibilidade.ipynb) |
| **Deploy e Monitoramento** | Batch vs online, feature store, contrato de dados, latência e o que instrumentar antes de ir para produção. | [PDF](02-deploy-e-monitoramento/teoria.pdf) | [1](02-deploy-e-monitoramento/01-servindo-um-modelo.ipynb), [2](02-deploy-e-monitoramento/02-monitoramento.ipynb) |
| **Data Drift e Retreino** | Drift de covariáveis, de conceito e de rótulo; testes estatísticos de detecção e políticas de retreino. | [PDF](03-drift-e-retreino/teoria.pdf) | [1](03-drift-e-retreino/01-deteccao-de-drift.ipynb), [2](03-drift-e-retreino/02-politicas-de-retreino.ipynb) |

---

Cada módulo tem um `teoria.pdf` (denso, com fórmulas e aplicações de mercado) e notebooks executáveis. Sugestão de uso: leia o PDF até o fim de um capítulo, depois rode o notebook correspondente mexendo nos parâmetros — o material foi escrito para ser alterado, não só lido.

[← voltar ao índice geral](../README.md)
