# 📊 Estatística

> A base sobre a qual todo o resto se apoia: descrever, modelar a incerteza, estimar e decidir a partir de dados.

**6 módulos** · leia de cima para baixo — cada um assume o anterior.

---

## 📚 Módulos deste tema

| # | Módulo | O que você vai aprender |
| :-: | :-- | :-- |
| **1** | **[Fundamentos e Estatística Descritiva](01-fundamentos-e-estatistica-descritiva/teoria.pdf)** | Tipos de variável, medidas de posição e dispersão, momentos, robustez e as armadilhas de resumir dados em um número. |
| **2** | **[Distribuições de Probabilidade](02-distribuicoes-de-probabilidade/teoria.pdf)** | O catálogo de modelos de incerteza: Bernoulli a Weibull, quando cada uma aparece e como reconhecê-la nos dados. |
| **3** | **[Inferência e Estimação](03-inferencia-e-estimacao/teoria.pdf)** | De amostra para população: máxima verossimilhança, viés, erro-padrão, intervalos de confiança e bootstrap. |
| **4** | **[Testes de Hipótese](04-testes-de-hipotese/teoria.pdf)** | p-valor, poder, erros tipo I e II, testes paramétricos e não-paramétricos, e o problema das comparações múltiplas. |
| **5** | **[Estatística Bayesiana](05-estatistica-bayesiana/teoria.pdf)** | Priori, verossimilhança e posteriori; conjugadas, MCMC artesanal e por que o mercado migrou para o pensamento bayesiano. |
| **6** | **[A/B Testing e Desenho Experimental](06-ab-testing-e-desenho-experimental/teoria.pdf)** | Aleatorização, métricas-guia, testes sequenciais, CUPED e os erros que invalidam experimentos em produção. |

---

## 🗂️ Arquivos de cada módulo

| Módulo | 📕 Teoria | 💻 Notebooks-guia | ✏️ Prática |
| :-- | :-: | :-- | :-: |
| Fundamentos e Estatística Descritiva | [PDF](01-fundamentos-e-estatistica-descritiva/teoria.pdf) | [1](01-fundamentos-e-estatistica-descritiva/01-medidas-resumo.ipynb) · [2](01-fundamentos-e-estatistica-descritiva/02-robustez-e-outliers.ipynb) · [3](01-fundamentos-e-estatistica-descritiva/03-visualizacao-distribuicoes.ipynb) | [abrir](01-fundamentos-e-estatistica-descritiva/99-exercicios.ipynb) |
| Distribuições de Probabilidade | [PDF](02-distribuicoes-de-probabilidade/teoria.pdf) | [1](02-distribuicoes-de-probabilidade/01-discretas.ipynb) · [2](02-distribuicoes-de-probabilidade/02-continuas.ipynb) · [3](02-distribuicoes-de-probabilidade/03-tlc-e-lei-dos-grandes-numeros.ipynb) · [4](02-distribuicoes-de-probabilidade/04-ajuste-a-dados-reais.ipynb) | [abrir](02-distribuicoes-de-probabilidade/99-exercicios.ipynb) |
| Inferência e Estimação | [PDF](03-inferencia-e-estimacao/teoria.pdf) | [1](03-inferencia-e-estimacao/01-estimadores-e-propriedades.ipynb) · [2](03-inferencia-e-estimacao/02-maxima-verossimilhanca.ipynb) · [3](03-inferencia-e-estimacao/03-intervalos-e-bootstrap.ipynb) | [abrir](03-inferencia-e-estimacao/99-exercicios.ipynb) |
| Testes de Hipótese | [PDF](04-testes-de-hipotese/teoria.pdf) | [1](04-testes-de-hipotese/01-logica-do-teste.ipynb) · [2](04-testes-de-hipotese/02-testes-classicos.ipynb) · [3](04-testes-de-hipotese/03-poder-e-tamanho-de-amostra.ipynb) · [4](04-testes-de-hipotese/04-comparacoes-multiplas.ipynb) | [abrir](04-testes-de-hipotese/99-exercicios.ipynb) |
| Estatística Bayesiana | [PDF](05-estatistica-bayesiana/teoria.pdf) | [1](05-estatistica-bayesiana/01-teorema-de-bayes.ipynb) · [2](05-estatistica-bayesiana/02-conjugadas-e-posteriori.ipynb) · [3](05-estatistica-bayesiana/03-mcmc-do-zero.ipynb) | [abrir](05-estatistica-bayesiana/99-exercicios.ipynb) |
| A/B Testing e Desenho Experimental | [PDF](06-ab-testing-e-desenho-experimental/teoria.pdf) | [1](06-ab-testing-e-desenho-experimental/01-desenho-e-aleatorizacao.ipynb) · [2](06-ab-testing-e-desenho-experimental/02-analise-de-experimento.ipynb) · [3](06-ab-testing-e-desenho-experimental/03-cuped-e-variancia.ipynb) | [abrir](06-ab-testing-e-desenho-experimental/99-exercicios.ipynb) |

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
