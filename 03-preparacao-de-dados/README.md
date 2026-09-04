# 🧹 Preparação de Dados

> Onde um cientista de dados sênior gasta a maior parte do tempo — e onde a maioria dos projetos falha silenciosamente.

**5 módulos** · leia de cima para baixo — cada um assume o anterior.

---

## 📚 Módulos deste tema

| # | Módulo | O que você vai aprender |
| :-: | :-- | :-- |
| **1** | **[Análise Exploratória (EDA)](01-analise-exploratoria/teoria.pdf)** | Um protocolo disciplinado de EDA: perfilamento, relações, hipóteses e o que procurar antes de qualquer modelo. |
| **2** | **[Dados Faltantes e Outliers](02-dados-faltantes-e-outliers/teoria.pdf)** | MCAR, MAR e MNAR; imputação simples, múltipla e por modelo; detecção e tratamento honesto de valores extremos. |
| **3** | **[Feature Engineering](03-feature-engineering/teoria.pdf)** | Transformações, interações, agregações temporais e features de domínio — o que ainda separa modelos bons de medianos. |
| **4** | **[Encoding, Escala e Vazamento de Dados](04-encoding-escala-e-vazamento/teoria.pdf)** | One-hot, ordinal, target encoding e escalonamento — sempre dentro de um Pipeline, para não vazar informação do futuro. |
| **5** | **[Dados Desbalanceados](05-dados-desbalanceados/teoria.pdf)** | Reamostragem, SMOTE, pesos de classe e ajuste de limiar — e por que acurácia é inútil quando a classe rara é o que importa. |

---

## 🗂️ Arquivos de cada módulo

| Módulo | 📕 Teoria | 💻 Notebooks-guia | ✏️ Prática |
| :-- | :-: | :-- | :-: |
| Análise Exploratória (EDA) | [PDF](01-analise-exploratoria/teoria.pdf) | [1](01-analise-exploratoria/01-protocolo-de-eda.ipynb) · [2](01-analise-exploratoria/02-relacoes-e-correlacao.ipynb) | [abrir](01-analise-exploratoria/99-exercicios.ipynb) |
| Dados Faltantes e Outliers | [PDF](02-dados-faltantes-e-outliers/teoria.pdf) | [1](02-dados-faltantes-e-outliers/01-mecanismos-de-ausencia.ipynb) · [2](02-dados-faltantes-e-outliers/02-estrategias-de-imputacao.ipynb) · [3](02-dados-faltantes-e-outliers/03-outliers.ipynb) | [abrir](02-dados-faltantes-e-outliers/99-exercicios.ipynb) |
| Feature Engineering | [PDF](03-feature-engineering/teoria.pdf) | [1](03-feature-engineering/01-transformacoes-numericas.ipynb) · [2](03-feature-engineering/02-features-temporais-e-agregacoes.ipynb) · [3](03-feature-engineering/03-selecao-de-features.ipynb) | [abrir](03-feature-engineering/99-exercicios.ipynb) |
| Encoding, Escala e Vazamento de Dados | [PDF](04-encoding-escala-e-vazamento/teoria.pdf) | [1](04-encoding-escala-e-vazamento/01-encoding-de-categoricas.ipynb) · [2](04-encoding-escala-e-vazamento/02-escalonamento.ipynb) · [3](04-encoding-escala-e-vazamento/03-vazamento-de-dados.ipynb) | [abrir](04-encoding-escala-e-vazamento/99-exercicios.ipynb) |
| Dados Desbalanceados | [PDF](05-dados-desbalanceados/teoria.pdf) | [1](05-dados-desbalanceados/01-o-problema-do-desbalanceamento.ipynb) · [2](05-dados-desbalanceados/02-tecnicas-de-reamostragem.ipynb) · [3](05-dados-desbalanceados/03-limiar-e-custo.ipynb) | [abrir](05-dados-desbalanceados/99-exercicios.ipynb) |

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
