# 🧠 Deep Learning

> Redes neurais do neurônio artificial ao Transformer, sempre implementadas antes de serem usadas por bibliotecas.

**7 módulos** · leia de cima para baixo — cada um assume o anterior.

---

## 📚 Módulos deste tema

| # | Módulo | O que você vai aprender |
| :-: | :-- | :-- |
| **1** | **[Fundamentos de Redes Neurais](01-fundamentos-de-redes-neurais/01-perceptron.ipynb)** | Perceptron, funções de ativação, a rede densa como composição de funções e o teorema da aproximação universal. |
| **2** | **[Backpropagation](02-backpropagation/01-regra-da-cadeia-e-grafo.ipynb)** | A regra da cadeia como grafo computacional: derivar, implementar e depurar o algoritmo que treina tudo. |
| **3** | **[Treinamento, Otimizadores e Regularização](03-treinamento-e-regularizacao/01-otimizadores.ipynb)** | SGD, momentum, Adam, agendadores de taxa, dropout, batch norm, early stopping e o diagnóstico de um treino que trava. |
| **4** | **[Redes Convolucionais](04-redes-convolucionais/01-convolucao-na-mao.ipynb)** | Convolução, campo receptivo, pooling e as arquiteturas que resolveram visão computacional. |
| **5** | **[Modelos Sequenciais: RNN, LSTM e GRU](05-modelos-sequenciais/01-rnn-do-zero.ipynb)** | Memória, gradientes que desaparecem e as portas que resolveram o problema — com aplicação em séries e texto. |
| **6** | **[Atenção e Transformers](06-atencao-e-transformers/01-atencao-do-zero.ipynb)** | Self-attention, multi-head, codificação posicional e a arquitetura que redefiniu a área inteira. |
| **7** | **[Embeddings e Transfer Learning](07-embeddings-e-transferencia/01-embeddings.ipynb)** | Representações densas, similaridade vetorial, fine-tuning e a economia de reaproveitar modelos pré-treinados. |

---

## 🗂️ Arquivos de cada módulo

| Módulo | 📕 Teoria | 💻 Notebooks-guia | ✏️ Prática |
| :-- | :-: | :-- | :-: |
| Fundamentos de Redes Neurais | embutida nos notebooks | [1](01-fundamentos-de-redes-neurais/01-perceptron.ipynb) · [2](01-fundamentos-de-redes-neurais/02-rede-densa-em-numpy.ipynb) · [3](01-fundamentos-de-redes-neurais/03-ativacoes.ipynb) | [abrir](01-fundamentos-de-redes-neurais/99-exercicios.ipynb) |
| Backpropagation | embutida nos notebooks | [1](02-backpropagation/01-regra-da-cadeia-e-grafo.ipynb) · [2](02-backpropagation/02-autograd-do-zero.ipynb) · [3](02-backpropagation/03-backprop-em-pytorch.ipynb) | [abrir](02-backpropagation/99-exercicios.ipynb) |
| Treinamento, Otimizadores e Regularização | embutida nos notebooks | [1](03-treinamento-e-regularizacao/01-otimizadores.ipynb) · [2](03-treinamento-e-regularizacao/02-regularizacao-em-redes.ipynb) · [3](03-treinamento-e-regularizacao/03-diagnostico-de-treino.ipynb) | [abrir](03-treinamento-e-regularizacao/99-exercicios.ipynb) |
| Redes Convolucionais | embutida nos notebooks | [1](04-redes-convolucionais/01-convolucao-na-mao.ipynb) · [2](04-redes-convolucionais/02-cnn-em-pytorch.ipynb) · [3](04-redes-convolucionais/03-aumento-de-dados.ipynb) | [abrir](04-redes-convolucionais/99-exercicios.ipynb) |
| Modelos Sequenciais: RNN, LSTM e GRU | embutida nos notebooks | [1](05-modelos-sequenciais/01-rnn-do-zero.ipynb) · [2](05-modelos-sequenciais/02-lstm-e-gru.ipynb) | [abrir](05-modelos-sequenciais/99-exercicios.ipynb) |
| Atenção e Transformers | embutida nos notebooks | [1](06-atencao-e-transformers/01-atencao-do-zero.ipynb) · [2](06-atencao-e-transformers/02-transformer-minimo.ipynb) · [3](06-atencao-e-transformers/03-escala-e-custo.ipynb) | [abrir](06-atencao-e-transformers/99-exercicios.ipynb) |
| Embeddings e Transfer Learning | embutida nos notebooks | [1](07-embeddings-e-transferencia/01-embeddings.ipynb) · [2](07-embeddings-e-transferencia/02-transfer-learning.ipynb) | [abrir](07-embeddings-e-transferencia/99-exercicios.ipynb) |

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
