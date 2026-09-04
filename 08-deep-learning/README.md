# Deep Learning

Redes neurais do neurônio artificial ao Transformer, sempre implementadas antes de serem usadas por bibliotecas.

| Módulo | Conteúdo | Teoria | Notebooks |
|---|---|---|---|
| **Fundamentos de Redes Neurais** | Perceptron, funções de ativação, a rede densa como composição de funções e o teorema da aproximação universal. | [PDF](01-fundamentos-de-redes-neurais/teoria.pdf) | [1](01-fundamentos-de-redes-neurais/01-perceptron.ipynb), [2](01-fundamentos-de-redes-neurais/02-rede-densa-em-numpy.ipynb), [3](01-fundamentos-de-redes-neurais/03-ativacoes.ipynb) |
| **Backpropagation** | A regra da cadeia como grafo computacional: derivar, implementar e depurar o algoritmo que treina tudo. | [PDF](02-backpropagation/teoria.pdf) | [1](02-backpropagation/01-regra-da-cadeia-e-grafo.ipynb), [2](02-backpropagation/02-autograd-do-zero.ipynb), [3](02-backpropagation/03-backprop-em-pytorch.ipynb) |
| **Treinamento, Otimizadores e Regularização** | SGD, momentum, Adam, agendadores de taxa, dropout, batch norm, early stopping e o diagnóstico de um treino que trava. | [PDF](03-treinamento-e-regularizacao/teoria.pdf) | [1](03-treinamento-e-regularizacao/01-otimizadores.ipynb), [2](03-treinamento-e-regularizacao/02-regularizacao-em-redes.ipynb), [3](03-treinamento-e-regularizacao/03-diagnostico-de-treino.ipynb) |
| **Redes Convolucionais** | Convolução, campo receptivo, pooling e as arquiteturas que resolveram visão computacional. | [PDF](04-redes-convolucionais/teoria.pdf) | [1](04-redes-convolucionais/01-convolucao-na-mao.ipynb), [2](04-redes-convolucionais/02-cnn-em-pytorch.ipynb), [3](04-redes-convolucionais/03-aumento-de-dados.ipynb) |
| **Modelos Sequenciais: RNN, LSTM e GRU** | Memória, gradientes que desaparecem e as portas que resolveram o problema — com aplicação em séries e texto. | [PDF](05-modelos-sequenciais/teoria.pdf) | [1](05-modelos-sequenciais/01-rnn-do-zero.ipynb), [2](05-modelos-sequenciais/02-lstm-e-gru.ipynb) |
| **Atenção e Transformers** | Self-attention, multi-head, codificação posicional e a arquitetura que redefiniu a área inteira. | [PDF](06-atencao-e-transformers/teoria.pdf) | [1](06-atencao-e-transformers/01-atencao-do-zero.ipynb), [2](06-atencao-e-transformers/02-transformer-minimo.ipynb), [3](06-atencao-e-transformers/03-escala-e-custo.ipynb) |
| **Embeddings e Transfer Learning** | Representações densas, similaridade vetorial, fine-tuning e a economia de reaproveitar modelos pré-treinados. | [PDF](07-embeddings-e-transferencia/teoria.pdf) | [1](07-embeddings-e-transferencia/01-embeddings.ipynb), [2](07-embeddings-e-transferencia/02-transfer-learning.ipynb) |

---

Cada módulo tem um `teoria.pdf` (denso, com fórmulas e aplicações de mercado) e notebooks executáveis. Sugestão de uso: leia o PDF até o fim de um capítulo, depois rode o notebook correspondente mexendo nos parâmetros — o material foi escrito para ser alterado, não só lido.

[← voltar ao índice geral](../README.md)
