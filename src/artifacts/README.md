## Explicação do que foi feito

O experimento avaliou o uso de **neuroevolução com NEAT** e de uma variante **HyperNEAT-like** em tarefas de classificação de imagens. A ideia principal foi investigar se redes geradas por evolução conseguem produzir **topologias úteis, compactas e interpretáveis** quando aplicadas a imagens reduzidas para **14x14 pixels**.

Foram utilizados três cenários experimentais:

| Dataset | Objetivo |
|---|---|
| **MNIST** | Cenário-base com dígitos manuscritos |
| **MNIST100** | Cenário de poucos dados, com amostra reduzida |
| **Fashion-MNIST** | Cenário mais difícil, com classes visuais mais complexas |

Em cada dataset, foram avaliadas diferentes variantes evolutivas:

| Variante | Ideia geral |
|---|---|
| `neat_mnist_14x14` | Variante NEAT principal |
| `neat_mnist_14x14_sparse` | Variante mais esparsa, com menos conexões |
| `neat_mnist_14x14_dense` | Variante mais densa, com mais conexões |
| `neat_mnist_14x14_hidden` | Variante com maior exploração de nós ocultos |
| `hyperneat_mnist_cppn` | Variante HyperNEAT-like baseada em CPPN |

A avaliação foi feita em duas etapas. Primeiro, foi medido o desempenho do **NEAT puro**, ou seja, a rede obtida diretamente por evolução, sem ajuste posterior por gradiente. Depois, a topologia evoluída foi reaproveitada e seus pesos foram refinados com otimizadores tradicionais de deep learning, usando **SGD** e **Adam**.

Assim, o experimento comparou três abordagens principais:

| Abordagem | Interpretação |
|---|---|
| **Baseline** | Rede neural tradicional treinada diretamente por gradiente |
| **NEAT puro** | Arquitetura e pesos obtidos apenas por evolução |
| **NEAT + retreino** | Topologia evoluída com pesos refinados por SGD ou Adam |

Além da acurácia, também foram analisados **mapas de oclusão**, que indicam quais regiões da imagem mais influenciam a decisão do modelo. Esses mapas permitem comparar se o baseline, o NEAT puro e o NEAT retreinado estão utilizando regiões semelhantes da imagem para realizar a classificação.

## Interpretação geral dos resultados

Os resultados mostram que o **NEAT puro** ainda apresentou desempenho limitado quando aplicado diretamente aos pixels das imagens em 14x14. Em geral, as melhores acurácias do NEAT puro ficaram entre aproximadamente **31% e 37%**, dependendo do dataset.

Por outro lado, quando as topologias evoluídas passaram por **retreino com gradiente**, os resultados melhoraram de forma expressiva. O destaque foi a variante **HyperNEAT-like com CPPN**, que obteve o melhor desempenho final em todos os cenários avaliados.

Isso sugere que, nesta etapa preliminar, a principal contribuição da neuroevolução não está em substituir completamente o treinamento por gradiente, mas em atuar como um mecanismo de **busca estrutural/topológica**. O gradiente, por sua vez, aparece como uma etapa importante de refinamento dos pesos.

## Interpretação por dataset

### MNIST

No MNIST, o baseline obteve desempenho razoável, mas o melhor resultado final foi alcançado pela variante **HyperNEAT-like + retreino**. O NEAT puro ficou abaixo do baseline, indicando que evoluir pesos e topologia diretamente ainda é uma tarefa difícil. Porém, após o retreino, a estrutura gerada pelo HyperNEAT-like alcançou o melhor desempenho geral.

**Leitura principal:**

> No MNIST, o NEAT puro ainda não foi competitivo, mas a estrutura gerada pelo HyperNEAT-like se mostrou muito eficiente após o refinamento dos pesos.

### MNIST100

No MNIST100, o cenário é mais interessante porque há menos dados disponíveis. Mesmo nesse contexto reduzido, a variante **HyperNEAT-like + retreino** apresentou desempenho muito alto, superando as demais variantes e o baseline.

**Leitura principal:**

> No cenário de poucos dados, a topologia evoluída e refinada por gradiente apresentou desempenho forte, especialmente com a variante HyperNEAT-like.

### Fashion-MNIST

No Fashion-MNIST, a tarefa é mais difícil porque as classes são visualmente mais parecidas do que dígitos manuscritos. Mesmo assim, o HyperNEAT-like continuou apresentando o melhor resultado final após o retreino.

**Leitura principal:**

> Mesmo em um dataset mais difícil, a combinação HyperNEAT-like + retreino manteve desempenho superior ao baseline, indicando que a abordagem não ficou restrita ao MNIST.

## Conclusão para apresentação

Os resultados preliminares mostram que o **NEAT puro**, quando aplicado diretamente aos pixels, ainda apresenta desempenho limitado. No entanto, quando a neuroevolução é usada para gerar estruturas e essas estruturas são posteriormente refinadas por gradiente, os resultados melhoram de forma expressiva.

O destaque foi a variante **HyperNEAT-like com CPPN**, que obteve os melhores resultados finais em **MNIST**, **MNIST100** e **Fashion-MNIST**. Isso sugere que, nesta etapa do trabalho, a neuroevolução tem maior potencial como mecanismo de busca estrutural do que como substituta completa do treinamento por gradiente.

## Tabela consolidada dos principais resultados

| Dataset | Otimizador | Baseline | Melhor NEAT puro | Melhor NEAT direto + retreino | HyperNEAT-like + retreino | Melhor modelo final |
|---|---|---:|---:|---:|---:|---|
| MNIST | SGD | 65,0% | 35,6% | 61,6% | **90,2%** | HyperNEAT-like |
| MNIST | Adam | 81,0% | 35,6% | 42,4% | **87,8%** | HyperNEAT-like |
| MNIST100 | SGD | 24,0% | 31,4% | 56,4% | **89,0%** | HyperNEAT-like |
| MNIST100 | Adam | 71,8% | 31,4% | 37,2% | **85,0%** | HyperNEAT-like |
| Fashion-MNIST | SGD | 66,6% | 37,4% | 68,2% | **75,0%** | HyperNEAT-like |
| Fashion-MNIST | Adam | 68,0% | 37,4% | 49,2% | **72,4%** | HyperNEAT-like |

## Síntese dos achados

| Achado | Interpretação |
|---|---|
| NEAT puro teve desempenho limitado | Evoluir pesos e topologia diretamente ainda é difícil em imagens |
| Retreino melhorou fortemente os resultados | O gradiente refinou estruturas geradas pela evolução |
| HyperNEAT-like foi o melhor modelo final | A codificação indireta via CPPN gerou estruturas mais promissoras |
| SGD foi melhor no retreino das topologias evoluídas | As topologias evoluídas pareceram responder melhor ao SGD do que ao Adam |
| Fashion-MNIST confirmou o padrão | O ganho do HyperNEAT-like não ficou restrito ao MNIST |