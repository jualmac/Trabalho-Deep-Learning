## Alguns Artigos sobre o assunto:

* Sobre performance de algoritmos evolutivos: Simple Evolutionary Optimization Can Rival Stochastic Gradient Descent in Neural Networks
* Artigo de Origem do Neat: [Evolving Neural Networks through Augmenting Topologies](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
* Revisão dos sucessores do NEAT: [A Systematic Literature Review of the
Successors of “NeuroEvolution
of Augmenting Topologies”](https://www.cse.unr.edu/~sushil/class/gas/papers/evco_a_00282.pdf)



# Divisão geral do paper em 5 partes

## 1. Contextualização e motivação

O paper parte do campo de neuroevolução, isto é, métodos que usam algoritmos evolutivos para otimizar redes neurais. A motivação central é que, em muitos métodos tradicionais, a arquitetura da rede precisa ser definida manualmente antes do treinamento. Isso limita a busca, porque uma arquitetura pequena demais pode não ser suficiente para resolver o problema, enquanto uma arquitetura grande demais aumenta desnecessariamente o espaço de busca.

Nesse contexto, o NEAT surge como um método importante porque evolui simultaneamente os pesos e a topologia da rede. Ele começa com redes simples e adiciona conexões e neurônios ao longo da evolução, conforme essas mudanças ajudam no desempenho. Seus princípios centrais são marcações históricas, especiação e complexificação incremental.

## 2. Cluster 1: espaço de busca e paisagem de fitness

O primeiro cluster reúne métodos que modificam o NEAT para lidar com dificuldades do processo de busca evolutiva. O foco está nos problemas que aparecem durante a otimização, como múltiplos objetivos, muitas features irrelevantes, paisagens enganosas, ambientes incertos, problemas open-ended e evolução online ou em tempo real.

A ideia geral desse grupo é mostrar que o NEAT original pode ser adaptado quando a busca evolutiva se torna mais difícil, instável ou enganosa. Esses métodos tentam tornar a evolução mais robusta em cenários nos quais simplesmente maximizar uma função de fitness pode não ser suficiente.

## 3. Cluster 2: métodos híbridos

O segundo cluster reúne métodos que combinam NEAT com outras técnicas de Machine Learning, Reinforcement Learning ou Computação Evolutiva. Nesse grupo, o NEAT deixa de ser visto como uma solução isolada e passa a ser usado em conjunto com outros métodos.

A ideia principal é aproveitar pontos fortes complementares. O NEAT pode ser usado para buscar arquiteturas ou estruturas globais, enquanto métodos como backpropagation, Q-learning, PSO ou outros algoritmos podem refinar pesos, políticas ou representações. Esse cluster é especialmente importante para conectar NEAT com deep learning moderno.

## 4. Cluster 3: redes com propriedades específicas

O terceiro cluster reúne métodos que usam princípios do NEAT para evoluir redes neurais com características particulares. O foco não está apenas em melhorar a busca, mas em produzir redes com propriedades desejáveis.

Entre essas propriedades estão modularidade, plasticidade, capacidade de transferência, memória, uso de diferentes tipos de neurônios, grandes topologias, substratos automáticos e arquiteturas profundas. Esse grupo mostra como os sucessores do NEAT expandiram a ideia original para redes mais complexas e mais próximas de problemas modernos.

## 5. Discussão, limitações e conexão com a proposta do trabalho

Na parte final, o paper destaca que comparar os sucessores do NEAT é difícil, porque os métodos usam datasets, métricas e protocolos experimentais diferentes. Além disso, muitos trabalhos comparam seus métodos apenas com o NEAT original, e não necessariamente com técnicas mais amplas do estado da arte.

Essa discussão se conecta bem com a proposta do trabalho. A ideia não é usar o NEAT apenas como um otimizador completo, mas como um possível descobridor de arquiteturas. Nesse cenário, o NEAT encontraria uma topologia de rede; depois, essa arquitetura seria reinicializada e treinada novamente com SGD.

A comparação ficaria entre três casos: uma rede convencional treinada com SGD, uma rede evoluída pelo próprio NEAT e uma arquitetura descoberta pelo NEAT, mas treinada do zero com SGD. Assim, a pergunta principal não seria apenas qual modelo tem maior acurácia, mas se a arquitetura encontrada por evolução faz o SGD aprender padrões diferentes de uma arquitetura convencional.

No caso de imagens, essa pergunta é interessante porque permite usar técnicas de interpretabilidade visual, como saliency maps, Grad-CAM, Integrated Gradients ou oclusão, para observar quais regiões da imagem influenciam mais a decisão do modelo.

A limitação prática é que o NEAT puro não escala bem para imagens grandes, pois o espaço de busca cresce muito com o número de pixels e conexões possíveis. Por isso, o caminho mais realista seria usar bases pequenas e controladas, como MNIST, Fashion-MNIST, uma versão simplificada do CIFAR-10 ou dados simulados.

A contribuição do trabalho, portanto, não seria competir com modelos estado da arte, mas investigar se uma arquitetura descoberta por neuroevolução pode induzir um treinamento por SGD a aprender soluções diferentes, tanto em desempenho quanto em interpretação visual.
