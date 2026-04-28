## Trabalho

A ideia que pensei para o trabalho é estudar o NEAT, que é um algoritmo de neuroevolucao. Diferente do jeito mais comum de treinar redes neurais, em que a gente escolhe uma arquitetura antes e depois treina os pesos com SGD ou Adam, o NEAT tenta evoluir tanto os pesos quanto a estrutura da rede.

Ou seja, ele comeca com redes bem simples e, ao longo do tempo, vai adicionando conexoes e neuronios. A ideia original do NEAT é que a rede va ficando mais complexa conforme isso ajuda na tarefa.

O ponto que eu penso em explorar é usar o NEAT nao exatamente como um otimizador “completo”, mas como um descobridor de arquitetura. Entao o papel dele seria encontrar uma topologia de rede. Depois disso, com essa arquitetura encontrada, reiniciamos os pesos e treinamos de novo usando SGD, como em deep learning tradicional.

A proposta ficaria mais ou menos assim:

Primeiro, uma rede convencional treinada com SGD. Essa seria a comparacao mais basica.

Segundo, uma rede treinada pelo proprio NEAT, com a arquitetura e os pesos encontrados por evolucao.

Terceiro, a arquitetura descoberta pelo NEAT, mas com os pesos zerados/reinicializados e treinados do zero usando SGD.

Com isso, a pergunta nao seria apenas “qual modelo tem maior acuracia?”. A pergunta mais interessante seria: esses modelos aprendem estrategias diferentes ou acabam chegando em solucoes parecidas?

Essa ideia é meio independente do dominio. Em principio, poderia ser aplicada em dados tabulares, dados simulados ou outros tipos de problema de classificacao. A escolha por imagens é mais interessante porque a interpretabilidade fica mais visual. Da para analisar nao so a saida do modelo, mas tambem quais regioes da imagem influenciaram mais a decisao da rede.

Para investigar isso, da para usar tecnicas de interpretabilidade visual, como mapas de saliencia, Grad-CAM, Integrated Gradients ou testes de oclusao. Essas tecnicas ajudam a visualizar quais regioes da imagem foram mais importantes para a classificacao.

A pergunta principal do trabalho seria:

A arquitetura descoberta por evolucao faz o SGD aprender padroes diferentes de uma arquitetura convencional?

No caso de imagens, isso pode ser visto de forma mais intuitiva:

Uma arquitetura evoluida pelo NEAT muda a forma como uma rede neural aprende e interpreta imagens, mesmo quando os pesos depois sao treinados com SGD?

A dificuldade principal é que o NEAT puro nao escala muito bem para imagens. Imagens tem muitos pixels, entao a entrada da rede fica muito grande. Isso torna a evolucao pesada, lenta e dificil de controlar. Alem disso, se a gente tentar evoluir redes muito grandes, o custo computacional pode ficar inviavel para um trabalho de disciplina de mestrado.

Por isso, a ideia nao é tentar resolver ImageNet nem criar uma arquitetura gigante. O caminho mais realista seria usar bases pequenas, como MNIST, Fashion-MNIST, talvez CIFAR-10 em uma versao bem controlada, ou ate dados simulados. O foco do trabalho nao precisa ser bater estado da arte. O foco seria observar se existe alguma diferenca interessante entre os modelos.

Para o seminario, uma boa estrutura seria primeiro apresentar o NEAT original: como ele comeca simples, como adiciona conexoes e neuronios, e como usa especiacao para proteger inovacoes. Depois, daria para comentar algumas variantes e trabalhos relacionados, como L-NEAT, HyperNEAT, CoDeepNEAT, EXACT, PropNEAT e LCoDeepNEAT. A ideia seria mostrar que o NEAT original tem limitacoes, principalmente em problemas grandes, mas que existem varias tentativas de adaptar essa familia de metodos para classificacao, deep learning e busca de arquiteturas.

Ver: [A Systematic Literature Review of the
Successors of “NeuroEvolution
of Augmenting Topologies”](https://www.cse.unr.edu/~sushil/class/gas/papers/evco_a_00282.pdf)