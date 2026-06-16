# MVP: NEAT e interpretabilidade em MNIST/CIFAR-10

Este MVP demonstra a ideia da proposta em uma versao pequena, colocando a
interpretacao das redes no centro da comparacao:

1. treinar uma MLP convencional com SGD;
2. evoluir uma rede pequena com NEAT;
3. reaproveitar a topologia encontrada pelo NEAT, reinicializar os pesos e treinar com SGD.

Por padrao o experimento roda no MNIST com imagens `26x26`. A mesma estrutura
tambem roda no CIFAR-10 com imagens `32x32`. A acuracia aparece como controle de
sanidade; a pergunta principal e se os mapas de importancia visual mudam entre a
rede convencional e a rede que usa a topologia descoberta por evolucao.

## O que observar

A interface mostra mapas de oclusao. Cada pixel e ocultado e medimos quanto a
confianca do modelo cai na classe originalmente prevista. Com isso, o MVP
compara:

- similaridade entre mapas de oclusao;
- sobreposicao dos pixels mais importantes;
- diferenca absoluta entre mapas;
- concordancia de predicao entre os modelos.

## Como executar do zero

Os passos abaixo assumem que voce acabou de clonar o repositorio e esta na raiz
do projeto.

```bash
git clone <url-do-repositorio>
cd Trabalho-Deep-Learning
```

### 1. Criar ambiente Python

Recomenda-se usar Python 3.10 ou superior. O projeto foi testado com Python
3.14 no ambiente local.

```bash
python -m venv .venv
source .venv/bin/activate
```

No Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Instalar dependencias

```bash
python -m pip install -r requirements.txt
```

### 3. Rodar o experimento

Este comando baixa o MNIST automaticamente, testa variantes de NEAT, escolhe a
melhor pelo desempenho do NEAT puro, retreina os pesos da arquitetura vencedora
com SGD e salva os artefatos usados pela interface.

O conjunto padrao usa imagens `26x26` e gera tres linhas de comparacao: o NEAT
default `neat_mnist_features_full`, uma variante Seeded NEAT
`neat_mnist_seeded_prototypes_full` e uma variante HyperNEAT-like
`hyperneat_mnist_prototypes_cppn`. Assim o experimento compara o default com duas
variantes realmente diferentes do paper/ecossistema NEAT. O modelo mostrado nos
exemplos e no PDF e sempre o vencedor por desempenho de `NEAT puro`.

```bash
PYTHONPATH=src python -m neuroevolution_mvp.cli
```

Para rodar no CIFAR-10:

```bash
PYTHONPATH=src python -m neuroevolution_mvp.cli --dataset cifar10
```

Tambem e possivel controlar a resolucao:

```bash
PYTHONPATH=src python -m neuroevolution_mvp.cli --dataset cifar10 --image-size 26
```

Para uma execucao ainda mais curta:

```bash
PYTHONPATH=src python -m neuroevolution_mvp.cli --generations 1 --train-limit 300 --test-limit 100 --epochs 1 --evolved-epochs 1
```

Para testar uma imagem maior no MNIST sem criar um genoma direto gigante, use a
variante HyperNEAT-like de pixels crus, que evolui uma CPPN geradora de pesos:

```bash
PYTHONPATH=src python -m neuroevolution_mvp.cli --image-size 32 --neat-variant hyperneat_mnist_cppn
```

No CIFAR-10, use as variantes baseadas em `features`, `prototypes` ou
`hyperneat_*_features/prototypes`. As variantes antigas de pixels crus foram
mantidas para MNIST monocromatico.

Os resultados ficam em:

- `src/artifacts/<dataset>/results.json`;
- `src/artifacts/<dataset>/winner_genome.pkl`;
- `src/artifacts/<dataset>/baseline_mlp.pt`;
- `src/artifacts/<dataset>/evolved_topology_sgd.pt`;
- `src/artifacts/<dataset>/<dataset>_interpretability_validation.pdf`.

## Datasets, modelos e otimizadores

Datasets:

- `mnist`: padrao, `1` canal, resolucao padrao `26x26`;
- `cifar10`: `3` canais RGB, resolucao padrao `32x32`.

Modelos comparados:

- `Baseline SGD`: CNN compacta treinada do zero com SGD. Apesar do nome interno
  `BaselineMLP`, o modelo atual nao e uma MLP pura; ele usa convolucoes;
- `NEAT puro`: evolui arquitetura e pesos, sem retreino por gradiente;
- `NEAT + SGD`: reaproveita a arquitetura vencedora do NEAT e retreina os pesos
  com SGD.

### Qual e o baseline?

O baseline e o ponto de referencia convencional do experimento. Ele responde a
pergunta: "se eu nao usar neuroevolucao, quanto uma rede neural pequena treinada
com SGD consegue aprender neste dataset?".

No codigo, o baseline e salvo como `baseline_mlp.pt` por compatibilidade com as
primeiras versoes do projeto, mas a arquitetura atual e uma CNN compacta:

1. a imagem achatada e reconstruida para o formato `canais x altura x largura`;
2. passa por uma convolucao `3x3`, seguida de `ReLU`;
3. passa por uma segunda convolucao `3x3`, seguida de `ReLU`;
4. passa por `MaxPool2d`, reduzindo a resolucao espacial;
5. passa por uma terceira convolucao `3x3`, seguida de `ReLU`;
6. passa por `AdaptiveAvgPool2d(3x3)`, que transforma qualquer resolucao de
   entrada em uma grade pequena e fixa;
7. a grade e achatada e enviada para um classificador linear com uma camada
   oculta de `64` unidades;
8. a saida final tem `10` logits, um para cada classe.

Esse baseline usa os canais corretos do dataset:

- MNIST: `1` canal, imagem em tons de cinza;
- CIFAR-10: `3` canais, imagem RGB.

Ele e treinado do zero com `SGD + momentum + Nesterov`, usando a mesma funcao de
perda de classificacao (`CrossEntropyLoss`). Ele nao recebe arquitetura do NEAT,
nao usa prototipos e nao usa features compactas. Portanto, ele mede uma solucao
classica de deep learning para comparar contra as solucoes neuroevolutivas.

Como interpretar o baseline:

- se o baseline tem acuracia alta, o dataset esta aprendivel pela arquitetura
  convencional;
- se o NEAT puro fica abaixo, isso nao invalida o experimento, porque o NEAT esta
  resolvendo uma tarefa mais dificil: buscar arquitetura e pesos;
- se `NEAT + SGD` se aproxima do baseline, isso sugere que a arquitetura evoluida
  pelo NEAT tem valor quando seus pesos sao refinados por gradiente;
- nos mapas de oclusao, o baseline serve como referencia visual: ele mostra para
  onde uma CNN treinada de forma tradicional tende a olhar.

Variantes NEAT padrao:

- `neat_mnist_features_full`: default, usa descritores compactos de imagem;
- `neat_mnist_seeded_prototypes_full`: Seeded/Prototype NEAT, inicia parte da
  populacao com conexoes fortes para prototipos de classe;
- `hyperneat_mnist_prototypes_cppn`: HyperNEAT-like, evolui uma CPPN que gera o
  substrato de pesos sobre atributos de prototipo.

Otimizador:

- O unico otimizador de retreino e `SGD`, para manter clara a comparacao entre
  pesos evoluidos pelo NEAT e pesos treinados por gradiente.

No CIFAR-10, as features compactas convertem a imagem RGB para intensidade media
antes dos descritores de forma. As features de prototipo usam o vetor RGB
completo, entao preservam cor na comparacao com os centroides de classe.

## Teoria dos otimizadores implementados

Nesta implementacao, a palavra "otimizador" aparece em dois sentidos:

- otimizador de pesos por gradiente: o `SGD`, usado no baseline e no retreino;
- otimizador evolutivo: as variantes de `NEAT`, que buscam pesos e arquitetura
  por selecao, mutacao e cruzamento, sem usar gradiente.

### SGD com momentum

O `SGD` significa `Stochastic Gradient Descent`. Ele treina uma rede ajustando
os pesos na direcao que reduz o erro de classificacao. Em vez de olhar o dataset
inteiro a cada passo, ele olha mini-lotes. Isso torna o treino mais barato e
introduz um pouco de ruido, que pode ajudar a escapar de solucoes ruins.

No codigo, o SGD usa:

- `CrossEntropyLoss`: erro padrao para classificacao com 10 classes;
- `learning_rate = 0.03`: tamanho do passo;
- `momentum = 0.9`: acumula parte da direcao anterior para estabilizar e acelerar;
- `nesterov = True`: calcula uma correcao olhando um passo "a frente" na direcao
  do momentum.

Intuicao para explicar em apresentacao: o SGD e como descer uma montanha olhando
a inclinacao local. O momentum evita que cada mini-lote mude a direcao de forma
muito brusca.

Por que usar apenas SGD no retreino? Porque o objetivo do trabalho nao e comparar
Adam, RMSProp e outros otimizadores. O objetivo e separar duas fontes de
aprendizado:

- `NEAT puro`: arquitetura e pesos vieram da evolucao;
- `NEAT + SGD`: a arquitetura veio da evolucao, mas os pesos foram ajustados por
  gradiente.

Assim, quando comparamos `NEAT puro` contra `NEAT + SGD`, a pergunta fica clara:
a topologia encontrada pelo NEAT continua util quando os pesos sao retreinados
por um metodo classico?

### NEAT puro

`NEAT` significa `NeuroEvolution of Augmenting Topologies`. A ideia central e
evoluir redes neurais como se fossem individuos de uma populacao. Cada individuo
tem um genoma que descreve:

- nos de entrada, saida e, quando aparecem, nos ocultos;
- conexoes entre nos;
- pesos das conexoes;
- vieses dos nos;
- quais conexoes estao ativas.

O NEAT comeca com redes simples e, ao longo das geracoes, pode adicionar
conexoes e nos. Por isso ele "aumenta topologias": ele nao treina apenas os
pesos de uma arquitetura fixa, ele tambem procura a propria arquitetura.

Fluxo no experimento:

1. cria-se uma populacao de redes candidatas;
2. cada rede faz predicoes em um subconjunto balanceado do dataset;
3. cada rede recebe um `fitness`, isto e, uma nota;
4. redes melhores tem mais chance de gerar descendentes;
5. mutacoes alteram pesos, conexoes e estrutura;
6. depois de algumas geracoes, o melhor genoma e escolhido.

O `fitness` implementado nao usa apenas acuracia. Ele combina acuracia,
confianca na classe correta, margem entre a melhor classe e as outras, e uma
penalizacao inspirada em perda logaritmica. Isso e importante porque, em
classificacao, duas redes podem ter a mesma acuracia, mas uma estar muito mais
confiante e separando melhor as classes.

Resposta curta para a plateia: o NEAT nao faz backpropagation. Ele testa muitas
redes, da nota para elas e evolui as melhores.

### NEAT com features compactas

A variante `neat_mnist_features_full` nao entrega todos os pixels crus ao NEAT.
Ela primeiro transforma a imagem em 69 atributos compactos. Esses atributos
resumem informacoes como:

- grade media da imagem;
- distribuicao de intensidade por linhas;
- distribuicao de intensidade por colunas;
- centro de massa da imagem;
- variancia horizontal e vertical;
- densidade e intensidade do traco.

Por que isso ajuda? Porque NEAT sofre quando a entrada e grande demais. Uma
imagem `32x32` RGB tem `3072` valores. Evoluir conexoes diretamente sobre tantos
pixels e caro. As features compactas reduzem o espaco de busca e deixam o NEAT
trabalhar com informacao mais organizada.

Limite dessa variante: ao compactar a imagem, perdemos detalhes locais. No
CIFAR-10, as features compactas usam intensidade media dos canais RGB, entao
parte da informacao de cor nao entra nessa variante.

### Seeded/Prototype NEAT

A variante `neat_mnist_seeded_prototypes_full` usa uma ideia simples: comparar a
imagem atual com prototipos medios de cada classe.

Um prototipo e a media das imagens de uma classe no conjunto de treino. Por
exemplo:

- media dos exemplos da classe `0`;
- media dos exemplos da classe `1`;
- ...
- media dos exemplos da classe `9`.

Para cada imagem, o codigo calcula atributos de similaridade com esses
prototipos. No MNIST, isso significa algo como: "este digito parece mais com o
prototipo do 3 ou com o prototipo do 8?". No CIFAR-10, a mesma ideia vale para
classes como aviao, carro, gato ou cachorro.

Essa variante tambem e "seeded": parte da populacao inicial recebe conexoes
fortes que ja ligam atributos de prototipo a suas respectivas classes. Isso nao
da a resposta pronta, mas fornece um ponto de partida melhor do que pesos
totalmente aleatorios.

Intuicao para explicar: em vez de pedir que a evolucao descubra tudo do zero, a
gente entrega uma pista inicial razoavel. O NEAT ainda pode ajustar, remover ou
expandir a solucao.

### HyperNEAT-like com CPPN

`HyperNEAT` e uma extensao do NEAT. Em vez de evoluir diretamente cada conexao da
rede final, ele evolui uma rede geradora, chamada `CPPN` (`Compositional Pattern
Producing Network`). Essa CPPN recebe coordenadas e gera pesos para um substrato.

No experimento, a variante `hyperneat_mnist_prototypes_cppn` usa essa ideia de
forma simplificada:

- a CPPN e evoluida pelo NEAT;
- ela recebe coordenadas abstratas dos atributos e das classes;
- ela gera uma matriz de pesos para o classificador final;
- essa matriz pode ser usada pura ou depois retreinada por SGD.

Por que isso pode ser interessante? Porque a CPPN pode gerar padroes regulares
com poucas conexoes. Por isso, as tabelas podem mostrar HyperNEAT com menos
conexoes diretas do que outra variante. Nesse caso, as conexoes contadas sao as
da CPPN evoluida, nao necessariamente todas as relacoes efetivas que ela gera no
substrato final.

Resposta curta para a plateia: NEAT evolui a rede diretamente; HyperNEAT evolui
uma rede que gera outra rede.

### Como o modelo final e escolhido

O CLI roda as variantes configuradas e escolhe como modelo final aquela com
melhor desempenho de `NEAT puro`. Depois disso, a arquitetura dessa variante e
usada para treinar `NEAT + SGD`.

Essa escolha e proposital: como a pergunta principal e melhorar e analisar o
NEAT puro, o criterio privilegia a topologia que ja funciona melhor antes do
retreino por gradiente.

## Teoria dos metodos de interpretabilidade

Interpretabilidade, aqui, significa tentar responder: "quais partes da imagem
influenciaram a predicao do modelo?".

O projeto nao tenta provar causalidade perfeita. Ele gera uma sonda visual
simples e comparavel entre modelos diferentes. Isso e importante porque o NEAT
puro, a rede treinada por SGD e os modelos gerados por HyperNEAT nao tem todos a
mesma estrutura interna. Por isso foi escolhido um metodo externo ao modelo.

### Mapa de oclusao

O principal metodo implementado e o mapa de oclusao.

Passo a passo:

1. o modelo recebe a imagem original;
2. registramos a classe prevista e a confianca nessa classe;
3. escolhemos um pixel da imagem;
4. ocultamos esse pixel, colocando seu valor como zero;
5. rodamos a predicao de novo;
6. medimos quanto a confianca na classe original caiu;
7. repetimos isso para todos os pixels.

Se esconder um pixel derruba muito a confianca, esse pixel recebe alta
importancia. Se esconder o pixel quase nao muda a predicao, ele recebe baixa
importancia.

No MNIST, cada pixel tem um canal. No CIFAR-10, cada posicao tem tres canais
RGB. Nesse caso, a oclusao zera os tres canais daquele mesmo ponto espacial,
porque queremos saber a importancia da regiao da imagem, nao de um canal isolado.

Vantagens:

- funciona para qualquer modelo que aceite a imagem como entrada;
- nao depende de gradientes;
- serve para NEAT, HyperNEAT e redes PyTorch comuns;
- e facil de explicar visualmente.

Limites:

- e mais lento que metodos por gradiente;
- mede sensibilidade local, nao uma explicacao causal completa;
- pode destacar regioes que afetam a confianca, mas nao necessariamente sao as
  regioes que um humano usaria;
- se o modelo for ruim, o mapa explica uma decisao ruim.

Resposta curta para a plateia: o mapa mostra onde a predicao mais sofre quando a
gente apaga partes da imagem.

### Comparacao entre mapas

O projeto compara os mapas de tres modelos:

- `Baseline SGD`;
- `NEAT puro`;
- `NEAT + SGD`.

As metricas calculadas sao:

- `cosine_similarity`: mede se dois mapas apontam para regioes parecidas. Quanto
  mais perto de `1`, mais parecidos;
- `top_pixel_overlap`: mede a sobreposicao entre os pixels mais importantes de
  dois mapas;
- `mean_absolute_difference`: mede a diferenca media entre os mapas. Quanto
  menor, mais parecidos;
- `prediction_agreement`: mede se `NEAT puro` e `NEAT + SGD` deram a mesma
  classe para as amostras analisadas.

Como interpretar:

- mapas parecidos e predicoes iguais sugerem que o retreino por SGD preservou
  parte do comportamento visual da arquitetura evoluida;
- mapas diferentes com predicoes iguais sugerem que os modelos chegaram a mesma
  resposta olhando para evidencias diferentes;
- mapas parecidos com baixa acuracia nao significam bom modelo, apenas
  consistencia entre explicacoes.

### PDF de validacao visual

O PDF gerado pelo CLI e uma forma de validacao humana. Ele mostra, para varias
amostras:

- imagem original;
- mapa de oclusao do baseline;
- mapa de oclusao do NEAT puro;
- mapa de oclusao do NEAT + SGD;
- mapa de diferenca entre NEAT puro e NEAT + SGD.

O objetivo do PDF nao e substituir metricas quantitativas. Ele serve para ver se
os mapas fazem sentido visualmente. Por exemplo, no MNIST, espera-se que regioes
do traco do digito sejam relevantes. No CIFAR-10, espera-se que regioes do objeto
tenham mais destaque do que fundo aleatorio.

## Perguntas provaveis da plateia

**Por que nao usar Adam no retreino?**

Porque o foco e comparar evolucao contra um otimizador classico e controlado. Se
usarmos varios otimizadores de gradiente, a pergunta muda para "qual otimizador
treina melhor?", e nao "a arquitetura evoluida pelo NEAT e util?".

**NEAT e melhor que SGD?**

Nao necessariamente. SGD costuma ser muito forte para redes diferenciaveis em
datasets como MNIST e CIFAR-10. O interesse do NEAT aqui e outro: buscar
arquiteturas e pesos sem gradiente e depois analisar se a topologia encontrada
produz padroes interpretaveis.

**Por que o NEAT puro pode ter acuracia menor?**

Porque ele precisa resolver um problema mais dificil: procurar arquitetura e
pesos ao mesmo tempo, usando avaliacao populacional. SGD recebe uma arquitetura
fixa e usa gradientes precisos para ajustar pesos.

**Se o HyperNEAT tem poucas conexoes, ele e necessariamente mais simples?**

Ele e mais compacto no genoma gerador. Mas a CPPN pode gerar uma matriz de pesos
maior no substrato final. Entao "poucas conexoes" significa simplicidade da regra
geradora, nao necessariamente poucas interacoes efetivas.

**O mapa de oclusao prova que o modelo entende a imagem?**

Nao. Ele mostra sensibilidade da predicao a regioes da imagem. E uma evidencia
visual util, mas deve ser lida junto com acuracia, exemplos e conhecimento do
dataset.

**Por que usar prototipos?**

Porque prototipos dao ao NEAT uma representacao mais amigavel: em vez de olhar
milhares de pixels, ele olha similaridades com exemplos medios de cada classe.
Isso reduz o espaco de busca e melhora o ponto de partida.

**O que significa `NEAT + SGD` exatamente?**

Significa que primeiro o NEAT encontra uma arquitetura. Depois congelamos a ideia
da arquitetura, mas permitimos que os pesos sejam treinados por SGD. Assim
avaliamos se a estrutura encontrada pela evolucao continua boa quando recebe
treino por gradiente.

### 4. Abrir a aplicacao

```bash
PYTHONPATH=src streamlit run src/app.py
```

Depois disso, abra o endereco mostrado no terminal. Normalmente sera:

```text
http://localhost:8501
```

Se a porta estiver ocupada, use outra:

```bash
PYTHONPATH=src streamlit run src/app.py --server.port 8502
```

## Fluxo recomendado para demonstracao

1. Execute `PYTHONPATH=src python -m neuroevolution_mvp.cli` antes da apresentacao.
2. Abra `PYTHONPATH=src streamlit run src/app.py`.
3. Na aba `Visao geral`, mostre que a pergunta principal e interpretabilidade.
4. Na aba `Exemplos de olhar`, clique nos botoes `0-9` para ver um exemplo de cada digito.
5. Compare `NEAT puro` com `NEAT + SGD`: mesma arquitetura, otimizadores de peso diferentes.
6. Na aba `Topologia NEAT`, mostre quantas conexoes/camadas a evolucao encontrou.

## Estrutura principal

- `src/app.py`: interface Streamlit.
- `src/neuroevolution_mvp/cli.py`: entrada de linha de comando.
- `src/neuroevolution_mvp/experiment.py`: orquestra o experimento completo.
- `src/neuroevolution_mvp/neat_runner.py`: evolucao e avaliacao com NEAT.
- `src/neuroevolution_mvp/models.py`: modelos PyTorch.
- `src/neuroevolution_mvp/interpretability.py`: mapas de oclusao e comparacao.
- `src/configs/neat_mnist_14x14*.ini`: variantes de configuracao do NEAT.

A interface tem abas para:

- resumo interpretavel do experimento;
- exemplos clicaveis mostrando onde cada rede olha;
- comparacao entre NEAT puro, que evolui arquitetura e pesos, e NEAT + SGD,
  que usa a arquitetura do NEAT mas treina os pesos com SGD;
- predicao e confianca abaixo de cada mapa visual;
- visao da topologia evoluida pelo NEAT;
- curvas e tabelas de acompanhamento.
