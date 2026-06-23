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

Como intuicao, o SGD e como descer uma montanha olhando
a inclinacao local. O momentum evita que cada mini-lote mude a direcao de forma
muito brusca.

Por que usar apenas SGD no retreino? Porque o objetivo do trabalho nao e comparar
Adam, RMSProp e outros otimizadores. O objetivo e separar duas fontes de
aprendizado:

- `NEAT puro`: arquitetura e pesos vieram da evolucao;
- `NEAT + SGD`: somente a arquitetura veio da evolucao; antes do SGD, os pesos
  evoluidos sao descartados e reinicializados aleatoriamente.

Assim, quando comparamos `NEAT puro` contra `NEAT + SGD`, a pergunta fica clara:
a topologia encontrada pelo NEAT continua util quando os pesos sao retreinados
por um metodo classico?

### Protocolo de reinicializacao dos pesos

O ramo `NEAT + SGD` nao inicia com os pesos vencedores do NEAT. Isso seria
fine-tuning e daria ao SGD uma vantagem inicial dificil de separar do efeito da
topologia. O protocolo implementado e:

1. o `NEAT puro` mantem arquitetura, pesos e vieses evoluidos;
2. copia-se apenas a topologia vencedora para o ramo `NEAT + SGD`;
3. os pesos das conexoes ativas sao reinicializados aleatoriamente;
4. os vieses sao zerados;
5. o SGD treina essa rede desde o inicio.

Nas variantes NEAT diretas, preservam-se os nos e as conexoes habilitadas pelo
genoma. Nas variantes HyperNEAT-like, preserva-se a mascara de conexoes expressas
pela CPPN: pesos ativos recebem inicializacao Xavier e conexoes inativas ficam
mascaradas durante todo o SGD. O baseline tambem nasce com pesos aleatorios, mas
sua arquitetura e uma CNN definida manualmente, nao a topologia do NEAT.

Assim, `NEAT puro vs NEAT + SGD` compara dois metodos de obtencao dos pesos sobre
a mesma estrutura. Artefatos produzidos antes desse protocolo devem ser gerados
novamente; em `results.json`, o campo `sgd_weight_initialization` deve ser
`random_reset_preserving_neat_topology`.

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

Em resumo, o NEAT nao faz backpropagation: ele testa muitas redes, atribui
fitness e evolui as melhores.

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
- no modelo puro, usam-se os pesos gerados;
- no ramo SGD, preserva-se apenas a mascara de conexoes nao nulas e os pesos
  ativos sao reinicializados antes do treino.

Por que isso pode ser interessante? Porque a CPPN pode gerar padroes regulares
com poucas conexoes. Por isso, as tabelas podem mostrar HyperNEAT com menos
conexoes diretas do que outra variante. Nesse caso, as conexoes contadas sao as
da CPPN evoluida, nao necessariamente todas as relacoes efetivas que ela gera no
substrato final.

Em resumo, NEAT evolui a rede diretamente; HyperNEAT evolui uma rede que gera
outra rede.

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

O principal metodo implementado e o mapa de oclusao, inspirado na analise de
sensibilidade por oclusao de Zeiler e Fergus (2014). O metodo perturba regioes da
entrada e mede a mudanca na saida do classificador, sem depender da arquitetura
interna ou de gradientes.

Passo a passo:

1. o modelo recebe a imagem original;
2. registramos a classe prevista e seu logit antes do softmax;
3. escolhemos um pixel da imagem;
4. ocultamos esse pixel, colocando seu valor como zero;
5. rodamos a predicao de novo;
6. medimos o valor absoluto da variacao do logit da classe original;
7. repetimos isso para todos os pixels.

Se esconder uma regiao altera muito o logit, ela recebe alta sensibilidade. Se a
oclusao quase nao muda o logit, ela recebe baixa sensibilidade. Usa-se valor
absoluto porque o objetivo e medir influencia local; tanto uma queda quanto um
aumento forte mostram que a regiao afeta a decisao.

O logit e usado no lugar da probabilidade softmax para evitar saturacao. Quando
o modelo apresenta probabilidade arredondada para `100%`, mudancas relevantes no
logit podem produzir variacoes numericamente quase nulas na probabilidade. A
implementacao anterior tambem descartava aumentos com `clamp(min=0)`, o que
podia gerar um heatmap visualmente vazio mesmo quando o modelo era sensivel a
oclusao. O mapa atual deve ser chamado de mapa de sensibilidade por oclusao, nao
de mapa de evidencia exclusivamente positiva.

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
- depende da escolha da perturbacao; zerar pixels pode criar entradas fora da
  distribuicao natural, uma limitacao conhecida dos metodos de perturbacao;
- pode destacar regioes que afetam a confianca, mas nao necessariamente sao as
  regioes que um humano usaria;
- se o modelo for ruim, o mapa explica uma decisao ruim.

Em termos operacionais, o mapa mostra onde a saida do modelo mais muda quando
regioes da imagem sao ocultadas.

### Comparacao entre mapas

O projeto compara os mapas de tres modelos:

- `Baseline SGD`;
- `NEAT puro`;
- `NEAT + SGD`.

As metricas calculadas sao:

- `cosine_similarity`: mede se dois mapas apontam para regioes parecidas. Quanto
  mais perto de `1`, mais parecidos;
- `top_pixel_overlap`: mede a sobreposicao entre os pixels mais importantes de
  dois mapas. Como os dois conjuntos tem o mesmo tamanho, o valor e a fracao de
  pixels-chave compartilhados;
- `top_pixel_iou`: divide a intersecao dos pixels-chave pela uniao. E uma versao
  mais conservadora da sobreposicao;
- `weighted_jaccard`: compara toda a massa dos mapas pela soma dos minimos
  dividida pela soma dos maximos. Nao depende de escolher apenas os top pixels;
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

### Metrica global do conjunto de teste

A metrica primaria do estudo e a `Dataset Attention Divergence` (`DAD`), uma
metrica operacional definida neste projeto a partir do Jaccard ponderado; o nome
nao representa um indice padronizado da literatura. A similaridade usada em
cada imagem e a mesma forma do `min-max kernel` para vetores nao negativos
formalizada por Li (2015), tambem relacionada ao Jaccard ponderado. Para
cada imagem `i`, calculamos o Jaccard ponderado entre o mapa do `NEAT puro` e o
mapa do `NEAT + SGD`:

```text
J_i = soma_p min(A_i,p, B_i,p) / soma_p max(A_i,p, B_i,p)
```

em que `p` percorre as posicoes espaciais, `A` e o mapa do NEAT puro e `B` e o
mapa da mesma imagem no NEAT + SGD. Em seguida:

```text
Dataset Attention Agreement  = media_i(J_i)
Dataset Attention Divergence = 1 - media_i(J_i)
```

Portanto, a parte estabelecida na literatura e a similaridade min-max entre dois
vetores nao negativos. A contribuicao metodologica deste experimento e aplicar
essa similaridade a pares de mapas normalizados da mesma imagem, tirar a media
no conjunto de teste e usar seu complemento como divergencia. A DAD deve ser
descrita no artigo como uma metrica proposta/operacional, nao como um benchmark
XAI previamente validado.

A divergencia varia de `0` a `1`:

- `0`: os dois otimizadores produziram padroes espaciais iguais nas imagens;
- valores maiores: maior diferenca media entre as regioes priorizadas;
- `1`: ausencia de massa de atencao compartilhada.

Essa e uma media de comparacoes pareadas: os dois modelos sao comparados na
mesma imagem antes da agregacao. Nao se calcula primeiro um heatmap medio do
dataset, pois objetos aparecem em posicoes diferentes e poderiam se cancelar.

O resultado inclui:

- `dataset_attention_divergence`: media do Jaccard ponderado por imagem, dando o
  mesmo peso a cada imagem valida;
- `dataset_attention_divergence_macro`: calcula a concordancia media dentro de
  cada classe e depois tira a media das classes;
- `dataset_attention_divergence_same_prediction`: considera apenas imagens em
  que os dois modelos previram a mesma classe, controlando a classe-alvo que o
  heatmap explica;
- `dataset_attention_divergence_ci95`: intervalo bootstrap de 95%, com 2.000
  reamostragens, seguindo a motivacao de reamostragem nao parametrica de Efron
  (1979);
- `valid_map_coverage`: fracao das imagens em que ambos os mapas possuem sinal
  positivo de oclusao.

Por padrao, a metrica usa todo o conjunto de teste carregado pelo experimento.
Com `test_limit=10000`, isso corresponde ao conjunto oficial de teste completo
do MNIST ou CIFAR-10. Para uma execucao exploratoria mais curta:

```bash
PYTHONPATH=src python -m neuroevolution_mvp.cli --test-limit 1000 --interpretability-samples 200
```

`--interpretability-samples 0` significa usar todas as imagens carregadas. Se o
valor for maior que zero, o codigo usa um subconjunto em rodizio balanceado por
classe e registra `evaluation_scope=class_balanced_subset`.

Para viabilizar a avaliacao completa, a metrica global usa oclusao regional em
uma grade `8x8`. Cada celula e ocultada e sua importancia e atribuida a regiao.
Isso e mais estavel para imagens naturais e reduz o custo para cerca de 64
perturbacoes por imagem. Os exemplos do PDF continuam usando oclusao por pixel
para maior detalhamento visual.

A DAD global e uma medida de divergencia comportamental: quando os modelos
preveem classes diferentes, os mapas tambem explicam alvos diferentes. Por isso,
a DAD restrita a `same_prediction` deve ser apresentada junto da global como
controle. A global responde "quanto o comportamento visual total difere?"; a
restrita responde "quanto as regioes diferem quando a decisao de classe e a
mesma?".

### Metricas condicionadas a acerto e erro

Para formalizar frases como "quando acertam, olham para os mesmos lugares; quando
erram, erram de forma diferente", o experimento nao calcula apenas uma media
global. Para cada par de modelos, as amostras sao separadas em:

- `both_correct`: os dois modelos acertaram o rotulo;
- `both_wrong`: os dois modelos erraram;
- `both_wrong_same_prediction`: os dois erraram e escolheram a mesma classe
  incorreta;
- `both_wrong_different_prediction`: os dois erraram, mas escolheram classes
  incorretas diferentes;
- `one_correct`: apenas um modelo acertou;
- `all`: todo o conjunto avaliado pela metrica global.

O par principal e `NEAT puro vs NEAT + SGD`, porque os dois usam a mesma
arquitetura evoluida. Assim, a diferenca principal esta nos pesos: evoluidos no
primeiro e retreinados por gradiente no segundo. Comparar `Baseline vs NEAT`
continua sendo util, mas mistura dois fatores, arquitetura e metodo de treino;
portanto, nao deve ser apresentado como efeito isolado do otimizador.

A metrica-resumo principal e:

```text
gap de concordancia = Jaccard ponderado medio em both_correct
                    - Jaccard ponderado medio em both_wrong
```

Leitura:

- `gap > 0`: os modelos concordam mais sobre onde olhar quando ambos acertam;
- `gap perto de 0`: nao ha diferenca clara entre acertos e erros;
- `gap < 0`: os mapas foram mais parecidos nos erros conjuntos.

O experimento tambem calcula `wrong_prediction_disagreement`: entre os casos em
que ambos erraram, qual fracao recebeu classes erradas diferentes. Essa metrica
mede diferenca de decisao; o gap mede diferenca espacial dos mapas. As duas nao
devem ser confundidas.

Existe ainda um `controlled_attention_gap`, que compara os acertos conjuntos
apenas com erros em que os dois modelos previram a mesma classe incorreta. Esse
e o contraste espacial mais rigoroso, porque em ambos os grupos os dois mapas da
mesma amostra explicam a mesma classe-alvo. Quando os modelos erram classes
diferentes, cada mapa explica uma saida diferente; nesse caso, menor concordancia
pode ser consequencia da classe escolhida, nao apenas do otimizador.

Cada media condicionada inclui:

- numero total de amostras do grupo;
- numero de pares de mapas validos;
- media;
- intervalo de confianca aproximado de 95% para a media.

O gap tambem recebe um intervalo de confianca bootstrap de 95%, com 2.000
reamostragens das imagens. Se esse intervalo incluir zero, os dados daquela
execucao nao sustentam uma diferenca clara entre acertos e erros.

Se um grupo tiver poucas imagens, a conclusao condicionada deve ser tratada como
exploratoria. Mapas sem sinal positivo de oclusao sao contados na cobertura, mas
excluidos das medias de
similaridade, evitando que pixels escolhidos arbitrariamente por empate sejam
interpretados como concordancia real.

As metricas top-k usam no maximo 15% dos pixels, mas nao completam o conjunto
com pixels de importancia zero. Isso evita sobreposicao artificial causada por
empates entre pixels sem sinal. O `weighted_jaccard` e a metrica primaria porque
usa o mapa inteiro e reduz a dependencia desse limiar.

Importante: cada mapa e normalizado pelo seu proprio valor maximo. Portanto, as
metricas comparam o padrao espacial relativo, nao a intensidade absoluta da
variacao do logit. A conclusao correta e "os modelos apresentam sensibilidade
espacial parecida", e nao "os pixels tiveram exatamente o mesmo efeito numerico".

Esses intervalos medem variacao entre imagens para um par de modelos ja
treinado. Eles nao medem a aleatoriedade do treinamento do SGD e da evolucao.
Para afirmar algo sobre os metodos, e nao apenas sobre uma execucao, repita o
experimento com varias sementes e reporte media e dispersao entre execucoes:

```bash
PYTHONPATH=src python -m neuroevolution_mvp.cli --seed 1 --artifact-dir src/artifacts/mnist_seed_1
PYTHONPATH=src python -m neuroevolution_mvp.cli --seed 2 --artifact-dir src/artifacts/mnist_seed_2
PYTHONPATH=src python -m neuroevolution_mvp.cli --seed 3 --artifact-dir src/artifacts/mnist_seed_3
```

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

## Como reexecutar para produzir resultados do artigo

Artefatos antigos devem ser descartados da analise porque foram gerados antes do
reset independente dos pesos e antes da correcao de saturacao dos heatmaps. Use
diretorios novos por dataset e semente.

Para confirmar que um resultado segue o protocolo atual, verifique em
`results.json`:

```text
summary.sgd_weight_initialization = random_reset_preserving_neat_topology
interpretation_summary.attribution_method = occlusion_absolute_predicted_logit_change_v2
interpretation_summary.dataset_metric = dad_weighted_jaccard_v1
```

Execucao completa para MNIST, com todas as 10.000 imagens de teste na DAD e 30
exemplos visuais no PDF:

```bash
PYTHONPATH=src python -m neuroevolution_mvp.cli \
  --dataset mnist \
  --seed 42 \
  --test-limit 10000 \
  --interpretability-samples 0 \
  --pdf-samples 30 \
  --artifact-dir src/artifacts/article_mnist_seed42
```

Para CIFAR-10:

```bash
PYTHONPATH=src python -m neuroevolution_mvp.cli \
  --dataset cifar10 \
  --seed 42 \
  --test-limit 10000 \
  --interpretability-samples 0 \
  --pdf-samples 30 \
  --artifact-dir src/artifacts/article_cifar10_seed42
```

Como NEAT e SGD sao estocasticos, o protocolo recomendado e repetir pelo menos
cinco sementes. Exemplo para MNIST:

```bash
for seed in 1 2 3 4 5; do
  PYTHONPATH=src python -m neuroevolution_mvp.cli \
    --dataset mnist \
    --seed "$seed" \
    --test-limit 10000 \
    --interpretability-samples 0 \
    --pdf-samples 20 \
    --artifact-dir "src/artifacts/article_mnist_seed${seed}"
done
```

Cada diretorio contem:

- `results.json`: metricas completas, intervalos, tamanhos dos grupos,
  acuracias e protocolo de inicializacao;
- `<dataset>_interpretability_validation.pdf`: exemplos qualitativos;
- `<dataset>_interpretability_summary.csv`: uma linha plana pronta para tabela,
  planilha ou agregacao entre sementes;
- `<dataset>_interpretability_summary.png`: painel quantitativo pronto para uso
  como figura no artigo;
- pesos do baseline, NEAT + SGD, genoma vencedor e centroides quando usados.

Dentro de `results.json`, use:

- `article_summary`: versao plana das metricas principais;
- `metric_definitions`: definicao, faixa e interpretacao de cada metrica;
- `interpretation_summary`: resultados completos e todos os recortes;
- `interpretation_summary.pairwise_conditioned.neat_pure_vs_neat_sgd`: analise
  detalhada do par principal.

O PDF comeca com capa, painel quantitativo, tabela de definicoes e depois os
exemplos visuais. A selecao nao e apenas por classe: ela reserva casos para
`apenas NEAT puro acerta`, `apenas NEAT + SGD acerta`, `ambos erram classes
diferentes`, `ambos acertam` e `ambos erram a mesma classe`. Se uma categoria
nao existir no conjunto de busca, as paginas restantes priorizam outras
discordancias e cobertura de classes.

Se o experimento ja foi executado com o protocolo atual, regenere somente os
artefatos editoriais sem treinar novamente:

```bash
PYTHONPATH=src python -m neuroevolution_mvp.cli \
  --dataset mnist \
  --artifact-dir src/artifacts/article_mnist_seed42 \
  --pdf-samples 30 \
  --report-only
```

Esse comando enriquece o `results.json` com `article_summary` e
`metric_definitions` e recria CSV, PNG e PDF. Ele recusa artefatos anteriores ao
reset independente dos pesos ou aos mapas de oclusao baseados em logits.

Para a tabela principal do artigo, reporte por dataset e por semente:

- acuracia de `Baseline`, `NEAT puro` e `NEAT + SGD`;
- DAD global e IC95%;
- DAD macro por classe;
- DAD condicionada a mesma classe prevista e seu `n`;
- cobertura de mapas validos;
- gap acerto-erro, gap controlado e respectivos IC95%;
- quantidade de imagens em `both_correct`, `both_wrong` e `one_correct`.

Uma conclusao formal deve seguir os intervalos. Exemplo: se a DAD for `0.31`
com IC95% `[0.28, 0.34]`, pode-se afirmar que houve divergencia espacial media
de `0.31` segundo a metrica proposta. Se o IC95% do gap acerto-erro incluir
zero, nao se deve afirmar que a concordancia muda entre acertos e erros. Os PDFs
servem como ilustracao dos resultados quantitativos, nao como evidencia isolada.

## Questoes metodologicas frequentes

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

## Referencias teoricas

- Stanley, K. O.; Miikkulainen, R. **Evolving Neural Networks through
  Augmenting Topologies**. *Evolutionary Computation*, 10(2), 99-127, 2002.
  DOI: [10.1162/106365602320169811](https://doi.org/10.1162/106365602320169811).
  Fundamenta o NEAT, incluindo complexificacao incremental, marcadores
  historicos e especiacao.
- Stanley, K. O.; D'Ambrosio, D. B.; Gauci, J. **A Hypercube-Based Encoding for
  Evolving Large-Scale Neural Networks**. *Artificial Life*, 15(2), 185-212,
  2009. DOI: [10.1162/artl.2009.15.2.15202](https://doi.org/10.1162/artl.2009.15.2.15202).
  Fundamenta HyperNEAT e o uso de CPPNs como codificacao indireta.
- Zeiler, M. D.; Fergus, R. **Visualizing and Understanding Convolutional
  Networks**. *ECCV*, 2014.
  [Artigo](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf). Fundamenta a
  analise de sensibilidade por oclusao de regioes da entrada.
- Li, P. **Min-Max Kernels**. 2015.
  [arXiv:1503.01737](https://arxiv.org/abs/1503.01737). Formaliza, para vetores
  nao negativos, a razao `soma(min)/soma(max)` usada como concordancia dos
  heatmaps antes da agregacao DAD.
- Efron, B. **Bootstrap Methods: Another Look at the Jackknife**. *The Annals of
  Statistics*, 7(1), 1-26, 1979.
  [Artigo](https://sites.stat.washington.edu/courses/stat527/s13/readings/ann_stat1979.pdf).
  Fundamenta os intervalos de confianca por reamostragem.
- Adebayo, J. et al. **Sanity Checks for Saliency Maps**. *NeurIPS*, 2018.
  [Artigo](https://papers.nips.cc/paper_files/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html).
  Motiva nao confiar apenas no apelo visual dos mapas e reportar verificacoes
  quantitativas e controles.
- Brunke, L.; Agrawal, P.; George, N. **Evaluating Input Perturbation Methods for
  Interpreting CNNs and Saliency Map Comparison**. 2021.
  [arXiv:2101.10977](https://arxiv.org/abs/2101.10977). Discute a sensibilidade
  dos mapas de perturbacao a escolha da oclusao e de seus hiperparametros.
