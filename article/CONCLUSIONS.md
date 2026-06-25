# Metodologia, resultados e conclusões do experimento MNIST

## 1. Objetivo

Este experimento investiga se procedimentos distintos de otimização produzem
padrões espaciais de sensibilidade diferentes em uma tarefa de classificação do
MNIST. Foram comparadas duas condições principais:

1. **NEAT puro:** a neuroevolução define simultaneamente a topologia e os pesos
   da rede;
2. **NEAT + SGD:** a topologia selecionada pelo NEAT é preservada, todos os
   pesos são reinicializados aleatoriamente e, em seguida, treinados por SGD.

A reinicialização dos pesos na condição **NEAT + SGD** é uma decisão
metodológica central. Caso o SGD herdasse os pesos evoluídos pelo NEAT, a solução
final refletiria uma combinação inseparável entre evolução e gradiente. Ao
preservar apenas a topologia, a comparação passa a contrastar dois mecanismos de
ajuste de pesos sobre a mesma estrutura de conectividade.

O estudo não compara diretamente “caminhos neurais”, representações internas ou
mecanismos causais das redes. A unidade de análise é mais restrita: comparam-se
**mapas de sensibilidade espacial da predição**, isto é, mapas que indicam quais
regiões da imagem alteram a evidência do modelo para a classe que ele próprio
predisse.

### Conclusão central

O resultado principal é claro: **quando NEAT puro e NEAT + SGD tomam a mesma
decisão, o método de otimização dos pesos não produz diferença material nos
padrões espaciais de sensibilidade medidos por oclusão**. A divergência observada
no conjunto completo é explicada principalmente pelos casos em que os modelos
predizem classes diferentes. Portanto, neste experimento, a diferença relevante
dos heatmaps acompanha a diferença da decisão de classe, e não o fato de os pesos
terem sido obtidos por evolução ou por gradiente.

As acurácias são utilizadas apenas como controle de comparabilidade. O objetivo
não é estabelecer qual otimizador é preditivamente superior, mas verificar se
modelos com desempenho suficiente para a tarefa recorrem a regiões semelhantes
ou distintas da entrada ao sustentar suas predições.

## 2. Configuração experimental

Foram utilizadas imagens do MNIST redimensionadas para `26 x 26`, com 5.000
amostras de treinamento e 10.000 imagens no conjunto de teste. O NEAT foi
executado por 35 gerações. O baseline foi treinado por 15 épocas, enquanto a
topologia NEAT reinicializada foi treinada por 25 épocas. Nos treinamentos por
gradiente, utilizou-se SGD com taxa de aprendizado `0,03` e lotes de 64 imagens.

A variante final, `neat_mnist_seeded_prototypes_full`, foi selecionada pela
maior acurácia obtida pelo NEAT puro. Esse critério favorece o melhor desempenho
evolutivo direto, mas não necessariamente a melhor acurácia posterior após
retreinamento da topologia por SGD.

## 3. Construção dos mapas de atribuição

Os mapas foram construídos por **oclusão**, uma abordagem de perturbação
associada à literatura de interpretabilidade visual [3]. Para cada imagem `x` e
modelo `m`, define-se primeiro a classe prevista na imagem intacta:

```text
c_m(x) = argmax_k z_m,k(x),
```

em que `z_m,k(x)` é o logit atribuído à classe `k`. Em seguida, cada região `r`
de uma grade de oclusão é removida da imagem. A atribuição da região é definida
como a variação absoluta do logit da classe originalmente prevista:

```text
A_m(x, r) = |z_m,c_m(x)(x) - z_m,c_m(x)(x sem a região r)|.
```

O uso do valor absoluto captura regiões cuja remoção reduz a evidência da classe
predita e também regiões cuja remoção a aumenta. Depois disso, as atribuições
regionais são projetadas para os pixels e normalizadas para soma unitária:

```text
P_m(x, p) = A_m(x, p) / soma_q A_m(x, q).
```

Com essa normalização, cada mapa válido pode ser interpretado como uma
distribuição espacial de sensibilidade. Isso permite comparar modelos cujos
logits tenham escalas diferentes. Nesta execução, todos os 10.000 pares de mapas
foram válidos, resultando em cobertura de `100%`.

## 4. Métricas

### 4.1 Jaccard ponderado

Para dois mapas normalizados `P_i` e `Q_i`, referentes à mesma imagem `i`, a
concordância espacial é calculada pelo **Jaccard ponderado**, também conhecido
como razão min-max [4]:

```text
WJ_i = soma_p min(P_i,p, Q_i,p) / soma_p max(P_i,p, Q_i,p).
```

O valor pertence ao intervalo `[0, 1]`. O valor `1` representa mapas idênticos; o
valor `0` representa ausência de massa de atribuição compartilhada. Valores
intermediários indicam concordância parcial. Não se assume um limiar universal
para classificar a magnitude como baixa, moderada ou alta; a interpretação é
feita por comparação interna entre condições do próprio experimento.

### 4.2 Dataset Attention Divergence

A **Dataset Attention Divergence (DAD)** é definida neste estudo como o
complemento da média do Jaccard ponderado no conjunto de imagens:

```text
DAD = 1 - (1/N) * soma_i WJ_i.
```

A DAD também pertence ao intervalo `[0, 1]`: `0` indica concordância espacial
completa e `1` indica divergência completa. A métrica resume a divergência média
entre mapas no conjunto de teste, mas não demonstra equivalência causal,
identidade funcional ou igualdade de representações internas entre redes.

A **DAD macro** calcula primeiro a média por classe verdadeira e, depois, atribui
o mesmo peso a cada classe. Sua proximidade com a DAD global indica que o
resultado agregado não foi determinado pela frequência relativa das classes.

A **DAD condicionada à mesma predição** considera apenas imagens nas quais os
dois modelos escolheram a mesma classe. Essa condição é essencial porque cada
heatmap explica o logit da classe prevista pelo próprio modelo. Se os modelos
predizem classes diferentes, os mapas explicam alvos distintos.

### 4.3 Concordância condicionada ao desfecho

Os pares de predição foram separados em quatro grupos:

- **ambos acertam:** os dois modelos predizem o rótulo verdadeiro;
- **ambos erram:** os dois modelos predizem rótulos incorretos;
- **ambos erram a mesma classe:** os dois modelos erram e escolhem o mesmo
  rótulo incorreto;
- **apenas um acerta:** somente um dos modelos prediz o rótulo verdadeiro.

O gap bruto de concordância é definido como:

```text
Gap_bruto = média(WJ | ambos acertam) - média(WJ | ambos erram).
```

O gap controlado substitui o grupo de todos os erros pelo subconjunto em que
ambos erram a mesma classe:

```text
Gap_controlado = média(WJ | ambos acertam)
                 - média(WJ | ambos erram a mesma classe).
```

Esse controle reduz o confundimento causado pela comparação de mapas que
explicam classes-alvo diferentes. Valores positivos indicam maior concordância
nos acertos; valores próximos de zero indicam concordância semelhante entre os
grupos comparados.

### 4.4 Incerteza estatística

Os intervalos de 95% foram estimados por bootstrap sobre as imagens do conjunto
de teste, com 2.000 reamostragens [5]. Esses intervalos quantificam a incerteza
da média para o par de modelos treinado nesta execução. Eles não estimam a
variação causada por novas sementes, novas inicializações ou novas execuções do
NEAT.

## 5. Controle de comparabilidade preditiva

| Modelo | Acurácia final | Melhor acurácia observada |
|---|---:|---:|
| Baseline MLP + SGD | 84,66% | 97,18% (época 14) |
| NEAT puro selecionado | 82,88% | 82,88% |
| Topologia NEAT + SGD | 77,54% | 78,64% (época 22) |

Esses valores são reportados para documentar que os modelos avaliados produzem
predições informativas sobre o MNIST. A acurácia, entretanto, não é utilizada
como desfecho central do estudo. Ela serve para contextualizar a comparação dos
mapas de atribuição e para evitar que a análise de interpretabilidade seja feita
sobre modelos sem capacidade mínima de classificação.

## 6. Variantes neuroevolutivas

| Variante | NEAT puro | Topologia + SGD | Conexões registradas |
|---|---:|---:|---:|
| Features | 37,64% | 85,63% | 618 |
| Seeded prototypes | 82,88% | 77,54% | 200 |
| HyperNEAT/CPPN | 82,84% | 77,92% | 16 |

A diferença entre `seeded prototypes` e HyperNEAT no NEAT puro foi de apenas
`0,04` ponto percentual. Para esta execução, os desempenhos devem ser tratados
como empiricamente equivalentes; não há evidência suficiente para ordenar as duas
variantes de forma geral.

As 16 conexões reportadas para o HyperNEAT pertencem ao CPPN gerador. Elas não
são diretamente comparáveis às 200 conexões explícitas da rede NEAT. A tabela,
portanto, não demonstra que a rede funcional gerada pelo HyperNEAT possua apenas
16 conexões.

O melhor fitness da variante selecionada permaneceu em `1,6272` da geração 0 à
geração 34. Isso indica que o melhor indivíduo já estava presente na população
inicial semeada e não foi superado ao longo da evolução. O resultado evidencia a
importância da inicialização por protótipos, mas não evidencia melhoria do melhor
indivíduo durante as gerações avaliadas.

## 7. Resultados de interpretabilidade

| Medida | Estimativa | Interpretação direta |
|---|---:|---|
| Jaccard ponderado global | 0,677 | Concordância min-max média |
| DAD global | 0,323 | IC95% `[0,321; 0,325]` |
| DAD macro por classe | 0,324 | Diferença de 0,002 em relação à DAD global |
| Concordância de predição | 85,28% | Mesma classe em 8.528 imagens |
| Jaccard com mesma predição | 0,714 | DAD condicionada de 0,286 |
| Jaccard quando ambos acertam | 0,714 | 7.438 imagens |
| Jaccard quando ambos erram | 0,658 | 1.396 imagens |
| Jaccard no erro com mesma classe | 0,714 | 1.090 imagens |
| Gap bruto acerto menos erro | 0,057 | IC95% `[0,049; 0,064]` |
| Gap controlado | -0,00007 | IC95% `[-0,0043; 0,0042]` |

A DAD global de `0,323` equivale, por definição, a um Jaccard ponderado médio de
`0,677`. Isso indica sobreposição espacial substancial, embora não identidade,
entre os mapas. Como a DAD é uma métrica operacional definida para este estudo,
seu valor não deve ser interpretado por categorias fixas de magnitude; sua
interpretação depende das comparações condicionais.

A DAD macro (`0,324`) praticamente coincide com a DAD global (`0,323`). A
diferença de aproximadamente `0,002` é desprezível na escala `[0, 1]`, indicando
que a agregação global não foi materialmente influenciada pelo número de exemplos
em cada classe.

Quando os modelos predizem a mesma classe, a DAD cai para `0,286`, isto é, o
Jaccard ponderado sobe para `0,714`. A redução em relação à DAD global é pequena
em magnitude absoluta, mas tem interpretação consistente: mapas que explicam a
mesma classe são mais semelhantes do que mapas que incluem também decisões
divergentes. Essa diferença não caracteriza uma mudança qualitativa de
estratégia espacial.

O Jaccard foi `0,714` quando ambos acertaram e `0,658` quando ambos erraram. O
gap bruto de `0,057` possui intervalo de bootstrap que não inclui zero; portanto,
essa diferença é estimada de forma estável para as imagens avaliadas. Entretanto,
sua magnitude é pequena e a comparação bruta mistura dois fatores: acerto/erro e
classe prevista.

Essa distinção é decisiva. Entre os 1.396 erros conjuntos, 1.090 produziram a
mesma classe incorreta e 306 produziram classes diferentes. O Jaccard foi `0,714`
quando ambos erraram a mesma classe, mas caiu para `0,456` quando erraram classes
diferentes. Ao comparar acertos com erros que mantêm a mesma classe-alvo, o gap
controlado foi `-0,00007`, com IC95% `[-0,0043; 0,0042]`. Na escala `[0, 1]`,
essa magnitude é desprezível. Assim, **não há evidência de diferença prática
entre a concordância espacial dos acertos e a dos erros quando a classe explicada
é controlada**.

## 8. Discussão

Os heatmaps de NEAT puro e NEAT + SGD não são idênticos: o Jaccard global foi
`0,677`. No entanto, essa diferença não sustenta a interpretação de que os dois
procedimentos aprenderam estratégias espaciais opostas ou substancialmente
distintas. Quando os modelos explicaram a mesma classe, o Jaccard foi `0,714`;
quando ambos acertaram, também foi `0,714`; e quando ambos erraram a mesma
classe, novamente foi `0,714`.

Essa convergência numérica é o resultado mais importante do estudo. Ela mostra
que a divergência adicional dos heatmaps não decorre do otimizador em si, mas da
classe que está sendo explicada. Quando a classe-alvo é a mesma, a concordância
espacial é praticamente a mesma nos acertos e nos erros. Quando a classe-alvo
difere, a concordância cai, como esperado, porque os mapas passam a explicar
decisões diferentes.

Portanto, a interpretação adequada é a seguinte: **NEAT puro e NEAT + SGD
produziram soluções paramétricas diferentes, mas recorreram a padrões espaciais
semelhantes de sensibilidade de entrada quando sustentaram a mesma predição**.
Essa conclusão não implica igualdade de pesos, equivalência de representações
internas ou identidade causal. Ela indica convergência funcional no nível
observável pelos mapas de oclusão.

Do ponto de vista metodológico, o contraste relevante não é a eficiência
computacional do treinamento, mas a origem dos parâmetros comparados. O modelo
NEAT puro tem pesos obtidos por neuroevolução, enquanto a condição NEAT + SGD
mantém a topologia evoluída, reinicializa os pesos e os ajusta por SGD via
backpropagation. Essa separação permite perguntar se a forma de obtenção dos
pesos altera o padrão espacial de sensibilidade, mantendo fixa a estrutura
evoluída.

## 9. Limitações

- Foi analisada uma única semente. Os intervalos de bootstrap representam a
  variação entre imagens para um par fixo de modelos, não a variação entre
  treinamentos independentes.
- As acurácias são usadas apenas como controle de comparabilidade. O estudo não
  foi desenhado para comparar de forma conclusiva desempenho preditivo entre
  otimizadores.
- O baseline e a topologia evoluída não possuem necessariamente a mesma
  arquitetura. A comparação de acurácia avalia sistemas completos, e não isola
  causalmente o efeito do otimizador.
- A DAD é uma métrica operacional proposta neste estudo, baseada no Jaccard
  ponderado. Seus valores não possuem categorias universais de magnitude.
- A oclusão mede sensibilidade da saída à remoção de regiões da entrada, mas não
  prova causalidade interna, fluxo de informação ou equivalência de
  representações.
- Mapas produzidos por modelos com predições diferentes explicam logits de
  classes diferentes. Por esse motivo, a DAD condicionada à mesma predição deve
  acompanhar a DAD global.
- As conclusões se restringem ao MNIST, à configuração descrita, à topologia
  selecionada e à semente 42.

## 10. Referências metodológicas

1. Bottou, L.; Curtis, F. E.; Nocedal, J. **Optimization Methods for
   Large-Scale Machine Learning**. *SIAM Review*, 60(2), 223-311, 2018.
   DOI: [10.1137/16M1080173](https://doi.org/10.1137/16M1080173).
2. Stanley, K. O.; Miikkulainen, R. **Evolving Neural Networks through
   Augmenting Topologies**. *Evolutionary Computation*, 10(2), 99-127, 2002.
   DOI: [10.1162/106365602320169811](https://doi.org/10.1162/106365602320169811).
3. Zeiler, M. D.; Fergus, R. **Visualizing and Understanding Convolutional
   Networks**. *ECCV*, 2014. Referência metodológica para análise por oclusão de
   regiões da entrada.
4. Li, P. **Min-Max Kernels**. 2015. [arXiv:1503.01737](https://arxiv.org/abs/1503.01737).
   Fundamenta a razão `soma(min)/soma(max)` para vetores não negativos.
5. Efron, B. **Bootstrap Methods: Another Look at the Jackknife**. *The Annals
   of Statistics*, 7(1), 1-26, 1979.

## 11. Conclusão

O experimento permite uma conclusão substantiva: **NEAT puro e NEAT + SGD
apresentam padrões espaciais de sensibilidade semelhantes quando sustentam a
mesma decisão de classe**. O Jaccard ponderado foi `0,714` tanto nos acertos
quanto nos erros em que ambos os modelos escolheram a mesma classe. O gap
controlado foi `-0,00007`, com IC95% `[-0,0043; 0,0042]`, valor praticamente nulo
na escala da métrica.

A DAD global de `0,323` não contradiz essa interpretação. Ela incorpora os casos
em que os modelos predizem classes diferentes e, consequentemente, explicam
logits diferentes. Quando esse fator é controlado, a diferença entre acertos e
erros desaparece. Assim, **a divergência espacial observada acompanha a
divergência das decisões, não o tipo de otimizador utilizado para ajustar os
pesos**.

As acurácias observadas são suficientes para contextualizar a comparação, mas
não constituem o achado principal. O resultado central é interpretativo: dentro
da configuração analisada, não há evidência de que a otimização evolutiva e o
treinamento por SGD sobre a topologia evoluída produzam padrões espacialmente
distintos de fundamentação da predição quando a classe prevista é a mesma.

Em forma sintética: **os mecanismos de busca são diferentes, mas, quando a
decisão é a mesma, os modelos olham para regiões semelhantes da imagem segundo a
métrica de oclusão adotada**.
