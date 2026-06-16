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

- `Baseline SGD`: CNN compacta treinada do zero com SGD;
- `NEAT puro`: evolui arquitetura e pesos, sem retreino por gradiente;
- `NEAT + SGD`: reaproveita a arquitetura vencedora do NEAT e retreina os pesos
  com SGD.

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
