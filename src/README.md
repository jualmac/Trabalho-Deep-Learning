# MVP: NEAT e interpretabilidade no MNIST

Este MVP demonstra a ideia da proposta em uma versao pequena, colocando a
interpretacao das redes no centro da comparacao:

1. treinar uma MLP convencional com SGD;
2. evoluir uma rede pequena com NEAT;
3. reaproveitar a topologia encontrada pelo NEAT, reinicializar os pesos e treinar com SGD.

As imagens do MNIST sao reduzidas para 14x14 para que a evolucao ainda rode em
tempo de demo, mas os digitos continuem reconheciveis a olho nu. A acuracia aparece apenas como controle de sanidade; a pergunta
principal e se os mapas de importancia visual mudam entre a rede convencional e
a rede que usa a topologia descoberta por evolucao.

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

Este comando baixa o MNIST automaticamente, evolui a topologia com NEAT, treina
os pesos com SGD e salva os artefatos usados pela interface.

```bash
PYTHONPATH=src python -m neuroevolution_mvp.cli
```

Para uma execucao ainda mais curta:

```bash
PYTHONPATH=src python -m neuroevolution_mvp.cli --generations 1 --train-limit 300 --test-limit 100 --epochs 1
```

Os resultados ficam em:

- `src/artifacts/results.json`;
- `src/artifacts/winner_genome.pkl`;
- `src/artifacts/baseline_mlp.pt`;
- `src/artifacts/evolved_topology_sgd.pt`.

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
- `src/configs/neat_mnist_14x14.ini`: configuracao do NEAT.

A interface tem abas para:

- resumo interpretavel do experimento;
- exemplos clicaveis mostrando onde cada rede olha;
- comparacao entre NEAT puro, que evolui arquitetura e pesos, e NEAT + SGD,
  que usa a arquitetura do NEAT mas treina os pesos com SGD;
- predicao e confianca abaixo de cada mapa visual;
- visao da topologia evoluida pelo NEAT;
- curvas e tabelas de acompanhamento.
