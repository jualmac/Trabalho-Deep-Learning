# Trabalho-Deep-Learning

O MVP do trabalho de neuroevolucao esta em [`src/README.md`](src/README.md).

Para executar a aplicacao:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=src python -m neuroevolution_mvp.cli
PYTHONPATH=src streamlit run src/app.py
```

Por padrao, o experimento roda no MNIST. Para rodar a mesma ideia no CIFAR-10:

```bash
PYTHONPATH=src python -m neuroevolution_mvp.cli --dataset cifar10
```

O CLI tambem gera um PDF separado para validacao visual das predicoes:

```bash
PYTHONPATH=src python -m neuroevolution_mvp.cli --pdf-samples 12
```

Os artefatos agora ficam separados por dataset. Exemplos:

- `src/artifacts/mnist/mnist_interpretability_validation.pdf`;
- `src/artifacts/cifar10/cifar10_interpretability_validation.pdf`.

O fluxo compara `NEAT puro` contra `NEAT + SGD`: primeiro o NEAT evolui
arquitetura e pesos, depois a arquitetura vencedora e reaproveitada com pesos
retreinados por SGD.

Por padrao o CLI testa todas as variantes configuradas. Por isso o terminal pode
mostrar `Running generation 0` varias vezes: cada reinicio e uma nova variante
NEAT, nao um loop infinito. Para rodar apenas uma variante:

```bash
PYTHONPATH=src python -m neuroevolution_mvp.cli --neat-variant neat_mnist_14x14_full
```

O conjunto padrao do MNIST roda em resolucao `26x26`; no CIFAR-10, o padrao e
`32x32`. O experimento tem tres linhas de comparacao:
`neat_mnist_features_full` como NEAT default, `neat_mnist_seeded_prototypes_full`
como Seeded NEAT e `hyperneat_mnist_prototypes_cppn` como HyperNEAT-like. Assim ha
duas variantes realmente diferentes alem do default. O modelo usado nos exemplos
e no PDF e sempre o que tiver melhor desempenho de `NEAT puro`.

Para testar a variante HyperNEAT-like antiga em pixels crus com imagem maior:

```bash
PYTHONPATH=src python -m neuroevolution_mvp.cli --image-size 32 --neat-variant hyperneat_mnist_cppn
```
