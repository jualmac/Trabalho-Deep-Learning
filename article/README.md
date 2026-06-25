# Material para escrita do artigo

Este diretório reúne os arquivos relevantes para redigir a seção de metodologia,
resultados, discussão e conclusão do artigo a partir da execução MNIST com seed
42.

## Arquivos principais

- `results_conclusions.tex`: trecho em LaTeX pronto para colar no Overleaf. É o
  arquivo principal para escrita do artigo. Contém formalização matemática das
  métricas, tabelas de resultados, discussão, limitações e conclusão.
- `CONCLUSIONS.md`: versão em Markdown da interpretação dos resultados, útil
  para leitura e revisão fora do LaTeX.
- `results_mnist_seed42.json`: saída completa do experimento, incluindo
  configuração, acurácias, histórico de treinamento, variantes NEAT e métricas
  agregadas de interpretabilidade.
- `mnist_interpretability_summary.csv`: tabela resumida das métricas principais.
- `mnist_interpretability_summary.png`: figura de sumarização quantitativa da
  interpretabilidade.

## Interpretação metodológica

A acurácia preditiva deve ser tratada como critério mínimo de comparabilidade, e
não como foco principal do estudo. A pergunta central é se os modelos, tendo
desempenho suficiente para produzir decisões relevantes, apresentam padrões
espaciais semelhantes ou diferentes nos mapas de sensibilidade.

O resultado principal está no bloco LaTeX: quando NEAT puro e NEAT + SGD predizem
a mesma classe, a concordância espacial dos heatmaps permanece praticamente igual
nos acertos e nos erros com a mesma classe prevista. Portanto, a divergência
global dos mapas acompanha principalmente a divergência das decisões de classe,
não o tipo de otimizador utilizado para ajustar os pesos.
