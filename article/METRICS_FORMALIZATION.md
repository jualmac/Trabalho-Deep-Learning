# Formalização matemática das métricas

## Notação

Seja `D = {(x_i, y_i)}_{i=1}^N` o conjunto de teste, em que `x_i` é a imagem e
`y_i` é o rótulo verdadeiro. Comparamos dois modelos:

- `m_E`: modelo obtido por NEAT puro;
- `m_G`: topologia selecionada pelo NEAT, com pesos reinicializados e treinados
  por SGD.

Para um modelo `m`, seja `z_m(x)` o vetor de logits. A classe prevista é:

```text
c_m(x) = argmax_k z_m,k(x).
```

## Mapa de sensibilidade por oclusão

Seja `R = {r_1, ..., r_L}` a grade de regiões ocluídas. Para cada região `r`, a
atribuição é:

```text
A_m(x, r) = |z_m,c_m(x)(x) - z_m,c_m(x)(x sem r)|.
```

Após projetar as atribuições regionais para os pixels `p em Omega`, normaliza-se
o mapa:

```text
P_m(x, p) = A_m(x, p) / soma_{q em Omega} A_m(x, q).
```

Assim, `P_m(x, .)` é interpretado como uma distribuição espacial de
sensibilidade.

## Jaccard ponderado

Para cada imagem `x_i`, defina:

```text
P_i = P_{m_E}(x_i, .)
Q_i = P_{m_G}(x_i, .)
```

O Jaccard ponderado é:

```text
WJ_i = soma_{p em Omega} min(P_i(p), Q_i(p))
       / soma_{p em Omega} max(P_i(p), Q_i(p)).
```

`WJ_i` varia de `0` a `1`; quanto maior, maior a concordância espacial entre os
mapas.

## Dataset Attention Divergence

A métrica principal é a Dataset Attention Divergence:

```text
DAD(D) = 1 - (1/N) * soma_{i=1}^N WJ_i.
```

Logo, `DAD = 0` indica concordância espacial completa e `DAD = 1` indica
divergência completa.

## DAD macro por classe

Se `D_c = {i : y_i = c}` e `C*` é o conjunto de classes presentes no teste:

```text
DAD_macro = 1 - (1/|C*|) * soma_{c em C*}
              [(1/|D_c|) * soma_{i em D_c} WJ_i].
```

Essa métrica controla a influência da frequência relativa das classes.

## DAD condicionada à mesma predição

Defina o conjunto:

```text
S = {i : c_{m_E}(x_i) = c_{m_G}(x_i)}.
```

A DAD condicionada à mesma predição é:

```text
DAD_same = 1 - (1/|S|) * soma_{i em S} WJ_i.
```

Essa métrica é essencial porque, quando os modelos predizem classes diferentes,
os heatmaps explicam logits diferentes.

## Grupos condicionais

Os subconjuntos usados na análise são:

```text
C_both = {i : c_{m_E}(x_i) = y_i e c_{m_G}(x_i) = y_i}
W_both = {i : c_{m_E}(x_i) != y_i e c_{m_G}(x_i) != y_i}
W_same = {i em W_both : c_{m_E}(x_i) = c_{m_G}(x_i)}
O_one  = {i : exatamente um dos modelos prediz y_i}
```

## Gap bruto

```text
Gap_bruto = (1/|C_both|) * soma_{i em C_both} WJ_i
            - (1/|W_both|) * soma_{i em W_both} WJ_i.
```

Ele compara concordância espacial nos acertos conjuntos contra todos os erros
conjuntos.

## Gap controlado

```text
Gap_controlado = (1/|C_both|) * soma_{i em C_both} WJ_i
                 - (1/|W_same|) * soma_{i em W_same} WJ_i.
```

Esse é o gap mais importante para a interpretação, pois controla a classe
explicada. Se ele é próximo de zero, então a diferença entre acertos e erros
desaparece quando ambos os modelos explicam a mesma classe.

## Papel da acurácia preditiva

A acurácia não é a pergunta central do estudo. Ela funciona como critério mínimo
de validade comparativa: os modelos precisam ter desempenho razoavelmente
próximo para que a comparação dos mapas de interpretabilidade seja informativa.

Assim, a conclusão principal não é que um otimizador é preditivamente superior ao
outro, mas que, quando os modelos sustentam a mesma decisão, seus mapas de
sensibilidade espacial são muito semelhantes segundo a métrica de oclusão.
