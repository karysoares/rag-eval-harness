# NLI e grounding de claims

## 1. Motivação

Verificar se uma **afirmação** é sustentada por uma **premissa** (evidência), além de similaridade lexical/embedding.

## 2. Intuição

Modelos NLI classificam pares (premissa, hipótese) em *entailment*, *neutral*, *contradiction*.

## 3. Definição operacional (extensão futura)

Unidade: sentença ou claim; premissa: chunk ou união de top-k.

## 4. Algoritmo

Não obrigatório no caminho feliz atual; *stub* ou integração opcional com `sentence-transformers` cross-encoder pode ser adicionada sem alterar contrato JSON da pipeline.

## 5. Hiperparâmetros

Escolha do modelo NLI; limiar de probabilidade.

## 6. Onde falha

Texto longo, OOD, paráfrases adversariais (Thorne et al., FEVER; surveys de NLI).

## 7. Neste repositório

Camada reservada; métricas descritas em `docs/metrics.md`. Testes usam apenas embedding + gold + judge.

## 8. Leituras

- [FEVER](https://arxiv.org/abs/1806.05564)

## 9. Exercícios

1. Porque é que “entailment” não é o mesmo que “semantic similarity”?
2. Como segmentarias um parágrafo longo para NLI sem perder contexto?
