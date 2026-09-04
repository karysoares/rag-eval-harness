# SPEC-012: Avaliação de recuperação sobre índice real

- **Estado:** implemented (BM25, denso, híbrido, cross-encoder)
- **Testes:** `tests/test_retrieval_eval.py`
- **Relacionado:** [SPEC-001](001-retrieval.md) (recuperação no pipeline), [SPEC-002](002-grounding.md) (grounding sobre o contexto recuperado), [SPEC-013](013-retrieval-generation-bridge.md) (efeito da recuperação na geração)

## Objetivo

Medir recuperação **contra um índice real**, com julgamentos de relevância humanos.

[SPEC-001](001-retrieval.md) mede recuperação dentro do pipeline, mas sobre um corpus
construído por item a partir do chunk ouro mais distratores conhecidos. Isso valida o
caminho do código; não mede recuperação. Recuperar de três candidatos sabendo qual é o
certo é uma verificação, não uma busca.

Esta spec fecha essa lacuna com o protocolo do BEIR: um índice de dezenas de milhares de
passagens, queries independentes, e qrels anotados por humanos.

## Conjunto e protocolo

Qualquer repositório no formato BEIR/MTEB serve — três configurações (`corpus`, `queries`,
`default` com os qrels). O caso de referência é `mteb/fiqa`: domínio financeiro, 57 638
passagens, 648 queries julgadas no split de teste, 2,63 relevantes por query em média.

O carregamento usa os **parquet convertidos** e não `datasets.load_dataset`: o loader não
resolve este layout na versão fixada, e o parquet traz 57k passagens em segundos sem
executar script de dataset.

`limite_queries` corta as queries avaliadas, **nunca o corpus**. Reduzir o índice tornaria
a recuperação artificialmente fácil e o número incomparável com a literatura.

## Métodos comparados

| Degrau | O que faz | Custo típico (FIQA, M3 16 GB) |
|--------|-----------|-------------------------------|
| `bm25` | índice esparso sobre o corpus completo | 2 s a indexar, 2 s as 648 queries |
| `denso_sobre_bm25` | reordena os primeiros 100 candidatos por coseno | ~230–270 s |
| `hibrido_rrf` | funde os dois rankings por Reciprocal Rank Fusion | igual ao denso |
| `cross_encoder_sobre_bm25` | lê query e passagem juntas, nos primeiros 50 | ~2 166 s |

O cross-encoder é o degrau que responde a «vale a pena?». Ao contrário do bi-encoder,
que compara vetores calculados em separado, lê o par **junto** e capta interação entre
termos que o coseno perde. O preço é não haver índice possível: cada par exige uma
passagem pelo modelo, por isso só se aplica a uma lista curta já filtrada.

O denso corre sobre os candidatos do BM25, que é o protocolo padrão do MS MARCO e do BEIR.
Indexar densamente 57k passagens é viável (89 MB), mas o re-ranking é o que a literatura
compara e o que escala para corpora maiores.

A cauda além da profundidade reordenada mantém a ordem do BM25. Descartá-la baixaria o
`recall@1000` por artefacto do protocolo, não por perda real de qualidade.

O RRF funde por **posição**, não por score, para não normalizar escalas incomparáveis
(o BM25 não tem tecto; o coseno vive em [-1, 1]).

## Métricas

Três, deliberadamente separadas, cada uma com denominador visível:

| Métrica | Pergunta |
|---------|----------|
| `recall@k` | dos relevantes, quantos apanhámos em k? |
| `nDCG@k` | apanhámos, e ficaram no topo? |
| `MRR@k` | a que distância está o primeiro acerto? |

Uma query sem julgamentos **não entra em nenhum denominador**: contá-la como zero seria
inventar um resultado negativo onde não há verdade. O agregado devolve
`n_queries_avaliadas` e `n_queries_sem_qrels` lado a lado.

## Porquê BM25 próprio

Implementado no repositório em vez de trazer dependência, pela mesma razão que
`statistics.py`: são vinte linhas de fórmula e o resultado é **verificável contra números
publicados**. Uma biblioteca daria o mesmo valor sem essa prova.

Parâmetros `k1=0.9`, `b=0.4` — os defaults do BEIR/Anserini, não os `1.2`/`0.75` do artigo
original, porque é assim que os números publicados foram produzidos.

**Validação:** esta implementação dá **nDCG@10 = 0,2322** no split de teste do FIQA.
Valores da mesma ordem são reportados para BM25 neste conjunto na literatura do BEIR, mas
**não confirmei a referência exacta**: até o fazer, o número acima vale como medição
própria e reprodutível, não como replicação verificada. Reproduzi-lo contra a tabela
oficial do BEIR é um critério de aceitação por cumprir.

## Resultados medidos

57 638 passagens, 648 queries julgadas, split de teste.

| Método | Embedder / modelo | nDCG@10 | recall@10 | MRR@10 | consulta |
|--------|-------------------|---------|-----------|--------|----------|
| BM25 | — | 0,2322 | 0,2922 | 0,2872 | 2 s |
| denso sobre BM25 | `paraphrase-multilingual-MiniLM-L12-v2` | 0,2323 | 0,2964 | 0,2881 | ~250 s |
| híbrido RRF | idem | 0,2702 | 0,3397 | 0,3361 | ~250 s |
| denso sobre BM25 | `all-MiniLM-L6-v2` | **0,3534** | **0,4083** | **0,4403** | 572 s |
| híbrido RRF | idem | 0,3201 | 0,3780 | 0,3974 | 573 s |
| cross-encoder sobre BM25 | `ms-marco-MiniLM-L-6-v2` | 0,3181 | 0,3711 | 0,3964 | 2 166 s |

### Diferenças com incerteza

Médias lado a lado não sustentam uma ordenação: os métodos correm sobre **as mesmas
648 queries**, logo a comparação é emparelhada. Bootstrap sobre o nDCG@10 por query:

| Par | Diferença | IC 95% | Leitura |
|-----|-----------|--------|---------|
| denso vs híbrido | +0,0333 | [+0,0164, +0,0502] | denso ganha |
| denso vs cross-encoder | +0,0352 | [+0,0191, +0,0526] | denso ganha |
| híbrido vs cross-encoder | +0,0020 | [−0,0118, +0,0164] | **indistinguíveis** |
| BM25 vs denso | −0,1211 | [−0,1452, −0,0975] | denso ganha |
| BM25 vs híbrido | −0,0879 | [−0,1021, −0,0750] | híbrido ganha |
| BM25 vs cross-encoder | −0,0859 | [−0,1042, −0,0671] | cross-encoder ganha |

Duas leituras que só a medição dá:

**O embedder do pipeline pt-BR não serve aqui.** O `paraphrase-multilingual` não acrescenta
nada ao BM25 em inglês financeiro (0,2323 vs 0,2322). Substituir o BM25 por ele seria uma
degradação silenciosa — o número não desce, portanto ninguém repara. O `all-MiniLM-L6-v2`,
treinado para recuperação, sobe 52% no nDCG.

**O híbrido nem sempre ganha.** Com o embedder fraco o RRF sobe 16% sobre o BM25; com o
embedder bom, **desce** de 0,3534 para 0,3201, e o IC da diferença exclui zero. Fundir um
ranking bom com um pior arrasta o resultado. A receita «híbrido é sempre melhor» é falsa e
só se vê medindo.

**O cross-encoder não é pior que o híbrido — é indistinguível e quatro vezes mais caro.**
A diferença de 0,0020 tem intervalo a atravessar zero: em qualidade não os separo. O que
os separa é 2 166 s contra 573 s. A afirmação defensável é sobre custo, não sobre ranking;
sem o teste emparelhado teria publicado uma ordenação a partir de ruído.

## Fora de âmbito

- Indexação densa do corpus completo (viável a esta escala, irrelevante para o protocolo).
- Latência de indexação em produção e índices persistentes em disco.
- Expansão de query, HyDE, e reescrita — outra spec.

## Critérios de aceitação

- [ ] BM25 confrontado com a tabela oficial do BEIR (medição própria feita: 0,2322).
- [x] Métricas com denominadores explícitos; query sem qrels fora do agregado.
- [x] Carregador agnóstico ao conjunto (qualquer repositório BEIR/MTEB).
- [x] Testes offline e determinísticos, sem rede.
- [x] Cross-encoder como quarto degrau, com custo e latência ao lado.
- [x] Comparações com teste emparelhado; ordenação escrita só onde o IC exclui zero.
- [x] Ligar recuperação real à geração no mesmo item — [SPEC-013](013-retrieval-generation-bridge.md).
