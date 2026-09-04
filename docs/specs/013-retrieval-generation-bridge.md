# SPEC-013: Ponte recuperação → geração

- **Estado:** implemented
- **Testes:** `tests/test_retrieval_ponte.py`, `tests/test_prompt_parity.py`
- **Config:** [`configs/hotpotqa_ponte.yaml`](../../configs/hotpotqa_ponte.yaml)
- **Script:** [`scripts/ablacao_recuperacao.py`](../../scripts/ablacao_recuperacao.py)
- **Relacionado:** [SPEC-001](001-retrieval.md), [SPEC-002](002-grounding.md), [SPEC-012](012-retrieval-evaluation.md)

## Objetivo

Medir se **recuperar melhor produz respostas melhor sustentadas**.

[SPEC-012](012-retrieval-evaluation.md) mede recuperação contra um índice real. O pipeline
de geração corre sobre um corpus construído por item. São duas medições desconexas:
nenhum item passa pelas duas, e portanto a pergunta que um harness de RAG existe para
responder ficava sem medição. Dava para dizer «o BM25 dá 0,2322 no FIQA» e «o juiz tem
ECE de X nos contos», e não dava para dizer que a recuperação boa importa — isso era
premissa, não resultado.

## O conjunto tem de trazer três coisas no mesmo item

| Precisa de | Para quê |
|------------|----------|
| corpus grande | haver onde procurar; recuperar de quatro candidatos é verificação, não busca |
| qrels humanos | saber o que era recuperável, independentemente do que foi recuperado |
| resposta ouro | avaliar a geração |

O **HotpotQA** é o candidato verificado. `mteb/hotpotqa` traz corpus, queries e qrels no
formato BEIR; `hotpotqa/hotpot_qa` (config `distractor`, split `validation`) traz as
respostas. Os ids coincidem — **7405 queries com qrels, 7405 com resposta, 7405 com
ambos**, medido e publicado em `diagnostico_juncao`. Uma junção que perdesse metade dos
itens é um defeito que, sem esta contagem, pareceria apenas um conjunto pequeno.

Descartados, com a razão:

| Candidato | Porque não |
|-----------|------------|
| `mteb/nq` | ids opacos (`test0`); não ligam às respostas do NQ original |
| `galileo-ai/ragbench` | anotado por GPT-4 (`annotating_model_name`), não por humanos; e sem corpus global |
| `neural-bridge/rag-dataset-12000` | contexto por item — exactamente o problema que se está a resolver |

## O índice é um subconjunto declarado

O corpus do HotpotQA tem 5 233 329 passagens, o que não cabe num BM25 em memória a esta
escala de máquina. O índice usado é: **todas** as passagens julgadas das queries
seleccionadas, mais uma amostra aleatória com semente fixa (150 000 por omissão), lida
shard a shard.

As julgadas entram todas por obrigação: deixar de fora uma passagem relevante tornaria a
query impossível e o `recall` mediria a amostragem em vez do recuperador.

**Isto não é comparável com a tabela do BEIR.** `ConjuntoPonte.resumo()` devolve
`comparavel_com_beir: false` e o tamanho do corpus original, ao lado de qualquer número.
A regra da [SPEC-012](012-retrieval-evaluation.md) — cortar queries, nunca o corpus —
existe para proteger a comparabilidade do FIQA; aqui abre-se a excepção de propósito,
porque o objectivo é outro.

## O desenho da ablação

Braços sobre **os mesmos itens**, com **uma** variável a mudar: a janela de candidatos
entregue ao gerador.

```
desvio=0   → candidatos das posições 0..k     recuperação normal
desvio=2   → candidatos das posições 2..2+k   recuperação degradada
desvio=50  → candidatos das posições 50..50+k praticamente sem evidência
```

Índice, recuperador, queries, gerador, juiz, prompts e semente são idênticos. É o que
torna a diferença atribuível à recuperação em vez de a ruído da geração.

**A cobertura é a variável independente** e vai publicada ao lado do resultado: sem ela,
uma diferença no grounding não se distingue de ruído. Medida sobre 100 queries, índice de
150k, `top_k=4`:

| braço | cobertura |
|-------|-----------|
| `desvio_0` | 0,96 |
| `desvio_2` | 0,16 |
| `desvio_50` | 0,01 |

Três braços dão **dose-resposta**, e não um binário: se o grounding cair monotonicamente
com a cobertura, a conclusão é muito mais forte do que «com contexto é melhor que sem».

### O defeito que anulou a primeira corrida

Vale documentado porque é instrutivo e porque voltará a tentar acontecer.

A primeira corrida completa deu o resultado ao contrário: o braço com cobertura
**0,01** teve a **maior** taxa de aprovação (0,930 contra 0,870 do braço com cobertura
0,96). Investigado em vez de publicado, a causa era do desenho:

`itens_para_pipeline` preenchia `rag_gold_chunk` **sempre**, e `build_chunks_for_item`
(`datasets_rag.py:18`) coloca o chunk ouro em **primeiro lugar** no contexto. O braço
degradado recebia de volta exactamente a passagem que a degradação lhe devia ter tirado:
**97 de 100 itens**. Os braços eram quase idênticos e a experiência não manipulou nada.

É a falha do corpus sintético da [SPEC-001](001-retrieval.md) a reaparecer dentro do
módulo construído para a eliminar. E estava consagrada num teste, defendida com o
argumento de que o chunk ouro ficava «para o pipeline distinguir *não recuperou* de
*recuperou e respondeu mal*» — raciocínio errado, porque anexar a passagem **torna-a
recuperada**. Essa informação é a cobertura, que já estava a ser calculada.

**A lição, generalizável:** verificar o ranking não verifica a experiência. O ranking
estava correcto; o que estava errado era o contexto entregue. `verifica_manipulacao`
aborta antes de qualquer chamada de geração se os braços não diferirem no **contexto
montado**, medido por `contexto_entregue_tem_relevante`, que reconstrói os chunks
exactamente como o pipeline faz:

| braço | cobertura | contexto com relevante (defeituoso) | contexto com relevante (corrigido) |
|-------|-----------|--------------------------------------|-------------------------------------|
| `desvio_0` | 0,96 | 96/100 | 96/100 |
| `desvio_2` | 0,16 | 97/100 | 16/100 |
| `desvio_50` | 0,01 | 97/100 | 1/100 |

Segundo defeito encontrado ao corrigir: os distractores excluíam **todas** as passagens
julgadas, portanto a segunda passagem relevante do HotpotQA era descartada quando caía
na janela. Estas perguntas exigem compor factos de duas passagens; remover uma remove
silenciosamente a evidência.

Os números da corrida defeituosa não são publicados.

### Escolhas de configuração que existem para não contaminar a comparação

| Definição | Valor | Porquê |
|-----------|-------|--------|
| `rag.min_score_recuperacao` | `null` | uma porta de recuperação cortaria o braço degradado antes do gerador — mediria a porta, não a geração |
| `generation.anti_refusal_repair` | `false` | repara recusas quando a recuperação é forte, logo actuaria de forma assimétrica entre braços |
| `generation.prompt_style` | `generic` | ver abaixo |
| `metricas_lexicas.modo_referencia` | `primeiro` | o HotpotQA tem uma resposta ouro por item |

## Os prompts tinham domínio fixado

O prompt do respondedor abria com «assistente de QA sobre **histórias e contos infantis**
(português do Brasil)», e o do juiz era igualmente específico. `generate_answer` recebia
um `prompt_style` e fazia `del prompt_style  # único estilo suportado`.

Correr outro corpus por esses prompts diz ao gerador que responde sobre contos infantis
enquanto lê Wikipédia em inglês. Para a **ablação** isso não invalidaria a comparação —
os braços partilham o prompt e o viés cancela-se na diferença — mas invalidaria qualquer
taxa absoluta publicada, e contradiz a alegação de que o harness é agnóstico ao corpus: a
arquitectura era, os prompts não.

O estilo `generic` acrescenta um par em inglês, sem domínio nomeado, espelhando a
estrutura existente: fronteira anti-injection, árvore de decisão, taxonomia de flags,
contrato JSON. `prompt_files_for_config` foi corrigido em conjunto — hasheava sempre o
respondedor português, o que dava uma corrida a declarar-se reproduzível com o hash do
prompt errado.

Efeito colateral corrigido: `rag_en` normalizava para `rag_pt`. Quem escrevesse `rag_en`
num config recebia exactamente o contrário do que pediu; passa a apontar para `generic`.

## Resultados medidos

100 queries, índice de 150 199 passagens, `top_k=4`, gerador `gpt-4o-mini`, juiz `gpt-4o`,
prompts `generic`. Zero itens excluídos por erro de execução ou fallback, nos três braços.

| braço | contexto com relevante | respondeu | **respondeu e sustentado** | recusou | alucinou |
|-------|------------------------|-----------|----------------------------|---------|----------|
| `desvio_0` | 96/100 | 0,830 | **0,690** | 0,170 | 0,090 |
| `desvio_2` | 16/100 | 0,180 | **0,080** | 0,820 | 0,070 |
| `desvio_50` | 1/100 | 0,030 | **0,000** | 0,970 | 0,020 |

Bootstrap emparelhado sobre «respondeu e sustentado», 100 pares:

| par | diferença | IC 95% | McNemar |
|-----|-----------|--------|---------|
| `desvio_0` vs `desvio_2` | +0,610 | [+0,510, +0,710] | p = 1,6 × 10⁻¹⁴ |
| `desvio_0` vs `desvio_50` | +0,690 | [+0,600, +0,780] | p = 2,7 × 10⁻¹⁶ |
| `desvio_2` vs `desvio_50` | +0,080 | [+0,030, +0,140] | p = 0,0078 |

**Dose-resposta monotónica, os três intervalos excluem zero.** A recuperação determina se
o sistema consegue responder de todo. É a primeira medição neste repositório que liga
qualidade de recuperação a qualidade de resposta nos mesmos itens.

**O sistema falha de forma segura.** Alucinação **desce** quando a recuperação piora
(0,090 → 0,070 → 0,020): sem contexto o modelo recusa em vez de inventar. Só o par
extremo é distinguível de ruído, mas o sinal é consistente e é o que se quer de um
sistema RAG — vale a pena porque nada no desenho o garantia.

### A métrica óbvia aponta ao contrário

A taxa de aprovação do juiz — o KPI que sairia naturalmente do `summary.json` — é:

| braço | taxa aprovada |
|-------|---------------|
| `desvio_0` | 0,910 |
| `desvio_2` | 0,930 |
| `desvio_50` | **0,980** |

**Sobe à medida que a recuperação piora.** Não é um defeito do juiz: a árvore de decisão
manda classificar como `sustentado` uma recusa honesta quando o contexto é genuinamente
insuficiente, e é o comportamento correcto. O defeito é usá-la como KPI de produto — ela
mistura «respondeu bem» com «recusou bem», e um sistema que não recupera nada recusa
sempre e pontua 0,98.

É a [regra 8 do `CLAUDE.md`](../../CLAUDE.md) — planos métricos não se misturam — a
aparecer com números. A variável dependente correcta separa os dois: **respondeu**
(tentou) e **respondeu e sustentado** (tentou e acertou o grounding). A recusa é
comportamento desejável e não é produto.

### O plano léxico não mede nada aqui

`exact_match` é **0,000 nos três braços** e o F1 de tokens fica em 0,225 no melhor. Não é
resultado: é artefacto de configuração. As respostas do HotpotQA são spans de 1 a 5
palavras («Yoruba»), e o prompt `generic` pede uma frase curta («The Yoruba people have
significant populations in…»). A sobreposição entre uma frase e um span é estruturalmente
baixa, e a igualdade exacta é impossível.

Os números léxicos desta corrida **não são publicáveis** e não entram em nenhuma
conclusão. Corrigir exige um modo de extracção de span no prompt, e é outra corrida.

## Estatística

Desenho emparelhado por construção — os braços correm sobre os mesmos ids. A comparação
usa **McNemar** e **bootstrap emparelhado**; um teste de duas proporções sobrestimaria o
erro-padrão e perderia poder.

Excluídos da comparação, contados por braço:

- itens com `processing_error` — medem infraestrutura, não o sistema ([regra 3](../../CLAUDE.md));
- itens em que o juiz caiu no `fallback_heuristico` — que responde `sustentado` e tornaria
  um juiz avariado indistinguível de um juiz permissivo ([regra 4](../../CLAUDE.md)).

Exclusão assimétrica entre braços é ela própria um achado, e por isso `n_excluidos_a` e
`n_excluidos_b` vão no relatório.

## Artefactos

Cada braço grava o seu `predictions.jsonl` completo, para o resultado ser reconferível
item a item. `outputs/ablacao/ablacao_recuperacao.json` traz o resumo do conjunto, os
parâmetros, a cobertura e a taxa por braço, e as comparações emparelhadas.

## Fora de âmbito

- Reranking denso ou cross-encoder dentro da ponte — a variável aqui é a **cobertura**, e
  os degraus de recuperação estão medidos na [SPEC-012](012-retrieval-evaluation.md).
- Multi-hop explícito: o HotpotQA exige compor factos de duas passagens, e este desenho
  não separa «falhou por não recuperar as duas» de «recuperou e não compôs». É a próxima
  pergunta, não esta.
- Estratificação por `type` (`comparison` vs `bridge`), que o conjunto permite e que
  precisa de mais itens por estrato.

## Critérios de aceitação

- [x] Junção verificada e contada, não assumida (`diagnostico_juncao`).
- [x] Subamostragem do índice declarada em todos os resumos, com `comparavel_com_beir: false`.
- [x] Cobertura publicada por braço como variável independente.
- [x] Comparação emparelhada; conclusão escrita só onde o IC exclui zero.
- [x] Exclusões contadas por braço e declaradas.
- [x] Prompts agnósticos ao corpus, com hashing de proveniência a acompanhar o estilo.
- [x] `predictions.jsonl` por braço.
- [x] Verificação do **contexto entregue** antes da geração, não só do ranking.
- [x] Dose-resposta medida, com IC a excluir zero nos três pares.
- [ ] Modo de extracção de span, para o plano léxico ser interpretável neste conjunto.
- [ ] Estratificação por tipo de pergunta (`comparison` vs `bridge`).
