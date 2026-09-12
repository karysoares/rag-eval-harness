# Veredito de defensabilidade

**Data:** 2026-09-11 · **Âmbito:** tudo o que o repositório publica · **Audiência:** comité técnico

A pergunta deste documento não é «o projecto é bom». É mais estreita e mais dura: **um
revisor que abra cada coisa publicada consegue defendê-la?** Um número medido mas
irreverificável, um braço experimental confundido apresentado como limpo, um ficheiro de
evidência que ninguém lê — nada disso é um bug de código, e tudo isso custa credibilidade
mais depressa do que um teste vermelho.

## Resumo

| Classe de defeito | Encontrados | Fechados | Abertos |
|---|---|---|---|
| Números publicados sem artefacto verificável | 8 | 8 | 0 |
| Comparações apresentadas como limpas quando não eram | 2 | 2 | 0 |
| Artefactos publicados que ninguém lê | 2 | 2 | 0 |
| Documentação a afirmar capacidade não entregue | 1 | 1 | 0 |
| Planos métricos declarados e vazios | 2 | 0 | **2** |
| Estilos de prompt entregues sem caracterização | 1 | 1 | 0 |
| Cobertura empírica abaixo do que a tese exige | 1 | 0 | **1** |

## Fechados

### 1. A evidência da tabela principal não existia no repositório
`README.md` linkava `docs/evidencia/judge_ab_fairytale_200.json` e esse directório **nunca
tinha sido commitado**. Para quem clonasse, o ficheiro que sustenta a tese central era um
404. Quebrava a invariante 5 (número publicado vem de corrida gravada) e a 7 (documentação
publicada não referencia o que não existe) ao mesmo tempo, no sítio que um comité verifica
primeiro. A auditoria anterior não o viu porque correu sobre a árvore de trabalho local.

**Fechado:** directório versionado, um único caminho de publicação
(`scripts/publish_run_evidence.py`), e a verificação passou a ser feita contra uma clonagem
limpa e não contra o disco local.

### 2. Os números publicados vinham do tokenizador quebrado
Este foi introduzido **durante** este trabalho: acrescentou-se `--rescore-lexical`,
demonstrou-se que altera 191–200 de 200 itens, e os READMEs continuaram a publicar os
valores anteriores. Como a exatidão e o κ derivam da referência léxica, um revisor que
corresse a ferramenta obtinha números diferentes dos publicados.

**Fechado:** todos os valores re-derivados; a reavaliação passou a re-derivar também a
meta-avaliação do juiz, porque re-pontuar as métricas e deixar o relatório intacto punha os
dois em desacordo nos mesmos itens.

| braço | exatidão antes | depois | κ antes | depois |
|---|---|---|---|---|
| `gpt-4o` | 0,561 | 0,545 | −0,028 | −0,006 |
| `gpt-5.4-nano` | 0,610 | 0,595 | 0,092 | 0,084 |

A correcção **enfraquece** o resultado publicado e é por isso que importa: com a
normalização correcta, o intervalo de κ do `gpt-5.4-nano` passa a conter zero, pelo que
nenhum braço de N completo concorda com a referência além do acaso.

### 3. Duas de quatro linhas da tabela não eram comparações de juízes
- `gpt-4o-mini` a julgar `gpt-4o-mini` é **auto-avaliação**. O protocolo já marcava
  `judge_same_as_generator: true`; o defeito era publicar a linha numa tabela sobre
  qualidade de juízes.
- `qwen2.5` a julgar `llama3.2` variava **juiz e gerador**, violando a regra de uma
  variável de cada vez que o próprio repositório impõe.

**Fechado:** a tabela de destaque só contém comparações válidas; as duas inválidas ficam na
tabela detalhada, marcadas, com uma secção a dizer o que as invalida. Retirá-las esconderia
o que foi corrido; apresentá-las sem marca era o erro.

### 4. Sete comparativos publicados eram irreverificáveis
`comparatives.json` publicava médias sobre N=1025 vindas de corridas que já não existem, e
os números estavam **hard-coded no script**, pelo que regenerar trazia-os de volta.

**Fechado:** esquema 2.0 com apenas o que uma clonagem limpa verifica; bloco `removidos` a
registar o que saiu e porquê; proveniência passa a dizer `caminho` e `versionado`.

### 5. Artefactos publicados que ninguém lia
`run_ci_fixture_kpi_lexical.json` tinha todos os valores `null`;
`run_ci_fixture_protocolo.json` descrevia uma fixture de uma linha com um limiar que
nenhum config usa. **Removidos.** O que o CI precisa de garantir está em
`tests/fixtures/ci_kpi_golden.json`, com KPIs reais e um gate que falha no desvio.

### 6. Uma spec afirmava capacidade não entregue
SPEC-A-NQ dizia `implemented` para um adaptador que não existe, citando configs e um teste
que não são distribuídos. **Marcada histórica.**

## Abertos — e declarados como tal

Estes não se fecham com código nesta sessão, e o veredito seria desonesto se os omitisse.

### A. Plano C (HITL) está documentado e vazio
`sumario_hitl` é `null` em todas as corridas. Os únicos rótulos humanos são 6 de fixture,
todos `correto`, o que torna κ indefinido (avaliador constante) e fica abaixo do portão
`MIN_ROTULOS_PARA_METRICAS = 10` que o próprio código impõe.

**Consequência a dizer em voz alta:** toda a tabela de juízes mede concordância com **F1
léxico**, nunca com um humano. O κ perto de zero significa sinais independentes, não juiz
incompetente — mas sem adjudicação humana não há como decidir qual dos dois sinais está
mais perto da verdade. Amostra de 24 itens exportada e pronta a rotular.

### A-bis. `generic_pt` está caracterizado, e a medição foi contra ele
Fechado desde a versão inicial deste documento, e vale a pena o registo porque o resultado
contraria o trabalho que o produziu: sobre 25 itens do FairytaleQA, o `generic_pt` é
estritamente mais estrito (McNemar p=0,0039, nove pares discordantes todos no mesmo sentido)
e **menos exato** que o `rag_pt` — 15/25 contra 20/25. O endurecimento concentra 11 itens em
`incompleto`, que não dispara anomalia, pelo que a rubrica mais estrita dá um detector mais
permissivo. Publicado com a recomendação de **não** o usar em narrativa. O caso de uso para
que foi feito — português não narrativo — continua sem corpus que o teste.

### B. Auto-consistência do juiz nunca foi medida
`scripts/judge_self_consistency.py` existe e nunca correu. Sem ela, o piso de ruído do
próprio juiz é desconhecido, o que limita a interpretação de qualquer efeito mínimo
detectável que o harness reporte.

### C. Cobertura empírica abaixo do que a tese exige
N=200 de um corpus de 1025; um corpus; uma família de gerador nos braços limpos. A
repetição do braço confundido ficou em 93 itens porque os créditos da API esgotaram-se, e
esses 93 são um **prefixo** da ordem do dataset, não uma amostra aleatória — no FairytaleQA,
onde os itens se agrupam por história, um prefixo cobre menos histórias.

### D. Português como corpus, não como problema resolvido
O normalizador léxico continua sem tratar a divergência pt-PT/pt-BR (`facto`/`fato`), pelo
que uma resposta correcta em pt-PT pontua abaixo contra ouro pt-BR. O prompt `generic_pt`
fecha essa lacuna **na camada do juiz** (regra dura, com exemplos negativos, verificada
contra 7 ataques), mas não na camada léxica. Não há corpus pt-PT nem ligação ao ecossistema
português (ASSIN-2, Napolab).

## Verdict

**O que se publica hoje é defensável; o que se reclama de âmbito não é ainda demonstrável.**

Cada número no README tem artefacto versionado e re-deriva-se sem chave de API. Cada
comparação inválida está nomeada como tal. `audit_run --strict` sai 0 e os quatro portões
estão verdes sobre 707 testes. Uma clonagem limpa verifica-se: 0 links quebrados, o gate de
avaliação corre, a tabela re-deriva-se da evidência.

O que **não** está demonstrado é a escala: um corpus, N=200, sem adjudicação humana e sem
auto-consistência do juiz. A tese — «caracterize o instrumento antes de confiar nos seus
números» — está demonstrada como método e apenas esboçada como resultado empírico.

Para um comité, a leitura honesta é: **infraestrutura de medição séria e auditável, com
cobertura empírica de projecto de portefólio.** Os três itens que mudam isso são
adjudicação humana (horas de uma pessoa), créditos de API (~$10 para N=1025 e para fechar o
braço parcial), e um segundo corpus. Nenhum deles é trabalho de arquitectura.
