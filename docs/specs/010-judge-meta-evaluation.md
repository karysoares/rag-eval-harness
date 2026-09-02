# SPEC-010: Meta-avaliação do juiz

- **Estado:** implemented
- **Testes:** `tests/test_judge_meta.py`, `tests/test_review_fixes.py`, `tests/test_cli_paths.py::TestJudgeReport`, `tests/test_statistics_paired.py`
- **Relacionado:** [SPEC-003](003-judge.md) (juiz como camada; Fase 3 "calibração humana"), [SPEC-005](005-reporting.md) (`sumario_juiz` — saúde operacional), [SPEC-008](008-hitl.md) (referência humana)

## Objetivo

Caracterizar o juiz LLM **como instrumento de medição**, não como camada de verificação.

A distinção é a razão de ser desta spec. [SPEC-005](005-reporting.md) já produz `sumario_juiz`, que responde a *o juiz correu bem?* — taxa de fallback, retries, schema inválido, truncagem de contexto. É saúde operacional. Esta spec responde a uma pergunta independente e mais dura: *podemos confiar no que o juiz mede?* Um juiz pode ter 0% de fallback, 0% de schema inválido e mesmo assim ser inútil — por estar mal calibrado, por premiar verbosidade, ou por ser instável entre amostras.

**Premissa herdada de `docs/PREMISSAS.md`:** nenhuma destas medidas prova viés por si só. São sondas de inspeção, e os relatórios declaram-no nos campos `nota`.

## Entradas e saídas

### Entrada

`predictions.jsonl` de uma corrida já concluída. Nenhuma chamada à API — a análise é inteiramente offline, exceto a auto-consistência (ver abaixo).

Itens elegíveis: `sinais.juiz` presente **e** sem `fallback_heuristico`. Um veredito de fallback é produzido por `llm_client.heuristic_judge_json`, não pelo juiz; incluí-lo mediria a heurística. Coerente com [SPEC-004](004-aggregation.md), que já o exclui da agregação.

A calibração exclui adicionalmente os itens marcados com `confianca_ausente`: quando `predictions.jsonl` não traz `confianca`, `prediction_row_to_run_record` preenche `0.5` para satisfazer o tipo de `JudgeResult`. Esse valor não é uma medição, e incluí-lo criaria um pico artificial no bin central lido como miscalibração real. O total excluído sai em `calibracao.n_excluidos_sem_confianca`.

### Política herdada da corrida

O relatório **não** fixa a polaridade dos vereditos: lê-a de `summary.json` → `protocolo_ativo`, preferindo `judge_aggregation_verdicts` e caindo para `negative_judge_verdicts`. "Aprovou" é o complemento desse conjunto.

Fixar `sustentado` como único positivo divergiria da política real: com o default, `incompleto` é consultivo e não dispara `flag_anomalia`, mas seria contado como reprovação — inflacionando os falsos negativos do juiz e deprimindo κ e ECE em itens que a corrida nunca considerou problemáticos. Pela mesma razão, o limiar léxico vem de `protocolo_ativo.pattern_settings.f1_fraca_min`, e não do default global.

Sem protocolo registado, aplica-se `DEFAULT_NEGATIVE_VERDICTS`, alinhado com `_default_judge_aggregation_verdicts`. A polaridade efectiva e a sua origem saem em `polaridade_vereditos`.

### CLI

```bash
uv run llm-eval --judge-report outputs/run_<id>
uv run llm-eval --judge-report outputs/run_<id> --judge-samples <jsonl>
```

| Argumento | Efeito |
|-----------|--------|
| `--judge-report RUN_DIR` | Grava `RUN_DIR/judge_report.json` |
| `--judge-samples JSONL` | Acrescenta a secção `autoconsistencia` |

Saída de erro `2`: diretório inexistente, sem `predictions.jsonl`, ou JSONL de amostras inexistente.

### Saída (`judge_report.json`)

| Campo | Conteúdo |
|-------|----------|
| `schema_version` | `JUDGE_META_SCHEMA_VERSION` (`"2"`: acrescenta `polaridade_vereditos`) |
| `n_itens` / `n_itens_com_veredito_real` / `n_itens_com_fallback_heuristico` | Denominadores explícitos |
| `tipo_referencia` | Lido de `summary.json` → `tipo_referencia_ativo` |
| `polaridade_vereditos` | Vereditos negativos efectivos, a sua origem e `f1_fraca_min` |
| `distribuicao_vereditos` | Contagem por veredito canónico |
| `calibracao` | ECE, MCE, tabela de fiabilidade |
| `concordancia_com_referencia` | Confusão 2×2, exatidão + IC de Wilson, κ de Cohen |
| `vies_verbosidade` | Médias por grupo + correlação ponto-bisserial |
| `vies_posicao` | Taxa de aprovação por rank do chunk ouro, com ICs |
| `autoconsistencia` | Só com `--judge-samples` |

Qualquer secção é `null` quando não há dados suficientes; o relatório nunca inventa um denominador.

## Comportamento

### Definição de "aprovou" e de "acertou"

- **Aprovou** ⇔ veredito ∉ conjunto negativo da corrida (ver *Política herdada da corrida*).
- **Referência ok** ⇔ `referencia_humana_incorreta(r) is False`, ou, na ausência de rótulo humano, `referencia_incorreta(r, reference_type, f1_fraca_min=…) is False` com o limiar da corrida.
- **Acertou** ⇔ `aprovou == referencia_ok`.

A **referência humana tem precedência** sobre a automática quando ambas existem: o plano C ([SPEC-008](008-hitl.md)) é a referência mais forte disponível, e usá-la em conjunto com a automática no mesmo denominador misturaria dois planos métricos.

### Calibração

`statistics.expected_calibration_error` sobre pares `(confianca, acertou)`, em 10 bins por omissão. Reporta ECE (média dos desvios ponderada pela ocupação), MCE (pior bin) e a tabela de fiabilidade. Confiança `1.0` entra no último bin (fechado à direita).

**Leitura:** ECE alto com exatidão alta significa juiz útil com confiança pouco informativa — nesse regime, `confianca` não serve de limiar de triagem, o que invalida o item 19 da Fase 3 de [SPEC-003](003-judge.md) enquanto assim for.

### Viés de verbosidade

`statistics.point_biserial` entre "aprovou" e `len(answer)`. Fecha o item de dívida conhecido em `docs/techniques/llm-as-judge.md`.

**Leitura:** `|r| > 0.3` justifica inspeção manual de uma amostra. Não é prova: num corpus onde respostas completas são naturalmente mais longas, a correlação é esperada e legítima.

### Viés de posição

Taxa de aprovação agrupada por `meta.metricas_recuperacao.rank_chunk_ouro` (chave `ausente` quando não há ouro no top-k), cada grupo com IC de Wilson.

**Leitura:** comparar rank 1 com os restantes. Quedas grandes com ICs disjuntos indicam sensibilidade à **posição** do contexto, não à qualidade da resposta — um risco directo para a validade de [SPEC-002](002-grounding.md) e [SPEC-003](003-judge.md).

### Auto-consistência

Exige N vereditos repetidos sobre o mesmo par (pergunta, resposta, contexto) e portanto novas chamadas ao juiz:

```bash
uv run python scripts/judge_self_consistency.py outputs/run_<id> --amostras 5 --limite 40
```

O script reutiliza um único cliente (e o seu pool HTTP), grava linha a linha (`{id_item, vereditos}`) e descarta amostras que caíram em fallback. `judge_meta.self_consistency` agrega: taxa de itens unânimes, taxa média de veredito modal e κ de Fleiss.

**O prompt tem de ser o da corrida.** `judge_meta.replay_config_from_run` lê `judge_prompt_style` e `judge_max_context_chars` de `protocolo_ativo` e o script imprime a origem de cada um. Isto não é zelo: o default de `run_judge_for_retrieved` é `max_context_chars=None` (sem tecto) enquanto o pipeline corta a 12000 caracteres, por isso reamostrar com os defaults mediria a estabilidade de uma configuração que nunca correu. Em corridas anteriores ao registo da chave, assume-se o default do pipeline — nunca o mais permissivo. `--prompt-style` continua disponível como sobreposição explícita.

Temperatura por omissão `0.7`, **não** `0`: a 0 mede-se sobretudo o não-determinismo residual do fornecedor. Para "este juiz é estável na configuração em que o uso?", correr também com a temperatura real da configuração.

**Leitura:** a auto-consistência impõe um piso ao efeito mínimo detetável. Uma diferença entre corridas menor que o ruído de amostragem do próprio juiz não é interpretável, por mais baixo que seja o p-valor do McNemar.

## Fora de âmbito

- **Corrigir** viés detetado (reescrita de rubrica, ensemble, debiasing) — sondas medem, não intervêm.
- Segundo juiz / ensemble — Fase 5 de [SPEC-003](003-judge.md).
- Alterar `flag_anomalia`: o relatório é meta-avaliação e **não** entra na agregação, tal como `meta.flag_critica`.
- Meta-avaliação do respondedor ou do embedder.
- Testes de significância entre corridas — [SPEC-005](005-reporting.md) / `--compare-runs`.

## Critérios de aceitação

- [x] `--judge-report` corre sem `OPENAI_API_KEY`.
- [x] Vereditos de fallback heurístico excluídos de todas as secções.
- [x] Polaridade dos vereditos e limiar léxico lidos de `protocolo_ativo`, com a origem no relatório.
- [x] Confiança de preenchimento (`confianca_ausente`) fora do ECE, com contagem do que foi excluído.
- [x] Auto-consistência reproduz `judge_prompt_style` e `judge_max_context_chars` da corrida.
- [x] Referência humana tem precedência sobre a automática, com teste dedicado.
- [x] Cada secção devolve `null` sem dados, sem inventar denominador.
- [x] Corrida sem juiz produz relatório válido e aviso em `stderr`.
- [x] ECE, κ de Fleiss e ponto-bisserial cobertos por testes de casos limite e degenerados.
- [x] `scripts/judge_self_consistency.py` reutiliza um cliente e grava incrementalmente.
- [x] `uv run pytest -q` verde; `mypy --strict` limpo.
