"""Carrega configuração YAML. Definições de métricas em `docs/metrics.md`."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from llm_evaluation.operational import OperationalThresholds

AggregationPolicy = Literal["qualquer_critico", "todos_criticos", "embedding_e_juiz"]
JudgePromptStyle = Literal["pt", "rag_pt"]
Orchestration = Literal["unico", "multiplo"]
DatasetMode = Literal["amostra_local", "hub"]
EmbedBackend = Literal["hash", "sentence_transformers"]
BaselineProfile = Literal["nenhum", "so_embeddings", "so_juiz", "hibrido"]
LexicalReferenceMode = Literal["primeiro", "mais_longo", "max_rouge_l"]
ReferenceType = Literal["none", "lexical", "answer_lists"]
PromptStyle = Literal["rag_pt"]


@dataclass
class DatasetConfig:
    name: str
    subset: str
    split: str
    limit: int
    mode: DatasetMode
    # Repositório Hugging Face (ex.: FairytaleQA pt-BR).
    hf_repo: str | None
    hf_subset: str | None
    question_column: str
    answer_column: str
    context_column: str | None
    incorrect_column: str | None
    id_column: str | None
    reference_type: ReferenceType
    #: Embaralhar linhas do Hub antes do ``limit`` (desligar em corrida 100% reprodutível).
    shuffle: bool = True


@dataclass
class RagConfig:
    enabled: bool
    top_k: int
    inject_retrieval_failure: bool
    chunk_max_chars: int
    #: Coseno pergunta↔melhor chunk (como ``RetrievedChunk.score``). ``None`` = gate off.
    #: Com backend ``hash`` os scores não são semânticos — use ``null``.
    min_retrieval_score: float | None = None


@dataclass
class EmbeddingsConfig:
    backend: EmbedBackend
    model_name: str


@dataclass
class VerificationConfig:
    verify_gold: bool
    verify_embedding: bool
    verify_judge: bool
    embedding_min_cosine: float
    #: Vereditos negativos para diagnóstico / padrões (pode incluir ``incompleto``).
    negative_judge_verdicts: list[str]
    #: Vereditos que disparam ``flag_anomalia`` via juiz (por omissão exclui ``incompleto``).
    judge_aggregation_verdicts: list[str]
    embedding_use_gold_chunk: bool = True
    judge_prompt_style: JudgePromptStyle = "pt"
    #: Persistir ``cadeia_de_pensamento`` no JSONL (debug); por omissão omitido.
    judge_return_chain_of_thought: bool = False
    #: Limite de caracteres do contexto no prompt do juiz; ``None`` = sem tecto extra.
    judge_max_context_chars: int | None = 12_000
    #: Tentativas de parse/schema após resposta do LLM (além do retry HTTP).
    judge_max_parse_retries: int = 2
    #: Gate de custo/latência: se embedding>=limiar e sem recusa, pode saltar juiz.
    judge_gate_embedding_max_cosine: float | None = None
    #: Quando ativo, só permite gate se a recuperação estiver forte (ouro no top-k + score).
    judge_gate_requires_strong_context: bool = False
    #: Limiar do melhor chunk para considerar contexto forte no gate.
    judge_gate_min_retrieval_score: float = 0.5
    #: Trata veredito ``incompleto`` como negativo quando contexto recuperado é forte.
    judge_incompleto_contexto_forte_negativo: bool = False
    #: Limiar mínimo de score do melhor chunk para activar a regra acima.
    judge_incompleto_contexto_forte_min_score: float = 0.5


@dataclass
class GenerationConfig:
    temperature: float
    max_tokens: int
    #: Resposta condicionada ao contexto recuperado (RAG).
    prompt_style: PromptStyle = "rag_pt"
    #: Com ``rag.min_retrieval_score`` definido: não chama respondedor se recuperação fraca.
    skip_llm_on_weak_retrieval: bool = False
    #: Mensagem fixa PT (vazia = texto por omissão da biblioteca).
    weak_retrieval_message: str = ""
    #: Em contexto forte, tenta reparar respostas de recusa genérica.
    anti_refusal_repair: bool = True
    #: Score mínimo do melhor chunk para acionar reparo de recusa.
    anti_refusal_min_retrieval_score: float = 0.5
    #: Quantas novas tentativas de geração após recusa.
    anti_refusal_max_attempts: int = 1


@dataclass
class LlmConfig:
    timeout_seconds: float


@dataclass
class AggregationConfig:
    policy: AggregationPolicy


@dataclass
class BaselinesConfig:
    profile: BaselineProfile


@dataclass
class LexicalMetricsConfig:
    """Métricas de sobreposição com respostas de referência (dataset)."""

    enabled: bool
    bleu: bool
    rouge_l: bool
    meteor: bool
    levenshtein: bool
    token_f1: bool
    reference_mode: LexicalReferenceMode


@dataclass
class PatternsConfig:
    """Overrides opcionais por ID de padrão (SPEC-007). Chaves vazias = defaults do registry."""

    overrides: dict[str, dict[str, Any]]


@dataclass
class AppConfig:
    seed: int
    orchestration: Orchestration
    output_dir: str
    dataset: DatasetConfig
    rag: RagConfig
    embeddings: EmbeddingsConfig
    verification: VerificationConfig
    generation: GenerationConfig
    llm: LlmConfig
    aggregation: AggregationConfig
    baselines: BaselinesConfig
    lexical_metrics: LexicalMetricsConfig
    patterns: PatternsConfig
    operational: OperationalThresholds


def _req(d: dict[str, Any], key: str) -> Any:
    if key not in d:
        msg = f"Chave de configuração em falta: {key}"
        raise KeyError(msg)
    return d[key]


def _optional_float(raw: object) -> float | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, bool):
        msg = f"Esperado número, recebido bool: {raw!r}"
        raise TypeError(msg)
    if isinstance(raw, int | float):
        return float(raw)
    if isinstance(raw, str):
        return float(raw.strip())
    msg = f"Tipo inválido para número opcional: {type(raw).__name__}"
    raise TypeError(msg)


def _expect_literal(name: str, value: str, allowed: tuple[str, ...]) -> str:
    if value not in allowed:
        msg = f"{name} inválido: {value!r}; esperado um de {list(allowed)!r}"
        raise ValueError(msg)
    return value


def _norm_orquestracao(s: str) -> str:
    return {"single": "unico", "multi": "multiplo"}.get(s, s)


def _norm_modo_dataset(s: str) -> str:
    return {
        "demo": "amostra_local",
        "demonstracao": "amostra_local",
        "hf": "hub",
    }.get(s, s)


def _norm_perfil_baseline(s: str) -> str:
    return {
        "none": "nenhum",
        "embedding_only": "so_embeddings",
        "judge_only": "so_juiz",
        "hybrid": "hibrido",
    }.get(s, s)


def _norm_agregacao(s: str) -> str:
    return {
        "any_critical": "qualquer_critico",
        "all_critical": "todos_criticos",
        "embedding_and_judge": "embedding_e_juiz",
    }.get(s, s)


def _norm_modo_referencia_lexical(s: str) -> str:
    return {"first": "primeiro", "longest": "mais_longo"}.get(s, s)


def _norm_reference_type(s: str) -> str:
    return {
        "answer_list": "answer_lists",
        "lexical_metrics": "lexical",
    }.get(s, s)


def _norm_prompt_style(s: str) -> str:
    low = s.strip().lower()
    if low in ("rag_pt", "pt_rag", "rag"):
        return "rag_pt"
    return low


def _norm_judge_prompt_style(s: str) -> str:
    low = s.strip().lower()
    if low in ("rag_pt", "pt_rag", "rag", "rag_en", "en"):
        return "rag_pt"
    if low == "pt":
        return "pt"
    return low


def _apply_reference_defaults(
    reference_type: ReferenceType,
    vf_raw: dict[str, Any],
    lx_enabled: bool,
) -> VerificationConfig:
    """Preenche verify_gold quando omitido, conforme tipo de referência."""
    v = vf_raw
    if "verify_gold" in v:
        verify_gold = bool(v["verify_gold"])
    elif reference_type == "answer_lists":
        verify_gold = True
    else:
        verify_gold = False

    j_style_raw = str(v.get("judge_prompt_style", v.get("estilo_prompt_juiz", "rag_pt")))
    j_style = _expect_literal(
        "verification.judge_prompt_style",
        _norm_judge_prompt_style(j_style_raw),
        ("pt", "rag_pt"),
    )

    j_cot = bool(
        v.get(
            "judge_return_chain_of_thought",
            v.get("devolver_cadeia_pensamento_juiz", False),
        ),
    )
    j_ctx_raw = v.get("judge_max_context_chars", v.get("max_chars_contexto_juiz", 12_000))
    j_ctx: int | None = None if j_ctx_raw is None or j_ctx_raw == "" else int(j_ctx_raw)
    j_retries = int(
        v.get("judge_max_parse_retries", v.get("max_retries_parse_juiz", 2)),
    )
    judge_gate = _optional_float(
        v.get("judge_gate_embedding_max_cosine", v.get("juiz_gate_embedding_max_coseno")),
    )
    judge_gate_requires_strong_context = bool(
        v.get(
            "judge_gate_requires_strong_context",
            v.get("juiz_gate_requer_contexto_forte", False),
        ),
    )
    judge_gate_min_retrieval_score = float(
        v.get(
            "judge_gate_min_retrieval_score",
            v.get("juiz_gate_min_score_recuperacao", 0.5),
        ),
    )
    judge_incomp_neg = bool(
        v.get(
            "judge_incompleto_contexto_forte_negativo",
            v.get("juiz_incompleto_contexto_forte_negativo", False),
        ),
    )
    judge_incomp_min = float(
        v.get(
            "judge_incompleto_contexto_forte_min_score",
            v.get("juiz_incompleto_contexto_forte_min_score", 0.5),
        ),
    )

    neg = [str(x) for x in _req(v, "negative_judge_verdicts")]
    agg_raw = v.get("judge_aggregation_verdicts", v.get("vereditos_juiz_agregacao"))
    if agg_raw is None:
        advisory = {"incompleto"}
        agg = [x for x in neg if x not in advisory]
    else:
        agg = [str(x) for x in agg_raw]

    return VerificationConfig(
        verify_gold=verify_gold,
        verify_embedding=bool(_req(v, "verify_embedding")),
        verify_judge=bool(_req(v, "verify_judge")),
        embedding_min_cosine=float(_req(v, "embedding_min_cosine")),
        negative_judge_verdicts=neg,
        judge_aggregation_verdicts=agg,
        embedding_use_gold_chunk=bool(v.get("embedding_use_gold_chunk", True)),
        judge_prompt_style=cast("JudgePromptStyle", j_style),
        judge_return_chain_of_thought=j_cot,
        judge_max_context_chars=j_ctx,
        judge_max_parse_retries=j_retries,
        judge_gate_embedding_max_cosine=judge_gate,
        judge_gate_requires_strong_context=judge_gate_requires_strong_context,
        judge_gate_min_retrieval_score=judge_gate_min_retrieval_score,
        judge_incompleto_contexto_forte_negativo=judge_incomp_neg,
        judge_incompleto_contexto_forte_min_score=judge_incomp_min,
    )


def _load_operational_config(raw: dict[str, Any]) -> OperationalThresholds:
    from llm_evaluation.operational import thresholds_from_mapping

    return thresholds_from_mapping(raw)


def _load_patterns_config(raw: dict[str, Any]) -> PatternsConfig:
    pt = raw.get("patterns")
    if not isinstance(pt, dict):
        pt = raw.get("padroes")
    if not isinstance(pt, dict):
        return PatternsConfig(overrides={})
    overrides: dict[str, dict[str, Any]] = {}
    for key, val in pt.items():
        if key in ("catalog_version", "versao_catalogo"):
            continue
        if isinstance(val, dict):
            overrides[str(key)] = dict(val)
    return PatternsConfig(overrides=overrides)


def _aliases_raiz_yaml(raw: dict[str, Any]) -> None:
    """Aceita sinónimos em português nas chaves de topo (retrocompatível com inglês)."""
    if "orchestration" not in raw and "orquestracao" in raw:
        raw["orchestration"] = raw["orquestracao"]
    if "output_dir" not in raw and "diretorio_saida" in raw:
        raw["output_dir"] = raw["diretorio_saida"]


def load_config(path: Path) -> AppConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = "A raiz da configuração tem de ser um mapa (dicionário)"
        raise TypeError(msg)

    _aliases_raiz_yaml(raw)

    ds = _req(raw, "dataset")
    rg = _req(raw, "rag")
    em = _req(raw, "embeddings")
    vf = _req(raw, "verification")
    gn = _req(raw, "generation")
    ll = _req(raw, "llm")
    ag = _req(raw, "aggregation")
    bl = _req(raw, "baselines")

    lim_raw = ds.get("limit")
    dataset_limit = 0 if lim_raw is None else int(lim_raw)

    hf_repo_raw = ds.get("hf_repo")
    hf_repo = None if hf_repo_raw in (None, "") else str(hf_repo_raw)
    hf_subset_raw = ds.get("hf_subset")
    hf_subset = None if hf_subset_raw in (None, "") else str(hf_subset_raw)
    ctx_col = ds.get("context_column")
    context_column = None if ctx_col in (None, "") else str(ctx_col)
    inc_col = ds.get("incorrect_column")
    incorrect_column = None if inc_col in (None, "") else str(inc_col)
    id_col = ds.get("id_column")
    id_column = None if id_col in (None, "") else str(id_col)

    lx = raw.get("metricas_lexicas") if isinstance(raw.get("metricas_lexicas"), dict) else {}
    if not lx and isinstance(raw.get("lexical_metrics"), dict):
        lx = raw["lexical_metrics"]
    lx_d = cast(dict[str, Any], lx)
    ref_mode = _expect_literal(
        "metricas_lexicas.modo_referencia",
        _norm_modo_referencia_lexical(
            str(lx_d.get("modo_referencia", lx_d.get("reference_mode", "max_rouge_l"))),
        ),
        ("primeiro", "mais_longo", "max_rouge_l"),
    )

    ref_type_raw = str(
        ds.get("reference_type", ds.get("tipo_referencia", "lexical")),
    )
    ref_type = cast(
        "ReferenceType",
        _expect_literal(
            "dataset.reference_type",
            _norm_reference_type(ref_type_raw),
            ("none", "lexical", "answer_lists"),
        ),
    )

    lx_enabled = bool(lx_d.get("habilitado", lx_d.get("enabled", False)))
    if ref_type == "lexical" and "habilitado" not in lx_d and "enabled" not in lx_d:
        lx_enabled = True
    token_f1_default = ref_type == "lexical"
    token_f1 = bool(lx_d.get("f1_token", lx_d.get("token_f1", token_f1_default)))

    verification = _apply_reference_defaults(ref_type, vf, lx_enabled)
    patterns = _load_patterns_config(raw)
    operational = _load_operational_config(raw)

    orchestration = cast(
        "Orchestration",
        _expect_literal(
            "orchestration",
            _norm_orquestracao(str(_req(raw, "orchestration"))),
            ("unico", "multiplo"),
        ),
    )
    dataset_mode = cast(
        "DatasetMode",
        _expect_literal(
            "dataset.mode",
            _norm_modo_dataset(str(_req(ds, "mode"))),
            ("amostra_local", "hub"),
        ),
    )
    embeddings_backend = cast(
        "EmbedBackend",
        _expect_literal(
            "embeddings.backend",
            str(_req(em, "backend")),
            ("hash", "sentence_transformers"),
        ),
    )
    generation_prompt_style = cast(
        "PromptStyle",
        _expect_literal(
            "generation.prompt_style",
            _norm_prompt_style(str(gn.get("prompt_style", gn.get("estilo_prompt", "rag_pt")))),
            ("rag_pt",),
        ),
    )
    aggregation_policy = cast(
        "AggregationPolicy",
        _expect_literal(
            "aggregation.policy",
            _norm_agregacao(str(_req(ag, "policy"))),
            ("qualquer_critico", "todos_criticos", "embedding_e_juiz"),
        ),
    )
    baseline_profile = cast(
        "BaselineProfile",
        _expect_literal(
            "baselines.profile",
            _norm_perfil_baseline(str(_req(bl, "profile"))),
            ("nenhum", "so_embeddings", "so_juiz", "hibrido"),
        ),
    )

    return AppConfig(
        seed=int(_req(raw, "seed")),
        orchestration=orchestration,
        output_dir=str(_req(raw, "output_dir")),
        dataset=DatasetConfig(
            name=str(_req(ds, "name")),
            subset=str(_req(ds, "subset")),
            split=str(_req(ds, "split")),
            limit=dataset_limit,
            mode=dataset_mode,
            hf_repo=hf_repo,
            hf_subset=hf_subset,
            question_column=str(ds.get("question_column", "question")),
            answer_column=str(ds.get("answer_column", "answer")),
            context_column=context_column,
            incorrect_column=incorrect_column,
            id_column=id_column,
            reference_type=ref_type,
            shuffle=bool(ds.get("shuffle", ds.get("embaralhar", True))),
        ),
        rag=RagConfig(
            enabled=bool(_req(rg, "enabled")),
            top_k=int(_req(rg, "top_k")),
            inject_retrieval_failure=bool(_req(rg, "inject_retrieval_failure")),
            chunk_max_chars=int(_req(rg, "chunk_max_chars")),
            min_retrieval_score=_optional_float(
                rg["min_retrieval_score"]
                if "min_retrieval_score" in rg
                else rg.get("min_score_recuperacao"),
            ),
        ),
        embeddings=EmbeddingsConfig(
            backend=embeddings_backend,
            model_name=str(_req(em, "model_name")),
        ),
        verification=verification,
        generation=GenerationConfig(
            temperature=float(_req(gn, "temperature")),
            max_tokens=int(_req(gn, "max_tokens")),
            prompt_style=generation_prompt_style,
            skip_llm_on_weak_retrieval=bool(
                gn.get(
                    "skip_llm_on_weak_retrieval",
                    gn.get("omitir_llm_se_recuperacao_fraca", False),
                ),
            ),
            weak_retrieval_message=str(
                gn.get("weak_retrieval_message", gn.get("mensagem_recuperacao_fraca", "")) or "",
            ),
            anti_refusal_repair=bool(
                gn.get(
                    "anti_refusal_repair",
                    gn.get("reparar_recusa_generica", True),
                ),
            ),
            anti_refusal_min_retrieval_score=float(
                gn.get(
                    "anti_refusal_min_retrieval_score",
                    gn.get("min_score_recuperacao_reparo_recusa", 0.5),
                ),
            ),
            anti_refusal_max_attempts=int(
                gn.get(
                    "anti_refusal_max_attempts",
                    gn.get("max_tentativas_reparo_recusa", 1),
                ),
            ),
        ),
        llm=LlmConfig(timeout_seconds=float(_req(ll, "timeout_seconds"))),
        aggregation=AggregationConfig(policy=aggregation_policy),
        baselines=BaselinesConfig(
            profile=baseline_profile,
        ),
        lexical_metrics=LexicalMetricsConfig(
            enabled=lx_enabled,
            bleu=bool(lx_d.get("bleu", True)),
            rouge_l=bool(lx_d.get("rouge_l", True)),
            meteor=bool(lx_d.get("meteor", True)),
            levenshtein=bool(lx_d.get("levenshtein", True)),
            token_f1=token_f1,
            reference_mode=cast("LexicalReferenceMode", ref_mode),
        ),
        patterns=patterns,
        operational=operational,
    )


def apply_baseline_profile(cfg: AppConfig, profile: str) -> AppConfig:
    """Devolve uma config atualizada superficialmente para corridas de baseline."""
    p = cast("BaselineProfile", profile)
    v = cfg.verification
    if p == "nenhum":
        new_v = replace(
            v,
            verify_gold=False,
            verify_embedding=False,
            verify_judge=False,
            negative_judge_verdicts=list(v.negative_judge_verdicts),
            judge_aggregation_verdicts=list(v.judge_aggregation_verdicts),
        )
    elif p == "so_embeddings":
        new_v = replace(
            v,
            verify_gold=False,
            verify_embedding=True,
            verify_judge=False,
            negative_judge_verdicts=list(v.negative_judge_verdicts),
            judge_aggregation_verdicts=list(v.judge_aggregation_verdicts),
        )
    elif p == "so_juiz":
        new_v = replace(
            v,
            verify_gold=False,
            verify_embedding=False,
            verify_judge=True,
            negative_judge_verdicts=list(v.negative_judge_verdicts),
            judge_aggregation_verdicts=list(v.judge_aggregation_verdicts),
        )
    else:
        new_v = replace(
            v,
            verify_gold=False,
            verify_embedding=True,
            verify_judge=True,
            negative_judge_verdicts=list(v.negative_judge_verdicts),
            judge_aggregation_verdicts=list(v.judge_aggregation_verdicts),
        )

    new_bl = BaselinesConfig(profile=p)
    return AppConfig(
        seed=cfg.seed,
        orchestration=cfg.orchestration,
        output_dir=cfg.output_dir,
        dataset=cfg.dataset,
        rag=cfg.rag,
        embeddings=cfg.embeddings,
        verification=new_v,
        generation=cfg.generation,
        llm=cfg.llm,
        aggregation=cfg.aggregation,
        baselines=new_bl,
        lexical_metrics=cfg.lexical_metrics,
        patterns=cfg.patterns,
        operational=cfg.operational,
    )
