from llm_evaluation.config import LexicalMetricsConfig
from llm_evaluation.lexical_metrics import compute_lexical_scores, pick_reference
from llm_evaluation.reporting import summarize
from llm_evaluation.types import JudgeResult, RunRecord, VerificationSignals


def test_pick_reference_first() -> None:
    ref, meta = pick_reference("x", ["a", "bb"], "primeiro")
    assert ref == "a"
    assert meta["indice_referencia"] == 0


def test_pick_reference_longest() -> None:
    ref, meta = pick_reference("x", ["a", "longer"], "mais_longo")
    assert ref == "longer"


def test_meteor_nltk_signature() -> None:
    """NLTK espera lista de referências tokenizadas, não tokens soltos como 1.º arg."""
    cfg = LexicalMetricsConfig(
        enabled=True,
        bleu=False,
        rouge_l=False,
        meteor=True,
        levenshtein=False,
        token_f1=False,
        reference_mode="primeiro",
    )
    out = compute_lexical_scores(
        "The cat sat on the mat.",
        ["The cat sat on the mat"],
        cfg,
    )
    assert "meteor" in out
    assert 0.0 <= float(out["meteor"]) <= 1.0


def test_compute_lexical_scores_smoke() -> None:
    cfg = LexicalMetricsConfig(
        enabled=True,
        bleu=True,
        rouge_l=True,
        meteor=False,
        levenshtein=True,
        token_f1=True,
        reference_mode="primeiro",
    )
    out = compute_lexical_scores(
        "A capital do Brasil é Brasília.",
        ["Brasília"],
        cfg,
    )
    assert "bleu" in out
    assert "f1_token" in out
    assert out["rouge_l_f"] is not None
    assert out["similaridade_levenshtein"] is not None
    assert 0.0 <= float(out["bleu"]) <= 1.0


def test_exact_match_requires_normalized_equality_not_substring() -> None:
    cfg = LexicalMetricsConfig(
        enabled=True,
        bleu=False,
        rouge_l=False,
        meteor=False,
        levenshtein=False,
        token_f1=False,
        reference_mode="primeiro",
    )
    out = compute_lexical_scores("Brasília é a capital do Brasil", ["Brasília"], cfg)
    assert out["exact_match"] is False
    assert out["exact_match_normalizado"] is False


def test_compute_lexical_scores_empty_refs() -> None:
    cfg = LexicalMetricsConfig(
        enabled=True,
        bleu=True,
        rouge_l=True,
        meteor=False,
        levenshtein=True,
        token_f1=False,
        reference_mode="primeiro",
    )
    out = compute_lexical_scores("resposta", [], cfg)
    assert out.get("note") == "sem_referencia"


def test_summarize_lexical_summary() -> None:
    r = RunRecord(
        item_id="1",
        question="q",
        answer="a",
        gold_correct=True,
        anomaly_flag=False,
        signals=VerificationSignals(
            gold_correct=True,
            gold_incorrect=False,
            is_refusal=False,
            embedding_max_cosine=0.9,
            embedding_low_support=False,
            judge=JudgeResult(veredito="sustentado", motivo_breve="", confianca=0.9),
            judge_negative=False,
        ),
        retrieved=[],
        baseline_profile="hibrido",
        meta={
            "metricas_lexicas": {
                "bleu": 0.4,
                "rouge_l_f": 0.5,
                "f1_token": 0.6,
                "texto_referencia": "ref",
            },
        },
    )
    s = summarize([r], reference_type="lexical")
    assert "sumario_lexical" in s
    assert s["sumario_lexical"]["media_bleu"] == 0.4


def test_summarize_uses_protocol_pattern_f1_threshold() -> None:
    r = RunRecord(
        item_id="1",
        question="q",
        answer="a",
        gold_correct=True,
        anomaly_flag=False,
        signals=VerificationSignals(
            gold_correct=True,
            gold_incorrect=False,
            is_refusal=False,
            embedding_max_cosine=0.9,
            embedding_low_support=False,
            judge=JudgeResult(veredito="sustentado", motivo_breve="", confianca=0.9),
            judge_negative=False,
        ),
        retrieved=[],
        baseline_profile="hibrido",
        meta={
            "metricas_lexicas": {
                "f1_token": 0.27,
                "texto_referencia": "ref",
            },
            "diagnostico": {"padroes": ["referencia_fraca"], "padrao_primario": "referencia_fraca"},
        },
    )
    protocol = {
        "verify_gold": False,
        "verify_embedding": False,
        "verify_judge": False,
        "aggregation_policy": "qualquer_critico",
        "negative_judge_verdicts": ["nao_sustentado", "contradicacao", "incompleto"],
        "judge_aggregation_verdicts": ["nao_sustentado", "contradicacao"],
        "pattern_settings": {"f1_fraca_min": 0.25},
    }
    s = summarize([r], reference_type="lexical", protocol=protocol)
    assert s["n_gold_corretos"] == 1
    assert s["n_gold_incorretos"] == 0
    camada = s["analise_camadas"]["por_camada_vs_referencia"]["sinal_ouro"]
    assert camada["vn"] == 1
