from llm_evaluation.generation import extract_answer_line


def test_extract_answer_line_prefers_marker() -> None:
    raw = """Passo 1: raciocinar
RESPOSTA: Paris is the capital."""
    assert extract_answer_line(raw) == "Paris is the capital."


def test_extract_answer_line_no_marker_returns_full() -> None:
    raw = "Não consigo responder só com o contexto dado."
    assert extract_answer_line(raw) == raw


def test_extract_answer_line_last_resposta_wins() -> None:
    raw = "RESPOSTA: antiga\nmais raciocínio\nRESPOSTA: só a final"
    assert extract_answer_line(raw) == "só a final"
