from llm_evaluation.llm_client import parse_json_object_lenient, parse_judge_json


def test_parse_json_object_lenient_prose_prefix() -> None:
    text = 'Segue o resultado:\n{"veredito": "sustentado", "x": 1}'
    d = parse_json_object_lenient(text)
    assert d["veredito"] == "sustentado"


def test_parse_judge_json_fenced() -> None:
    text = '```json\n{"veredito": "inseguro", "nota": "x"}\n```'
    d = parse_judge_json(text)
    assert d["veredito"] == "inseguro"
