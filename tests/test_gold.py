from llm_evaluation.verification.gold import (
    gold_correct,
    gold_incorrect,
    is_refusal,
    normalize_answer,
)


def test_normalize_answer() -> None:
    assert normalize_answer("  Hello, World!  ") == "hello, world"


def test_gold_correct_simple() -> None:
    assert gold_correct("Nothing happens.", ["Nothing happens"], []) is True


def test_gold_incorrect_myth() -> None:
    ans = "A watermelon will grow in your stomach."
    correct = ["They pass through your digestive system"]
    incorrect = ["A watermelon grows in your belly"]
    assert gold_incorrect(ans, correct, incorrect) is True


def test_refusal() -> None:
    assert is_refusal("I cannot answer from the context.") is True
    assert is_refusal("Não consigo responder só com esse contexto.") is True
    assert is_refusal("Não há informações suficientes.") is True
