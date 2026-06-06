from llm_evaluation.retrieval import HashEmbedder
from llm_evaluation.verification.embedding_verify import max_cosine_answer_to_chunks


def test_max_cosine_identical() -> None:
    emb = HashEmbedder()
    chunks = ["Paris is the capital of France."]
    answer = "The capital is Paris."
    sim = max_cosine_answer_to_chunks(answer, chunks, emb)
    assert -1.0 <= sim <= 1.0
