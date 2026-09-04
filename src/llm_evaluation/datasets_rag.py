"""Construção de corpus RAG e injeção de falha de recuperação.

Chunks por item, com **proveniência**: saber se um chunk veio da passagem ouro
ou de um distractor é o que distingue recuperação de verificação. Derivar essa
resposta por comparação de texto a jusante falha em dois casos reais — corpus
sem distratores (tudo é ouro por construção) e passagem ouro curta contida num
distractor — e a proveniência é conhecida aqui, de graça.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_evaluation.types import EvalItem


@dataclass(frozen=True)
class CorpusChunk:
    """Chunk do corpus do item, com a origem que o produziu."""

    texto: str
    #: ``True`` quando o chunk saiu de ``item.rag_gold_chunk``.
    e_ouro: bool


def chunk_text(text: str, max_chars: int) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def build_corpus_for_item(item: EvalItem, chunk_max_chars: int) -> list[CorpusChunk]:
    """Corpus do item com proveniência preservada.

    A de-duplicação mantém a **primeira** ocorrência. Como o ouro é adicionado
    primeiro, um texto que apareça em ambas as origens fica marcado como ouro —
    a leitura conservadora: a passagem relevante está presente.
    """
    parts: list[CorpusChunk] = []
    if item.rag_gold_chunk:
        parts.extend(CorpusChunk(t, True) for t in chunk_text(item.rag_gold_chunk, chunk_max_chars))
    for d in item.rag_distractors:
        parts.extend(CorpusChunk(t, False) for t in chunk_text(d, chunk_max_chars))

    seen: set[str] = set()
    out: list[CorpusChunk] = []
    for p in parts:
        if p.texto not in seen:
            seen.add(p.texto)
            out.append(p)
    return out


def build_chunks_for_item(item: EvalItem, chunk_max_chars: int) -> list[str]:
    """Só os textos do corpus do item (compatibilidade com chamadores existentes)."""
    return [c.texto for c in build_corpus_for_item(item, chunk_max_chars)]


def corpus_tem_distratores(item: EvalItem, chunk_max_chars: int) -> bool:
    """``True`` se o corpus do item contém algo que não seja a passagem ouro.

    Sem distratores, «o ouro está no top-k?» tem uma única resposta possível e a
    métrica correspondente não mede recuperação. Quem publica precisa de saber.
    """
    return any(not c.e_ouro for c in build_corpus_for_item(item, chunk_max_chars))
