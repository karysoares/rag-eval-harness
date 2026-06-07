"""Harness de avaliação para sistemas de linguagem com recuperação aumentada.

Inclui adaptadores de dataset, pipeline de verificação multicamada, relatórios
agregados e componentes de análise de corridas. Ver README.md.
"""

from llm_evaluation.eval_items_load import load_eval_items

__all__: list[str] = ["load_eval_items"]
