# Referências bibliográficas e mapa técnica → literatura

Cada técnica usada neste repositório liga-se a uma pergunta de pesquisa e a uma limitação conhecida.

| Técnica no pipeline | Pergunta científica que endereça | Referência (link) | Limitação principal |
|---------------------|----------------------------------|-------------------|---------------------|
| Listas correct/incorrect (`answer_lists`) | Como medir respostas plausíveis mas factualmente incorretas? | Literatura de fact-checking / QA adversarial (ex. Lin et al., 2022) | Paráfrases correctas não listadas; recusas ambíguas. |
| Recuperação densa + chunking | Como obter evidência textual para RAG? | [Dense Passage Retrieval (Karpukhin et al.)](https://arxiv.org/abs/2004.04906) | *Domain gap*; mau chunking esconde a evidência mesmo com corpus correto. |
| Similaridade embedding (grounding) | A resposta está semanticamente próxima do contexto recuperado? | BEIR / bi-encoders (literatura de retrieval) | Embeddings confundem negação, paráfrase adversarial e respostas vagas “seguras”. |
| Verificação estilo NLI / entailment | A premissa sustenta a hipótese? | [FEVER (Thorne et al., 2018)](https://arxiv.org/abs/1806.05564) | NLI em frases curtas não cobre bem texto longo nem OOD. *(NLI opcional no código; ver `docs/metrics.md`.)* |
| Métricas RAG / faithfulness | Como quantificar aderência ao contexto? | [RAGAS (Es et al., 2023)](https://arxiv.org/abs/2312.10997) | Métricas com LLM-juiz herdam viés e variância do avaliador. |
| LLM-as-judge | Julgamento escalável com rubrica? | [Judging LLM-as-a-Judge (Zheng et al., 2023)](https://arxiv.org/abs/2306.05685) | Viés de posição, verbosidade, sycophancy; juiz alinhado ao gerador infla acordo. |
| Self-consistency | Reduzir erro por votação de amostras? | [Self-Consistency (Wang et al., ICLR 2023)](https://arxiv.org/abs/2203.11171) | Não corrige erro sistemático se todas as amostras erram igual. |
| Multi-agente / crítico | Decomorrer geração e verificação? | [Reflexion (Shinn et al.)](https://arxiv.org/abs/2303.11366) | Propagação de erro e custo de múltiplas chamadas. |

## Licenças de dados (resumo)

- **FairytaleQA pt-BR** ([`benjleite/FairytaleQA-translated-ptBR`](https://huggingface.co/datasets/benjleite/FairytaleQA-translated-ptBR)): Apache-2.0.

Não publique *outputs* brutos de APIs comerciais se as políticas do fornecedor o proibirem; ver `docs/SECURITY.md`.
