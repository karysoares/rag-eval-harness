# Glossário

| Termo | Definição neste repositório |
|-------|------------------------------|
| **Anomalia** | Item onde `anomaly_flag=True` após agregação (ver ADR 0002). |
| **Baseline A** | Avaliação sem camadas de verificação (só geração + gold opcional). |
| **Baseline B** | Uma única camada de verificação (embedding-only ou judge-only). |
| **Chunk** | Unidade textual indexada para RAG (parágrafo ou passagem). |
| **Faithfulness (proxy)** | Grau em que a resposta é sustentada pelo contexto; aproximado por similaridade embedding máxima e juiz LLM. |
| **Falso alarme** | Gold correto mas `anomaly_flag=True`. |
| **Gold** | Rótulo derivado do adaptador (`reference_type`: listas correct/incorrect, F1 lexical, etc.). |
| **LLM-as-judge** | Modelo avaliador com rubrica e saída JSON estruturada. |
| **Recall de flag** | Fração de itens gold-incorretos com `anomaly_flag=True`. |
| **Refusal** | Resposta de recusa/heurística; tratada separadamente de erro factual. |
| **RAG** | Retrieval-Augmented Generation: recuperar chunks → gerar resposta condicionada. |
