# Overview & Strategy

LoRA-fine-tuned Llama 3 8B model optimised for answering questions about internal HR policy documents. Deployed as a RAG (Retrieval-Augmented Generation) pipeline with strict context grounding.

## Inputs

- `user_query`: Natural language question (string)
- `retrieved_contexts`: Top-K document chunks from the vector store (list)
- `conversation_history`: Prior turns for multi-turn coherence (list)

## Outputs

- `response`: Generated answer grounded in retrieved context (string)
- `source_citations`: Document references supporting the response (list)
- `confidence_score`: Self-assessed answer confidence [0, 1]
