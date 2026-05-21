# Internal Copilot — LLM Prototype

Retrieval-Augmented Generation (RAG) prototype for internal HR policy
Q&A, using LoRA fine-tuning on Llama 3 8B with constitutional guardrails.

## Quick Start

```bash
pip install -r requirements.txt
python -m src.pipelines.train --data /path/to/hr_qa.jsonl
```

> **Note:** This is an alpha-stage prototype requiring GPU resources
> and the Llama 3 model weights. See the model card for deployment status.
