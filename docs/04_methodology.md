# Methodology & Assumptions

LoRA fine-tuning (rank 16, alpha 32) applied to the attention projection layers of Llama 3 8B. The model is constrained to an 8192 token context window. RLHF bounds are strictly applied via a constitutional AI filter to prevent hallucination and off-topic responses.

### Guard Rails

| Guard | Method |
| --- | --- |
| Hallucination prevention | Context-grounding score threshold (> 0.7) |
| Toxicity filtering | Perspective API pre/post-check |
| Scope enforcement | System prompt + fine-tuned refusal behaviour |
