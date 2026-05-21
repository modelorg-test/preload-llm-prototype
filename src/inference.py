"""Inference pipeline for the Internal Copilot.

Handles retrieval, context injection, generation with constitutional
guardrails, and confidence scoring for the Slack bot integration.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from rag_pipeline import GROUNDING_THRESHOLD, PolicyVectorIndex

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.70
FALLBACK_RESPONSE = (
    "I'm not confident enough to answer this reliably. "
    "Please contact the HR team directly at hr@modelorg.com."
)
SYSTEM_PROMPT = (
    "You are an internal HR policy assistant. Answer ONLY based on the "
    "provided context. If the answer is not in the context, say you don't know."
)


@dataclass
class CopilotResponse:
    """Response from the internal copilot."""

    response: str
    source_citations: list[str]
    confidence_score: float
    latency_ms: float
    is_fallback: bool


class CopilotInference:
    """End-to-end inference engine for the Internal Copilot."""

    def __init__(
        self,
        base_model_id: str,
        adapter_path: str,
        index_path: str,
        metadata_path: str,
    ) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self._model = PeftModel.from_pretrained(base, adapter_path)
        self._model.eval()
        self._index = PolicyVectorIndex(index_path, metadata_path)
        self._pipe = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
        )
        logger.info("Copilot inference engine ready")

    def embed(self, text: str) -> np.ndarray:
        """Embed a query string using the tokenizer's mean-pooling."""
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self._model.base_model(**inputs, output_hidden_states=True)
        # Mean-pool the last hidden state as a simple embedding
        return outputs.hidden_states[-1].mean(dim=1).squeeze().float().numpy()

    def answer(
        self,
        user_query: str,
        conversation_history: list[dict] | None = None,
    ) -> CopilotResponse:
        """Generate a grounded answer to a user query.

        Args:
            user_query: Natural language question from the user.
            conversation_history: Prior conversation turns for multi-turn coherence.

        Returns:
            CopilotResponse with generated answer and metadata.
        """
        t0 = time.perf_counter()
        query_embedding = self.embed(user_query)
        rag_result = self._index.retrieve(query_embedding)

        if not rag_result.is_grounded:
            return CopilotResponse(
                response=FALLBACK_RESPONSE,
                source_citations=[],
                confidence_score=rag_result.grounding_score,
                latency_ms=(time.perf_counter() - t0) * 1000,
                is_fallback=True,
            )

        context = "\n\n".join(c.text for c in rag_result.chunks)
        prompt = (
            f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {user_query}\nAnswer:"
        )
        output = self._pipe(prompt)[0]["generated_text"]
        answer = output.split("Answer:", 1)[-1].strip()

        return CopilotResponse(
            response=answer,
            source_citations=[c.source for c in rag_result.chunks],
            confidence_score=rag_result.grounding_score,
            latency_ms=(time.perf_counter() - t0) * 1000,
            is_fallback=False,
        )
