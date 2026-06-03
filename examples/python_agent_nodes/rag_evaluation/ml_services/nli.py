"""
Natural Language Inference (NLI) Service

Uses DeBERTa-v3-base-mnli-fever-anli for entailment checking.
Determines if a hypothesis is entailed by, contradicts, or is neutral to a premise.
"""

from typing import List, Dict, Literal, Optional
import numpy as np

# Lazy loading
_model = None
_tokenizer = None


def _get_model():
    """Lazy load the NLI model."""
    global _model, _tokenizer
    if _model is None:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            _model = AutoModelForSequenceClassification.from_pretrained(model_name)
            _model.eval()

            # Move to CPU explicitly for consistency
            _model = _model.to("cpu")

        except ImportError:
            raise ImportError(
                "transformers and torch required. Install with: "
                "pip install transformers torch"
            )
    return _model, _tokenizer


class NLIService:
    """
    Natural Language Inference service for entailment checking.

    Checks if a hypothesis (claim) is:
    - entailed: Supported by the premise (context)
    - contradiction: Contradicted by the premise
    - neutral: Neither supported nor contradicted

    Usage:
        service = NLIService()
        result = service.check_entailment(
            premise="The sky is blue.",
            hypothesis="The sky has color."
        )
        # Returns: {"label": "entailment", "score": 0.95}
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None

    def _ensure_model(self):
        if self._model is None:
            self._model, self._tokenizer = _get_model()

    def check_entailment(
        self,
        premise: str,
        hypothesis: str
    ) -> Dict[str, float]:
        """
        Check entailment between premise and hypothesis.

        Args:
            premise: The context/source text
            hypothesis: The claim to verify

        Returns:
            Dict with 'label' and 'score' keys
        """
        self._ensure_model()
        import torch

        # Tokenize
        inputs = self._tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # Inference
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        # DeBERTa-MNLI labels: 0=contradiction, 1=neutral, 2=entailment
        labels = ["contradiction", "neutral", "entailment"]
        scores = probs.numpy()

        best_idx = int(np.argmax(scores))

        return {
            "label": labels[best_idx],
            "score": float(scores[best_idx]),
            "all_scores": {
                "contradiction": float(scores[0]),
                "neutral": float(scores[1]),
                "entailment": float(scores[2])
            }
        }

    def batch_check_entailment(
        self,
        premise: str,
        hypotheses: List[str]
    ) -> List[Dict[str, float]]:
        """
        Check entailment for multiple hypotheses against same premise.

        Args:
            premise: The context/source text
            hypotheses: List of claims to verify

        Returns:
            List of entailment results
        """
        return [
            self.check_entailment(premise, h)
            for h in hypotheses
        ]

    def verify_claim(
        self,
        context: str,
        claim: str,
        entailment_threshold: float = 0.7,
        contradiction_threshold: float = 0.7
    ) -> Dict:
        """
        Verify a claim against context with configurable thresholds.

        Args:
            context: Source context
            claim: Claim to verify
            entailment_threshold: Score needed to consider claim supported
            contradiction_threshold: Score needed to consider claim contradicted

        Returns:
            Dict with verification status and details
        """
        result = self.check_entailment(context, claim)

        if result["label"] == "entailment" and result["score"] >= entailment_threshold:
            status = "verified"
            confidence = result["score"]
        elif result["label"] == "contradiction" and result["score"] >= contradiction_threshold:
            status = "contradicted"
            confidence = result["score"]
        elif result["all_scores"]["entailment"] > result["all_scores"]["contradiction"]:
            status = "uncertain_leaning_supported"
            confidence = result["all_scores"]["entailment"]
        elif result["all_scores"]["contradiction"] > result["all_scores"]["entailment"]:
            status = "uncertain_leaning_contradicted"
            confidence = result["all_scores"]["contradiction"]
        else:
            status = "uncertain"
            confidence = result["score"]

        return {
            "status": status,
            "confidence": confidence,
            "needs_llm_review": status.startswith("uncertain"),
            "raw_result": result
        }


# Singleton instance
_service_instance: Optional[NLIService] = None


def get_nli_service() -> NLIService:
    """Get singleton NLI service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = NLIService()
    return _service_instance
