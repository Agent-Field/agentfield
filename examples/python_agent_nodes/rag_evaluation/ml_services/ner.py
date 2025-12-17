"""
Named Entity Recognition (NER) Service

Uses BERT-based NER for extracting entities from text.
Identifies: persons, organizations, locations, dates, numbers, etc.
"""

from typing import List, Dict, Optional
import re

# Lazy loading
_nlp = None


def _get_spacy():
    """Lazy load spaCy model."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            try:
                _nlp = spacy.load("en_core_web_sm")
            except OSError:
                # Model not installed, try downloading
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                _nlp = spacy.load("en_core_web_sm")
        except ImportError:
            raise ImportError(
                "spacy required. Install with: "
                "pip install spacy && python -m spacy download en_core_web_sm"
            )
    return _nlp


class NERService:
    """
    Named Entity Recognition service for extracting entities.

    Usage:
        service = NERService()
        entities = service.extract_entities("Apple Inc. is based in Cupertino.")
        # Returns: [{"text": "Apple Inc.", "label": "ORG", "start": 0, "end": 10}, ...]
    """

    def __init__(self):
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = _get_spacy()
        return self._nlp

    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract named entities from text.

        Args:
            text: Input text

        Returns:
            List of entity dicts with text, label, start, end
        """
        doc = self.nlp(text)

        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })

        return entities

    def extract_factual_claims(self, text: str) -> List[str]:
        """
        Extract sentences containing factual claims (entities, numbers, dates).

        Args:
            text: Input text

        Returns:
            List of sentences with factual content
        """
        doc = self.nlp(text)

        factual_sentences = []
        for sent in doc.sents:
            # Check if sentence contains entities or numbers
            sent_doc = self.nlp(sent.text)
            has_entities = len(sent_doc.ents) > 0
            has_numbers = bool(re.search(r'\d+', sent.text))

            if has_entities or has_numbers:
                factual_sentences.append(sent.text.strip())

        return factual_sentences

    def extract_numbers_with_context(self, text: str) -> List[Dict]:
        """
        Extract numbers with their surrounding context.

        Args:
            text: Input text

        Returns:
            List of dicts with number, context, and position
        """
        doc = self.nlp(text)

        numbers = []
        for token in doc:
            if token.like_num or token.pos_ == "NUM":
                # Get surrounding context (5 tokens before and after)
                start = max(0, token.i - 5)
                end = min(len(doc), token.i + 6)
                context = doc[start:end].text

                numbers.append({
                    "number": token.text,
                    "context": context,
                    "position": token.idx
                })

        return numbers

    def split_into_claims(self, text: str) -> List[str]:
        """
        Split text into individual claim-like statements.

        Uses sentence boundaries and conjunction splitting.

        Args:
            text: Input text

        Returns:
            List of claim strings
        """
        doc = self.nlp(text)

        claims = []
        for sent in doc.sents:
            sent_text = sent.text.strip()

            # Check for compound sentences with conjunctions
            if " and " in sent_text.lower() or " but " in sent_text.lower():
                # Try to split on conjunctions while preserving meaning
                parts = re.split(r'\s+(?:and|but)\s+', sent_text, flags=re.IGNORECASE)
                for part in parts:
                    part = part.strip()
                    if len(part) > 10:  # Skip very short fragments
                        claims.append(part)
            else:
                claims.append(sent_text)

        return claims

    def find_entity_in_text(
        self,
        entity: str,
        text: str,
        fuzzy: bool = False
    ) -> Optional[Dict]:
        """
        Find an entity mention in text.

        Args:
            entity: Entity to search for
            text: Text to search in
            fuzzy: Allow fuzzy matching

        Returns:
            Dict with match details or None
        """
        # Exact match first
        if entity.lower() in text.lower():
            idx = text.lower().find(entity.lower())
            return {
                "found": True,
                "match_type": "exact",
                "position": idx,
                "matched_text": text[idx:idx+len(entity)]
            }

        if fuzzy:
            # Try to find similar entities using spaCy
            doc = self.nlp(text)
            entity_doc = self.nlp(entity)

            for ent in doc.ents:
                # Check similarity
                if ent.text.lower() in entity.lower() or entity.lower() in ent.text.lower():
                    return {
                        "found": True,
                        "match_type": "fuzzy",
                        "position": ent.start_char,
                        "matched_text": ent.text
                    }

        return None


# Singleton instance
_service_instance: Optional[NERService] = None


def get_ner_service() -> NERService:
    """Get singleton NER service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = NERService()
    return _service_instance
