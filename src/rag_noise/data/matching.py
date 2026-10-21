"""Answer-string checks used only for quality control.

The stored passage is never rewritten. Normalization follows the SQuAD-style
check in NoiserBench ``utils/normalize_answer.py`` (lowercase, punctuation
removed, articles removed) and is applied to a copy.
"""

from __future__ import annotations

import re
import unicodedata

_ARTICLE = re.compile(r"\b(a|an|the)\b")
_NON_ALNUM = re.compile(r"[^a-z0-9\s]+")
_SPACE = re.compile(r"\s+")


def normalize_for_match(text: str) -> str:
    lowered = text.lower()
    no_punct = _NON_ALNUM.sub(" ", lowered)
    no_articles = _ARTICLE.sub(" ", no_punct)
    return _SPACE.sub(" ", no_articles).strip()


def normalize_question(text: str) -> str:
    return _SPACE.sub(" ", text.lower()).strip()


def loose_question_key(text: str) -> str:
    """Letters and digits only, for punctuation-level near-duplicate checks."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _answer_phrases(answer: object) -> list[str]:
    if isinstance(answer, bool):
        return ["yes", "true"] if answer else ["no", "false"]
    phrase = normalize_for_match(str(answer))
    return [phrase] if phrase else []


def contains_answer(text: str, answers: object) -> bool:
    """Return True when any gold answer occurs in ``text`` after normalization.

    Booleans map to the tokens yes/true or no/false. Those tokens must match
    a whole word so that ``no`` does not match inside ``known``.
    """
    if answers is None:
        return False
    items = answers if isinstance(answers, list) else [answers]
    normalized_text = normalize_for_match(text)
    if not normalized_text:
        return False
    padded = f" {normalized_text} "
    for answer in items:
        for phrase in _answer_phrases(answer):
            if isinstance(answer, bool):
                if f" {phrase} " in padded:
                    return True
            elif phrase in normalized_text:
                return True
    return False


def nfc_or_none(text: str, preserve_exact: bool) -> str | None:
    """Return NFC text, or None when NFC would change a noise class we must preserve."""
    converted = unicodedata.normalize("NFC", text)
    if preserve_exact and converted != text:
        return None
    return converted


def serialize_answers(answers: object) -> list[str]:
    if answers is None:
        return []
    items = answers if isinstance(answers, list) else [answers]
    serialized: list[str] = []
    for item in items:
        if isinstance(item, bool):
            serialized.append("true" if item else "false")
        else:
            serialized.append(str(item))
    return serialized
