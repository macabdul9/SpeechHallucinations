"""Map raw LLM-judge responses to canonical error labels."""

from __future__ import annotations

import re

HALLUCINATION = "Hallucination"
NO_ERROR = "No Error"

COARSE_LABELS = (HALLUCINATION, "Non-Hallucination", NO_ERROR)
FINE_LABELS = (HALLUCINATION, "Language", "Oscillation", "Phonetic", NO_ERROR)
LABELS = {"coarse": COARSE_LABELS, "fine": FINE_LABELS}


def _clean(response: str) -> str:
    text = response.strip().casefold()
    text = re.sub(r"^(\*\*)?output:?(\*\*)?:?", "", text).strip()
    text = text.strip("*\"'`. \n\t")
    text = re.sub(r"[\s_-]+", " ", text)
    return re.sub(r"(?<!^no) error$", "", text)


def parse_label(response, granularity: str) -> str | None:
    """Return the canonical label, or None for API errors and off-format answers.

    Accepts the prompt's label strings ("Hallucination Error", "No Error", ...)
    with surrounding whitespace, quotes, markdown bold, a leading "Output:", or a
    trailing period. Anything else is unparseable rather than guessed.
    """
    if not isinstance(response, str):
        return None
    cleaned = _clean(response)
    for label in LABELS[granularity]:
        if cleaned == _clean(label):
            return label
    return None
