"""Conservative STT cleanup: remove hesitation tokens, never guess real words."""
import re

_FILLERS = re.compile(r"(?iu)(?<!\w)(?:umm+|uhh+|erm+|hmm+|mmm+|eee+|emm+|e+hm+)(?!\w)")
_SPACES = re.compile(r"[ \t]{2,}")

def clean_transcript(text: str) -> str:
    """Remove only unmistakable standalone non-lexical hesitation sounds."""
    if not text:
        return ""
    result = _FILLERS.sub(" ", text)
    result = _SPACES.sub(" ", result)
    result = re.sub(r"\s+([,.;!?])", r"\1", result)
    return result.strip(" \t,;")
