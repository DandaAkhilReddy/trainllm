"""Pure-Python text helpers: cleaning, PII scrubbing, dedupe, memorization checks.

No heavy dependencies so this module is unit-testable on any machine.
"""

from __future__ import annotations

import hashlib
import re

# Cover-slide headers like "#DesiLolliConfessions #2026 - #88"
_HEADER_RE = re.compile(r"#\w*confessions?\b|#\d{4}\s*-\s*#\d+", re.IGNORECASE)
_URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b\w+\.(?:com|in|gle|ly|io|me)/\S*)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{8,}\d)(?!\d)")
_HANDLE_RE = re.compile(r"(?<![\w@])@[A-Za-z0-9_.]{2,30}")
_HASHTAG_RE = re.compile(r"(?<!\w)#\w+")
_WS_RE = re.compile(r"[ \t]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")

THEME_KEYWORDS: dict[str, tuple[str, ...]] = {
    "visa & immigration": ("h1b", "h-1b", "visa", "opt", "stem", "green card", "gc ", "i-20", "f1", "lottery"),
    "jobs & layoffs": ("layoff", "laid off", "manager", "job", "offer", "interview", "office", "salary", "consult"),
    "parents & family": ("amma", "nanna", "mom", "dad", "parents", "family", "in-laws", "mother", "father"),
    "marriage & arranged match": ("marriage", "wedding", "alliance", "arranged", "husband", "wife", "pelli"),
    "love & relationships": ("love", "crush", "girlfriend", "boyfriend", "breakup", "dating", "ex "),
    "roommates & shared living": ("roommate", "room mate", "apartment", "lease", "flatmate"),
    "college & masters": ("university", "college", "masters", "ms ", "professor", "semester", "campus"),
    "money & loans": ("loan", "debt", "money", "emi", "rent", "savings", "credit card"),
    "friends & betrayal": ("friend", "betray", "cheat", "trust"),
}


def fix_mojibake(text: str) -> str:
    """Instagram JSON exports encode UTF-8 as latin-1 escapes; undo that when possible."""
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def scrub_pii(text: str) -> str:
    text = _URL_RE.sub("[link]", text)
    text = _EMAIL_RE.sub("[email]", text)
    text = _PHONE_RE.sub("[phone]", text)
    text = _HANDLE_RE.sub("@someone", text)
    return text


def clean_story(text: str) -> str:
    """Strip cover headers, hashtags, PII and noisy whitespace from one story."""
    lines = []
    for line in text.replace("\r\n", "\n").split("\n"):
        stripped = line.strip()
        if stripped.upper() == "NONE":
            continue
        # Drop lines that are only headers/hashtags (e.g. the cover slide)
        if stripped and not _HASHTAG_RE.sub("", _HEADER_RE.sub("", stripped)).strip(" -|•.,"):
            continue
        lines.append(_HASHTAG_RE.sub("", stripped))
    text = "\n".join(lines)
    text = scrub_pii(text)
    text = _WS_RE.sub(" ", text)
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def detect_themes(text: str, limit: int = 2) -> list[str]:
    lowered = f" {text.lower()} "
    scored = [
        (sum(lowered.count(k) for k in kws), theme) for theme, kws in THEME_KEYWORDS.items()
    ]
    scored = [s for s in scored if s[0] > 0]
    scored.sort(key=lambda s: -s[0])
    return [theme for _, theme in scored[:limit]]


def normalize_for_hash(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def content_hash(text: str) -> str:
    return hashlib.sha1(normalize_for_hash(text).encode()).hexdigest()


def word_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def ngram_overlap(candidate: str, reference_ngrams: set[tuple[str, ...]], n: int = 8) -> float:
    """Fraction of the candidate's n-grams that appear verbatim in the reference set."""
    cand = word_ngrams(candidate, n)
    if not cand:
        return 0.0
    return len(cand & reference_ngrams) / len(cand)
