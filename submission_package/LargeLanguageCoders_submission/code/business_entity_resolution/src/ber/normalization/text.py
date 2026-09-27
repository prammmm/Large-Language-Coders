"""Low-level, script-aware text normalization primitives.

Design rules (DEC-003):
* Never strip combining marks globally: Indic vowel signs / viramas are marks, and
  removing them destroys words (e.g. ``प्राइवेट``). Accents are stripped ONLY when the
  mark sits on a Latin base letter.
* Punctuation/symbols become spaces, zero-width format characters (ZWJ/ZWNJ) are
  dropped, and every Unicode decimal digit is mapped to ASCII.
* Functions are pure and deterministic; they never look anything up externally.
"""

from __future__ import annotations

import unicodedata

# Latin letter code-point ranges (Basic Latin letters, Latin-1 Supplement letters,
# Latin Extended-A/B, IPA, Latin Extended Additional, Extended-C/D/E).
_LATIN_RANGES = (
    (0x0041, 0x005A),
    (0x0061, 0x007A),
    (0x00C0, 0x024F),
    (0x0250, 0x02AF),
    (0x1E00, 0x1EFF),
    (0x2C60, 0x2C7F),
    (0xA720, 0xA7FF),
    (0xAB30, 0xAB6F),
)

# Latin letters that do not decompose under NFD but have a plain-ASCII reading.
_LATIN_SPECIAL = str.maketrans({"ø": "o", "ł": "l", "đ": "d", "ħ": "h", "æ": "ae", "œ": "oe", "ı": "i"})

# Characters removed entirely (not turned into a space) so words are not split.
_DELETE_CHARS = {"'", "’", "‘", "`", "´"}


def is_latin_char(ch: str) -> bool:
    cp = ord(ch)
    for lo, hi in _LATIN_RANGES:
        if lo <= cp <= hi:
            return True
    return False


class _CharMap(dict):
    """Lazy ``str.translate`` table: punctuation/symbols/separators/controls -> space,
    format chars (ZWJ/ZWNJ, BOM) -> removed, Unicode decimal digits -> ASCII digits,
    everything else (letters, marks, other numbers) unchanged. Results are cached."""

    def __missing__(self, cp: int) -> str:
        ch = chr(cp)
        if ch in _DELETE_CHARS:
            out = ""
        else:
            cat = unicodedata.category(ch)
            if cat == "Nd":
                out = str(unicodedata.digit(ch))
            elif cat == "Cf":
                out = ""
            elif cat[0] in "PSZC":
                out = " "
            else:
                out = ch
        self[cp] = out
        return out


_CHARMAP = _CharMap()


def unicode_normalize(text: str) -> str:
    """NFKC: folds compatibility forms (full-width letters, ligatures, superscripts)."""
    return unicodedata.normalize("NFKC", text)


def case_normalize(text: str) -> str:
    return text.casefold()


def strip_latin_accents(text: str) -> str:
    """Remove combining marks only when they follow a Latin base letter.

    ``"Àmicale Léarning"`` -> ``"Amicale Learning"``; Devanagari/Tamil/etc. are
    returned unchanged (their vowel signs are marks on non-Latin bases).
    """
    if text.isascii():
        return text
    out = []
    prev_latin = False
    for ch in unicodedata.normalize("NFD", text):
        cat = unicodedata.category(ch)
        if cat[0] == "M":
            if prev_latin and cat == "Mn":
                continue  # accent on a Latin letter
            out.append(ch)
            continue
        prev_latin = is_latin_char(ch)
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out)).translate(_LATIN_SPECIAL)


def punct_to_space(text: str) -> str:
    return text.translate(_CHARMAP)


def squeeze_whitespace(text: str) -> str:
    return " ".join(text.split())


def normalize_text(text: str | None) -> str:
    """Full generic normalization used by every view.

    NFKC -> casefold -> ``&``->``and`` -> Latin-only accent strip -> punctuation/symbols
    to spaces (digits to ASCII) -> whitespace squeeze.
    """
    if not text:
        return ""
    s = case_normalize(unicode_normalize(text))
    s = s.replace("&", " and ")
    s = strip_latin_accents(s)
    s = punct_to_space(s)
    return squeeze_whitespace(s)


def tokenize(normalized: str) -> list[str]:
    """Whitespace tokenization of an already-normalized string."""
    return normalized.split()


def has_digit(token: str) -> bool:
    return any(ch.isdigit() for ch in token)


def numeric_tokens(tokens: list[str]) -> list[str]:
    """Tokens containing at least one digit (house numbers, ``b4``, ``45nd``, postcodes)."""
    return [t for t in tokens if has_digit(t)]


def script_class(text: str) -> int:
    """0 = Latin letters only, 1 = mixed Latin + non-Latin letters, 2 = non-Latin only,
    3 = no letters at all."""
    latin = other = False
    for ch in text:
        if ch.isalpha():
            if is_latin_char(ch):
                latin = True
            else:
                other = True
            if latin and other:
                return 1
    if latin:
        return 0
    if other:
        return 2
    return 3
