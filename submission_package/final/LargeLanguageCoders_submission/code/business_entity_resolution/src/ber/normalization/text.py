from __future__ import annotations
import unicodedata
_LATIN_RANGES = ((65, 90), (97, 122), (192, 591), (592, 687), (7680, 7935), (11360, 11391), (42784, 43007), (43824, 43887))
_LATIN_SPECIAL = str.maketrans({'ø': 'o', 'ł': 'l', 'đ': 'd', 'ħ': 'h', 'æ': 'ae', 'œ': 'oe', 'ı': 'i'})
_DELETE_CHARS = {"'", '’', '‘', '`', '´'}

def is_latin_char(ch: str) -> bool:
    cp = ord(ch)
    for lo, hi in _LATIN_RANGES:
        if lo <= cp <= hi:
            return True
    return False

class _CharMap(dict):
    """Lazy ``str.translate`` table: punctuation/symbols/separators/controls -> space,"""

    def __missing__(self, cp: int) -> str:
        ch = chr(cp)
        if ch in _DELETE_CHARS:
            out = ''
        else:
            cat = unicodedata.category(ch)
            if cat == 'Nd':
                out = str(unicodedata.digit(ch))
            elif cat == 'Cf':
                out = ''
            elif cat[0] in 'PSZC':
                out = ' '
            else:
                out = ch
        self[cp] = out
        return out
_CHARMAP = _CharMap()

def unicode_normalize(text: str) -> str:
    """NFKC: folds compatibility forms (full-width letters, ligatures, superscripts)."""
    return unicodedata.normalize('NFKC', text)

def case_normalize(text: str) -> str:
    return text.casefold()

def strip_latin_accents(text: str) -> str:
    """Remove combining marks only when they follow a Latin base letter."""
    if text.isascii():
        return text
    out = []
    prev_latin = False
    for ch in unicodedata.normalize('NFD', text):
        cat = unicodedata.category(ch)
        if cat[0] == 'M':
            if prev_latin and cat == 'Mn':
                continue
            out.append(ch)
            continue
        prev_latin = is_latin_char(ch)
        out.append(ch)
    return unicodedata.normalize('NFC', ''.join(out)).translate(_LATIN_SPECIAL)

def punct_to_space(text: str) -> str:
    return text.translate(_CHARMAP)

def squeeze_whitespace(text: str) -> str:
    return ' '.join(text.split())

def normalize_text(text: str | None) -> str:
    """Full generic normalization used by every view."""
    if not text:
        return ''
    s = case_normalize(unicode_normalize(text))
    s = s.replace('&', ' and ')
    s = strip_latin_accents(s)
    s = punct_to_space(s)
    return squeeze_whitespace(s)

def tokenize(normalized: str) -> list[str]:
    """Whitespace tokenization of an already-normalized string."""
    return normalized.split()

def has_digit(token: str) -> bool:
    return any((ch.isdigit() for ch in token))

def numeric_tokens(tokens: list[str]) -> list[str]:
    """Tokens containing at least one digit (house numbers, ``b4``, ``45nd``, postcodes)."""
    return [t for t in tokens if has_digit(t)]

def script_class(text: str) -> int:
    """0 = Latin letters only, 1 = mixed Latin + non-Latin letters, 2 = non-Latin only,"""
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
