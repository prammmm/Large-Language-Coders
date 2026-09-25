import unicodedata

from ber.normalization.text import (
    normalize_text,
    numeric_tokens,
    punct_to_space,
    script_class,
    strip_latin_accents,
    tokenize,
    unicode_normalize,
)
from ber.normalization.views import address_views, build_views, name_views


def test_unicode_nfkc_folds_compatibility_forms():
    assert unicode_normalize("ＡＢＣ１２３") == "ABC123"
    assert unicode_normalize("ﬁne") == "fine"


def test_case_and_whitespace():
    assert normalize_text("  Summit   PARTNERS\tInc ") == "summit partners inc"


def test_latin_accents_stripped():
    assert strip_latin_accents("Àmicale Léarning Pàrenthese") == "Amicale Learning Parenthese"
    assert normalize_text("Café Crème Œuvre") == "cafe creme oeuvre"


def test_indic_marks_preserved():
    for s in ["प्राइवेट लिमिटेड", "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ்", "মহারাষ্ট্র", "ಬೆಂಗಳೂರು"]:
        out = normalize_text(s)
        assert out == unicodedata.normalize("NFC", s)  # nothing removed
        assert strip_latin_accents(s) == s


def test_mixed_script_only_latin_part_changes():
    assert normalize_text("Raj Invéstments एलएलपी") == "raj investments एलएलपी"


def test_zero_width_joiner_removed_not_split():
    assert normalize_text("क्‍ष") == "क्ष"


def test_punctuation_and_ampersand():
    assert normalize_text("Wave & Brothers (Ltd.)") == "wave and brothers ltd"
    assert punct_to_space("a--b<<c##d") == "a  b  c  d"
    assert normalize_text("McDonald’s") == "mcdonalds"


def test_unicode_digits_to_ascii():
    assert normalize_text("प्लॉट १२३") == "प्लॉट 123"


def test_tokenize_and_numeric_extraction():
    toks = tokenize(normalize_text("Plot No. 65, B4 Wing, 14/15 MG Road"))
    assert toks == ["plot", "no", "65", "b4", "wing", "14", "15", "mg", "road"]
    assert numeric_tokens(toks) == ["65", "b4", "14", "15"]


def test_script_class():
    assert script_class("summit") == 0
    assert script_class("raj एलएलपी") == 1
    assert script_class("एलएलपी") == 2
    assert script_class("123") == 3


def test_name_views():
    norm, core, concat, script = name_views("M/s. -- APAR (Limited)")
    assert norm == "m s apar limited" and core == "apar" and concat == "apar" and script == 0
    assert name_views("truefactory.com")[1] == "truefactory"
    assert name_views("McDonald's & Co")[1] == "mcdonalds"
    assert name_views("Limited")[1] == "limited"  # never empty: falls back to name_norm
    assert name_views("")[0] == ""


def test_address_views():
    norm, nums, words = address_views("KANSAS CITY, MO, 630 45ND TERRACE, null")
    assert norm == "kansas city mo 630 45nd terrace"
    assert nums == "630 45nd"
    assert words == "kansas city mo terrace"
    assert address_views("12 Main St, N/A")[0] == "12 main street"
    assert address_views("") == ("", "", "")


def test_build_views_batch():
    v = build_views(["Apex Inc", "एपेक्स"], ["1 Oak Rd", ""])
    assert v["name_core"] == ["apex", "एपेक्स"]
    assert v["addr_norm"] == ["1 oak road", ""]
    assert v["name_script"] == [0, 2]
