from __future__ import annotations
import re
from ber.normalization.text import normalize_text, numeric_tokens, script_class
LEGAL_FORMS = frozenset('\n    inc incorporated llc llp lllp lp ltd limited pvt private corp corporation co company\n    plc pllc pc sarl sas sasu sa sci eurl snc gmbh ag bv nv pty\n    '.split())
WEB_TOKENS = frozenset('www http https com net org'.split())
ADDRESS_ABBREVIATIONS = {'rd': 'road', 'st': 'street', 'str': 'street', 'ave': 'avenue', 'av': 'avenue', 'blvd': 'boulevard', 'bd': 'boulevard', 'dr': 'drive', 'ln': 'lane', 'hwy': 'highway', 'pkwy': 'parkway', 'ct': 'court', 'pl': 'place', 'sq': 'square', 'apt': 'apartment', 'ste': 'suite', 'fl': 'floor', 'flr': 'floor', 'bldg': 'building', 'opp': 'opposite', 'nr': 'near', 'rte': 'route', 'chem': 'chemin'}
ADDRESS_STOPWORDS = frozenset('\n    road street avenue lane drive boulevard highway parkway court place square suite floor\n    apartment building unit block sector phase main cross near opposite behind beside next\n    no number plot door house flat shop the of and at in on to rue route chemin allee impasse\n    de la le les du des et\n    '.split())
_NULL_RE = re.compile('(?<![\\w/])(?:n/a|null|none|nan)(?![\\w/])')
_MS_RE = re.compile('^m s\\b ?')
_EDGE_CONNECTORS = frozenset({'and', 'et', 'the'})

def name_views(raw: str | None) -> tuple[str, str, str, int]:
    norm = normalize_text(raw)
    tokens = norm.split()
    core_tokens = [t for t in tokens if t not in LEGAL_FORMS and t not in WEB_TOKENS]
    while core_tokens and core_tokens[-1] in _EDGE_CONNECTORS:
        core_tokens.pop()
    while core_tokens and core_tokens[0] in _EDGE_CONNECTORS:
        core_tokens.pop(0)
    core = ' '.join(core_tokens)
    core = _MS_RE.sub('', core)
    if not core:
        core = norm
    concat = core.replace(' ', '')
    return (norm, core, concat, script_class(norm))

def address_views(raw: str | None) -> tuple[str, str, str]:
    if not raw:
        return ('', '', '')
    s = _NULL_RE.sub(' ', raw.casefold())
    tokens = [ADDRESS_ABBREVIATIONS.get(t, t) for t in normalize_text(s).split()]
    nums = numeric_tokens(tokens)
    words = [t for t in tokens if t not in nums]
    return (' '.join(tokens), ' '.join(nums), ' '.join(words))
VIEW_COLUMNS = ('name_norm', 'name_core', 'name_concat', 'name_script', 'addr_norm', 'addr_nums', 'addr_words')

def build_views(names: list[str], addresses: list[str]) -> dict[str, list]:
    """Compute all views for a batch (pure function; safe for multiprocessing)."""
    out: dict[str, list] = {c: [] for c in VIEW_COLUMNS}
    for name, addr in zip(names, addresses):
        n_norm, n_core, n_concat, n_script = name_views(name)
        a_norm, a_nums, a_words = address_views(addr)
        out['name_norm'].append(n_norm)
        out['name_core'].append(n_core)
        out['name_concat'].append(n_concat)
        out['name_script'].append(n_script)
        out['addr_norm'].append(a_norm)
        out['addr_nums'].append(a_nums)
        out['addr_words'].append(a_words)
    return out
