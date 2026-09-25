"""Open-set normalisation. No rule is keyed on a country name: every rule is a text pattern,
so an unseen country (France in test) gets the same treatment.

Two passes:
  1. `clean_components` splits each address on commas and cleans each component.
  2. `fit_admin_vocab` (unsupervised, per country, on S1 + S2/S3 of the SAME split) finds
     - state_vocab: components that are frequent in S1 and usually the last one (US 'tx',
       India 'haryana', France 'hauts de france');
     - noise_comps: components frequent in S2/S3 but never seen in S1 (state spelled another way:
       'texas', native-script 'महाराष्ट्र', 'keralam', French departments, 'null').
     Both are removed from the address text and kept as a separate `state` field.
  3. `prepare` builds every normalised field used by blocking and features.
"""
import re
import unicodedata
from collections import Counter

from anyascii import anyascii

LEGAL = {  # canonical legal-form tokens (removed from the name edges, kept as a feature)
    "incorporated": "inc", "inc": "inc", "lnc": "inc", "nc": "inc", "corporation": "corp", "corp": "corp",
    "company": "co", "co": "co", "llc": "llc", "llp": "llp", "lp": "lp", "pllc": "pllc", "pc": "pc",
    "limited": "ltd", "ltd": "ltd", "plc": "plc", "private": "pvt", "pvt": "pvt", "pte": "pvt",
    "pr": "pvt", "opc": "opc", "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl",
    "sci": "sci", "snc": "snc", "scp": "scp", "selarl": "selarl", "gmbh": "gmbh", "srl": "srl",
    "cie": "co", "compagnie": "co", "ets": "ets", "etablissements": "ets", "dds": "dds", "md": "md",
    "pa": "pa", "l": "l", "c": "c", "p": "p",
}
LEAD_LEGAL = {"sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "scp", "selarl", "ets", "etablissements"}
HONORIFIC = {"the", "sri", "shri", "shree", "smt", "mr", "mrs", "ms", "dr", "m", "s", "messrs", "le", "la", "les"}
NAME_STOP = {"the", "and", "of", "de", "du", "des", "la", "le", "les", "et", "a", "l", "d", "en", "null"}
MARKER_RE = re.compile(r"\b(?:d\s*/\s*b\s*/\s*a|dba|formerly known as|formerly|f\s*/\s*k\s*/\s*a|fka|"
                       r"a\s*/\s*k\s*/\s*a|aka|also known as|trading as|t\s*/\s*a)\b\s*:?", re.I)
WEB_RE = re.compile(r"(?:https?://)?(?:www\.)?([a-z0-9-]+)\.(?:com|in|net|org|co|fr|us|biz|info|io)\b")

ADDR = {  # long -> short canonical; seed list extended with rewrites mined from train labels
    "street": "st", "str": "st", "saint": "st", "sainte": "ste", "road": "rd", "avenue": "ave",
    "av": "ave", "boulevard": "blvd", "bd": "blvd", "bvd": "blvd", "drive": "dr", "lane": "ln",
    "court": "ct", "circle": "cir", "place": "pl", "square": "sq", "highway": "hwy", "parkway": "pkwy",
    "terrace": "ter", "trail": "trl", "suite": "ste", "apartment": "apt", "building": "bldg",
    "floor": "fl", "flr": "fl", "north": "n", "south": "s", "east": "e", "west": "w", "number": "no",
    "num": "no", "nagar": "ngr", "sector": "sec", "chemin": "ch", "allee": "all", "impasse": "imp",
    "route": "rte", "r": "rue", "near": "nr", "opposite": "opp", "behind": "bhd", "mount": "mt",
    "fort": "ft", "cross": "x", "first": "1", "second": "2", "third": "3", "centre": "center",
    "bengaluru": "bangalore", "gurugram": "gurgaon", "bombay": "mumbai", "calcutta": "kolkata",
    "madras": "chennai", "orissa": "odisha", "keralam": "kerala",
}
# tokens that carry no identity in an address (house-number prefixes, generator's city suffixes)
ADDR_STOP = {"no", "nos", "door", "h", "hn", "hno", "ndeg", "deg", "plt", "plot", "shopno", "house", "null", "township", "cdp", "city", "county",
             "de", "du", "des", "la", "le", "les", "et", "of", "the", "and", "a", "d", "l", "n0"}
BOX_RE = re.compile(r"\b(?:p\s*o\s*box|post box|pmb|box)\s*#?\s*\d+\b")
LANDMARK_RE = re.compile(r"\b(?:near|nr|opp|opposite|behind|beside|next to|adjacent to|adj|in front of|"
                         r"pres de|pres du|face a|en face de|a cote de)\b.*")
UNIT_RE = re.compile(r"\b(?:suite|ste|apt|unit|room|rm|shop|flat|office|bureau|#)\s*(?:no\s*)?([a-z0-9-]+)")


NUMSIGN_RE = re.compile(r"\b[Nn]\s*[°º]\s*|[°º]")


def to_ascii(s):
    # 'N°15' -> ' 15' first (anyascii would glue it into 'Ndeg15'); then NFKC + anyascii:
    # native scripts -> Latin ('हरि' -> 'hri'), accents stripped, native digits -> 0-9
    s = NUMSIGN_RE.sub(" ", str(s))
    return anyascii(unicodedata.normalize("NFKC", s))


def base_clean(s):
    s = s.lower()
    s = s.replace("<null>", " ").replace("m/s", " ")
    s = re.sub(r"\b(?:[a-z]\.){2,}", lambda m: m.group(0).replace(".", "") + " ", s)  # s.a.s. -> sas
    s = s.replace("&", " and ").replace("@", " ")
    s = re.sub(r"['`]", "", s)                # o'brien -> obrien
    s = re.sub(r"(?<=\d)(st|nd|rd|th|er|eme|e)\b", "", s)  # 5th / 8e / 8eme -> 5 / 8
    s = re.sub(r"(?<=\d)(bis|ter)\b", r" \1", s)            # 15bis -> 15 bis
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


TRANSLIT = {}  # romanised-native token -> English token, mined from train labels (mine.py)
LEAD_STRIP = {k for k, v in LEGAL.items() if len(k) > 2 and v in {"pvt", "ltd", "llc", "inc", "corp", "llp", "pllc"}}


def set_translit(d):
    TRANSLIT.clear()
    TRANSLIT.update(d)


def norm_name(raw, native=None):
    """-> (name_n, legal, alt_name, web). name_n is the business name with honorifics,
    legal forms at the edges, and 'X DBA:'/'X formerly' prefixes removed.
    Native-script names are romanised and then mapped token-by-token with TRANSLIT."""
    native = is_native(raw) if native is None else native
    s = to_ascii(raw).lower()
    s = re.sub(r"\b(?:[a-z]\.){2,}", lambda m: m.group(0).replace(".", "") + " ", s)  # d.b.a. -> dba
    web = ""
    parts = s.split("|")
    if len(parts) > 1:
        s = parts[0]
        m = WEB_RE.search("|".join(parts[1:]))
        web = m.group(1) if m else ""
    alt = ""
    m = MARKER_RE.search(s)
    if m:
        pre, post = s[:m.start()], s[m.end():]
        if base_clean(post) and base_clean(pre):
            s, alt = post, base_clean(pre)
    m = WEB_RE.fullmatch(s.strip())
    if m:
        s = m.group(1)
        web = web or m.group(1)
    toks = base_clean(s).split()
    if native and TRANSLIT:
        toks = [TRANSLIT.get(t, t) for t in toks]
    while len(toks) > 1 and toks[0] in HONORIFIC:
        toks.pop(0)
    legal = []
    while len(toks) > 1 and toks[0] in LEAD_STRIP:   # generator moves legal words to the front
        legal.append(LEGAL[toks.pop(0)])
    while len(toks) > 1 and toks[-1] in LEGAL:
        legal.append(LEGAL[toks.pop()])
    while len(toks) > 1 and toks[0] in LEAD_LEGAL:
        legal.append(LEGAL[toks.pop(0)])
    core = [t for t in toks if t not in NAME_STOP] or toks
    return " ".join(core), " ".join(sorted(set(legal) - {"l", "c", "p"})), alt, web


def clean_components(raw):
    """Address -> list of cleaned comma components (PO boxes / PMB removed)."""
    s = re.sub(r"\bn\s*/\s*a\b|<null>|\bnull\b", " ", to_ascii(raw).lower())
    out = []
    for c in s.split(","):
        c = BOX_RE.sub(" ", c)
        c = " ".join(ADDR.get(t, t) for t in base_clean(c).split())
        if c and c != "null":
            out.append(c)
    return out


def fit_admin_vocab(s1_comps, s1_ctry, c_comps, c_ctry, min_state_frac=0.0005, min_noise=100,
                    min_comp_frac=0.0005):
    """Unsupervised per-country vocabulary (no labels). Returns {country: (state_vocab, noise_tokens)}.
    state_vocab : S1 address components that are frequent and usually last ('tx', 'haryana').
    noise_tokens: address TOKENS frequent in S2/S3 (>= min_noise) that never occur in any S1 address of
                  that country, i.e. systematic format noise: 'texas', romanised native state names,
                  'keralam', French departments ('gironde'), 'null'. Such a token can never help match
                  an S1 record, and it inflates dissimilarity."""
    out = {}
    for ctry in set(s1_ctry) | set(c_ctry):
        allc, lastc, tok1, n1 = Counter(), Counter(), set(), 0
        for comps, k in zip(s1_comps, s1_ctry):
            if k != ctry or not comps:
                continue
            n1 += 1
            allc.update(set(comps))
            lastc[comps[-1]] += 1
            for c in comps:
                tok1.update(c.split())
        state = {c for c, n in allc.items() if n >= max(20, min_state_frac * n1) and lastc[c] >= 0.5 * n}
        tc, cc, n2 = Counter(), Counter(), 0
        for comps, k in zip(c_comps, c_ctry):
            if k == ctry:
                n2 += 1
                tc.update({t for c in comps for t in c.split()})
                cc.update(set(comps))
        noise = {t for t, n in tc.items() if n >= min_noise and t not in tok1 and not t.isdigit()}
        # whole components (no digits) frequent in S2/S3 but never an S1 component: another spelling of
        # the admin area ('montana', 'gj', 'gironde', 'loire atlantique') -> treated like state
        alias = {c for c, n in cc.items() if n >= max(min_noise, min_comp_frac * n2) and c not in allc
                 and not any(ch.isdigit() for ch in c)}
        out[ctry] = (state | alias, noise)
    return out


def _num(t):
    return t.lstrip("0") or "0"


def house_postal(nums):
    """nums: the address's numbers in reading order (leading zeros stripped).
    House = the FIRST number with at most 5 digits, by position: US suburban house numbers have 5 digits
    ('11900 54th Avenue' -> 11900, not 54). Postal = every 6-digit number (Indian PIN) and every OTHER
    5-digit number (US ZIP / French code postal after the street). Measured on 518k true train pairs:
    US S1s with a house number 92.3% -> 99.96%, S1 house found in the matched record 72.1% -> 78.0%."""
    hi = next((i for i, t in enumerate(nums) if len(t) <= 5), None)
    house = nums[hi] if hi is not None else ""
    postal = sorted({t for i, t in enumerate(nums) if len(t) == 6 or (len(t) == 5 and i != hi)})
    return house, postal


def nums_in_order(addr):
    """Numbers of a normalised `addr` string in reading order (the same list norm_addr builds)."""
    return [_num(x) for t in addr.split() for x in re.findall(r"\d+", t)]


def norm_addr(comps, state_vocab=frozenset(), noise=frozenset()):
    state = [c for c in comps if c in state_vocab]
    body = [c for c in comps if c not in state_vocab]
    if noise:
        body = [" ".join(t for t in c.split() if t not in noise) for c in body]
        body = [c for c in body if c]
    text = " , ".join(body)
    landmark = " ".join(m.group(0) for c in body for m in [LANDMARK_RE.search(c)] if m)
    units = sorted(set(UNIT_RE.findall(text)))
    toks = []
    for t in text.split():
        if t == ",":
            continue
        t = ADDR.get(t, t)
        if t in ADDR_STOP:
            continue
        if t.isdigit():
            t = _num(t)
        toks.append(t)
    nums = [_num(x) for t in toks for x in re.findall(r"\d+", t)]
    house, postal = house_postal(nums)
    words = [t for t in toks if not any(ch.isdigit() for ch in t)]
    return dict(addr=" ".join(toks), addr_words=" ".join(words), state=" ".join(state[:1]),
                postal=" ".join(postal), house=house, units=" ".join(units), landmark=landmark,
                nums=" ".join(sorted(set(nums))))


_SK = [("aa", "a"), ("ee", "i"), ("oo", "u"), ("ou", "u"), ("sh", "s"), ("ch", "c"), ("th", "t"),
       ("dh", "d"), ("bh", "b"), ("kh", "k"), ("gh", "g"), ("ph", "f"), ("w", "v"), ("z", "j"),
       ("q", "k"), ("ck", "k"), ("x", "ks")]


def skeleton(s):
    """Transliteration-robust key: Agarwal/Aggarwal/Agrawal -> agrvl."""
    out = []
    for t in s.split():
        for a, b in _SK:
            t = t.replace(a, b)
        t = t[:1] + re.sub(r"[aeiouy]", "", t[1:])
        t = re.sub(r"(.)\1+", r"\1", t)
        out.append(t)
    return " ".join(out)


def is_native(s):
    return any(ord(ch) > 0x24F and unicodedata.category(ch).startswith("L") for ch in str(s))


FIELDS = ["name_n", "legal", "name_alt", "web", "addr", "addr_words", "state", "postal", "house",
          "units", "landmark", "nums", "name_sk", "name_acr", "name_nums", "native"]


def prepare_record(name, addr_comps, state_vocab, noise):
    nn, legal, alt, web = norm_name(name)
    a = norm_addr(addr_comps, state_vocab, noise)
    return (nn, legal, alt, web, a["addr"], a["addr_words"], a["state"], a["postal"], a["house"],
            a["units"], a["landmark"], a["nums"], skeleton(nn),
            "".join(t[0] for t in nn.split() if not t.isdigit()),
            " ".join(sorted(t for t in nn.split() if any(c.isdigit() for c in t))),
            int(is_native(name)))
