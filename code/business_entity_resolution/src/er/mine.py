"""Rules mined from TRAINING LABELS only (no external dictionaries).

mine_translit: native-script S2/S3 names are whole-name transliterations of the S1 name
('जैन इंटरनेशनल प्राइवेट लिमिटेड' <-> 'Jain International Private Limited'). After
romanising, the tokens come out as consistent misspellings ('imtrnesnl', 'praivet', 'phuds').
We align them position-by-position with the S1 name when the token counts agree, and keep
(romanised -> english) pairs that are frequent and unambiguous."""
from collections import Counter, defaultdict

from .normalize import base_clean, is_native, to_ascii


def mine_translit(s1, cands, gt_map, min_count=3, min_purity=0.6, max_pairs=2_000_000):
    name1 = dict(zip(s1.entity_id, s1.business_name))
    namec = dict(zip(cands.entity_id, cands.business_name))
    cnt = defaultdict(Counter)
    n = 0
    for s, cs in gt_map.items():
        a = name1.get(s)
        if a is None:
            continue
        A = base_clean(to_ascii(a)).split()
        for c in cs:
            b = namec.get(c)
            if b is None or not is_native(b):
                continue
            B = base_clean(to_ascii(b)).split()
            if len(A) != len(B):
                continue
            for x, y in zip(B, A):
                if x != y and not x.isdigit():
                    cnt[x][y] += 1
            n += 1
            if n >= max_pairs:
                break
    out = {}
    for x, c in cnt.items():
        y, k = c.most_common(1)[0]
        tot = sum(c.values())
        if k >= min_count and k / tot >= min_purity:
            out[x] = y
    return out
