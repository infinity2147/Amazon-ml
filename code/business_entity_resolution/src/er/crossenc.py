"""Cross-encoder pair scorer (microsoft/mdeberta-v3-base, MIT licence, ~280M parameters).

Input: the RAW texts of both records as one sequence pair,
    "<S1 name> | <S1 address>"  [SEP]  "<record name> | <record address>"
(raw, not normalised, so the model sees native script, casing, punctuation and the generator's noise).
Output: one logit = evidence that the two records are the same business. Used only as an extra LightGBM feature.

Training rows: uncertain-band pairs (stage-1 p in (0.005, 0.995)) of a set of train S1s, label 1/0 from the answer
key. Two models are trained on disjoint halves of those S1s (split by a seeded hash of the S1 id); an S1 in half A is
scored by model B and vice versa; every other S1 (audit, test, other train S1s) gets the mean of both logits."""
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

BACKBONE = "microsoft/mdeberta-v3-base"


def half_of(s1_ids, seed=13):
    """0/1 half of each S1 id, stable across runs (seeded hash)."""
    h = pd.util.hash_array(np.asarray(s1_ids, dtype=object), hash_key=f"{seed:016d}")
    return (h % 2).astype(np.int8)


def pair_texts(P, s1raw, cpraw):
    """P: [s1_id, cand_id(base)] -> two lists of strings."""
    a = s1raw.loc[P.s1_id.values]
    b = cpraw.loc[P.cand_id.values]
    ta = (a.business_name + " | " + a.business_address).tolist()
    tb = (b.business_name + " | " + b.business_address).tolist()
    return ta, tb


class _Collate:
    def __init__(self, tok, max_len):
        self.tok, self.max_len = tok, max_len

    def __call__(self, batch):
        ta, tb, y = zip(*batch)
        enc = self.tok(list(ta), list(tb), truncation=True, max_length=self.max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(y, dtype=torch.float32)
        return enc


def train_ce(ta, tb, y, out_dir, backbone_path, log, epochs=1, bs=64, lr=2e-5, max_len=128, seed=0, device="cuda:0"):
    torch.manual_seed(seed); np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(backbone_path)
    model = AutoModelForSequenceClassification.from_pretrained(backbone_path, num_labels=1, dtype=torch.float32).to(device)
    data = list(zip(ta, tb, y.astype(np.float32)))
    dl = DataLoader(data, batch_size=bs, shuffle=True, collate_fn=_Collate(tok, max_len), num_workers=2,
                    generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * len(dl)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train(); t0 = time.time(); step = 0
    for ep in range(epochs):
        run = 0.0
        for batch in dl:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(**batch).logits.squeeze(-1)
            loss = lossf(logit.float(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            run += loss.item(); step += 1
            if step % 500 == 0:
                log(f"ce train step {step}/{steps} loss {run / 500:.4f} ({(step * bs) / (time.time() - t0):.0f} pairs/s)"); run = 0.0
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    return out_dir


@torch.no_grad()
def score_ce(ta, tb, model_dir, log, bs=512, max_len=128, device="cuda:0"):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, dtype=torch.float32).to(device).eval()
    n = len(ta)
    order = np.argsort([len(x) + len(z) for x, z in zip(ta, tb)])          # length-sorted batches: less padding
    out = np.zeros(n, np.float32); t0 = time.time()
    for i in range(0, n, bs):
        ix = order[i:i + bs]
        enc = tok([ta[j] for j in ix], [tb[j] for j in ix], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out[ix] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
        if (i // bs) % 2000 == 0:
            log(f"ce score {i:,}/{n:,} ({i / max(time.time() - t0, 1e-9):.0f} pairs/s)")
    return out
