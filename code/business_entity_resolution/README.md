# Business Entity Resolution: reproduction guide

## Environment
- Python 3.13, `pip install -r requirements.txt` (versions pinned).
- Tested on a shared 64-core / 125 GB RAM Linux machine; **no GPU is used**.
- Libraries and licences: pandas, numpy, scipy, scikit-learn (BSD-3), LightGBM (MIT), RapidFuzz (MIT),
  sparse_dot_topn (Apache-2.0), pyarrow (Apache-2.0), anyascii (ISC, used as a library to romanise text).
- **No pretrained model, external dataset, API, geocoder or gazetteer is used.** Every rule is either a
  small generic seed (street-type and legal-form abbreviations) or mined from the provided training files.

## Run end to end
```bash
python src/run.py --data <student_resource/dataset> --work <cache dir> --out <output dir>
```
Stages (run in this order; each caches its result in `--work` and is skipped when already present):

| Stage | What it does | Module | Time on the test machine |
|---|---|---|---|
| `prep` | transliteration rules mined from train labels; clean every record | `er/mine.py`, `er/normalize.py`, `er/prep.py` | ~5 min |
| `block` | three TF-IDF search views per country, top-k in both directions | `er/blocking.py` | ~70 min per split |
| `rank` | LightGBM ranker on search scores (trained on a seeded 30% of train S1s), keep each S1's top 20 + each record's top 1 | `er/stages.py` | ~20 min per split |
| `keyview` | add exact "same cleaned name + same house number" candidates | `er/stages.py` | ~5 min |
| `feats` | ~72 pair features per country | `er/features.py` | ~15 min per split |
| `tlv` | test-like validation rows: clone each unowned train record to test decoy density | `er/stages.py` | ~8 min |
| `final` | final LightGBM + untouched-audit score + test predictions | `er/final.py` | ~60 min |
| `write` | decode, write both TSVs, run every submission check | `run.py`, `er/decode.py`, `er/io_metric.py` | ~5 min |

The submission in `output/` was produced with
```bash
python src/run.py --data ../student_resource/dataset --work cache/v2 --out <out> \
    --stages final,write --name sub5 --extra 1 --stage2 0 --folds 3 --lr 0.1 --threads 32
```
on top of cached `prep … tlv` outputs. Then validate:
```bash
python3 student_resource/utils/validate_submission.py --matching <out>/matching_results.tsv \
    --candidate <out>/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids
```
Seeds are fixed: ranker S1s 42, clones 11, audit S1s 2026, folds 0, LightGBM `seed=0` with
`deterministic=True`, `force_row_wise=True`.

## Method in one page
1. **Clean** names and addresses: romanise native scripts and apply 534 word rules learned from train pairs;
   strip `NULL`/`N/A`/PO boxes; recover the real name from `X DBA: Y`, `formerly`, `name.com`; split legal forms;
   canonicalise street types; detect state/department spellings per country without labels; parse the house
   number (the first number of the address, up to 5 digits) and the other numbers.
2. **Shortlist**: three TF-IDF views (name+address words, name 5-grams, address 5-grams) searched in both
   directions per country; a small LightGBM ranker keeps 20 candidates per S1 plus each record's best S1; plus
   exact name+house matches. Share of true pairs kept (non-ranker train S1s): US 99.05%, India 97.2%.
3. **Features** (77): name/address similarities, rare-word overlap, house-number relation (exact, digit dropped,
   distance, one-digit substitution), state agreement, pseudo-word detector, competition ranks and gaps, and
   **density** features (how many S1s of the same country share this name or this address).
4. **Model**: LightGBM trained on the original train pairs only. Pairs whose record belongs to no S1 (decoys)
   are weighted **1 + q** (US 0.891, India 0.941): test has about twice the decoys per S1 of train, so the model
   learns at test density without ever seeing synthetic records. 3 folds grouped by S1; each fold calibrated with
   weighted isotonic regression on its own held-out S1s; test = mean of the calibrated fold models.
5. **Decision rule**: each S2/S3 record goes to exactly one S1 (highest probability; ties broken by the
   ranker score), then per S1 the top-k with the highest expected F0.5 (k may be 0).

## Validation used for every decision
- **Test-like validation (TLV)**: train S1s' candidate lists with every unowned record cloned at rate q, so decoys
  per S1 match test. Scoring the leaderboard model on it reproduced its leaderboard drop (0.978 → 0.9655).
- **Audit set**: 220,682 train S1s (10%) drawn once, never used for training, calibration, early stopping or any
  choice; scored once per candidate, exactly as test is scored.
- **Leave-one-country-out** at test density as the France stand-in.
- Known limitation: the 534 transliteration rules were mined from all train labels, audit S1s included.
