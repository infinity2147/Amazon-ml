# Business Entity Resolution: reproduction guide

## Environment
- Python 3.13, `pip install -r requirements.txt`
- Tested on 64 CPU cores and 125 GB RAM. No GPU is needed for the current pipeline.
- Libraries and licences: pandas, numpy, scipy, scikit-learn (BSD), LightGBM (MIT), RapidFuzz (MIT),
  sparse_dot_topn (Apache-2.0), anyascii (ISC), pyarrow (Apache-2.0).
- No pretrained models, external data, APIs or gazetteers are used. All rules are either small generic
  seeds (street-type and legal-form abbreviations) or mined from the training labels and records.

## Run end to end
```bash
python src/run.py --data <path to student_resource/dataset> --out output --work work
```
This writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`, then runs the in-pipeline
format check. Afterwards, run the official validator:
```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```
Every stage caches its output as parquet in `--work`, so a rerun skips finished stages. Pass `--force`
to recompute everything. Seeds are fixed (LightGBM `seed=0`, sampling `random_state=0`).

## Pipeline
1. **Transliteration rules** (`er/mine.py`): native-script S2/S3 names are romanised with anyascii and
   aligned word by word with their true S1 name. This yields about 530 rules such as `praivet→private`
   and `phuds→food`.
2. **Normalisation** (`er/normalize.py`, `er/prep.py`):
   - romanise; strip accents; remove `NULL`/`N/A`/PO boxes;
   - handle `DBA`/`formerly`/`t/a` names and glued website names;
   - extract legal forms at the name's edges;
   - canonicalise street types;
   - detect admin-area spellings (states, departments, native-script states) per country, unsupervised;
   - parse numbers, units and landmarks.
3. **Blocking** (`er/blocking.py`): per country, TF-IDF top-k in both directions over word tokens
   (name + address), character 5-grams of the name, and character 5-grams of the address. The most
   common keys are pruned to a fixed compute budget. The union is capped per S1 / per record.
4. **Pair features** (`er/features.py`): about 55 country-agnostic features (similarities, number-set
   agreement, rarity, pseudo-word detection, competition ranks).
5. **Model** (`er/model.py`): LightGBM. Out-of-fold predictions by S1 group, isotonic calibration.
6. **Decoding** (`er/decode.py`): each S2/S3 record keeps only its best S1 (the one-owner rule holds
   exactly in train). Then, per S1, the set of top-k candidates with the highest expected F0.5.
