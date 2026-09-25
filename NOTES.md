# NOTES.md: Day-0 recon (measured 2026-09-25)

All numbers are [MEASURED] on the real data unless marked otherwise. Scripts: `work/recon*.py`, cache: `work/cache/*.parquet`.

## Sizes
| split | S1 | S2 | S3 |
|---|---|---|---|
| train | 2,206,821 | 5,034,616 | 5,285,603 |
| test  | 1,732,544 | 4,887,273 | 5,082,316 |

- Test S2+S3 by country: India 4.72M, US 3.82M, France 1.43M. Test S1: India 810k, US 663k, France 259k.
- TSVs load cleanly with the safe loader: row counts equal `wc -l` − 1, and every file has unique IDs.

## Ground-truth facts (train)
- **Singleton rate is only 5.6%**, in both US and India. An all-empty submission scores about 0.056.
- **Match sets are large:** the mode is 3 and the mean is about 3.5. Distribution: 0 = 5.6%, 1 = 5.4%, 2 = 17%, 3 = 24%, 4 = 22%, 5 = 15%, 6 = 7.5%, 7 = 2.9%, 8+ = 1.2%, max 11. The two countries have the same distribution, which points to a synthetic generator.
  - Consequence: most of the score comes from recall inside multi-match sets, not from singletons. For example, with gold of size 4, predicting 3 correct gives 0.94.
- Per S1 there are 0–5 S2 matches and 0–6 S3 matches.
- **One-owner rule holds exactly:** 0 of the 7.64M matched S2/S3 IDs belong to more than one S1.
- **0 cross-country links.** Block within country.
- **Distractors:** 26.6% of S2 records and 25.4% of S3 records match no S1. They look like noisy copies of businesses that are *not* in S1. Only 0.9% of them share an address prefix with any S1 record, against 21% for matched records.
- Singletons: 38% have a record in S2/S3 with the exact same normalised name (74% for matched S1s). Name alone is not enough.

## Script
- S1 is 100% Latin.
- India S2/S3 **names** in native scripts: S2 23%, S3 12%. Scripts: Devanagari ≫ Telugu ≈ Kannada ≈ Tamil ≈ Bengali ≈ Gujarati > Malayalam > Oriya > Gurmukhi. These are whole-name transliterations, e.g. `हरि फाउंडेशन प्राइवेट लिमिटेड`.
- India **addresses** are mostly Latin. About 23% of matched India records have the *state* in native script (`दिल्ली`, `ಕರ್ನಾಟಕ`, `महाराष्ट्र`).
- US and France are 100% Latin, but have **injected accent noise** (`Cómmittee`, `Ínc`, `Àmicale`, `Frànce`), and France also has real accents.

## Noise operators on matched pairs (sample of 60k S1s, about 200k pairs)
| | India S2 | India S3 | US S2 | US S3 |
|---|---|---|---|---|
| native-script name | 23.6% | 13.1% | 0 | 0 |
| website/hashtag name (`x.com`, `#x`, `xcom`) | 4.8% | 4.8% | 6.3% | 5.8% |
| name identical (case-insensitive) | 6% | 7% | 14% | 13% |
| name token_set mean | 71 | 80 | 93 | 93 |
| name **replaced by a random pseudo-word** (`UMBRATAVO`, `Zephcalo`, `Sri Zephgild`) | about 1–3% | | | |
| address empty | 3.7% | 4.0% | 4.9% | 4.5% |
| house/number sets fully disjoint | 2.2% | 2.5% | 6.9% | 5.9% |
| PO Box / PMB added | 0 | 0 | 3.1% | 3.1% |
| `<NULL>`/`NULL` tokens | 2% | 2% | 3% | 3% |

**Name operators**
- Legal-suffix drop, swap or expand (`llc→l`, `inc→incorporated`, `limited→ltd`, `private→pr`, `pvt ltd`, bracketed `[Private-Limited]`).
- Filler words added: `Services, Center, Partners, Group, Holdings, Enterprises, Council`.
- Prefixes: `The, Sri, Shri, Smt, Mr, Dr, M/s`, `DBA`, `aka`, `Formerly X`.
- Junk prefixes: `--`, `***`, `<<`, `#`.
- Token reorder or duplication (`Commission Commission on`).
- Character typos: dropped letters (`ssociates`), l↔1, o↔0, I↔l (`lncorporated`), inserted letters.
- A core word is replaced (`Halcyon Productions → Halcyon Partners`).
- Acronym (`Sun Om Infra → SOI`).

**Address operators**
- Street-type abbreviation (`street→st`, `road→rd`, and so on).
- **State: abbreviation ↔ full name ↔ native script.** India also has `orissa→odisha` and `kerala→keralam`.
- Component reorder, and dropped components (street, city, unit, or everything except the house number).
- House-number noise:
  - zero-padding (`004514`, `01335`);
  - prefixes `#`, `##`, `No`, `H.NO`, `Door No`, `N°`;
  - suffix (`1335-C`);
  - digit drop (`1490→490`, `4175→417`);
  - ±small (`360→351`, `19327→19328`);
  - extra fake house number (`No 941 3/365`).
- City substitution: alias or neighbouring city (`Salemburg→Godwin`, `Gurgaon↔Gurugram`, `Bombay`, `Bengaluru`) and `CITY`/`TOWNSHIP`/`CDP` suffixes.
- Typos in street names.
- Casing: S2 addresses are UPPERCASE, S3 addresses are title case with full state names.

## France (test only), eyeballed
- S1 is concentrated in about 10 cities: Bordeaux, Nantes, Lille, Tourcoing, Dunkerque, Pessac, Mérignac, Saint-Nazaire, and others.
- The vocabulary is extremely repetitive: `Club, Ecole, Amis, Parents, Comite, Maison, Sante, (France), SARL/SAS/EURL/SCI`.
- **Many near-twin distractors share the same street with different house numbers**, e.g. `Locataires & Cie SARL, 27 Rue de l'Avenir, Pessac` vs `Locataires (France) Parents SARL, 12 Rue de l'Avenir, Bordeaux`.
  - The house number plus the full name-token set are the key discriminators.
  - Train shows house numbers are also noisy (digit drop, ±1), so this is the main France precision risk.
- French address variants:
  - `RUE / R / R.`, `BD / Boulevard`, `AV / Avenue`;
  - `N° / Nº / #` prefixes, `BIS / TER`;
  - department (`Gironde`, `Nord`, `Loire-Atlantique`) ↔ region (`Nouvelle-Aquitaine`, `Hauts-de-France`, `Pays de la Loire`);
  - postcode occasionally.
- French name variants: `S.A.S / S.A.S.U / Sàrl / S.A.R.L.`, `& Cie ↔ & Compagnie`, `(France)`, `Ets`, leading legal forms (`SARL Bruges`), and the same filler and website operators as train.

## Deep dive 2 (hands-on pass over raw rows, scripts `work/dd1..dd7.py`)

**Match groups (S1 + all true S2/S3), extra operators not listed above**
- `N/A` is a missing-value token like `NULL`/`<NULL>` (about 0.7–1% of S2/S3 addresses; France has none of the three).
- A junk word replaces the city: `528 ELSIE AVENUE, INCORPORATED, IL`. City typos/neighbours: `MADIOSN` for Huntsville, `PRESCOT TVALLEY`.
- Names are comma-inverted (`crouch, cynde keystone coca inc`), legal forms bracketed (`[Inc]`, `[PLLC]`, `(Managed)`), and legal words moved to the front (`PVT RADHA SYSTEMS LTD`, `LLC Ellis Innovative Era`).
- Markers come dotted or slashed: `D.B.A.`, `t/a`, `a/k/a`, `fka`, `trading as`, `formerly known as`. The real name always follows the marker; before it is a pseudo-word (`Avidelta`, `Umbraonyx`).
- Name can be a whole glued website: `RAMOSPEAKRAIL.COM | www.ramospeakr.com`, `lofficinegroupementsas.com` → need a no-space name comparison.
- Name can be an acronym of the S1 name (`TB` = TAW Buildcon, `RC` = Raid Comite) or a pure pseudo-word (`Gildkortavo`, `Nylazetatavo`), with the address intact.
- Address numbers: fake extra numbers are prepended (`NO. 42 SHOPNO.-32`, `HN 259- 44-387/1A`), digits dropped (`836→36`), ±1 (`204→205`), `B-##3`. **S1 itself** has shuffled/duplicated components (`Agra, Uttar Pradesh, Agra, 2`), so "first number = house number" is unreliable → compare number *sets* fuzzily.
- Addresses empty in 2.3–3.7% of S2/S3 records (never in S1).

**Chains and singletons**
- 39% of train S1 names are shared with another S1 (`primary care group` ×253, all different addresses); France 34% (`bordeaux club sarl` ×205). S1 never repeats name+address.
- A singleton's exact name often exists in S2/S3, but it belongs to a different S1 at another address. **The address decides**.
- Decoys (S2/S3 with no owner) are mostly unrelated businesses. Only 5% share an exact name with some S1, against 26% for matched records.

**Native scripts**
- The romanised native names (anyascii) come out as consistent misspellings: `praivet/praibhet/piraivet/pra` (private), `limitet/limirrd/limtid/li` (limited), `elelpi` (LLP), `phuds` (foods), `imtrnesnl` (international).
- Aligning them with S1 names in train pairs yields **534 romanised→English rules** (`er/mine.py`), so `जैन इंटरनेशनल प्राइवेट लिमिटेड` → `jain international` + legal `ltd pvt`.

**Leak check:** corr(S1 id number, matched id number) = 0.008. No ID leak.

**Train vs test shift (important)**
- S2/S3 per S1: train 4.7 (US and India); test 5.8 (US, India), 5.5 (France).
- "Obvious owner" proxy (same name without legal form + a shared address number), per S1: train 1.36 / 1.31, test 1.36 / 1.35 (India/US), France 1.56.
- So test has the **same number of true matches per S1** and about **2× the decoys** (≈40% of S2/S3 against 26%). Precision matters more on test than CV shows.
- Whether the extra decoys are "hard" (at S1 addresses) is **not yet established**: the crude address-key proxy was dominated by coincidental `22 … Street` hits. To be measured with the blocking and model score distributions on test.

**France (test)** is clearly the same generator: DBA/trading-as, pseudo-words (`Dovawex`), acronyms, glued websites, `N°`, `R.`/`RUE`, `BIS`, and region↔department swaps (`Nouvelle-Aquitaine`↔`Gironde`, `Hauts-de-France`↔`Nord`/`Pas-de-Calais`, `Pays de la Loire`↔`Loire-Atlantique`). S1 addresses mostly have 3 components (street, city, region).
