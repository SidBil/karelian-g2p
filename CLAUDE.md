# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Karelian grapheme-to-phoneme (G2P) conversion: turning Karelian orthography into IPA
transcription. It contains two parallel, independently-scored approaches over the same
dataset (`karelian_terms.csv`, ~orthography/gold-IPA pairs scraped from Wiktionary):

1. A **rule-based baseline** (`g2p.py`) — a fixed pipeline of hand-written string-rewrite
   rules. Scores 4.43% average phoneme error rate (PER).
2. A **neural seq2seq model** (`karelian_g2p_seq2seq.ipynb`, plus `learning_curve.py`) —
   a bidirectional-GRU-encoder + attention-GRU-decoder trained character-by-character,
   which only beats the rule-based baseline once trained on ~50%+ of available data.

There is no git repo here (not `git init`'d) and no package manifest (no `requirements.txt`
/ `pyproject.toml`) — dependencies live only in `venv/`.

## Running things

Use the local venv's Python directly (it already has torch, matplotlib, requests, bs4,
jupyter, etc. installed):

```bash
venv/bin/python g2p.py                # run + score the rule-based pipeline
venv/bin/python learning_curve.py     # train neural model across data fractions, sweep learning curve
venv/bin/python plot_learning_curve.py  # re-plot learning_curve_results.csv without retraining
venv/bin/python process_karelian.py   # clean karelian_terms.csv (drop hyphen-prefixed / single-char entries)
venv/bin/python scrape_wiktionary.py  # re-scrape Karelian/Livonian/Ingrian terms from Wiktionary
venv/bin/jupyter notebook karelian_g2p_seq2seq.ipynb  # interactive neural model notebook
```

There is no test suite. `g2p.py` and `learning_curve.py` are self-scoring scripts: running
them *is* the verification step (they print PER/accuracy on the data they run over).

## Architecture

### Data flow

`scrape_wiktionary.py` → `karelian_terms.csv` (raw orthography,IPA pairs) →
`process_karelian.py` (filters bad rows in place) → consumed identically by both the
rule-based and neural pipelines, each producing their own PER metric.

`karelian_terms.csv` rows are `orthography,/ipa/` with no header, IPA wrapped in `/.../`.

### Rule-based pipeline (`g2p.py`)

`g2p(word)` applies six string-rewrite rules in a **fixed, order-dependent** sequence —
each rule assumes the string shape left by the previous one:

```
apply_stress → gemination → mappings → palatilization → desyllabification → velarization
```

- `apply_stress` prepends `ˈ` (primary stress only, no secondary stress support).
- `gemination` must run on *raw orthography* (before `mappings`) since it only detects
  literal doubled letters.
- `mappings` is a flat orthography→IPA character substitution table (e.g. `ä→æ`, `v→ʋ`).
  Note `v` maps to `ʋ`, which is *not* in the `consonants` list used by palatalization —
  so `ʋ` can never trigger palatalization, by design of the table.
- `palatilization` inserts `ʲ` after consonants (except `h,k,j,b,r,ʃ,p,m`) before front
  vowels — must run *after* `mappings` since it checks against post-mapping IPA vowels.
- `desyllabification` marks the second vowel of a VV sequence non-syllabic (only handles
  pairwise runs, not 3-vowel sequences).
- `velarization` converts orthographic `n` before `g` to `ŋ` — note `g` itself is *never*
  mapped to IPA `ɡ` anywhere in the pipeline (a known gap).

Known modeling limitations (no secondary stress, no `š`→`ʒ` voicing alternation, `g`≠`ɡ`,
only pairwise gemination/desyllabification) are documented in detail with real-word traces
in `README_rule_based_g2p.md` — read that file before modifying `g2p.py`'s rules, since
each limitation is a deliberate/known gap, not an oversight.

`min_edit_distance` in `g2p.py` computes Levenshtein distance normalized by gold-string
length (PER), after stripping the wrapping `/.../` from both predicted and gold strings.

### Neural pipeline (`karelian_g2p_seq2seq.ipynb` / `learning_curve.py`)

Character-level seq2seq: `Encoder` (bidirectional GRU) → `Attention` (Bahdanau-style,
concatenative) → `Decoder` (GRU with attention context) → `Seq2Seq` wrapper handling
teacher forcing during training and greedy decoding at inference. `learning_curve.py`
duplicates this architecture verbatim from the notebook (kept in sync manually — there's
no shared module) so it can retrain from scratch across `FRACTIONS` of the training pool
while holding a fixed val/test split (seeded via `SEED = 42`).

`best_g2p_model.pt` is the notebook's checkpoint (best validation loss during training),
not something `learning_curve.py` reads — that script trains fresh models per data
fraction and never persists them.

Evaluation metric (PER + exact-match accuracy) is the same Levenshtein-based approach as
the rule-based pipeline, reimplemented locally in both the notebook and `learning_curve.py`.

### Output artifacts

- `karelian_output.csv` — sorted rule-based predictions vs. gold, from `g2p.py`.
- `neural_karelian_output.csv` — equivalent output from the notebook's neural model.
- `learning_curve_results.csv` / `learning_curve.png` — PER vs. training-data-fraction,
  from `learning_curve.py` / `plot_learning_curve.py`.

These CSVs/PNGs are regenerated by running the corresponding script — treat them as build
output, not source of truth.
