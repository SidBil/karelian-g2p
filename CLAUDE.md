# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Grapheme-to-phoneme (G2P) conversion for three closely related, low-resource Finnic
varieties — **Karelian**, **Livonian**, and **Ingrian** — using orthography/gold-IPA pairs
scraped from Wiktionary (`data/<language>_terms.csv`). It currently contains two parallel,
independently-scored approaches:

1. A **rule-based baseline** (`src/g2p.py`), Karelian-only — a fixed pipeline of
   hand-written string-rewrite rules. Scores 4.43% average phoneme error rate (PER).
2. A **neural seq2seq model** (architecture in `src/g2p_model.py`, originally prototyped
   in `src/karelian_g2p_seq2seq.ipynb`), covering all three languages — a
   bidirectional-GRU-encoder + attention-GRU-decoder trained character-by-character. On
   Karelian it only overtakes the rule-based baseline once trained on ~90% of the (small,
   552-example) training pool; on the larger Livonian and Ingrian sets it clears a lower
   dev PER with far less data.

A **cross-lingual transfer** experiment (`src/cross_lingual_eval.py`) evaluates all three
trained models against all three languages' test sets (9 combinations total) as a
zero-shot transfer baseline — see `paper/research_plan.md` Q3. A companion **joint
multilingual** experiment (`src/train_multilingual.py`) trains a single model on all three
languages' pooled training data instead, as the in-domain counterpart to that zero-shot
grid.

A **rule-guided ("residual") hybrid** (`src/train_residual_model.py`,
`src/residual_learning_curve.py`) feeds the rule-based output into the same seq2seq
encoder as a hint — the NN+HR experiment, `paper/research_plan.md` Q2.

This is an active research project — see `paper/research_plan.md` for the full research
questions (rule-based vs. neural vs. LLM-induced rules, hybrid/residual models,
cross-lingual transfer between the three languages and from Finnish) and `paper/abstract.md`
for the current framing. Not all of that is implemented yet; check `research_plan.md`
before assuming a described experiment already exists in code.

There is no package manifest (no `requirements.txt` / `pyproject.toml`) — dependencies
live only in `venv/`.

## Running things

Use the local venv's Python directly (it already has torch, matplotlib, requests, bs4,
jupyter, etc. installed). **The notebook is the one holdout for path handling** — every
`src/*.py` script now resolves `data/`, `outputs/`, `figures/`, and `models/` relative to
the repo root (via `Path(__file__).resolve().parent.parent`), so they can all be run from
anywhere. `src/karelian_g2p_seq2seq.ipynb` still uses a bare relative `DATA_PATH =
Path("karelian_terms.csv")` and writes `best_g2p_model.pt` / `neural_karelian_output.csv`
into the current directory — it must be run **with `data/` as the working directory**, and
its outputs then need to be moved into `outputs/`/`models/` by hand.

```bash
# rule-based pipeline (reads data/karelian_terms.csv, writes outputs/karelian_output.csv)
venv/bin/python src/g2p.py

# (re)generate data/splits/<language>_{train,dev,test}.csv from data/<language>_terms.csv
# -- only needed once, or again if a language's raw *_terms.csv changes
venv/bin/python src/make_splits.py

# neural learning-curve sweep, any language (reads data/splits/, needs matplotlib --
# see the DYLD_LIBRARY_PATH note below)
DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib venv/bin/python src/learning_curve.py --language karelian
DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib venv/bin/python src/plot_learning_curve.py --language karelian

# train + save one final model per language to models/<language>_g2p_model.pt
venv/bin/python src/train_g2p_model.py                  # all three languages
venv/bin/python src/train_g2p_model.py --language ingrian  # just one

# 3x3 cross-lingual transfer grid -- needs all three models/*_g2p_model.pt to exist first
venv/bin/python src/cross_lingual_eval.py

# joint multilingual model: one model trained on all three languages' pooled
# train splits, evaluated separately on each language's test split
venv/bin/python src/train_multilingual.py

# rule-guided (residual) model: input = orthography + "|" + g2p.py's prediction.
# Writes models/<language>_residual_g2p_model.pt and outputs/residual_results.csv
# (rule-only vs plain-NN vs guided PER/acc side by side). plain-NN column needs
# models/<language>_g2p_model.pt to already exist.
venv/bin/python src/train_residual_model.py [--language ingrian]

# rules-on vs rules-off learning curve (same train subsets for both conditions);
# the plot script needs matplotlib, hence the DYLD prefix
DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib venv/bin/python src/residual_learning_curve.py --language karelian
DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib venv/bin/python src/plot_residual_learning_curve.py --language karelian

# dataset cleanup / scraping (defaults target data/<language>_terms.csv)
venv/bin/python src/process_karelian.py
venv/bin/python src/scrape_wiktionary.py

# interactive neural notebook (Karelian-only prototype; still expects cwd = data/ for
# DATA_PATH to resolve, and does not share code with src/g2p_model.py)
cd data && ../venv/bin/jupyter notebook ../src/karelian_g2p_seq2seq.ipynb
```

There is no test suite. `g2p.py`, `learning_curve.py`, `train_g2p_model.py`, and the
notebook are self-scoring: running them *is* the verification step (they print
PER/accuracy on the data they run over).

**This venv's matplotlib import needs a library path workaround.** Homebrew's Python 3.14
build expects a newer `libexpat` than the macOS system one ships (`pyexpat` fails with
`Symbol not found: _XML_SetAllocTrackerActivationThreshold`), which matplotlib's
`font_manager` pulls in transitively via `plistlib`. `expat` is now installed via Homebrew
(keg-only, so it's not on the default library path) — prefix any matplotlib-importing
command with `DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib`:

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib venv/bin/python src/learning_curve.py --language karelian
DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib venv/bin/python src/plot_learning_curve.py --language karelian
```

This sometimes appears unnecessary because matplotlib only hits the broken import path
when it needs to rebuild its macOS font-list cache — once that cache is warm the crash
doesn't recur for a while, then resurfaces unpredictably. Always use the prefix for
matplotlib-importing scripts rather than relying on that. `src/g2p_model.py`,
`train_g2p_model.py`, and `cross_lingual_eval.py` avoid importing matplotlib entirely, so
they don't need it.

## Architecture

### Data flow

`scrape_wiktionary.py` → `data/<language>_terms.csv` (raw orthography,IPA pairs, one file
per language: karelian, livonian, ingrian) → `process_karelian.py` (filters bad rows) →
`make_splits.py` writes the fixed 80/10/10 split to
`data/splits/<language>_{train,dev,test}.csv` → consumed by the rule-based pipeline
(Karelian only, which still reads the whole `data/karelian_terms.csv` rather than a split)
and the neural pipeline (all three languages, which reads only from `data/splits/`).
`process_karelian.py`'s filtering has only been applied to the Karelian file so far —
Livonian/Ingrian have not been checked for the same hyphen-prefix/single-char issues.

`data/*_terms.csv` and `data/splits/*.csv` rows are both `orthography,/ipa/` with no
header, IPA wrapped in `/.../` — the split files are just a partition of the same format,
written once by `make_splits.py` (seed 42) rather than re-derived by reshuffling on every
run. **Regenerate `data/splits/` if a language's raw `*_terms.csv` changes**, and retrain
any model whose split changed underneath it.

### Rule-based pipeline (`src/g2p.py`)

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
each limitation is a deliberate/known gap, not an oversight. This rule-based approach has
not been extended to Livonian or Ingrian.

`min_edit_distance` in `g2p.py` computes Levenshtein distance normalized by gold-string
length (PER), after stripping the wrapping `/.../` from both predicted and gold strings.

### Neural pipeline (`src/g2p_model.py` + `learning_curve.py` / `train_g2p_model.py` / `cross_lingual_eval.py`)

`src/g2p_model.py` is the **single shared module** for the character-level seq2seq
architecture: `Encoder` (bidirectional GRU) → `Attention` (Bahdanau-style, concatenative)
→ `Decoder` (GRU with attention context) → `Seq2Seq` wrapper handling teacher forcing
during training and greedy decoding at inference, plus vocab building, the
`G2PDataset`/collate helpers, and PER/accuracy evaluation (`evaluate`, `transcribe`,
`levenshtein`). It has no matplotlib dependency. The scripts below import from it rather than
redefining any of this:

- `learning_curve.py` — retrains from scratch across `FRACTIONS` (10%–100%) of a
  language's persisted train split, evaluating dev PER at each fraction; the test split is
  never touched. Only this script (plus `plot_learning_curve.py`) imports matplotlib, for
  plotting the curve.
- `train_g2p_model.py` — trains **one** model per language (all three by default, or
  `--language`) on the full persisted train split, checkpointing on best dev loss, then
  evaluates it on that language's held-out test split. Saves state dict *and* vocab
  (`src_stoi`/`src_itos`/`tgt_stoi`/`tgt_itos`) to `models/<language>_g2p_model.pt` — the
  vocab has to be saved because it's required to run inference on a checkpoint later, and
  each language's model has a different character vocabulary.
- `cross_lingual_eval.py` — loads all three `models/<language>_g2p_model.pt` checkpoints
  and evaluates each one on all three languages' test splits (3×3 = 9 combinations),
  writing `outputs/cross_lingual_results.csv` (aggregate PER/accuracy per combination) and
  `outputs/cross_lingual_failures.csv` (every non-exact-match prediction across all 9
  combinations, for inspecting individual error cases). Each model uses its own saved
  vocab, so a foreign language's characters that the model never trained on fall back to
  `<unk>` — that's the actual zero-shot transfer condition being measured, not a bug.
- `train_multilingual.py` — the in-domain counterpart to `cross_lingual_eval.py`: trains
  **one** model on the pooled train+dev splits of all three languages (a shared vocab
  covers every language's characters, so no `<unk>` fallback), then evaluates it
  separately on each language's held-out test split. Saves to
  `models/multilingual_g2p_model.pt` and `outputs/multilingual_results.csv`. Compare its
  per-language test PER against the diagonal of `outputs/cross_lingual_results.csv` (each
  language's own monolingual model) to see whether joint training helps or hurts relative
  to training separately.

- `train_residual_model.py` — NN+HR hybrid. Runs `g2p.py`'s rule-based `g2p()` over every
  word, then trains the same architecture on `orthography|rule_prediction → gold IPA`
  (`SEP = "|"`, built by `guided_pairs()`). `g2p.py` is Karelian-only, so on
  Livonian/Ingrian the hint is mostly wrong (rule-only test PER ≈ 38% / 11%) — that's
  part of what's being measured, not a bug. Note it imports `g2p.py` at module load,
  so changes to the rules change this model's inputs: retrain after editing `g2p.py`.
- `residual_learning_curve.py` — reuses `train_and_eval()` from `learning_curve.py`
  (generic over any `(src, tgt)` string pairs) to train plain and guided models on the
  *same* subset at each fraction; plotted by `plot_residual_learning_curve.py`.

`models/best_g2p_model.pt` is the original notebook's checkpoint (best validation loss
during training) for Karelian only — a bare state dict with no vocab attached, so it can't
be loaded for inference the way `train_g2p_model.py`'s checkpoints can. The notebook
(`src/karelian_g2p_seq2seq.ipynb`) remains a self-contained, Karelian-only prototype; it
does not import from `g2p_model.py` and was the original (now superseded) source the
architecture was copied from.

### Output artifacts

- `outputs/karelian_output.csv` — sorted rule-based predictions vs. gold, from `g2p.py`.
- `outputs/neural_karelian_output.csv` — equivalent output from the notebook's neural model.
- `outputs/<language>_learning_curve_results.csv` / `figures/<language>_learning_curve.png`
  — dev PER vs. training-data-fraction per language, from `learning_curve.py` /
  `plot_learning_curve.py`.
- `outputs/learning_curve_run.log` — console log from a past learning-curve sweep across
  all three languages; referenced in `paper/research_plan.md` for the summary PER table.
- `outputs/cross_lingual_results.csv` — the 3×3 (trained_on, tested_on) PER/accuracy grid
  from `cross_lingual_eval.py`; `outputs/cross_lingual_failures.csv` — every non-exact-match
  prediction underlying that grid, from the same script.
- `outputs/multilingual_results.csv` — per-language test PER/accuracy for the single joint
  model, from `train_multilingual.py`.
- `outputs/residual_results.csv` — rule-only / plain-NN / rule-guided comparison per
  language, from `train_residual_model.py`;
  `outputs/<language>_residual_learning_curve_results.csv` +
  `figures/<language>_residual_learning_curve.png` — rules-on vs rules-off curves.
  `outputs/nn_failures.csv` — error cases from a plain-NN run, for inspection.
- `models/<language>_g2p_model.pt` — one final trained checkpoint per language (state dict
  + vocab), from `train_g2p_model.py`. `models/multilingual_g2p_model.pt` — the single
  joint checkpoint (state dict + shared vocab), from `train_multilingual.py`.

These CSVs/PNGs/logs/checkpoints are regenerated by running the corresponding script —
treat them as build output, not source of truth. `data/splits/*.csv` is the one exception:
it's committed as a fixed, reviewable artifact of `make_splits.py`'s seed-42 partition, not
something to casually regenerate (regenerating it invalidates every trained model's
train/dev/test membership).
