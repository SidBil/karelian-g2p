# Finnic G2P — Research Plan & Summary

*Compiled 2026-08-23, from the RQs/task list you gave, the `karelian` repo as it stands, and a first pass on the reading list.*

## 1. Framing

The project builds and compares grapheme-to-phoneme (G2P) systems for three closely related, low-resource Finnic varieties — **Karelian**, **Livonian**, and **Ingrian** — using orthography/IPA pairs scraped from Wiktionary. You already have a working rule-based Karelian system and neural learning-curve results for all three languages; what's missing is an LLM-induced-rules approach, guided/hybrid neural variants, a cross-lingual transfer study, and the shared evaluation/writing infrastructure to compare all of it fairly.

## 2. Research Questions

**Q1 — How well does each approach do G2P (rule-based vs. neural vs. LLM-induced rules)?**
This is the core bake-off: same three languages, same train/dev/test splits, PER + WER (see §7) for the hand-written rules (`g2p.py`), the neural seq2seq model, and whatever an LLM produces when asked to induce G2P rules from examples.

**Q2 — Can a rule-based or LLM-rule signal guide/residual-correct the neural network?**
The hypothesis is that a hybrid (NN + HR, NN + LR) trains faster, needs less data, or generalizes better than a plain NN — e.g. the network learns a residual correction on top of the rule-based output rather than the mapping from scratch. The existing Karelian learning curve is the natural baseline to beat: it's noisy and needs ~90% of only 552 training examples to beat the rule-based system's 4.43% PER, so "less data needed" has a concrete, already-measured target.

**Q3 — Can we do cross-lingual transfer?**
Not currently represented as its own task in your list (see §6) — worth adding explicitly. With three related low-resource orthographies in hand plus Finnish/Estonian as higher-resource relatives, there are two transfer directions worth testing: (a) among the three target languages (train on Livonian's larger 2,483-pair set, fine-tune/test on Karelian's 690), and (b) from a well-resourced relative (Finnish, which Epitran/eSpeak already cover) into the target languages.

**Q4 — How good are the LLM-induced rules specifically?**
A narrower cut of Q1: not just LLM-rules' raw PER/WER, but whether the rules it induces are inspectable/sensible (comparable to the hand-written rule README's rule-by-rule trace), how they fail, and whether they capture things the hand-written rules explicitly punt on (secondary stress, š→ʒ voicing — see §4).

## 3. Data & Languages in Hand

All three datasets come from `scrape_wiktionary.py` (Wiktionary category-page scraping, `orthography,ipa` pairs) and are already sitting in `data/`:

| Language | ISO 639-3 | Unique pairs | Rule-based baseline | Neural dev PER @ 100% data | Neural dev PER @ 10% data |
|---|---|---:|---|---:|---:|
| Karelian | krl | 690 | **4.43%** (whole-corpus PER, `g2p.py`) | 4.26% | 32.66% |
| Livonian | liv | 2,483 | none yet | 0.71% | 4.65% |
| Ingrian | izh | 8,241 | none yet | 2.68% | 7.12% |

(Neural numbers are from `outputs/learning_curve_run.log` / `*_learning_curve_results.csv` — dev-set PER from `learning_curve.py`, fixed 80/10/10 split, seed 42, held-out test set never touched.)

Two things worth flagging for the paper's data section (task 8):

- **Karelian's curve is small and noisy** — non-monotonic PER as training data increases (e.g. 5.07% at 50%, 5.88% at 60%, then down to 3.04% at 90%, back up to 4.26% at 100%), because the training pool is only 552 examples. That noise is itself a data point for Q2: a guided/residual approach has the clearest room to help exactly where data is scarcest.
- **Ingrian is by far the largest set** (8,241 pairs) and gives the smoothest curve — useful as the "high(er)-resource-in-context" anchor for the cross-lingual transfer experiments in Q3.

`process_karelian.py` (hyphen-prefixed and single-character entries dropped) has only been applied to Karelian so far — worth confirming Livonian/Ingrian don't need the same cleaning pass before you build on them further.

## 4. What's Already Built vs. What's Net-New

**Already built (`karelian` repo):**
- Rule-based Karelian G2P: `g2p.py`, six ordered string-rewrite rules (stress → gemination → mappings → palatalization → desyllabification → velarization), 4.43% PER, with known/documented gaps in `README_rule_based_g2p.md` (no secondary stress, no š→ʒ voicing alternation, `g` never mapped to IPA `ɡ`, only pairwise gemination/desyllabification handled). That README is genuinely useful raw material for the rule-based section of the paper — it already has real-word traces through every rule.
- Neural seq2seq: bidirectional-GRU encoder + Bahdanau-attention GRU decoder, character-level (`karelian_g2p_seq2seq.ipynb`, duplicated verbatim into `learning_curve.py` since there's no shared module — worth refactoring into one before you add more languages/approaches). `learning_curve.py` already takes `--language`, so extending it to Livonian/Ingrian (already done) and to new languages later is low-friction.
- Learning-curve sweeps (10%–100% of training pool) already run and logged for all three languages.

**Net-new (not started yet):**
- LLM-induced-rules pipeline (task 2 / "LR").
- Guided/hybrid pipelines, NN+HR and NN+LR (tasks 4–5 / Q2).
- Cross-lingual transfer code (Q3 — currently has no task number; see §6).
- A shared evaluation harness: right now PER is computed independently in `g2p.py` and `learning_curve.py` (two separate re-implementations of Levenshtein-based PER), neither script records training time or inference time, and "WER" isn't computed anywhere yet — `learning_curve.py` reports exact-match accuracy, whose complement (1 − accuracy) is the natural word error rate, but nothing currently logs it under that name or persists it alongside PER, timing, and the model artifact in one place. Since the brief says *every* task should save the model, train time, inference time, PER, and WER, standardizing this into one small results-schema module before running the other four pipelines will save you from reconciling five different logging formats in October.
- Rule-based baselines for Livonian and Ingrian (currently Karelian-only).

**Note on `panini-probing`:** this is a separate, unrelated project in your account (Hindi kāraka/vibhakti probing on mBERT/MuRIL/XLM-R — a different paper entirely). It doesn't feed into the G2P work directly, though its layer-wise linear-probing methodology is a plausible technique to borrow *if* Q2's "guide the network" ends up meaning "intervene on internal representations" rather than "combine outputs" — flagging as an idea, not a dependency.

## 5. Baseline-Tool Check (relevant to Task 1 / "HR")

You mentioned trying Epitran, eSpeak, and GE2PE for Finnic. Checked current language coverage:

- **Epitran** and **eSpeak-ng** both cover only **Finnish and Estonian** among Finnic languages — neither has Karelian, Livonian, or Ingrian rule files. [[Epitran](https://aclanthology.org/L18-1429.pdf)] [[eSpeak-ng languages](https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md)]
- **GE2PE** is a **Persian-specific** end-to-end G2P system (EMNLP 2024 Findings, built around a homograph-disambiguation dataset called HomoRich) — it isn't multilingual and won't apply to Finnic at all. [[GE2PE](https://aclanthology.org/2024.findings-emnlp.196/)] This is very likely a name mix-up worth resolving before Task 1 starts.

Two ways to still get value from "existing rule-based tools" for Task 1:
1. **Run Epitran/eSpeak on Finnish as an approximate baseline**, since Karelian orthography is close enough to Finnish that this becomes a genuine (if imperfect) point of comparison — and doubles as Q3 transfer-source data, since Finnish is the well-resourced relative.
2. **Add Karelian/Livonian/Ingrian rule files to Epitran** rather than just running it — Epitran's whole design is per-language hand-written rule files, so this is close to what `g2p.py` already does, just in Epitran's format. That would make the "rule-based" arm of Q1 a fairer multi-tool comparison instead of one bespoke script.
3. Consider swapping GE2PE for something that's actually multilingual: **Gi2Pi** (rule-based, index-preserving, built for exactly this kind of low-resource/Indigenous-language use case — see §6) or **transphone** (`xinjli/transphone`, phoneme tokenizer + G2P claiming coverage of ~8,000 languages) are both closer fits than GE2PE.

## 6. Reading List, Organized

Your one-line notes are in quotes; corrections/additions from a first check are unmarked.

**Framing & evaluation precedent**
- [1] *Fast, Not Fancy: Rethinking G2P with Rich Data and Rule-Based Models* (2025) — worth a correction: this is a **Persian-focused** paper about building a homograph-annotated dataset (HomoRich) and a faster homograph-aware eSpeak variant (HomoFast eSpeak), not a broad multilingual model comparison. Still useful for *how* to build/report a dataset-plus-tool-improvement paper, less useful as an eval-methodology template for a multi-language, multi-approach bake-off — that role may be better filled by [8] (SIGMORPHON 2020 shared task), which is explicitly a multilingual G2P benchmark with a standard PER/WER protocol.
- [3] Tamil dialectal phonetic transcription tool — good framing precedent for "a phonetic transcription tool for an under-resourced/variety-rich language," matches your framing plan.
- [4] Plains Cree speech synthesizer — good precedent for how to write up the target language's phonology section.

**Rule-based**
- [2] Japanese rule-based G2P + multilingual NE/IPA dataset.
- [14] Letter-to-sound rules for accented lexicon compression (older; general LTS-rule lineage, useful historical grounding).
- [15] **Gi2Pi** — rule-based, index-preserving G2P, built at NRC Canada for Indigenous-language tooling (ReadAlong Studio). This is less "background reading" and more a real candidate tool for Task 1 — directly relevant given the GE2PE mismatch above.

**Neural / multilingual neural**
- [7] ByT5 for massively multilingual G2P.
- [8] SIGMORPHON 2020 shared task on multilingual G2P — the closest thing to a standard eval protocol in this literature; worth reading before finalizing your PER/WER harness.
- [10] Massively multilingual neural G2P.
- [11] G2P models for (almost) any language.
- [12] Token-level ensemble distillation for G2P.
- [13] Transformer-based G2P.

**Tools**
- [9] Epitran — already checked (§5): no Finnic-minority coverage, but its per-language rule-file design is a plausible extension path.

**LLM-induced rules**
- [5] C-LARA / Kanak languages, built on ChatGPT — the closest existing precedent for Task 2 ("LR"); read this one first, before designing the LLM-rules pipeline, since it's the one paper on your list that's actually doing what Q4 asks about.

**Survey**
- [6] Survey of G2P conversion methods — good general orientation for the literature review (Task 6).

## 7. Task List Mapped to Research Questions

| # | Task | Tag | Primary RQ(s) | Notes |
|---|---|---|---|---|
| 1 | Rule-based pipeline | HR | Q1, Q4 (contrast) | Extend `g2p.py`-style baseline to Livonian/Ingrian; resolve tool question in §5 |
| 2 | LLM pipeline | LR | Q1, Q4 | Read [5] first |
| 3 | Neural network pipeline | NN | Q1, Q2 (baseline) | Refactor shared model code out of the notebook; add timing + WER logging |
| 4 | NN + HR pipeline | — | Q2 | Residual/guided design against the rule-based output |
| 5 | NN + LR pipeline | — | Q2, Q4 | Residual/guided design against LLM-induced rules |
| — | *(ungrouped)* Cross-lingual transfer | — | **Q3** | Not currently a numbered task — needs one, likely built on top of 3/4/5 once those exist |
| 6 | Literature review | — | supports all | Read [1],[6],[8],[9],[15] early — they double as methodology inputs, not just citations |
| 7 | Data exploration script | — | supports Q3, Task 8 | Also a natural place to check/apply `process_karelian.py`-style cleaning to Livonian/Ingrian |
| 8 | Data & statistics section | — | writing | Feed directly from Task 7 + the table in §3 |

## 8. Proposed Timeline

One correction first: **Aug 23 → Oct 11 is 49 days, i.e. 7 weeks, not 10.** If Oct 11 is a hard deadline, the plan below assumes 7 weeks; if you actually have 10 weeks of work time, either the deadline is later (~Oct 31) or the "10 weeks" figure was from a different start date — worth pinning down before committing to a week-by-week plan.

Draft 7-week version:

- **Week 1 (Aug 24–30):** Resolve the tool question (§5), build the shared eval harness (PER, WER, timing, model-artifact schema — §4), start the data exploration script (Task 7).
- **Week 2 (Aug 31–Sep 6):** Extend the rule-based (HR) pipeline to Livonian and Ingrian; start the data/statistics section (Task 8) from Task 7's output.
- **Week 3 (Sep 7–13):** Refactor + formalize the NN pipeline (checkpointing, timing, WER) across all three languages; start the literature review (Task 6).
- **Week 4 (Sep 14–20):** Build the LLM-rules (LR) pipeline; read [5] first.
- **Week 5 (Sep 21–27):** Build the guided/hybrid pipelines, NN+HR and NN+LR (Q2).
- **Week 6 (Sep 28–Oct 4):** Cross-lingual transfer experiments (Q3 — needs its own task, see §7); consolidate all results tables across the five pipelines.
- **Week 7 (Oct 5–11):** Writing sprint — assemble the full paper, error analysis, polish lit review and data sections, buffer for reruns.

## 9. Open Questions / Decisions Needed

1. **GE2PE mismatch** — it's Persian-only; pick a real substitute (Gi2Pi and transphone are the strongest candidates from the reading list/search) or reframe Task 1 around Epitran-rule-file extension.
2. **7 vs. 10 weeks** — confirm the actual deadline/duration.
3. **Q3 has no task number** — cross-lingual transfer needs an explicit pipeline task; right now it would fall through the cracks of the 8-item list.
4. **WER definition** — proposing "WER = 1 − exact-match word accuracy" (already computed as `dev_acc` in `learning_curve.py`) rather than a new metric; confirm that's what you mean by WER for a word-level (not sentence-level) G2P task.
5. **Rule-based coverage** — confirm you want HR extended to Livonian/Ingrian (currently Karelian-only), since Task 1 as worded ("a pipeline to evaluate all rules") implies all three languages.
