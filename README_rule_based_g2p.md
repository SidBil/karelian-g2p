# Rule-based Karelian G2P (`g2p.py`)

`g2p.py` converts Karelian orthography to IPA transcription using a fixed
pipeline of hand-written string-rewriting rules, then scores itself against
gold transcriptions scraped from Wiktionary (`karelian_terms.csv`). It's the
non-neural baseline for the seq2seq model in
`karelian_g2p_seq2seq.ipynb` — it currently scores **4.43% average phoneme
error rate (PER)** across the dataset.

## Pipeline

For every input word, `g2p()` (`g2p.py:95`) runs six rules in this fixed
order, each rewriting the string produced by the previous one:

```python
def g2p(word):
    word = apply_stress(word)
    word = gemination(word)
    word = mappings(word)
    word = palatilization(word)
    word = desyllabification(word)
    word = velarization(word)
    return f"/{word}/"
```

Below, each rule is explained with a real word traced step by step. Reading
top to bottom in a row shows exactly what each rule changed.

### 1. `apply_stress` — primary stress on the first syllable

```python
def apply_stress(word):
    word = list(word)
    word.insert(0, "ˈ")
    return "".join(word)
```

Karelian (like Finnish) stresses the initial syllable, so this unconditionally
prepends `ˈ` to the word. It runs first because every later rule assumes the
stress mark is already sitting at index 0 and skips over it implicitly (it's
never a vowel/consonant, so it's ignored by the rest of the rules).

```
heikko -> ˈheikko
```

Note this rule only ever inserts *primary* stress. It has no notion of
*secondary* stress (`ˌ`) on later syllables of compounds — see Limitations.

### 2. `gemination` — doubled letters become a length mark

```python
def gemination(word):
    word = list(word)
    for i in range(len(word) - 1):
        if word[i] == word[i + 1]:
            word[i + 1] = "ː"
    return "".join(word)
```

Scans adjacent character pairs; if two are identical, the second is replaced
with the IPA length mark `ː`. This runs on the *raw orthography* (before
`mappings`), so it only catches literal doubled letters like `kk`, `aa`.

```
aakkua -> ˈaakkua -> ˈaːkːua      (aa -> aː,  kk -> kː)
heikko -> ˈheikko -> ˈhei kːo     (kk -> kː; the "ei" here is not doubled, untouched)
```

Because it just compares neighboring characters left to right, three-in-a-row
would still only geminate the second character (there's no run of 3 identical
letters in the data, so this doesn't come up in practice).

### 3. `mappings` — orthography-to-IPA symbol substitution

```python
def mappings(word):
    mappings = {"ä": "æ", "ö": "ø", "š": "ʃ", "č": "t͡ʃ",
                "ž": "ʒ", "v": "ʋ", "ʹ": "ʲ", "a": "ɑ"}
    ...
```

A straight character-for-character lookup table, applied after gemination
(so a doubled `a` → `aː` from step 2 still gets its remaining `a` mapped to
`ɑ` here). Everything not in the table (`p, b, t, d, k, h, l, m, n, r, s, ...`)
is assumed to already be valid IPA and passes through unchanged.

```
aakkua  -> ˈaːkːua  -> ˈɑːkːuɑ      (a -> ɑ, both occurrences)
heinä   -> ˈheinä   -> ˈheinæ       (ä -> æ)
dekabrʹa -> ˈdekabrʹa -> ˈdekɑbrʲɑ  (a -> ɑ, ʹ -> ʲ)
šygyžkuu -> ˈšygyžkuu -> ˈʃygyʒkuu  (š -> ʃ, ž -> ʒ)
```

The `dekabrʹa` example is worth calling out: the apostrophe `ʹ` is a
Cyrillic-derived orthographic palatalization marker some Karelian spelling
conventions use directly, so it's mapped straight to IPA `ʲ` here — this is a
*different* mechanism from the automatic palatalization in the next step.

Note `v` is mapped to `ʋ` (a labiodental approximant, not the `v` in the
`consonants` list used by the next rule) — meaning `ʋ` can never trigger
palatalization, since the palatalization rule only checks against the
original `consonants` list, which contains `v`, not `ʋ`.

### 4. `palatilization` — automatic palatalization before front vowels

```python
def palatilization(word):
    word = list(word)
    for i in range(len(word)-1):
        if word[i] in consonants and word[i] not in ["h","k","j","b","r","ʃ","p","m"]:
            if word[i+1] in ["i","y","e","ø","æ"]:
                word.insert(i+1, "ʲ")
    return "".join(word)
```

For every consonant immediately followed by a front vowel (`i, y, e, ø, æ`),
insert `ʲ` — *unless* the consonant is one of `h, k, j, b, r, ʃ, p, m`, which
are exempted and stay plain.

```
kieli -> ˈkielʲi (after mappings) -> ˈkielʲi
```
Traced fully: `k` precedes the front vowel `i` in `kie` but **k is exempt**,
so it stays bare `k`; `l` precedes `i` in `li` and **is** palatalized:
```
kieli -> /ˈkie̯lʲi/    (matches gold exactly)
```

```
ahašhengine -> ˈɑhɑʃhengine -> ˈɑhɑʃhenginʲe
```
Here `n` before the final `e` gets palatalized (`n` is not on the exemption
list) → `nʲe`.

```
abuiäni -> ˈɑbuiæni -> ˈɑbuiænʲi
```
`n` before `i` → `nʲi`.

`kirja` stays `ˈkirjɑ` with no palatalization at all: `k` precedes `i` but is
exempt, and neither `r` nor `j` precedes a front vowel (`j` and `ɑ` aren't in
the front-vowel trigger set). This exemption is why words with `k` or `h`
before a front vowel (`kieli`, `heinä`) never get a palatalized `kʲ`/`hʲ`
from this rule, which is phonologically correct for Karelian/Finnish, where
velars and a few other consonants pattern differently from palatalizable
coronals.

### 5. `desyllabification` — second vowel of a VV sequence becomes non-syllabic

```python
def desyllabification(word):
    desyllabified = {"i": "i̯", "y": "y̯", "u": "u̯", "e": "e̯",
                      "ø": "ø̯", "o": "o̯", "æ": "æ̯", "ɑ": "ɑ̯"}
    for i in range(len(word)-1):
        if word[i] in vowels and word[i+1] in vowels:
            word[i+1] = desyllabified[word[i+1]]
    return "".join(word)
```

When two vowels sit next to each other, the second is marked non-syllabic
(the under-arc diacritic `◌̯`) — i.e. treated as gliding off the first vowel
rather than forming its own syllable, modeling Karelian diphthongs.

```
aida    -> ˈɑidɑ    -> ˈɑi̯dɑ     (a+i -> a + i̯)
koira   -> ˈkoirɑ   -> ˈkoi̯rɑ    (o+i -> o + i̯)
aakkua  -> ˈɑːkːuɑ  -> ˈɑːkːuɑ̯   (u+a -> u + ɑ̯)
abuiäni -> ˈɑbuiænʲi -> ˈɑbui̯ænʲi (u+i -> u + i̯; note æ afterward is
                                    untouched since i̯ is no longer a plain
                                    vowel character for the *next* pair check)
```

Because the loop only ever looks at immediate left/right neighbors and only
mutates `word[i+1]`, three-vowel runs only get their *second* vowel
desyllabified, not the third — visible in `abuiäni`, where gold marks a
*separate* syllable break with secondary stress (`ˌiæ̯`) that this rule can't
produce (see Limitations).

### 6. `velarization` — /n/ before /g/ becomes /ŋ/

```python
def velarization(word):
    word = list(word)
    for i in range(len(word)-1):
        if word[i] == "n" and word[i+1] == "g":
            word[i] = "ŋ"
    return "".join(word)
```

Runs last, on literal lowercase `n` followed by literal `g` (note: this is
the orthographic `g`, not IPA `ɡ` — nothing upstream ever converts `g` to
`ɡ`, so it always stays as plain `g` in the output; see Limitations).

```
ahingo -> ˈɑhingo -> ˈɑhiŋgo
ahašhengine -> ˈɑhɑʃhenginʲe -> ˈɑhɑʃheŋginʲe
```

## Scoring: phoneme error rate

`min_edit_distance` (`g2p.py:60`) computes Levenshtein (character) edit
distance between the predicted and gold IPA strings (after stripping the
wrapping `/ .../` slashes from both), normalized by the length of the gold
string — the same PER metric used to evaluate the neural model. The main
script body applies `g2p()` to every word in `karelian_terms.csv`, computes
PER per word, sorts ascending (best matches first), prints everything, prints
the corpus-average PER, and writes the full sorted comparison to
`karelian_output.csv`.

## Limitations (visible directly in the data)

- **No secondary stress.** `apply_stress` only ever inserts one `ˈ` at
  position 0. Compounds that gold-mark a secondary stress are systematically
  wrong on stress alone, e.g.:
  ```
  abuiäni  -> predicted /ˈɑbui̯ænʲi/   gold /ˈɑbuˌiæ̯nʲi/
  šygyžkuu -> predicted /ˈʃygyʒkuː/   gold /ˈʃyɡyʃˌkuː/
  ```
- **No voicing alternation.** `š` always maps to `ʃ`, never the voiced `ʒ`
  it sometimes surfaces as between vowels/voiced consonants in gold, e.g.
  `šygyžkuu` above (gold `ʃ` for the second `š`, predicted keeps it as `ʒ`
  from the unrelated `ž`) and `ahašhengine` → predicted `/ˈɑhɑʃheŋginʲe/`
  vs. gold `/ˈɑhɑʒˌheŋɡinʲe/` (gold voices the `š` to `ʒ` before `h`).
- **`g` is never mapped to IPA `ɡ`.** Compare the same `ahašhengine` example:
  predicted keeps plain `g` (`heŋginʲe`), gold uses proper IPA `ɡ`
  (`heŋɡinʲe`). This is a straightforward gap in the `mappings` table rather
  than a phonological modeling limitation.
- **Only pairwise gemination/desyllabification.** Both rules look one
  character ahead at a time, so they can't correctly handle 3-vowel or
  3-consonant runs beyond desyllabifying/geminating the second character in
  the run.

Despite these gaps, the rule-based system is quite strong on the majority of
(mostly disyllabic, non-compound) words — the corpus-average PER is 4.43%,
which the neural model only surpasses once trained on roughly 50%+ of the
available training data (see the learning curve).
