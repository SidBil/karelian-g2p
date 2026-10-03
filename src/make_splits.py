"""
Generate the fixed 80/10/10 train/dev/test split for each language and write
it to data/splits/<language>_{train,dev,test}.csv.

Every other script (learning_curve.py, train_g2p_model.py,
cross_lingual_eval.py) reads these files directly instead of re-deriving the
split by reshuffling data/<language>_terms.csv on every run. Re-run this only
when a language's raw *_terms.csv changes -- not as part of a normal training
run.
"""

import csv
import random

from g2p_model import ROOT, SEED, LANGUAGES, load_pairs, split_paths


def split_pairs(pairs, seed=SEED):
    shuffled = pairs[:]
    random.seed(seed)
    random.shuffle(shuffled)
    n = len(shuffled)
    n_dev = max(1, round(0.1 * n))
    n_test = max(1, round(0.1 * n))
    dev_pairs = shuffled[:n_dev]
    test_pairs = shuffled[n_dev:n_dev + n_test]
    train_pairs = shuffled[n_dev + n_test:]
    return train_pairs, dev_pairs, test_pairs


def write_pairs(path, pairs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for grapheme, phoneme in pairs:
            writer.writerow([grapheme, f"/{phoneme}/"])


def main():
    for language in LANGUAGES:
        data_path = ROOT / "data" / f"{language}_terms.csv"
        pairs = load_pairs(data_path)
        train_pairs, dev_pairs, test_pairs = split_pairs(pairs)

        paths = split_paths(language)
        write_pairs(paths["train"], train_pairs)
        write_pairs(paths["dev"], dev_pairs)
        write_pairs(paths["test"], test_pairs)

        print(f"[{language}] {len(pairs)} pairs -> train {len(train_pairs)} / "
              f"dev {len(dev_pairs)} / test {len(test_pairs)}  -> {paths['train'].parent}")


if __name__ == "__main__":
    main()
