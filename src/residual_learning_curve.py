"""
Rules-on vs rules-off learning curve.

Compares data efficiency of the plain neural G2P model ("rules off": orthography
-> IPA) against the rule-guided NN+HR model ("rules on": orthography + "|" +
rule-based prediction -> IPA, see train_residual_model.py) across the same
training-data fractions used by learning_curve.py. This directly tests the
paper/research_plan.md Q2 hypothesis -- that guiding the network with the
rule-based output improves data efficiency -- by training both conditions on
the *same* sampled train_subset at each fraction, so any gap is attributable
to the rule hint rather than to different training data.

Reuses train_and_eval() from learning_curve.py (generic over any (src, tgt)
string pairs -- it doesn't care whether src is plain orthography or a guided
orthography+rule-hint string) and guided_pairs() from train_residual_model.py,
rather than reimplementing the training loop. The test split is never touched.

Select the language with --language/-l (default karelian).
"""

import argparse
import csv
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from g2p_model import ROOT, SEED, RULE_BASED_PER, load_splits
from learning_curve import FRACTIONS, train_and_eval
from train_residual_model import guided_pairs


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-l", "--language", default="karelian",
                     help="Language name; reads data/splits/<language>_{train,dev}.csv "
                          "(default: karelian)")
    return ap.parse_args()


def main():
    args = parse_args()
    language = args.language
    results_path = ROOT / "outputs" / f"{language}_residual_learning_curve_results.csv"
    plot_path = ROOT / "figures" / f"{language}_residual_learning_curve.png"

    splits = load_splits(language)
    train_pool, dev_pairs, test_pairs = splits["train"], splits["dev"], splits["test"]
    print(f"[{language}] train pool: {len(train_pool)}  dev (fixed): {len(dev_pairs)}  "
          f"test (fixed, held out, unused here): {len(test_pairs)}")

    guided_dev = guided_pairs(dev_pairs)

    results = []
    for frac in FRACTIONS:
        n_train = max(1, round(frac * len(train_pool)))
        rng = random.Random(SEED + round(frac * 1000))
        train_subset = rng.sample(train_pool, n_train)

        off_per, off_acc = train_and_eval(train_subset, dev_pairs, seed=SEED)

        guided_subset = guided_pairs(train_subset)
        on_per, on_acc = train_and_eval(guided_subset, guided_dev, seed=SEED)

        print(f"[{language}] frac={frac:.1f}  n_train={n_train:4d}  "
              f"rules_off dev_PER={off_per*100:6.2f}% dev_acc={off_acc*100:6.2f}%  |  "
              f"rules_on dev_PER={on_per*100:6.2f}% dev_acc={on_acc*100:6.2f}%")
        results.append((frac, n_train, off_per, off_acc, on_per, on_acc))

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fraction", "n_train", "dev_per_rules_off", "dev_acc_rules_off",
                          "dev_per_rules_on", "dev_acc_rules_on"])
        for row in results:
            writer.writerow(row)

    fracs, n_trains, off_pers, off_accs, on_pers, on_accs = zip(*results)
    plt.figure(figsize=(7, 5))
    plt.plot([p * 100 for p in off_pers], marker="o", label="rules off (plain NN)")
    plt.plot([p * 100 for p in on_pers], marker="o", label="rules on (NN + HR, guided)")
    if language in RULE_BASED_PER:
        rb = RULE_BASED_PER[language]
        plt.axhline(rb, linestyle="--", color="gray", label=f"rule-based ({rb}%)")
    plt.gca().set_xticks(range(len(fracs)))
    plt.gca().set_xticklabels([f"{int(f*100)}%" for f in fracs])
    plt.xlabel("training data used (% of train pool)")
    plt.ylabel("dev phoneme error rate (%)")
    plt.title(f"{language.capitalize()} learning curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    print(f"saved {plot_path} and {results_path}")


if __name__ == "__main__":
    main()
