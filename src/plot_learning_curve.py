"""Re-plot a language's learning curve from its saved results CSV (no retraining).

Reads outputs/<language>_learning_curve_results.csv and writes
figures/<language>_learning_curve.png. Select the language with
--language/-l (default karelian).
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

# Rule-based g2p.py PER, per language -- only measured for languages that have
# a hand-written rule-based baseline (currently just Karelian; see g2p.py).
RULE_BASED_PER = {"karelian": 4.43}  # %


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-l", "--language", default="karelian",
                     help="Language name; reads outputs/<language>_learning_curve_results.csv "
                          "(default: karelian)")
    return ap.parse_args()


def main():
    args = parse_args()
    language = args.language
    results_path = ROOT / "outputs" / f"{language}_learning_curve_results.csv"
    plot_path = ROOT / "figures" / f"{language}_learning_curve.png"

    fracs, n_trains, pers = [], [], []
    with open(results_path) as f:
        for row in csv.DictReader(f):
            fracs.append(float(row["fraction"]))
            n_trains.append(int(row["n_train"]))
            pers.append(float(row["dev_per"]) * 100)

    plt.figure(figsize=(7, 5))
    plt.plot(pers, marker="o", label="neural seq2seq (dev)")
    if language in RULE_BASED_PER:
        rb = RULE_BASED_PER[language]
        plt.axhline(rb, linestyle="--", color="gray", label=f"rule-based ({rb}%)")
    plt.gca().set_xticks(range(len(fracs)))
    plt.gca().set_xticklabels([f"{int(f*100)}%" for f in fracs])
    plt.xlabel("training data used (% of train pool)")
    plt.ylabel("dev phoneme error rate (%)")
    plt.title(f"{language.capitalize()} G2P learning curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    print(f"saved {plot_path}")


if __name__ == "__main__":
    main()
