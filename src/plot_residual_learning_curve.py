"""Re-plot a language's rules-on-vs-rules-off learning curve from its saved
results CSV (no retraining).

Reads outputs/<language>_residual_learning_curve_results.csv (see
residual_learning_curve.py) and writes
figures/<language>_residual_learning_curve.png. Select the language with
--language/-l (default karelian).
"""

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from g2p_model import ROOT, RULE_BASED_PER


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-l", "--language", default="karelian",
                     help="Language name; reads "
                          "outputs/<language>_residual_learning_curve_results.csv "
                          "(default: karelian)")
    return ap.parse_args()


def main():
    args = parse_args()
    language = args.language
    results_path = ROOT / "outputs" / f"{language}_residual_learning_curve_results.csv"
    plot_path = ROOT / "figures" / f"{language}_residual_learning_curve.png"

    fracs, off_pers, on_pers = [], [], []
    with open(results_path) as f:
        for row in csv.DictReader(f):
            fracs.append(float(row["fraction"]))
            off_pers.append(float(row["dev_per_rules_off"]) * 100)
            on_pers.append(float(row["dev_per_rules_on"]) * 100)

    plt.figure(figsize=(7, 5))
    plt.plot(off_pers, marker="o", label="rules off (plain NN)")
    plt.plot(on_pers, marker="o", label="rules on (NN + HR, guided)")
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
    print(f"saved {plot_path}")


if __name__ == "__main__":
    main()
