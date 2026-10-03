"""
Learning curve for a G2P seq2seq model, for any of the scraped languages.

Trains an attention-based GRU encoder-decoder (architecture in g2p_model.py)
on increasing fractions (10%, 20%, ..., 100%) of the language's persisted
train split (data/splits/<language>_train.csv, see make_splits.py), evaluating
phoneme error rate (PER) on the fixed dev split at each fraction. The test
split is never touched by this sweep.

Select the language with --language/-l (default karelian).
"""

import argparse
import csv
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from g2p_model import (
    ROOT, SEED, DEVICE, EMB_DIM, HID_DIM, N_EPOCHS, LEARNING_RATE,
    TEACHER_FORCING_RATIO, BATCH_SIZE, PAD, SOS, EOS,
    load_splits, build_vocab, G2PDataset, make_collate_fn,
    Encoder, Decoder, Seq2Seq, run_epoch, evaluate,
)

print("device:", DEVICE)

FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def train_and_eval(train_subset, dev_pairs, seed):
    random.seed(seed)
    torch.manual_seed(seed)

    src_stoi, src_itos = build_vocab([g for g, _ in train_subset])
    tgt_stoi, tgt_itos = build_vocab([p for _, p in train_subset])
    pad_idx = src_stoi[PAD]
    tgt_pad_idx = tgt_stoi[PAD]
    sos_idx = tgt_stoi[SOS]
    eos_idx = tgt_stoi[EOS]

    collate_fn = make_collate_fn(pad_idx, tgt_pad_idx)
    train_dl = DataLoader(
        G2PDataset(train_subset, src_stoi, tgt_stoi),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn,
    )
    dev_dl = DataLoader(
        G2PDataset(dev_pairs, src_stoi, tgt_stoi),
        batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
    )

    encoder = Encoder(len(src_itos), EMB_DIM, HID_DIM, pad_idx).to(DEVICE)
    decoder = Decoder(len(tgt_itos), EMB_DIM, HID_DIM, tgt_pad_idx).to(DEVICE)
    model = Seq2Seq(encoder, decoder, pad_idx, sos_idx, eos_idx, DEVICE).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)

    best_dev_loss = float("inf")
    best_state = None
    for epoch in range(1, N_EPOCHS + 1):
        run_epoch(model, train_dl, optimizer, criterion, train=True,
                  teacher_forcing_ratio=TEACHER_FORCING_RATIO)
        dev_loss = run_epoch(model, dev_dl, optimizer, criterion, train=False)
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    dev_per, dev_acc = evaluate(model, dev_pairs, src_stoi, tgt_stoi, tgt_itos, sos_idx, eos_idx)
    return dev_per, dev_acc


# ---------------------------------------------------------------------------
# Main: sweep fractions of the persisted training split
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-l", "--language", default="karelian",
                     help="Language name; reads data/splits/<language>_{train,dev}.csv "
                          "(default: karelian)")
    return ap.parse_args()


def main():
    args = parse_args()
    language = args.language
    results_path = ROOT / "outputs" / f"{language}_learning_curve_results.csv"
    plot_path = ROOT / "figures" / f"{language}_learning_curve.png"

    splits = load_splits(language)
    train_pool, dev_pairs, test_pairs = splits["train"], splits["dev"], splits["test"]
    print(f"[{language}] train pool: {len(train_pool)}  dev (fixed): {len(dev_pairs)}  "
          f"test (fixed, held out, unused here): {len(test_pairs)}")

    results = []
    for frac in FRACTIONS:
        n_train = max(1, round(frac * len(train_pool)))
        rng = random.Random(SEED + round(frac * 1000))
        train_subset = rng.sample(train_pool, n_train)

        dev_per, dev_acc = train_and_eval(train_subset, dev_pairs, seed=SEED)
        print(f"[{language}] frac={frac:.1f}  n_train={n_train:4d}  dev_PER={dev_per*100:6.2f}%  "
              f"dev_acc={dev_acc*100:6.2f}%")
        results.append((frac, n_train, dev_per, dev_acc))

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fraction", "n_train", "dev_per", "dev_acc"])
        for frac, n_train, per, acc in results:
            writer.writerow([frac, n_train, per, acc])

    fracs, n_trains, pers, accs = zip(*results)
    plt.figure(figsize=(7, 5))
    plt.plot([n * 100 for n in pers], marker="o")
    plt.gca().set_xticks(range(len(fracs)))
    plt.gca().set_xticklabels([f"{int(f*100)}%" for f in fracs])
    plt.xlabel("training data used (% of train pool)")
    plt.ylabel("dev phoneme error rate (%)")
    plt.title(f"{language.capitalize()} learning curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    print(f"saved {plot_path} and {results_path}")


if __name__ == "__main__":
    main()
