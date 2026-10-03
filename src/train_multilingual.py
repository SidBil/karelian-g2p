"""
Train ONE multilingual G2P model on the pooled train splits of all three
languages, then evaluate it on each language's held-out test split separately.

This is the "joint multilingual" counterpart to the zero-shot transfer grid in
cross_lingual_eval.py: instead of training three monolingual models and testing
each on foreign languages (where foreign characters fall back to <unk>), here a
single model sees all three languages' training data at once, so its shared
vocab covers every character and each test language is in-domain. Compare the
per-language test PER printed here against the diagonal of
outputs/cross_lingual_results.csv (each language's own monolingual model).

Reuses the shared architecture / training loop from g2p_model.py. Saves the
model + shared vocab to models/multilingual_g2p_model.pt and a per-language
results table to outputs/multilingual_results.csv.
"""

import csv
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from g2p_model import (
    ROOT, SEED, DEVICE, LANGUAGES, EMB_DIM, HID_DIM, N_EPOCHS, LEARNING_RATE,
    TEACHER_FORCING_RATIO, BATCH_SIZE, PAD, SOS, EOS,
    load_splits, build_vocab, G2PDataset, make_collate_fn,
    Encoder, Decoder, Seq2Seq, run_epoch, evaluate,
)

print("device:", DEVICE)


def main():
    # Load every language's split, then pool train and dev across languages.
    per_lang = {lang: load_splits(lang) for lang in LANGUAGES}
    train_pairs, dev_pairs = [], []
    for lang in LANGUAGES:
        train_pairs += per_lang[lang]["train"]
        dev_pairs += per_lang[lang]["dev"]
        print(f"[{lang}] train {len(per_lang[lang]['train'])} / "
              f"dev {len(per_lang[lang]['dev'])} / test {len(per_lang[lang]['test'])}")
    print(f"[pooled] train {len(train_pairs)} / dev {len(dev_pairs)}")

    random.seed(SEED)
    torch.manual_seed(SEED)

    # Shared vocab over the pooled training data -> covers all three languages'
    # characters, so no <unk> fallback for any of the test languages.
    src_stoi, src_itos = build_vocab([g for g, _ in train_pairs])
    tgt_stoi, tgt_itos = build_vocab([p for _, p in train_pairs])
    pad_idx = src_stoi[PAD]
    tgt_pad_idx = tgt_stoi[PAD]
    sos_idx = tgt_stoi[SOS]
    eos_idx = tgt_stoi[EOS]
    print(f"[pooled] src vocab {len(src_itos)} / tgt vocab {len(tgt_itos)}")

    collate_fn = make_collate_fn(pad_idx, tgt_pad_idx)
    train_dl = DataLoader(G2PDataset(train_pairs, src_stoi, tgt_stoi),
                          batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dev_dl = DataLoader(G2PDataset(dev_pairs, src_stoi, tgt_stoi),
                        batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    encoder = Encoder(len(src_itos), EMB_DIM, HID_DIM, pad_idx).to(DEVICE)
    decoder = Decoder(len(tgt_itos), EMB_DIM, HID_DIM, tgt_pad_idx).to(DEVICE)
    model = Seq2Seq(encoder, decoder, pad_idx, sos_idx, eos_idx, DEVICE).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)

    best_dev_loss = float("inf")
    best_state = None
    for epoch in range(1, N_EPOCHS + 1):
        train_loss = run_epoch(model, train_dl, optimizer, criterion, train=True,
                               teacher_forcing_ratio=TEACHER_FORCING_RATIO)
        dev_loss = run_epoch(model, dev_dl, optimizer, criterion, train=False)
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(f"[pooled] epoch {epoch:3d} | train_loss {train_loss:.4f} | "
                  f"dev_loss {dev_loss:.4f}")

    model.load_state_dict(best_state)

    # Evaluate the single pooled model on each language's own test split.
    rows = []
    print(f"\n[pooled] best_dev_loss={best_dev_loss:.4f}")
    print(f"{'tested_on':<10} {'n_test':>6} {'test_PER':>9} {'test_acc':>9}")
    for lang in LANGUAGES:
        test_pairs = per_lang[lang]["test"]
        test_per, test_acc = evaluate(model, test_pairs, src_stoi, tgt_stoi,
                                      tgt_itos, sos_idx, eos_idx)
        print(f"{lang:<10} {len(test_pairs):>6} {test_per*100:>8.2f}% {test_acc*100:>8.2f}%")
        rows.append({
            "trained_on": "multilingual",
            "tested_on": lang,
            "n_test": len(test_pairs),
            "test_per": test_per,
            "test_acc": test_acc,
        })

    out_path = ROOT / "outputs" / "multilingual_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["trained_on", "tested_on", "n_test",
                                               "test_per", "test_acc"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out_path}")

    model_path = ROOT / "models" / "multilingual_g2p_model.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "src_stoi": src_stoi, "src_itos": src_itos,
        "tgt_stoi": tgt_stoi, "tgt_itos": tgt_itos,
        "emb_dim": EMB_DIM, "hid_dim": HID_DIM,
        "languages": LANGUAGES,
        "results": rows,
    }, model_path)
    print(f"saved {model_path}")


if __name__ == "__main__":
    main()
