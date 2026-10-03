"""
Rule-guided ("residual") G2P: apply the hand-written rule-based pipeline
(g2p.py) to every word first, then train the same seq2seq architecture as
train_g2p_model.py to map (orthography + "|" + rule-based prediction) -> gold
IPA instead of orthography -> gold IPA alone.

This is the NN+HR hybrid from paper/research_plan.md Q2: the rule-based
output is fed into the encoder as a hint, so the network's job is to learn a
residual correction on top of it (copy the parts the rules got right, fix the
parts they didn't) rather than the orthography-to-IPA mapping from scratch.
g2p.py's rules were written for Karelian only, so on Livonian/Ingrian the
"hint" is frequently wrong/irrelevant -- that's part of what this experiment
measures, not an error.

Saves state dict + vocab to models/<language>_residual_g2p_model.pt and a
comparison table (rule-only PER/acc, plain-neural test PER/acc from an
existing models/<language>_g2p_model.pt if present, and this guided model's
dev/test PER/acc) to outputs/residual_results.csv. Select languages with
--language/-l (default: all three).
"""

import argparse
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from g2p_model import (
    ROOT, SEED, DEVICE, LANGUAGES, EMB_DIM, HID_DIM, N_EPOCHS, LEARNING_RATE,
    TEACHER_FORCING_RATIO, BATCH_SIZE, PAD, SOS, EOS,
    load_splits, build_vocab, G2PDataset, make_collate_fn,
    Encoder, Decoder, Seq2Seq, run_epoch, evaluate, levenshtein,
)
from g2p import g2p as rule_g2p

print("device:", DEVICE)

SEP = "|"


def guided_pairs(pairs):
    """(grapheme, phoneme) -> (grapheme + SEP + rule_prediction, phoneme)."""
    out = []
    for grapheme, phoneme in pairs:
        rule_pred = rule_g2p(grapheme).strip("/")
        out.append((f"{grapheme}{SEP}{rule_pred}", phoneme))
    return out


def rule_baseline_per_acc(pairs):
    """PER/accuracy of the rule-based pipeline alone on these pairs, using
    the same total-edits/total-length definition as evaluate_detailed() in
    g2p_model.py, so it's directly comparable to the neural numbers."""
    total_edits, total_len, exact = 0, 0, 0
    for grapheme, phoneme in pairs:
        pred = rule_g2p(grapheme).strip("/")
        edits = levenshtein(pred, phoneme)
        total_edits += edits
        total_len += len(phoneme)
        exact += int(pred == phoneme)
    return total_edits / total_len, exact / len(pairs)


def train_residual_model(language: str):
    splits = load_splits(language)
    train_pairs, dev_pairs, test_pairs = splits["train"], splits["dev"], splits["test"]
    print(f"[{language}] train {len(train_pairs)} / dev {len(dev_pairs)} / test {len(test_pairs)}")

    rule_dev_per, rule_dev_acc = rule_baseline_per_acc(dev_pairs)
    rule_test_per, rule_test_acc = rule_baseline_per_acc(test_pairs)
    print(f"[{language}] rule-only   dev_PER={rule_dev_per*100:.2f}% dev_acc={rule_dev_acc*100:.2f}%  "
          f"test_PER={rule_test_per*100:.2f}% test_acc={rule_test_acc*100:.2f}%")

    guided_train = guided_pairs(train_pairs)
    guided_dev = guided_pairs(dev_pairs)
    guided_test = guided_pairs(test_pairs)

    random.seed(SEED)
    torch.manual_seed(SEED)

    src_stoi, src_itos = build_vocab([g for g, _ in guided_train])
    tgt_stoi, tgt_itos = build_vocab([p for _, p in guided_train])
    pad_idx = src_stoi[PAD]
    tgt_pad_idx = tgt_stoi[PAD]
    sos_idx = tgt_stoi[SOS]
    eos_idx = tgt_stoi[EOS]

    collate_fn = make_collate_fn(pad_idx, tgt_pad_idx)
    train_dl = DataLoader(G2PDataset(guided_train, src_stoi, tgt_stoi),
                           batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dev_dl = DataLoader(G2PDataset(guided_dev, src_stoi, tgt_stoi),
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
            print(f"[{language}] epoch {epoch:3d} | train_loss {train_loss:.4f} | "
                  f"dev_loss {dev_loss:.4f}")

    model.load_state_dict(best_state)
    dev_per, dev_acc = evaluate(model, guided_dev, src_stoi, tgt_stoi, tgt_itos, sos_idx, eos_idx)
    test_per, test_acc = evaluate(model, guided_test, src_stoi, tgt_stoi, tgt_itos, sos_idx, eos_idx)
    print(f"[{language}] guided(NN+HR) best_dev_loss={best_dev_loss:.4f}  dev_PER={dev_per*100:.2f}%  "
          f"test_PER={test_per*100:.2f}%  test_acc={test_acc*100:.2f}%")

    plain_test_per, plain_test_acc = None, None
    plain_model_path = ROOT / "models" / f"{language}_g2p_model.pt"
    if plain_model_path.exists():
        ckpt = torch.load(plain_model_path, map_location="cpu")
        plain_test_per = ckpt.get("test_per")
        plain_test_acc = ckpt.get("test_acc")

    model_path = ROOT / "models" / f"{language}_residual_g2p_model.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "src_stoi": src_stoi, "src_itos": src_itos,
        "tgt_stoi": tgt_stoi, "tgt_itos": tgt_itos,
        "emb_dim": EMB_DIM, "hid_dim": HID_DIM,
        "separator": SEP,
        "dev_per": dev_per, "dev_acc": dev_acc,
        "test_per": test_per, "test_acc": test_acc,
    }, model_path)
    print(f"[{language}] saved {model_path}")

    return {
        "language": language,
        "n_train": len(train_pairs), "n_dev": len(dev_pairs), "n_test": len(test_pairs),
        "rule_test_per": rule_test_per, "rule_test_acc": rule_test_acc,
        "plain_nn_test_per": plain_test_per, "plain_nn_test_acc": plain_test_acc,
        "residual_dev_per": dev_per, "residual_dev_acc": dev_acc,
        "residual_test_per": test_per, "residual_test_acc": test_acc,
    }


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-l", "--language", choices=LANGUAGES, default=None,
                     help="Language to train (default: all three)")
    return ap.parse_args()


def main():
    import csv
    args = parse_args()
    rows = []
    for language in ([args.language] if args.language else LANGUAGES):
        rows.append(train_residual_model(language))

    out_path = ROOT / "outputs" / "residual_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["language", "n_train", "n_dev", "n_test",
                  "rule_test_per", "rule_test_acc",
                  "plain_nn_test_per", "plain_nn_test_acc",
                  "residual_dev_per", "residual_dev_acc",
                  "residual_test_per", "residual_test_acc"]
    write_header = not out_path.exists()
    existing = []
    if out_path.exists():
        with open(out_path, newline="") as f:
            existing = [r for r in csv.DictReader(f) if r["language"] not in {r2["language"] for r2 in rows}]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in existing:
            writer.writerow(r)
        for r in rows:
            writer.writerow(r)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
