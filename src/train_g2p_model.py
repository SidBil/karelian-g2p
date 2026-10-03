"""
Train and save one final G2P model per language.

learning_curve.py trains fresh models per data fraction purely to measure the
learning curve and never persists them, so models/ was missing per-language
checkpoints. This script trains a single model per language on its full
persisted train split (data/splits/<language>_train.csv, see make_splits.py)
and saves state dict + vocab to models/<language>_g2p_model.pt (vocab is
required to run inference on a saved checkpoint, so it's included here unlike
the notebook's models/best_g2p_model.pt). Select languages with
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
    Encoder, Decoder, Seq2Seq, run_epoch, evaluate,
)

print("device:", DEVICE)


def train_final_model(language: str):
    splits = load_splits(language)
    train_pairs, dev_pairs, test_pairs = splits["train"], splits["dev"], splits["test"]
    print(f"[{language}] train {len(train_pairs)} / dev {len(dev_pairs)} / test {len(test_pairs)}")

    random.seed(SEED)
    torch.manual_seed(SEED)

    src_stoi, src_itos = build_vocab([g for g, _ in train_pairs])
    tgt_stoi, tgt_itos = build_vocab([p for _, p in train_pairs])
    pad_idx = src_stoi[PAD]
    tgt_pad_idx = tgt_stoi[PAD]
    sos_idx = tgt_stoi[SOS]
    eos_idx = tgt_stoi[EOS]

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
            print(f"[{language}] epoch {epoch:3d} | train_loss {train_loss:.4f} | "
                  f"dev_loss {dev_loss:.4f}")

    model.load_state_dict(best_state)
    dev_per, dev_acc = evaluate(model, dev_pairs, src_stoi, tgt_stoi, tgt_itos, sos_idx, eos_idx)
    test_per, test_acc = evaluate(model, test_pairs, src_stoi, tgt_stoi, tgt_itos, sos_idx, eos_idx)
    print(f"[{language}] best_dev_loss={best_dev_loss:.4f}  dev_PER={dev_per*100:.2f}%  "
          f"test_PER={test_per*100:.2f}%  test_acc={test_acc*100:.2f}%")

    model_path = ROOT / "models" / f"{language}_g2p_model.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "src_stoi": src_stoi, "src_itos": src_itos,
        "tgt_stoi": tgt_stoi, "tgt_itos": tgt_itos,
        "emb_dim": EMB_DIM, "hid_dim": HID_DIM,
        "dev_per": dev_per, "dev_acc": dev_acc,
        "test_per": test_per, "test_acc": test_acc,
    }, model_path)
    print(f"[{language}] saved {model_path}")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-l", "--language", choices=LANGUAGES, default=None,
                     help="Language to train (default: all three)")
    return ap.parse_args()


def main():
    args = parse_args()
    for language in ([args.language] if args.language else LANGUAGES):
        train_final_model(language)


if __name__ == "__main__":
    main()
