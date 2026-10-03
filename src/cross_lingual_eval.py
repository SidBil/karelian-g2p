"""
Cross-lingual G2P transfer evaluation.

Loads each language's saved model (models/<language>_g2p_model.pt, produced by
train_g2p_model.py) and evaluates it on every language's persisted TEST split
(data/splits/<language>_test.csv, see make_splits.py), for all 3x3 = 9
(trained_on, tested_on) combinations. The 3 in-language cells (trained_on ==
tested_on) reproduce that model's own test PER/accuracy; the 6 off-diagonal
cells are the actual cross-lingual transfer numbers (see
paper/research_plan.md, Q3).

Each model is evaluated with its OWN vocabulary (src_stoi/tgt_stoi/tgt_itos
saved in its checkpoint) -- this is a zero-shot transfer test, not
fine-tuning. Characters in a foreign language's orthography that the model
never saw during training fall back to <unk>, which is itself part of what's
being measured.

Writes outputs/cross_lingual_results.csv (aggregate PER/accuracy per
combination) and outputs/cross_lingual_failures.csv (every mismatched
prediction -- gold != prediction -- across all 9 combinations, so individual
error cases can be inspected rather than just the aggregate rate). Run only
after all three models/<language>_g2p_model.pt files exist
(train_g2p_model.py).
"""

import csv

import torch

from g2p_model import (
    ROOT, LANGUAGES, DEVICE, PAD, SOS, EOS,
    load_splits, Encoder, Decoder, Seq2Seq, evaluate_detailed,
)


def load_model(language):
    model_path = ROOT / "models" / f"{language}_g2p_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No saved model for {language}: {model_path} "
            f"(run train_g2p_model.py first)"
        )
    ckpt = torch.load(model_path, map_location=DEVICE)
    src_stoi, src_itos = ckpt["src_stoi"], ckpt["src_itos"]
    tgt_stoi, tgt_itos = ckpt["tgt_stoi"], ckpt["tgt_itos"]
    pad_idx = src_stoi[PAD]
    tgt_pad_idx = tgt_stoi[PAD]
    sos_idx = tgt_stoi[SOS]
    eos_idx = tgt_stoi[EOS]

    encoder = Encoder(len(src_itos), ckpt["emb_dim"], ckpt["hid_dim"], pad_idx).to(DEVICE)
    decoder = Decoder(len(tgt_itos), ckpt["emb_dim"], ckpt["hid_dim"], tgt_pad_idx).to(DEVICE)
    model = Seq2Seq(encoder, decoder, pad_idx, sos_idx, eos_idx, DEVICE).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, src_stoi, tgt_stoi, tgt_itos, sos_idx, eos_idx


def main():
    test_sets = {lang: load_splits(lang)["test"] for lang in LANGUAGES}
    for lang, pairs in test_sets.items():
        print(f"[{lang}] test set: {len(pairs)} pairs (held out, never used in training)")

    results = []
    failures = []
    for trained_on in LANGUAGES:
        model, src_stoi, tgt_stoi, tgt_itos, sos_idx, eos_idx = load_model(trained_on)
        for tested_on in LANGUAGES:
            pairs = test_sets[tested_on]
            per, acc, records = evaluate_detailed(
                model, pairs, src_stoi, tgt_stoi, tgt_itos, sos_idx, eos_idx
            )
            tag = "in-language" if trained_on == tested_on else "cross-lingual"
            print(f"[{tag:13s}] trained={trained_on:10s} tested={tested_on:10s} "
                  f"n={len(pairs):4d}  test_PER={per*100:6.2f}%  test_acc={acc*100:6.2f}%")
            results.append({
                "trained_on": trained_on,
                "tested_on": tested_on,
                "n_test": len(pairs),
                "test_per": per,
                "test_acc": acc,
            })
            for rec in records:
                if rec["exact_match"]:
                    continue
                failures.append({
                    "trained_on": trained_on,
                    "tested_on": tested_on,
                    "grapheme": rec["grapheme"],
                    "gold": rec["gold"],
                    "prediction": rec["prediction"],
                    "edit_distance": rec["edit_distance"],
                })

    assert len(results) == len(LANGUAGES) ** 2, "expected 9 (trained_on, tested_on) combinations"

    out_path = ROOT / "outputs" / "cross_lingual_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["trained_on", "tested_on", "n_test", "test_per", "test_acc"]
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"saved {out_path}")

    failures_path = ROOT / "outputs" / "cross_lingual_failures.csv"
    with open(failures_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["trained_on", "tested_on", "grapheme", "gold", "prediction", "edit_distance"]
        )
        writer.writeheader()
        writer.writerows(failures)
    print(f"saved {failures_path} ({len(failures)} failures across all 9 combinations)")


if __name__ == "__main__":
    main()
