"""
Learning curve for a G2P seq2seq model, for any of the scraped languages.

Trains an attention-based GRU encoder-decoder (architecture copied from
karelian_g2p_seq2seq.ipynb) on increasing fractions of the training data
(10%, 20%, ..., 100%), keeping a fixed 80/10/10 train/dev/test split, and
plots phoneme error rate (PER) on the constant dev set vs. amount of training
data. The test set is held out and never touched by this sweep.

Expects data/<language>_terms.csv (orthography,ipa rows, gold IPA optionally
wrapped in /.../). Select the language with --language/-l (default karelian).
"""

import argparse
import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
HEADER_LABELS = {"orthography", "grapheme", "word", "term"}
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
)
print("device:", DEVICE)

EMB_DIM = 64
HID_DIM = 256
N_EPOCHS = 60
LEARNING_RATE = 1e-3
TEACHER_FORCING_RATIO = 0.5
BATCH_SIZE = 32
FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"
SPECIALS = [PAD, SOS, EOS, UNK]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_pairs(path: Path):
    seen = set()
    pairs = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if len(row) < 2:
                continue
            grapheme = row[0].strip().lower()
            phoneme = row[1].strip().strip("/")
            if i == 0 and grapheme in HEADER_LABELS:
                continue
            if not grapheme or not phoneme:
                continue
            key = (grapheme, phoneme)
            if key in seen:
                continue
            seen.add(key)
            pairs.append((grapheme, phoneme))
    return pairs


def build_vocab(sequences):
    chars = sorted(set(ch for seq in sequences for ch in seq))
    itos = SPECIALS + chars
    stoi = {ch: i for i, ch in enumerate(itos)}
    return stoi, itos


def encode(seq, stoi):
    ids = [stoi.get(ch, stoi[UNK]) for ch in seq]
    return [stoi[SOS]] + ids + [stoi[EOS]]


class G2PDataset(Dataset):
    def __init__(self, pairs, src_stoi, tgt_stoi):
        self.pairs = pairs
        self.src_stoi = src_stoi
        self.tgt_stoi = tgt_stoi

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        grapheme, phoneme = self.pairs[idx]
        src = torch.tensor(encode(grapheme, self.src_stoi), dtype=torch.long)
        tgt = torch.tensor(encode(phoneme, self.tgt_stoi), dtype=torch.long)
        return src, tgt


def make_collate_fn(pad_idx, tgt_pad_idx):
    def collate_fn(batch):
        srcs, tgts = zip(*batch)
        src_lens = torch.tensor([len(s) for s in srcs])
        tgt_lens = torch.tensor([len(t) for t in tgts])
        src_pad = nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=pad_idx)
        tgt_pad = nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=tgt_pad_idx)
        return src_pad, src_lens, tgt_pad, tgt_lens
    return collate_fn


# ---------------------------------------------------------------------------
# Model (identical architecture to the notebook)
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src, src_lens):
        embedded = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        hidden = torch.tanh(self.fc(torch.cat((hidden[0], hidden[1]), dim=1)))
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs, mask):
        seq_len = enc_outputs.size(1)
        dec_hidden_rep = dec_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((dec_hidden_rep, enc_outputs), dim=2)))
        scores = self.v(energy).squeeze(2)
        scores = scores.masked_fill(mask == 0, -1e10)
        return F.softmax(scores, dim=1)


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.attention = Attention(hidden_dim)
        self.rnn = nn.GRU(emb_dim + hidden_dim * 2, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim * 3 + emb_dim, vocab_size)

    def forward(self, input_tok, hidden, enc_outputs, mask):
        embedded = self.embedding(input_tok).unsqueeze(1)
        attn_weights = self.attention(hidden, enc_outputs, mask)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        hidden = hidden.squeeze(0)
        output = output.squeeze(1)
        embedded = embedded.squeeze(1)
        context = context.squeeze(1)
        pred = self.fc_out(torch.cat((output, context, embedded), dim=1))
        return pred, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, sos_idx, eos_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = device

    def create_mask(self, src):
        return src != self.src_pad_idx

    def forward(self, src, src_lens, tgt=None, max_len=40, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        enc_outputs, hidden = self.encoder(src, src_lens)
        mask = self.create_mask(src)

        out_len = tgt.size(1) if tgt is not None else max_len
        vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, out_len, vocab_size, device=self.device)

        input_tok = torch.full((batch_size,), self.sos_idx, dtype=torch.long, device=self.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for t in range(1, out_len):
            pred, hidden, _ = self.decoder(input_tok, hidden, enc_outputs, mask)
            outputs[:, t] = pred
            top1 = pred.argmax(1)
            if tgt is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input_tok = tgt[:, t]
            else:
                input_tok = top1
            finished = finished | (top1 == self.eos_idx)
            if tgt is None and finished.all():
                break
        return outputs


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def levenshtein(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


@torch.no_grad()
def transcribe(model, word, src_stoi, tgt_stoi, tgt_itos, sos_idx, eos_idx, max_len=40):
    model.eval()
    word = word.strip().lower()
    src_ids = encode(word, src_stoi)
    src = torch.tensor(src_ids, device=DEVICE).unsqueeze(0)
    src_lens = torch.tensor([len(src_ids)])

    enc_outputs, hidden = model.encoder(src, src_lens)
    mask = model.create_mask(src)

    input_tok = torch.tensor([sos_idx], device=DEVICE)
    result_chars = []
    for _ in range(max_len):
        pred, hidden, _ = model.decoder(input_tok, hidden, enc_outputs, mask)
        top1 = pred.argmax(1).item()
        if top1 == eos_idx:
            break
        result_chars.append(tgt_itos[top1])
        input_tok = torch.tensor([top1], device=DEVICE)
    return "".join(result_chars)


def evaluate(model, pairs, src_stoi, tgt_stoi, tgt_itos, sos_idx, eos_idx):
    total_edits = 0
    total_len = 0
    exact = 0
    for grapheme, phoneme in pairs:
        pred = transcribe(model, grapheme, src_stoi, tgt_stoi, tgt_itos, sos_idx, eos_idx)
        total_edits += levenshtein(pred, phoneme)
        total_len += len(phoneme)
        exact += int(pred == phoneme)
    per = total_edits / total_len
    acc = exact / len(pairs)
    return per, acc


def run_epoch(model, dl, optimizer, criterion, train, teacher_forcing_ratio=0.0):
    model.train() if train else model.eval()
    total_loss = 0.0
    with torch.set_grad_enabled(train):
        for src, src_lens, tgt, tgt_lens in dl:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            if train:
                optimizer.zero_grad()
            outputs = model(src, src_lens, tgt, teacher_forcing_ratio=teacher_forcing_ratio)
            vocab_size = outputs.shape[-1]
            loss = criterion(outputs[:, 1:].reshape(-1, vocab_size), tgt[:, 1:].reshape(-1))
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item()
    return total_loss / len(dl)


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
# Main: build fixed splits, sweep fractions of the training pool
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-l", "--language", default="karelian",
                     help="Language name; expects data/<language>_terms.csv (default: karelian)")
    return ap.parse_args()


def main():
    args = parse_args()
    language = args.language
    data_path = ROOT / "data" / f"{language}_terms.csv"
    results_path = ROOT / "outputs" / f"{language}_learning_curve_results.csv"
    plot_path = ROOT / "figures" / f"{language}_learning_curve.png"
    if not data_path.exists():
        raise FileNotFoundError(f"No such data file: {data_path}")

    pairs = load_pairs(data_path)
    print(f"[{language}] {len(pairs)} unique (grapheme, phoneme) pairs")

    shuffled = pairs[:]
    random.seed(SEED)
    random.shuffle(shuffled)

    n = len(shuffled)
    n_dev = max(1, round(0.1 * n))
    n_test = max(1, round(0.1 * n))

    dev_pairs = shuffled[:n_dev]
    test_pairs = shuffled[n_dev:n_dev + n_test]
    train_pool = shuffled[n_dev + n_test:]
    print(f"train pool: {len(train_pool)} (~80%)  dev (fixed): {len(dev_pairs)} (~10%)  "
          f"test (fixed, held out): {len(test_pairs)} (~10%)")

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
    plt.gca().set_xticklabels([f"{int(f*100)}%\n(n={nt})" for f, nt in zip(fracs, n_trains)])
    plt.xlabel("training data used (% of train pool)")
    plt.ylabel("dev phoneme error rate (%)")
    plt.title(f"{language.capitalize()} G2P learning curve\n"
              f"(fixed 80/10/10 split; test set held out, unused here)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    print(f"saved {plot_path} and {results_path}")


if __name__ == "__main__":
    main()
