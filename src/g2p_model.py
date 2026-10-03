"""
Shared G2P neural model: character-level seq2seq architecture, vocab building,
dataset/dataloader helpers, and PER/accuracy evaluation.

Single source of truth for the bidirectional-GRU-encoder + Bahdanau-attention
GRU-decoder architecture -- imported by learning_curve.py, train_g2p_model.py,
and cross_lingual_eval.py instead of being copy-pasted into each. Deliberately
has no matplotlib import (plotting stays in learning_curve.py /
plot_learning_curve.py) so scripts that only train/evaluate don't pull in a
plotting dependency.

Data is read from the pre-split files in data/splits/ (see make_splits.py),
not re-shuffled from data/<language>_terms.csv on every run.
"""

import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
HEADER_LABELS = {"orthography", "grapheme", "word", "term"}
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
)

LANGUAGES = ["karelian", "livonian", "ingrian"]

# Rule-based PER, per language, for plotting as a reference line against the
# neural learning curves. Karelian's is measured directly (g2p.py, run
# against its own orthography); Livonian's and Ingrian's are external
# reference figures, since g2p.py's rules are Karelian-specific and were
# never designed for those two languages' phonology.
RULE_BASED_PER = {"karelian": 4.43, "livonian": 1.97, "ingrian": 7.48}  # %

EMB_DIM = 64
HID_DIM = 256
N_EPOCHS = 60
LEARNING_RATE = 1e-3
TEACHER_FORCING_RATIO = 0.5
BATCH_SIZE = 32

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"
SPECIALS = [PAD, SOS, EOS, UNK]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_pairs(path: Path):
    """Read (grapheme, phoneme) pairs from an `orthography,/ipa/`-formatted
    CSV (no header, IPA optionally wrapped in slashes), lowercased and
    deduplicated. Used both for raw data/<language>_terms.csv and the
    pre-split data/splits/<language>_{train,dev,test}.csv files, which share
    the format."""
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


def split_paths(language):
    base = ROOT / "data" / "splits"
    return {
        "train": base / f"{language}_train.csv",
        "dev": base / f"{language}_dev.csv",
        "test": base / f"{language}_test.csv",
    }


def load_splits(language):
    """Load the persisted 80/10/10 train/dev/test split for a language from
    data/splits/ (see make_splits.py). Raises if it hasn't been generated."""
    paths = split_paths(language)
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing split file(s) for {language}: {missing} -- run make_splits.py first."
        )
    return {name: load_pairs(path) for name, path in paths.items()}


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
# Model
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
# Eval / training helpers
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
    per, acc, _ = evaluate_detailed(model, pairs, src_stoi, tgt_stoi, tgt_itos, sos_idx, eos_idx)
    return per, acc


def evaluate_detailed(model, pairs, src_stoi, tgt_stoi, tgt_itos, sos_idx, eos_idx):
    """Like evaluate(), but also returns a per-example record for every pair
    (grapheme, gold, prediction, edit_distance, exact_match) -- used to pull
    out the failure cases (exact_match == False) without a second pass over
    the model."""
    total_edits = 0
    total_len = 0
    exact = 0
    records = []
    for grapheme, phoneme in pairs:
        pred = transcribe(model, grapheme, src_stoi, tgt_stoi, tgt_itos, sos_idx, eos_idx)
        edits = levenshtein(pred, phoneme)
        is_exact = pred == phoneme
        total_edits += edits
        total_len += len(phoneme)
        exact += int(is_exact)
        records.append({
            "grapheme": grapheme,
            "gold": phoneme,
            "prediction": pred,
            "edit_distance": edits,
            "exact_match": is_exact,
        })
    per = total_edits / total_len
    acc = exact / len(pairs)
    return per, acc, records


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
