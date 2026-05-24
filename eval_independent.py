# eval_independent.py
# Evaluate InteractionCNNPlus on an independent CSR-1 CLASH dataset
# Files needed in the same directory:
#   - best_fold1.pt
#   - CSR-1_CLASH_positive.csv
#   - CSR-1_CLASH_negative.csv

import os, math, time, argparse, random, pickle
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------ Repro ------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# ------------------ I/O ------------------
POS_PATH = "CSR-1_CLASH_positive.csv"
NEG_PATH = "CSR-1_CLASH_negative.csv"
WEIGHTS  = "best_fold1.pt"
OUT_PRED = "independent_predictions.csv"
OUT_ROC  = "independent_roc.png"

# ------------------ Sequence utils ------------------
NUC2IDX = {'A':0, 'C':1, 'G':2, 'T':3, 'U':3}
VALID_NUCS = set(NUC2IDX.keys())

def clean_seq(s: str) -> str:
    s = (s or "").strip().upper()
    s = ''.join([ch for ch in s if ch in VALID_NUCS])
    return s.replace('U','T')

def one_hot(seq: str, max_len: int) -> np.ndarray:
    seq = clean_seq(seq)
    if len(seq) > max_len: seq = seq[:max_len]
    arr = np.zeros((max_len, 4), dtype=np.uint8)
    for i, ch in enumerate(seq):
        j = NUC2IDX.get(ch, None)
        if j is not None and i < max_len:
            arr[i, j] = 1
    return arr

# ------------------ CSV loading ------------------
P_COLS = ["piRNA_seq", "pirna_seq", "piRNA_sequence", "pirna"]
M_COLS = ["site_seq", "mRNA_seq", "mrna_seq", "target_seq", "target_site", "mRNA_sequence"]

def get_first_present(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of {candidates} found in {list(df.columns)}")

def load_pair_csv(path, label):
    df = pd.read_csv(path)
    pcol = get_first_present(df, P_COLS)
    mcol = get_first_present(df, M_COLS)
    df = df[[pcol, mcol]].rename(columns={pcol:"piRNA_seq", mcol:"site_seq"})
    df["label"] = int(label)
    return df

@dataclass
class SeqLens:
    pi_max: int
    mr_max: int

# ------------------ Simple Nussinov unpaired mask ------------------
_PAIR_OK = {("A","U"),("U","A"),("C","G"),("G","C"),("G","U"),("U","G")}
def _nussinov_unpaired_mask(seq: str, min_loop: int = 3) -> np.ndarray:
    s = seq.replace('T','U').upper()
    n = len(s)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    dp = np.zeros((n, n), dtype=np.float32)

    for i in range(n-2, -1, -1):
        for j in range(i+1, n):
            best = max(dp[i+1, j], dp[i, j-1])
            for k in range(i+1, j):
                cand = dp[i, k] + dp[k+1, j]
                if cand > best: best = cand
            if (j - i) > min_loop and (s[i], s[j]) in _PAIR_OK:
                best = max(best, dp[i+1, j-1] + 1.0)
            dp[i, j] = best

    paired = np.zeros(n, dtype=bool)
    def trace(i, j):
        if i >= j: return
        if dp[i, j] == dp[i+1, j]: trace(i+1, j); return
        if dp[i, j] == dp[i, j-1]: trace(i, j-1); return
        if (j - i) > min_loop and (s[i], s[j]) in _PAIR_OK and dp[i, j] == dp[i+1, j-1] + 1.0:
            paired[i] = paired[j] = True
            trace(i+1, j-1); return
        for k in range(i+1, j):
            if dp[i, j] == dp[i, k] + dp[k+1, j]:
                trace(i, k); trace(k+1, j); return
    trace(0, n-1)
    return (~paired).astype(np.float32)  # 1=unpaired

def _pad_vec(v: np.ndarray, L: int, fill: float = 1.0) -> np.ndarray:
    if len(v) >= L: return v[:L].astype(np.float32)
    out = np.full(L, fill, dtype=np.float32); out[:len(v)] = v.astype(np.float32)
    return out

# ------------------ Dataset/Collate (21 channels) ------------------
class PairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, lens: SeqLens):
        self.pi_max = lens.pi_max; self.mr_max = lens.mr_max
        self.df = df.reset_index(drop=True)
        self.pirnas = self.df["piRNA_seq"].astype(str).map(clean_seq).tolist()
        self.sites  = self.df["site_seq"].astype(str).map(clean_seq).tolist()
        self.y      = self.df["label"].astype(np.float32).to_numpy()
        self.P = np.stack([one_hot(s, self.pi_max) for s in self.pirnas], axis=0)
        self.M = np.stack([one_hot(s, self.mr_max) for s in self.sites ], axis=0)
        uP, uM = [], []
        for pi, mr in zip(self.pirnas, self.sites):
            up = _pad_vec(_nussinov_unpaired_mask(pi), self.pi_max, fill=1.0)
            um = _pad_vec(_nussinov_unpaired_mask(mr), self.mr_max, fill=1.0)
            uP.append(up); uM.append(um)
        self.uP = np.stack(uP, axis=0); self.uM = np.stack(uM, axis=0)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.P[idx], self.M[idx], self.uP[idx], self.uM[idx], self.y[idx]

def collate_struct_delta(batch, seed_len=7):
    P_list, M_list, uP_list, uM_list, y_list = zip(*batch)
    P = torch.from_numpy(np.stack(P_list)).float()    # (B, Lp, 4)
    M = torch.from_numpy(np.stack(M_list)).float()    # (B, Lm, 4)
    uP = torch.from_numpy(np.stack(uP_list)).float()  # (B, Lp)
    uM = torch.from_numpy(np.stack(uM_list)).float()  # (B, Lm)
    y  = torch.tensor(np.stack(y_list)).float()

    B, Lp, _ = P.shape
    _, Lm, _ = M.shape

    # 16-channel pair identity
    inter = torch.einsum('bpi,bmj->bpmij', P, M).reshape(B, Lp, Lm, 16).permute(0,3,1,2).contiguous()

    # compatibility prior
    S = torch.tensor([
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.6],
        [1.0, 0.0, 0.6, 0.0],
    ], dtype=torch.float32)
    PS = torch.matmul(P, S)                           # (B,Lp,4)
    compat = torch.einsum('bik,bjk->bij', PS, M).clamp(0,1).unsqueeze(1)  # (B,1,Lp,Lm)

    # helix run (diag + anti)
    k = seed_len
    Kdiag = torch.eye(k, dtype=torch.float32).view(1,1,k,k)
    Kanti = torch.flip(Kdiag, dims=[-1])
    pad = k // 2
    diag_run = F.conv2d(compat, Kdiag, padding=pad) / float(k)
    anti_run = F.conv2d(compat, Kanti, padding=pad) / float(k)

    # positional delta
    i_coords = torch.linspace(0, 1, steps=Lp, dtype=torch.float32).view(1,1,Lp,1).expand(B,1,Lp,Lm)
    j_coords = torch.linspace(0, 1, steps=Lm, dtype=torch.float32).view(1,1,1,Lm).expand(B,1,Lp,Lm)
    delta = (i_coords - j_coords).abs()

    # structure A = uP ⊗ uM
    A = (uP.unsqueeze(2) * uM.unsqueeze(1)).unsqueeze(1)

    X = torch.cat([inter, compat, diag_run, anti_run, delta, A], dim=1)  # (B,21,Lp,Lm)
    return X, y

class CollateStructDelta:
    def __init__(self, seed_len=7):
        self.seed_len = seed_len
    def __call__(self, batch):
        return collate_struct_delta(batch, seed_len=self.seed_len)

# ------------------ Model (matches training) ------------------
class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.gate = nn.Sigmoid()
    def forward(self, x):
        s = self.avg(x)
        s = self.fc2(self.act(self.fc1(s)))
        return x * self.gate(s)

class InteractionCNNPlus(nn.Module):
    def __init__(self, in_ch=21, dropout=0.50):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            SEBlock(64, reduction=8),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SEBlock(128, reduction=8),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SEBlock(256, reduction=8),
        )
        self.head = nn.Sequential(
            nn.AdaptiveMaxPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x).squeeze(1)

# ------------------ Utils ------------------
def choose_threshold(y_true, y_prob):
    # Youden's J (for optional on-test reporting)
    best_t, best_j = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        tpr = tp / (tp + fn + 1e-9)
        fpr = fp / (fp + tn + 1e-9)
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, t
    return float(best_t), float(best_j)

def select_device():
    if torch.cuda.is_available():
        print("Using CUDA")
        torch.backends.cudnn.benchmark = True
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass
        return torch.device("cuda")
    print("Using CPU")
    return torch.device("cpu")

# ------------------ Evaluate ------------------
def main(args):
    assert os.path.exists(WEIGHTS), f"Missing weights: {WEIGHTS}"
    assert os.path.exists(POS_PATH), f"Missing: {POS_PATH}"
    assert os.path.exists(NEG_PATH), f"Missing: {NEG_PATH}"

    df_pos = load_pair_csv(POS_PATH, label=1)
    df_neg = load_pair_csv(NEG_PATH, label=0)

    for df in (df_pos, df_neg):
        df["piRNA_seq"] = df["piRNA_seq"].astype(str).map(clean_seq)
        df["site_seq"]  = df["site_seq"].astype(str).map(clean_seq)

    df_pos = df_pos[(df_pos["piRNA_seq"].str.len()>0) & (df_pos["site_seq"].str.len()>0)].reset_index(drop=True)
    df_neg = df_neg[(df_neg["piRNA_seq"].str.len()>0) & (df_neg["site_seq"].str.len()>0)].reset_index(drop=True)
    df_all = pd.concat([df_pos, df_neg], ignore_index=True)

    # lengths
    pi_max = max(df_all["piRNA_seq"].map(len).max(), 1)
    mr_max = max(df_all["site_seq"].map(len).max(), 1)
    lens = SeqLens(pi_max=pi_max, mr_max=mr_max)
    print(f"Independent set: N={len(df_all)} (pos={int(df_all['label'].sum())}, neg={len(df_all)-int(df_all['label'].sum())})")
    print(f"Lengths -> pi_max={pi_max}, mr_max={mr_max}")

    # data
    ds = PairDataset(df_all, lens)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0 if os.name=="nt" else 2,
                        pin_memory=torch.cuda.is_available(),
                        collate_fn=CollateStructDelta(seed_len=args.seed_len))

    # model
    device = select_device()
    model = InteractionCNNPlus(in_ch=21, dropout=0.50).to(device)
    state = torch.load(WEIGHTS, map_location=device)
    # handle DataParallel or plain
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.","",1): v for k,v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()

    # inference
    all_probs, all_truth = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device, non_blocking=True)
            logits = model(X)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_truth.append(y.numpy())
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_truth)

    # AUC + ROC
    auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    # thresholded metrics
    thr = args.thr
    if args.youden:
        thr, _ = choose_threshold(y_true, y_prob)  # NOTE: optimal-on-test; report separately
        print(f"[Info] Using Youden-optimal threshold on test: {thr:.3f}")

    y_pred = (y_prob >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    print("\n=== Independent Test Metrics ===")
    print(f"AUC:       {auc:.4f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Threshold used: {thr:.3f}")

    # save predictions CSV (useful for auditing)
    pd.DataFrame({
        "piRNA_seq": df_all["piRNA_seq"].to_numpy(),
        "site_seq":  df_all["site_seq"].to_numpy(),
        "y_true":    y_true,
        "y_prob":    y_prob,
        "y_pred":    y_pred,
        "thr":       np.full_like(y_true, thr, dtype=float)
    }).to_csv(OUT_PRED, index=False)
    print(f"Saved predictions to {OUT_PRED}")

    # plot ROC
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})", linewidth=2)
    plt.plot([0,1],[0,1], 'k--', linewidth=1)
    plt.xlabel("1 - Specificity (FPR)")
    plt.ylabel("Sensitivity (TPR)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT_ROC, dpi=300)
    plt.close()
    print(f"Saved ROC curve to {OUT_ROC}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed_len",   type=int, default=7, help="seed length used in helix-run filters")
    ap.add_argument("--thr",        type=float, default=0.50, help="decision threshold (ignored if --youden)")
    ap.add_argument("--youden",     action="store_true", help="compute Youden-optimal threshold on test (report separately)")
    args = ap.parse_args()
    main(args)
