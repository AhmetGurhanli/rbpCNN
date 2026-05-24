"""
Biophysics-informed Interaction-CNN

"""

import os
import csv
import math
import random
import time
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------- Reproducibility -------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# ------------------------- Optional RAM reporting -------------------------
try:
    import psutil
    _PSUTIL = psutil.Process(os.getpid())
except Exception:
    _PSUTIL = None

def _bytes_to_mb(b): return float(b) / (1024.0**2)
def _ram_mb():
    if _PSUTIL is None: return None
    try: return _PSUTIL.memory_info().rss / (1024.0**2)
    except Exception: return None
def _fmt_seconds(s):
    if s < 60: return f"{s:.2f}s"
    m, sec = divmod(s, 60)
    if m < 60: return f"{int(m)}m {sec:.1f}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m)}m {int(sec)}s"
def _print_profile(phase, t_sec, n_samples, device):
    thrpt = (n_samples / t_sec) if t_sec > 0 else float('nan')
    parts = [f" [PROFILE] {phase:<5}: {_fmt_seconds(t_sec):>8} | {n_samples:,} samp | {thrpt:,.0f} samp/s"]
    if device.type == "cuda":
        try: torch.cuda.synchronize()
        except Exception: pass
        alloc = _bytes_to_mb(torch.cuda.max_memory_allocated())
        rese  = _bytes_to_mb(torch.cuda.max_memory_reserved())
        parts.append(f"| GPU peak {alloc:,.1f} MB alloc / {rese:,.1f} MB reserved")
    ram = _ram_mb()
    if ram is not None:
        parts.append(f"| RAM {ram:,.1f} MB")
    print(" ".join(parts))

# ------------------------- Sequence utilities -------------------------
NUC2IDX = {'A':0, 'C':1, 'G':2, 'T':3, 'U':3}
VALID_NUCS = set(NUC2IDX.keys())

def clean_seq(s: str) -> str:
    s = (s or "").strip().upper()
    s = ''.join([ch for ch in s if ch in VALID_NUCS])
    return s.replace('U','T')

def one_hot(seq: str, max_len: int) -> np.ndarray:
    seq = clean_seq(seq)
    if len(seq) > max_len:
        seq = seq[:max_len]
    arr = np.zeros((max_len, 4), dtype=np.uint8)
    for i, ch in enumerate(seq):
        j = NUC2IDX.get(ch, None)
        if j is not None and i < max_len:
            arr[i, j] = 1
    return arr

# ------------------------- Threshold selection -------------------------
def choose_threshold(y_true, y_prob, mode="youden"):
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        y_pred = (y_prob >= t).astype(int)
        if mode == "accuracy":
            score = accuracy_score(y_true, y_pred)
        else:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
            tpr = tp / (tp + fn + 1e-9)
            fpr = fp / (fp + tn + 1e-9)
            score = tpr - fpr  # Youden's J
        if score > best_score:
            best_score, best_t = score, t
    return float(best_t), float(best_score)

def make_plateau_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )

def default_num_workers():
    if os.name == "nt":  # safer on Windows
        return 0
    return min(4, (os.cpu_count() or 2))

# ------------------------- CSV loading -------------------------
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

# ------------------------- Nussinov (paired/unpaired) -------------------------
_PAIR_OK = {("A","U"),("U","A"),("C","G"),("G","C"),("G","U"),("U","G")}  # includes wobble

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
    return (~paired).astype(np.float32)  # 1=unpaired, 0=paired

def _pad_vec(v: np.ndarray, L: int, fill: float = 1.0) -> np.ndarray:
    if len(v) >= L:
        return v[:L].astype(np.float32)
    out = np.full(L, fill, dtype=np.float32)
    out[:len(v)] = v.astype(np.float32)
    return out

# ------------------------- Nussinov Cache -------------------------
NUSS_POS_PATH = "nussinov_pos.pkl"
NUSS_NEG_PATH = "nussinov_neg.pkl"

def _pair_key(pi: str, mr: str) -> str:
    return f"{clean_seq(pi)}|{clean_seq(mr)}"

def _load_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}

def _save_cache(path: str, cache: dict):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)

def build_or_load_nussinov_cache(df: pd.DataFrame, out_path: str) -> dict:
    cache = _load_cache(out_path)
    needed = 0
    computed = 0
    for _, row in df.iterrows():
        pi = clean_seq(str(row["piRNA_seq"]))
        mr = clean_seq(str(row["site_seq"]))
        key = _pair_key(pi, mr)
        needed += 1
        if key in cache:
            continue
        up = _nussinov_unpaired_mask(pi)
        um = _nussinov_unpaired_mask(mr)
        cache[key] = (up.astype(np.float32), um.astype(np.float32))
        computed += 1
    if computed > 0:
        _save_cache(out_path, cache)
        print(f"[CACHE] Wrote {computed} new entries to {out_path} (total {len(cache)}/{needed} present)")
    else:
        print(f"[CACHE] Loaded {out_path} with {len(cache)}/{needed} entries")
    return cache

# ------------------------- Dataset -------------------------
class PairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, lens: SeqLens, pos_cache: dict, neg_cache: dict):
        self.pi_max = lens.pi_max
        self.mr_max = lens.mr_max
        self.df = df.reset_index(drop=True)
        self.pos_cache = pos_cache
        self.neg_cache = neg_cache

        self.pirnas = self.df["piRNA_seq"].astype(str).map(clean_seq).tolist()
        self.sites  = self.df["site_seq"].astype(str).map(clean_seq).tolist()
        self.y      = self.df["label"].astype(np.float32).to_numpy()

        self.P = np.stack([one_hot(s, self.pi_max) for s in self.pirnas], axis=0)
        self.M = np.stack([one_hot(s, self.mr_max) for s in self.sites ], axis=0)

        uP, uM = [], []
        for pi, mr, y in zip(self.pirnas, self.sites, self.y):
            key = _pair_key(pi, mr)
            src = self.pos_cache if int(y) == 1 else self.neg_cache
            if key in src:
                up_raw, um_raw = src[key]
            else:
                up_raw = _nussinov_unpaired_mask(pi)
                um_raw = _nussinov_unpaired_mask(mr)
            up = _pad_vec(up_raw, self.pi_max, fill=1.0)
            um = _pad_vec(um_raw, self.mr_max, fill=1.0)
            uP.append(up); uM.append(um)
        self.uP = np.stack(uP, axis=0)
        self.uM = np.stack(uM, axis=0)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return self.P[idx], self.M[idx], self.uP[idx], self.uM[idx], self.y[idx]

# ------------------------- Collate: build 21-channel tensor -------------------------
def collate_struct_delta(batch, seed_len=7):
    P_list, M_list, uP_list, uM_list, y_list = zip(*batch)
    P = torch.from_numpy(np.stack(P_list)).float()
    M = torch.from_numpy(np.stack(M_list)).float()
    uP = torch.from_numpy(np.stack(uP_list)).float()
    uM = torch.from_numpy(np.stack(uM_list)).float()
    y  = torch.tensor(np.stack(y_list)).float()

    B, Lp, _ = P.shape
    _, Lm, _ = M.shape

    inter = torch.einsum('bpi,bmj->bpmij', P, M).reshape(B, Lp, Lm, 16).permute(0,3,1,2).contiguous()

    S = torch.tensor([
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.6],
        [1.0, 0.0, 0.6, 0.0],
    ], dtype=torch.float32)
    PS = torch.matmul(P, S)
    compat = torch.einsum('bik,bjk->bij', PS, M).clamp(0,1).unsqueeze(1)

    k = seed_len
    Kdiag = torch.eye(k, dtype=torch.float32).view(1,1,k,k)
    Kanti = torch.flip(Kdiag, dims=[-1])
    pad = k // 2
    diag_run = F.conv2d(compat, Kdiag, padding=pad) / float(k)
    anti_run = F.conv2d(compat, Kanti, padding=pad) / float(k)

    i_coords = torch.linspace(0, 1, steps=Lp, dtype=torch.float32).view(1,1,Lp,1).expand(B,1,Lp,Lm)
    j_coords = torch.linspace(0, 1, steps=Lm, dtype=torch.float32).view(1,1,1,Lm).expand(B,1,Lp,Lm)
    delta = (i_coords - j_coords).abs()

    A = (uP.unsqueeze(2) * uM.unsqueeze(1)).unsqueeze(1)

    X = torch.cat([inter, compat, diag_run, anti_run, delta, A], dim=1)
    return X, y

class CollateStructDelta:
    def __init__(self, seed_len=7):
        self.seed_len = seed_len
    def __call__(self, batch):
        return collate_struct_delta(batch, seed_len=self.seed_len)

# ------------------------- SE Attention block -------------------------
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
        w = self.gate(s)
        return x * w

# ------------------------- Model -------------------------
class InteractionCNNPlus(nn.Module):
    def __init__(self, in_ch=21, dropout=0.50):  # stronger default dropout
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
        logits = self.head(x)
        return logits.squeeze(1)

# ------------------------- Device selection -------------------------
def select_device():
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        names = []
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            names.append(f"{props.name} ({_bytes_to_mb(props.total_memory)/(1024):.1f} GB VRAM)")
        print(f"Using CUDA with {n} GPU(s): {', '.join(names)}")
        torch.backends.cudnn.benchmark = True
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass
        return torch.device("cuda"), n
    else:
        print("CUDA not available; using CPU.")
        return torch.device("cpu"), 0

# ------------------------- Focal BCE (optional) -------------------------
class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, gamma: float = 1.5, pos_weight: torch.Tensor = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        if pos_weight is not None and pos_weight.dim() == 0:
            pos_weight = pos_weight.view(1)
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else torch.tensor([1.0]))
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none', pos_weight=self.pos_weight)
        p = torch.sigmoid(logits)
        p_t = p*targets + (1-p)*(1-targets)
        alpha_t = self.alpha*targets + (1-self.alpha)*(1-targets)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * bce
        return loss.mean()

# ------------------------- Epoch runner -------------------------
def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None,
              use_amp=True, record_batches: bool = False):
    """
    If record_batches=True, also returns a 4th value: np.array of per-batch accuracies
    computed at threshold 0.5. Otherwise returns (avg_loss, y_true, y_prob).
    """
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    probs, truths = [], []
    batch_accs = [] if record_batches else None

    amp_enabled = use_amp and (device.type == "cuda")

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type='cuda', enabled=amp_enabled, dtype=torch.float16):
            logits = model(X)
            loss = criterion(logits, y)

        # collect per-batch accuracy (threshold 0.5) BEFORE optimizer.step()
        if record_batches:
            with torch.no_grad():
                p_batch = torch.sigmoid(logits).detach().cpu().numpy()
                y_batch = y.detach().cpu().numpy()
                y_hat = (p_batch >= 0.5).astype(int)
                from sklearn.metrics import accuracy_score
                batch_accs.append(accuracy_score(y_batch, y_hat))

        if training:
            optimizer.zero_grad(set_to_none=True)
            if amp_enabled and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss += loss.item() * y.size(0)
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        truths.append(y.detach().cpu().numpy())

    y_prob = np.concatenate(probs)
    y_true = np.concatenate(truths)

    if record_batches:
        return total_loss / len(loader.dataset), y_true, y_prob, np.array(batch_accs, dtype=np.float32)
    else:
        return total_loss / len(loader.dataset), y_true, y_prob


# ------------------------- Fit one model -------------------------
def fit_one_model(df_subtrain, df_innerval, lens, device, n_gpus,
                  pos_cache, neg_cache,
                  batch_size=64, max_epochs=40, patience=6, lr=1e-3,
                  use_focal=True, seed_len=7,  dropout=0.3, weight_decay=0.0):

    ds_tr = PairDataset(df_subtrain, lens, pos_cache, neg_cache)
    ds_va = PairDataset(df_innerval, lens, pos_cache, neg_cache)
    pin = (device.type == "cuda")
    num_workers = default_num_workers()
    collate_fn = CollateStructDelta(seed_len)

    tr_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                           pin_memory=pin, num_workers=num_workers, collate_fn=collate_fn)
    va_loader = DataLoader(ds_va, batch_size=batch_size, shuffle=False,
                           pin_memory=pin, num_workers=num_workers, collate_fn=collate_fn)

    pos = df_subtrain["label"].sum(); neg = len(df_subtrain) - pos
    pos_weight = torch.tensor([(neg + 1e-6)/(pos + 1e-6)], dtype=torch.float32, device=device)

    model = InteractionCNNPlus(in_ch=21, dropout=dropout)
    if n_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion = FocalBCEWithLogitsLoss(alpha=0.5, gamma=1.5, pos_weight=pos_weight) if use_focal \
                else nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = make_plateau_scheduler(optimizer)

    try:
        scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    unwrap = model.module if isinstance(model, nn.DataParallel) else model
    best_auc, best_state, best_epoch, no_imp = -1.0, None, -1, 0

    # history containers
    train_acc_history, val_acc_history, val_auc_history, epoch_times = [], [], [], []
    # batch-level histories for epoch 1
    epoch1_train_batch_acc, epoch1_val_batch_acc = None, None

    print("\n--- Training (profiling enabled) ---")
    fold_train_start = time.perf_counter()
    for epoch in range(1, max_epochs+1):

        # TRAIN
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()

        if epoch == 1:
            tr_loss, y_tr_true, y_tr_prob, epoch1_train_batch_acc = run_epoch(
                model, tr_loader, device, criterion, optimizer, scaler, use_amp=True, record_batches=True
            )
            # ensure ndarray
            epoch1_train_batch_acc = np.asarray(epoch1_train_batch_acc, dtype=np.float32)
        else:
            tr_loss, y_tr_true, y_tr_prob = run_epoch(
                model, tr_loader, device, criterion, optimizer, scaler, use_amp=True, record_batches=False
            )

        tr_sec = time.perf_counter() - t0
        _print_profile("Train", tr_sec, len(ds_tr), device)

        y_tr_pred = (y_tr_prob >= 0.5).astype(int)
        tr_acc = accuracy_score(y_tr_true, y_tr_pred)
        train_acc_history.append(tr_acc)

        # VAL
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()

        if epoch == 1:
            va_loss, y_va, p_va, epoch1_val_batch_acc = run_epoch(
                model, va_loader, device, criterion, optimizer=None, scaler=None, use_amp=True, record_batches=True
            )
        else:
            va_loss, y_va, p_va = run_epoch(
                model, va_loader, device, criterion, optimizer=None, scaler=None, use_amp=True, record_batches=False
            )

        va_sec = time.perf_counter() - t0
        _print_profile("Val", va_sec, len(ds_va), device)

        y_va_pred = (p_va >= 0.5).astype(int)
        va_acc = accuracy_score(y_va, y_va_pred)
        try:
            va_auc = roc_auc_score(y_va, p_va)
        except Exception:
            va_auc = float('nan')

        val_acc_history.append(va_acc)
        val_auc_history.append(va_auc)
        epoch_times.append(tr_sec + va_sec)

        scheduler.step(va_auc if not math.isnan(va_auc) else 0.0)

        print(f"  Epoch {epoch:02d} | Train {tr_loss:.4f} | Val {va_loss:.4f} | Val AUC {va_auc:.4f} | LR {optimizer.param_groups[0]['lr']:.2e}")

        if not math.isnan(va_auc) and va_auc > best_auc + 1e-4:
            best_auc = va_auc
            best_state = {k: v.cpu().clone() for k,v in unwrap.state_dict().items()}
            best_epoch = epoch; no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"  Early stop @ epoch {epoch} (best AUC {best_auc:.4f} @ {best_epoch})")
                break

    if best_state is not None:
        unwrap.load_state_dict(best_state)

    fold_train_sec = time.perf_counter() - fold_train_start
    print(f"--- Training finished in {_fmt_seconds(fold_train_sec)} (best AUC {best_auc:.4f} @ epoch {best_epoch}) ---")

    # Threshold on inner-val with best weights
    _, y_va_final, p_va_final = run_epoch(model, va_loader, device, criterion, optimizer=None, scaler=None, use_amp=True)
    thr, _ = choose_threshold(y_va_final, p_va_final, mode="youden")

    logs = {
        "best_val_auc": best_auc,
        "best_epoch": best_epoch,
        "train_sec": fold_train_sec,
        "train_acc_history": np.array(train_acc_history),
        "val_acc_history": np.array(val_acc_history),
        "val_auc_history": np.array(val_auc_history),
        "epoch_times": np.array(epoch_times),
        "val_y": y_va_final,
        "val_p": p_va_final,
        # NEW: batch-level epoch 1 curves
        "epoch1_train_batch_acc": None if epoch1_train_batch_acc is None else np.asarray(epoch1_train_batch_acc),
        "epoch1_val_batch_acc":   None if epoch1_val_batch_acc   is None else np.asarray(epoch1_val_batch_acc),
    }
    return model, thr, logs


# ------------------------- Evaluation -------------------------
def evaluate_on_df(model, df_eval, lens, device, thr, pos_cache, neg_cache,
                   batch_size=64, seed_len=7, profile=True):
    ds = PairDataset(df_eval, lens, pos_cache, neg_cache)
    pin = (device.type == "cuda")
    num_workers = default_num_workers()
    collate_fn = CollateStructDelta(seed_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        pin_memory=pin, num_workers=num_workers, collate_fn=collate_fn)
    criterion = nn.BCEWithLogitsLoss()

    if profile and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    _, y_true, y_prob = run_epoch(model, loader, device, criterion, optimizer=None, scaler=None, use_amp=True)
    if device.type == "cuda": torch.cuda.synchronize()
    eval_sec = time.perf_counter() - t0
    if profile:
        _print_profile("Test", eval_sec, len(ds), device)

    y_pred = (y_prob >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    cm  = confusion_matrix(y_true, y_pred, labels=[0,1])
    return {"y_true": y_true, "y_prob": y_prob, "y_pred": y_pred,
            "accuracy": float(acc), "auc": float(auc), "confusion_matrix": cm,
            "eval_sec": eval_sec}

# ------------------------- K-Fold driver -------------------------
def kfold_training(df_all, df_pos, df_neg, K=5, inner_val_size=0.15, batch_size=64, max_epochs=40,
                   patience=6, lr=1e-3, use_focal=True, seed_len=7,
                   dropout=0.3, weight_decay=0.0):


    overall_start = time.perf_counter()
    device, n_gpus = select_device()

    # Lens based on cleaned sequences
    pi_max = max(df_all["piRNA_seq"].astype(str).map(lambda s: len(clean_seq(s))).max(), 1)
    mr_max = max(df_all["site_seq"].astype(str).map(lambda s: len(clean_seq(s))).max(), 1)
    lens = SeqLens(pi_max=pi_max, mr_max=mr_max)
    print(f"Input channels = 21 (16 interaction + 1 compat + 2 helix-run + 1 delta + 1 structure) | "
          f"pi_len={pi_max} | mr_len={mr_max}")

    # Build/load caches once
    pos_cache = build_or_load_nussinov_cache(df_pos, NUSS_POS_PATH)
    neg_cache = build_or_load_nussinov_cache(df_neg, NUSS_NEG_PATH)

    X = df_all[["piRNA_seq","site_seq"]].to_numpy()
    y = df_all["label"].to_numpy()

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)

    # per-fold metrics + OOF storage
    fold_mets = []
    all_oof_true, all_oof_prob = [], []
    all_oof_pi,   all_oof_mr   = [], []
    all_oof_fold  = []

    # epoch-wise learning curves across folds
    all_train_hist, all_val_hist = [], []

    # ROC plot storage
    roc_entries = []

    # NEW: batch-level epoch-1 curves across folds
    epoch1_train_batches_all = []
    epoch1_val_batches_all   = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        fold_start = time.perf_counter()
        print(f"\n========== Fold {fold}/{K} ==========")

        df_train = df_all.iloc[tr_idx].reset_index(drop=True)
        df_test  = df_all.iloc[te_idx].reset_index(drop=True)

        # inner validation split from training
        X_tr, X_va, y_tr, y_va = train_test_split(
            df_train[["piRNA_seq","site_seq"]],
            df_train["label"].to_numpy(),
            test_size=inner_val_size,
            random_state=SEED,
            stratify=df_train["label"].to_numpy()
        )
        # Preserve column names explicitly
        df_subtrain = pd.DataFrame(X_tr, columns=['piRNA_seq','site_seq'])
        df_subtrain['label'] = y_tr
        df_innerval = pd.DataFrame(X_va, columns=['piRNA_seq','site_seq'])
        df_innerval['label'] = y_va

        # ---- train this fold ----
        model, thr, logs = fit_one_model(
            df_subtrain, df_innerval, lens, device, n_gpus,
            pos_cache=pos_cache, neg_cache=neg_cache,
            batch_size=batch_size, max_epochs=max_epochs, patience=patience, lr=lr,
            use_focal=use_focal, seed_len=seed_len
        )

        # store epoch-wise histories & epoch-1 batch curves
        all_train_hist.append(logs["train_acc_history"])
        all_val_hist.append(logs["val_acc_history"])
        if logs.get("epoch1_train_batch_acc") is not None:
            epoch1_train_batches_all.append(logs["epoch1_train_batch_acc"])
        if logs.get("epoch1_val_batch_acc") is not None:
            epoch1_val_batches_all.append(logs["epoch1_val_batch_acc"])

        print(f"  Threshold (inner-val): {thr:.3f} | Best Val AUC {logs['best_val_auc']:.4f} @ epoch {logs['best_epoch']}")
        unwrap = model.module if isinstance(model, nn.DataParallel) else model
        torch.save(unwrap.state_dict(), f"best_fold{fold}.pt")

        # ---- evaluate on inner-val & test ----
        val_eval = evaluate_on_df(model, df_innerval, lens, device, thr,
                                  pos_cache=pos_cache, neg_cache=neg_cache,
                                  batch_size=batch_size, seed_len=seed_len, profile=False)
        test_eval = evaluate_on_df(model, df_test, lens, device, thr,
                                   pos_cache=pos_cache, neg_cache=neg_cache,
                                   batch_size=batch_size, seed_len=seed_len, profile=True)

        # detailed test metrics
        y_true = test_eval["y_true"]
        y_prob = test_eval["y_prob"]
        y_pred = test_eval["y_pred"]

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        acc  = accuracy_score(y_true, y_pred)
        try:
            auc_fold = roc_auc_score(y_true, y_prob)
        except Exception:
            auc_fold = float('nan')

        cm = test_eval["confusion_matrix"]
        print(f"  Fold {fold} | Test Acc: {acc:.4f} | Test AUC: {auc_fold:.4f} | "
              f"Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
        print("  Confusion matrix [rows true 0/1, cols pred 0/1]:")
        print(cm)

        fold_sec = time.perf_counter() - fold_start
        print(f"========== Fold {fold} time: {_fmt_seconds(fold_sec)} ==========")

        # record per-fold metrics
        fold_mets.append({
            "fold": fold, "acc": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc_fold, "thr": thr,
            "train_sec": logs.get("train_sec", float('nan')),
            "test_sec": test_eval.get("eval_sec", float('nan')),
            "fold_sec": fold_sec
        })

        # OOF collections (test split of this fold)
        all_oof_true.append(y_true)
        all_oof_prob.append(y_prob)
        all_oof_pi.append(df_test["piRNA_seq"].to_numpy())
        all_oof_mr.append(df_test["site_seq"].to_numpy())
        all_oof_fold.append(np.full_like(y_true, fill_value=fold, dtype=int))

        # ROC traces for plotting
        fpr_val,  tpr_val,  _ = roc_curve(val_eval["y_true"],  val_eval["y_prob"])
        fpr_test, tpr_test, _ = roc_curve(test_eval["y_true"], test_eval["y_prob"])
        roc_entries.append({
            "fold": fold,
            "fpr_val": fpr_val, "tpr_val": tpr_val,
            "fpr_test": fpr_test, "tpr_test": tpr_test,
            "auc_val": roc_auc_score(val_eval["y_true"],  val_eval["y_prob"]),
            "auc_test": roc_auc_score(test_eval["y_true"], test_eval["y_prob"])
        })

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---------- aggregate ----------
    mets = pd.DataFrame(fold_mets)

    acc_mean,  acc_std  = mets["acc"].mean(),        mets["acc"].std(ddof=0)
    prec_mean, prec_std = mets["precision"].mean(),  mets["precision"].std(ddof=0)
    rec_mean,  rec_std  = mets["recall"].mean(),     mets["recall"].std(ddof=0)
    f1_mean,   f1_std   = mets["f1"].mean(),         mets["f1"].std(ddof=0)
    auc_mean,  auc_std  = mets["auc"].mean(),        mets["auc"].std(ddof=0)

    y_true_all = np.concatenate(all_oof_true)
    y_prob_all = np.concatenate(all_oof_prob)
    pi_all     = np.concatenate(all_oof_pi)
    mr_all     = np.concatenate(all_oof_mr)
    fold_all   = np.concatenate(all_oof_fold)

    auc_global = roc_auc_score(y_true_all, y_prob_all)
    total_sec  = time.perf_counter() - overall_start

    print("\n========== K-Fold Summary ==========")
    print(mets[["fold","acc","precision","recall","f1","auc","thr","train_sec","test_sec","fold_sec"]])
    print(f"\nMean Acc: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"Mean Precision: {prec_mean:.4f} ± {prec_std:.4f}")
    print(f"Mean Recall: {rec_mean:.4f} ± {rec_std:.4f}")
    print(f"Mean F1: {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"Mean AUC: {auc_mean:.4f} ± {auc_std:.4f}")
    print(f"Global OOF AUC: {auc_global:.4f}")
    print(f"Total wall time: {_fmt_seconds(total_sec)}")

    # ---- save CSVs ----
    mets.to_csv("cv_fold_metrics_struct_delta21.csv", index=False)
    pd.DataFrame({
        "fold": fold_all,
        "piRNA_seq": pi_all,
        "site_seq": mr_all,
        "y_true": y_true_all,
        "y_prob": y_prob_all
    }).to_csv("oof_predictions_struct_delta21.csv", index=False)
    print("Saved: cv_fold_metrics_struct_delta21.csv, oof_predictions_struct_delta21.csv")

    # ---------------- PLOT epoch-wise learning curves (mean ± SD) ----------------
    max_len = max(h.shape[0] for h in all_train_hist)
    train_matrix = np.stack([np.pad(h, (0, max_len - h.shape[0]), 'edge') for h in all_train_hist])
    val_matrix   = np.stack([np.pad(h, (0, max_len - h.shape[0]), 'edge') for h in all_val_hist])
    train_mean, train_std = train_matrix.mean(axis=0), train_matrix.std(axis=0)
    val_mean,   val_std   = val_matrix.mean(axis=0),   val_matrix.std(axis=0)

    epochs = np.arange(1, max_len+1)
    plt.figure(figsize=(6,4))
    plt.plot(epochs, train_mean, color="green", label="Training Accuracy")
    plt.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.15, color="green")
    plt.plot(epochs, val_mean, color="orange", label="Validation Accuracy")
    plt.fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.15, color="orange")
    plt.xlabel("epoch"); plt.ylabel("Accuracy"); plt.ylim(0.5,1.0)
    plt.legend(); plt.tight_layout()
    plt.savefig("learning_curves.tiff", dpi=300); plt.close()
    print("Saved learning curves: learning_curves.tiff")

    # ---------------- PLOT ROC curves (validation + test per fold) ----------------
    plt.figure(figsize=(6,6))
    for e in roc_entries:
        plt.plot(e["fpr_val"], e["tpr_val"], color="orange", alpha=0.25, linewidth=1)
    for e in roc_entries:
        plt.plot(e["fpr_test"], e["tpr_test"], color="blue", alpha=0.25, linewidth=1)

    grid = np.linspace(0,1,100)
    interp_val, interp_test = [], []
    for e in roc_entries:
        interp_val.append(np.interp(grid, e["fpr_val"],  e["tpr_val"]))
        interp_test.append(np.interp(grid, e["fpr_test"], e["tpr_test"]))
    if len(interp_val) > 0:
        mean_val  = np.mean(interp_val,  axis=0)
        mean_test = np.mean(interp_test, axis=0)
        plt.plot(grid, mean_val,  color="darkorange",
                 label=f"Val ROC mean (AUC={np.mean([e['auc_val'] for e in roc_entries]):.3f})", linewidth=2)
        plt.plot(grid, mean_test, color="blue",
                 label=f"Test ROC mean (AUC={np.mean([e['auc_test'] for e in roc_entries]):.3f})", linewidth=2)
    plt.plot([0,1],[0,1], linestyle='--', color='gray')
    plt.xlabel("1 - Specificity (FPR)"); plt.ylabel("Sensitivity (TPR)")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig("roc_curves.tiff", dpi=300); plt.close()
    print("Saved ROC curves: roc_curves.tiff")

    # ---------------- PLOT batch-level training curve for epoch 1 (robust) ----------------
    if len(epoch1_train_batches_all) > 0:
        # Normalize inputs: keep only non-empty arrays and flatten
        train_lists = []
        for arr in epoch1_train_batches_all:
            if arr is None:
                continue
            a = np.asarray(arr).ravel()
            if a.size > 0 and np.isfinite(a).any():
                train_lists.append(a)
    
        if len(train_lists) > 0:
            max_b_tr = max(len(a) for a in train_lists)
            # pad within training group only (so we can avg across folds)
            tr_mat = np.vstack([
                np.pad(a, (0, max_b_tr - len(a)), mode="edge")
                for a in train_lists
            ])
            tr_mean, tr_std = tr_mat.mean(axis=0), tr_mat.std(axis=0)
            batches_tr = np.arange(1, max_b_tr + 1)
    
            plt.figure(figsize=(6,4))
            h1, = plt.plot(batches_tr, tr_mean, label="Training Acc (epoch 1)")
            plt.fill_between(batches_tr, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15)
    
            plt.xlabel("batch index (epoch 1)")
            plt.ylabel("Accuracy")
            plt.ylim(0.5, 1.0)
    
            # only show legend if we actually plotted something
            if h1 is not None:
                plt.legend()
    
            plt.tight_layout()
            plt.savefig("learning_curve_epoch1_train_only.tiff", dpi=300)
            plt.close()
            print("Saved batch-level training curve for epoch 1: learning_curve_epoch1_train_only.tiff")
        else:
            print("No non-empty epoch-1 training batch arrays to plot.")
    else:
        print("epoch1_train_batches_all is empty; nothing to plot for epoch-1 batches.")

    return {
        "per_fold": mets,
        "acc_mean": acc_mean, "acc_std": acc_std,
        "precision_mean": prec_mean, "precision_std": prec_std,
        "recall_mean": rec_mean, "recall_std": rec_std,
        "f1_mean": f1_mean, "f1_std": f1_std,
        "auc_mean": auc_mean, "auc_std": auc_std,
        "auc_global_oof": auc_global,
        "total_sec": total_sec
    }



def can_modify_csv(filename="test_permission.csv"):
    try:
        # 1. Try to open file in append mode (creates if doesn't exist)
        with open(filename, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["test"])  # write a dummy row

        # 2. Try to delete the file if it was just a test
        if filename == "test_permission.csv":
            os.remove(filename)

        return True
    except Exception as e:
        print("Error:", e)
        return False
    
# ------------------------- Main -------------------------
if __name__ == "__main__":
    if can_modify_csv():
        print("Script CAN modify CSV files in this folder.")
    else:
        print("Script CANNOT modify CSV files in this folder.")
    POS_PATH = "WT_CLASH_positive.csv"
    NEG_PATH = "WT_CLASH_negative.csv"
    assert os.path.exists(POS_PATH), f"Not found: {POS_PATH}"
    assert os.path.exists(NEG_PATH), f"Not found: {NEG_PATH}"

    df_pos = load_pair_csv(POS_PATH, label=1)
    df_neg = load_pair_csv(NEG_PATH, label=0)

    for df in (df_pos, df_neg):
        df["piRNA_seq"] = df["piRNA_seq"].astype(str).map(clean_seq)
        df["site_seq"]  = df["site_seq"].astype(str).map(clean_seq)

    df_pos = df_pos[(df_pos["piRNA_seq"].str.len()>0) & (df_pos["site_seq"].str.len()>0)].reset_index(drop=True)
    df_neg = df_neg[(df_neg["piRNA_seq"].str.len()>0) & (df_neg["site_seq"].str.len()>0)].reset_index(drop=True)

    df_all = pd.concat([df_pos, df_neg], ignore_index=True)

    print(f"Dataset size: {len(df_all)} | Positives: {int(df_all['label'].sum())} | "
          f"Negatives: {len(df_all)-int(df_all['label'].sum())}")

    df_all = df_all.sample(frac=1, random_state=SEED).reset_index(drop=True)

    kfold_training(
        df_all=df_all,
        df_pos=df_pos,
        df_neg=df_neg,
        K=5,
        inner_val_size=0.15,
        batch_size=64,
        max_epochs=30,     # shorter default than before
        patience=4,        # earlier stop to avoid overfitting
        lr=1e-3,
        use_focal=True,
        seed_len=7,
        dropout=0.50,      # stronger dropout
        weight_decay=1e-4  # L2 regularization
    )


