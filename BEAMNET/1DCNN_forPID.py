#!/usr/bin/env python3
"""
train_pid_cnn.py
=================
Train a compact 1D-CNN on the EXACT same 3-channel windowed waveforms
used by waveform_shape_pid.py:
  channels  : Quartz=104, Plastic=010, Scintillator=107  (3 mm)
  window    : [t50 - 15 ns,  t50 + 25 ns]  →  200 bins @ 0.2 ns/bin
  same cuts : peak >= 100 ADC, trough >= -100 ADC, t50 valid

Classes: pion (0)  vs  positron (1)

Outputs (all in <outdir>/)
--------------------------
  best_model.pt
  score_distributions.pdf   — P(positron) overlaid for both classes
  roc_curve.pdf
  loss_accuracy.pdf         — train/val loss + accuracy vs epoch
  run_summary.txt

Usage
-----
  python train_pid_cnn.py --outdir ./pid_cnn_out --n-events 5000 --epochs 40
"""

import os, re, argparse, warnings
import numpy as np
import uproot
import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")
try:
    import mplhep as hep
    plt.style.use(hep.style.CMS)
except ImportError:
    pass

# =============================================================================
# MIRROR EXACTLY waveform_shape_pid.py settings
# =============================================================================
TREE_NAME     = "EventTree"
TIME_PER_BIN  = 0.2
BASELINE_BINS = 30
TIMING_SUFFIX = "_LP2_50"
AMP_THRESHOLD = 100.0
MIN_ADC_CUT   = -100.0
TWINDOW_LEFT  = 15.0
TWINDOW_RIGHT = 25.0
N_BINS        = int((TWINDOW_LEFT + TWINDOW_RIGHT) / TIME_PER_BIN)  # 200

CHANNELS_3MM  = {"Quartz": "104", "Plastic": "010", "Scintillator": "107"}
FAMILIES      = ["Quartz", "Plastic", "Scintillator"]

RUN_FILES = {
    "positron": "/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/"
                "run1355_250924165834_converted_timingskim.root",
    "pion":     "/lustre/research/hep/akshriva/Dream-Timing/PostTimingFitsNtuples/"
                "run1429_250926183919_converted_timingskim.root",
}

PID_BRANCH_MAP = {
    "PSD":         "DRS_Board7_Group1_Channel1",
    "TTUMuonVeto": "DRS_Board7_Group2_Channel4",
    "Cer474":      "DRS_Board7_Group2_Channel5",
    "Cer519":      "DRS_Board7_Group2_Channel6",
    "Cer537":      "DRS_Board7_Group2_Channel7",
}
_SVC_CUTS = {
    "PSD":         (100, 400, -3500., "Sum"),
    "TTUMuonVeto": (200, 400, -2e3,   "Sum"),
    "Cer474":      (800, 900, -2000., "Sum"),
    "Cer519":      (450, 550, -1000., "Sum"),
    "Cer537":      (400, 500, -500.,  "Sum"),
}
_PARTICLE_REQS = {
    "positron": {"TTUMuonVeto": False, "PSD": True,
                 "Cer474": True, "Cer519": True, "Cer537": True},
    "pion":     {"TTUMuonVeto": False, "PSD": False,
                 "Cer474": True, "Cer519": True, "Cer537": True},
}

COLORS = {"positron": "#E45C3A", "pion": "#3A8FE4"}

# =============================================================================
# PID  (identical to analysis script)
# =============================================================================

def code_to_branch(code):
    s = str(code).zfill(3)
    return f"DRS_Board{s[0]}_Group{s[1]}_Channel{s[2]}"


def compute_pid_mask(tree, particle):
    reqs = _PARTICLE_REQS.get(particle, {})
    mask = np.ones(tree.num_entries, dtype=bool)
    for det, must_fire in reqs.items():
        br = PID_BRANCH_MAP.get(det)
        if not br or br not in tree.keys():
            continue
        ts_min, ts_max, val_cut, _ = _SVC_CUTS.get(det, (0, 1000, -5e4, "Sum"))
        try:
            waves   = tree[br].array(library="ak")
            bl      = ak.mean(waves[:, :BASELINE_BINS], axis=1)
            wsum    = ak.sum((waves - bl)[:, int(ts_min):int(ts_max)], axis=1)
            fired   = ak.to_numpy(wsum) < val_cut
            mask   &= fired if must_fire else ~fired
        except Exception:
            pass
    return mask

# =============================================================================
# DATA LOADING  — same window extraction as the plotting script
# =============================================================================

def load_particle(fpath, particle, n_max):
    """
    Returns X: (N, 3, 200)  float32
    Channel order: [Quartz, Plastic, Scintillator]
    Extraction logic is byte-for-byte identical to collect_metrics().
    """
    samples = []

    with uproot.open(fpath) as f:
        tree     = f[TREE_NAME]
        pid_mask = compute_pid_mask(tree, particle)
        pid_idxs = np.where(pid_mask)[0]
        print(f"  [{particle}] PID pass {len(pid_idxs)}/{tree.num_entries}")

        # Load all 3 families upfront
        family_arrays = {}
        for fam in FAMILIES:
            code  = CHANNELS_3MM[fam]
            br    = code_to_branch(code)
            t50br = br + TIMING_SUFFIX
            if br not in tree.keys() or t50br not in tree.keys():
                print(f"  [WARN] {fam} branch missing")
                family_arrays[fam] = None
                continue
            waves_ak = tree[br].array(library="ak")
            bl       = ak.mean(waves_ak[:, :BASELINE_BINS], axis=1)
            w_np     = ak.to_numpy(waves_ak - bl)
            t50_arr  = tree[t50br].array(library="np")
            peak_np  = ak.to_numpy(ak.max(waves_ak - bl, axis=1))
            trough_np= ak.to_numpy(ak.min(waves_ak - bl, axis=1))
            family_arrays[fam] = dict(w=w_np, t50=t50_arr,
                                      peak=peak_np, trough=trough_np)

        n_collected = 0
        for ev in pid_idxs:
            if n_collected >= n_max:
                break

            # Reference t50: first family with a valid pulse (same logic as plotter)
            ref_t50 = None
            for fam in FAMILIES:
                fd = family_arrays.get(fam)
                if fd is None:
                    continue
                t50 = float(fd["t50"][ev])
                if (np.isfinite(t50) and t50 > 0
                        and fd["peak"][ev] >= AMP_THRESHOLD
                        and fd["trough"][ev] >= MIN_ADC_CUT):
                    ref_t50 = t50
                    break
            if ref_t50 is None:
                continue

            i_lo = int((ref_t50 - TWINDOW_LEFT) / TIME_PER_BIN)
            i_hi = i_lo + N_BINS

            channels = []
            skip = False
            for fam in FAMILIES:
                fd = family_arrays.get(fam)
                if fd is None:
                    channels.append(np.zeros(N_BINS, dtype=np.float32))
                    continue
                wf = fd["w"][ev]
                if i_lo < 0 or i_hi > len(wf):
                    skip = True; break
                # zero-fill if this family's pulse is bad
                if (fd["peak"][ev] < AMP_THRESHOLD
                        or fd["trough"][ev] < MIN_ADC_CUT):
                    channels.append(np.zeros(N_BINS, dtype=np.float32))
                else:
                    channels.append(wf[i_lo:i_hi].astype(np.float32))

            if skip:
                continue

            samples.append(np.stack(channels, axis=0))  # (3, 200)
            n_collected += 1

    print(f"  [{particle}] collected {n_collected} events")
    return np.stack(samples, axis=0) if samples else np.empty((0, 3, N_BINS))


def load_dataset(n_max):
    X_pos = load_particle(RUN_FILES["positron"], "positron", n_max)
    X_pi  = load_particle(RUN_FILES["pion"],     "pion",     n_max)

    X = np.concatenate([X_pos, X_pi], axis=0)
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_pi))]).astype(np.float32)

    # Peak-normalise each channel of each event independently
    # (same spirit as the histogram peak-normalisation in the plotting script)
    peak = np.abs(X).max(axis=2, keepdims=True).clip(min=1.0)
    X    = X / peak

    return X.astype(np.float32), y

# =============================================================================
# MODEL  — compact 1D CNN
# =============================================================================

class PIDNet(nn.Module):
    """
    Input : (B, 3, 200)  — 3 fiber families × 200 time bins
    Output: (B,)         — logit for positron

    Architecture (quick but effective):
      Stem  : Conv(k=11) → BN → ReLU → MaxPool/2   [200→100 bins, 3→32 ch]
      Block1: Conv(k=7)  → BN → ReLU               [100 bins, 32 ch]
      Block2: Conv(k=5)  → BN → ReLU → MaxPool/2   [100→50 bins, 64 ch]
      Block3: Conv(k=3)  → BN → ReLU               [50 bins, 128 ch]
      Head  : GlobalAvgPool → Dropout(0.4) → Linear(1)
    """
    def __init__(self, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            # stem
            nn.Conv1d(3,   32, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),

            # block 1
            nn.Conv1d(32,  32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(),

            # block 2
            nn.Conv1d(32,  64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),

            # block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.net(x)               # (B, 128, 50)
        x = x.mean(dim=-1)            # global average pool → (B, 128)
        return self.head(x).squeeze(1)  # (B,)

# =============================================================================
# TRAINING
# =============================================================================

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            
            logits = model(xb)
            loss   = criterion(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(yb)
            correct    += ((logits > 0) == (yb > 0.5)).sum().item()
            n          += len(yb)
    return total_loss / n, correct / n

# =============================================================================
# PLOTS
# =============================================================================

def plot_loss_accuracy(history, outdir):
    epochs = range(1, len(history["tl"]) + 1)
    pdf    = os.path.join(outdir, "loss_accuracy.pdf")
    with PdfPages(pdf) as pp:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(epochs, history["tl"], lw=2,   label="Train loss")
        ax1.plot(epochs, history["vl"], lw=2, ls="--", label="Val loss")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("BCE Loss")
        ax1.legend(); ax1.grid(alpha=0.3)
        ax1.set_title("Loss")

        ax2.plot(epochs, history["ta"], lw=2,   label="Train acc")
        ax2.plot(epochs, history["va"], lw=2, ls="--", label="Val acc")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0.5, 1.02)
        ax2.legend(); ax2.grid(alpha=0.3)
        ax2.set_title("Accuracy")

        fig.suptitle("Training history", fontsize=13)
        fig.tight_layout()
        pp.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    print(f"  [OK] {pdf}")


def plot_scores(scores, labels, outdir):
    pdf = os.path.join(outdir, "score_distributions.pdf")
    with PdfPages(pdf) as pp:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, yscale in zip(axes, ["linear", "log"]):
            for name, label, color in [("Positron", 1, COLORS["positron"]),
                                        ("Pion",     0, COLORS["pion"])]:
                vals = scores[labels == label]
                counts, edges = np.histogram(vals, bins=50, range=(0, 1))
                norm   = counts / counts.max().clip(min=1)
                centers= 0.5 * (edges[:-1] + edges[1:])
                ax.fill_between(centers, norm, step="mid",
                                color=color, alpha=0.25)
                ax.step(centers, norm, where="mid",
                        color=color, lw=2, label=name)

            ax.set_xlabel("P(positron)")
            ax.set_ylabel("Peak-normalised events")
            ax.set_yscale(yscale)
            ax.set_xlim(0, 1)
            ax.legend(fontsize=11)
            ax.set_title(f"Score distribution  ({yscale} scale)")
            ax.grid(alpha=0.3)

        fig.tight_layout()
        pp.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    print(f"  [OK] {pdf}")


def plot_roc(scores, labels, auc, outdir):
    fpr, tpr, _ = roc_curve(labels, scores)
    pdf = os.path.join(outdir, "roc_curve.pdf")
    with PdfPages(pdf) as pp:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, lw=2.5, color=COLORS["positron"],
                label=f"CNN  AUC = {auc:.4f}")
        ax.fill_between(fpr, tpr, alpha=0.10, color=COLORS["positron"])
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random")
        ax.set_xlabel("False positive rate  (pion → positron)")
        ax.set_ylabel("True positive rate  (positron efficiency)")
        ax.set_title("ROC curve  —  Pion vs Positron\n"
                     "3-channel waveform CNN  (Quartz | Plastic | Scintillator)")
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        pp.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    print(f"  [OK] {pdf}")


# =============================================================================
# INTERPRETABILITY: CHANNEL IMPORTANCE / RANKING
# =============================================================================

def evaluate_model_on_arrays(model, X_eval, y_eval, device, batch_size=256):
    """
    Returns AUC, accuracy, and scores for a given X/y array.
    """
    model.eval()

    ds = TensorDataset(
        torch.from_numpy(X_eval.astype(np.float32)),
        torch.from_numpy(y_eval.astype(np.float32)),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            scores = torch.sigmoid(logits).cpu().numpy()

            all_scores.extend(scores)
            all_labels.extend(yb.numpy())

    all_scores = np.asarray(all_scores)
    all_labels = np.asarray(all_labels)

    auc = roc_auc_score(all_labels, all_scores)
    acc = ((all_scores > 0.5) == all_labels).mean()

    return auc, acc, all_scores


def channel_importance_ranking(model, X_val, y_val, device, outdir, batch_size=256, seed=42):
    """
    Ranks Quartz / Plastic / Scintillator importance.

    Method 1:
      Ablation: set one channel to zero.
      Bigger AUC drop = more important channel.

    Method 2:
      Permutation: shuffle one channel across events.
      Bigger AUC drop = more important channel.
    """
    rng = np.random.default_rng(seed)

    channel_names = list(FAMILIES)

    base_auc, base_acc, base_scores = evaluate_model_on_arrays(
        model, X_val, y_val, device, batch_size=batch_size
    )

    rows = []

    print("\n" + "=" * 70)
    print("[INTERPRETABILITY] Channel importance ranking")
    print("=" * 70)
    print(f"Baseline AUC = {base_auc:.5f}")
    print(f"Baseline Acc = {base_acc:.5f}")

    for ch_idx, ch_name in enumerate(channel_names):
        # ------------------------------------------------------------
        # 1. Ablation importance
        # ------------------------------------------------------------
        X_zero = X_val.copy()
        X_zero[:, ch_idx, :] = 0.0

        auc_zero, acc_zero, _ = evaluate_model_on_arrays(
            model, X_zero, y_val, device, batch_size=batch_size
        )

        delta_auc_zero = base_auc - auc_zero
        delta_acc_zero = base_acc - acc_zero

        rows.append({
            "channel": ch_name,
            "method": "zero_ablation",
            "auc_modified": auc_zero,
            "acc_modified": acc_zero,
            "delta_auc": delta_auc_zero,
            "delta_acc": delta_acc_zero,
        })

        # ------------------------------------------------------------
        # 2. Permutation importance
        # ------------------------------------------------------------
        X_perm = X_val.copy()
        perm_idx = rng.permutation(len(X_perm))
        X_perm[:, ch_idx, :] = X_perm[perm_idx, ch_idx, :]

        auc_perm, acc_perm, _ = evaluate_model_on_arrays(
            model, X_perm, y_val, device, batch_size=batch_size
        )

        delta_auc_perm = base_auc - auc_perm
        delta_acc_perm = base_acc - acc_perm

        rows.append({
            "channel": ch_name,
            "method": "permutation",
            "auc_modified": auc_perm,
            "acc_modified": acc_perm,
            "delta_auc": delta_auc_perm,
            "delta_acc": delta_acc_perm,
        })

    # ------------------------------------------------------------
    # Print ranked tables
    # ------------------------------------------------------------
    for method in ["zero_ablation", "permutation"]:
        method_rows = [r for r in rows if r["method"] == method]
        method_rows = sorted(method_rows, key=lambda r: r["delta_auc"], reverse=True)

        print("\n" + "-" * 70)
        print(f"Ranking by {method}: larger Delta AUC means more important")
        print("-" * 70)
        print(f"{'Rank':<6} {'Channel':<16} {'AUC after change':<18} {'Delta AUC':<12} {'Delta Acc':<12}")
        print("-" * 70)

        for rank, r in enumerate(method_rows, start=1):
            print(
                f"{rank:<6} "
                f"{r['channel']:<16} "
                f"{r['auc_modified']:<18.5f} "
                f"{r['delta_auc']:<12.5f} "
                f"{r['delta_acc']:<12.5f}"
            )

    # ------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------
    csv_path = os.path.join(outdir, "channel_importance_ranking.csv")
    with open(csv_path, "w") as f:
        f.write("channel,method,auc_modified,acc_modified,delta_auc,delta_acc\n")
        for r in rows:
            f.write(
                f"{r['channel']},"
                f"{r['method']},"
                f"{r['auc_modified']:.8f},"
                f"{r['acc_modified']:.8f},"
                f"{r['delta_auc']:.8f},"
                f"{r['delta_acc']:.8f}\n"
            )

    print(f"\n  [OK] saved {csv_path}")

    # ------------------------------------------------------------
    # Make ranking plot
    # ------------------------------------------------------------
    pdf_path = os.path.join(outdir, "channel_importance_ranking.pdf")

    with PdfPages(pdf_path) as pp:
        for method in ["zero_ablation", "permutation"]:
            method_rows = [r for r in rows if r["method"] == method]
            method_rows = sorted(method_rows, key=lambda r: r["delta_auc"], reverse=True)

            names = [r["channel"] for r in method_rows]
            drops = [r["delta_auc"] for r in method_rows]

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(names, drops)

            ax.set_ylabel("AUC drop")
            ax.set_xlabel("Channel removed / shuffled")
            ax.set_title(f"Channel importance ranking: {method}")
            ax.grid(axis="y", alpha=0.3)

            for i, v in enumerate(drops):
                ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=12)

            fig.tight_layout()
            pp.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"  [OK] saved {pdf_path}")

    # ------------------------------------------------------------
    # Add compact text summary
    # ------------------------------------------------------------
    txt_path = os.path.join(outdir, "channel_importance_summary.txt")

    perm_rows = [r for r in rows if r["method"] == "permutation"]
    perm_rows = sorted(perm_rows, key=lambda r: r["delta_auc"], reverse=True)

    abl_rows = [r for r in rows if r["method"] == "zero_ablation"]
    abl_rows = sorted(abl_rows, key=lambda r: r["delta_auc"], reverse=True)

    with open(txt_path, "w") as f:
        f.write("Channel importance summary\n")
        f.write("==========================\n\n")
        f.write(f"Baseline AUC = {base_auc:.6f}\n")
        f.write(f"Baseline Acc = {base_acc:.6f}\n\n")

        f.write("Permutation ranking, preferred:\n")
        for i, r in enumerate(perm_rows, start=1):
            f.write(
                f"  {i}. {r['channel']}: "
                f"Delta AUC = {r['delta_auc']:.6f}, "
                f"AUC after shuffle = {r['auc_modified']:.6f}\n"
            )

        f.write("\nZero-ablation ranking:\n")
        for i, r in enumerate(abl_rows, start=1):
            f.write(
                f"  {i}. {r['channel']}: "
                f"Delta AUC = {r['delta_auc']:.6f}, "
                f"AUC after zeroing = {r['auc_modified']:.6f}\n"
            )

    print(f"  [OK] saved {txt_path}")

    return rows

# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir",     default="./pid_cnn_out")
    ap.add_argument("--n-events",   type=int,   default=5000,
                    help="Max events per class")
    ap.add_argument("--epochs",     type=int,   default=30)
    ap.add_argument("--batch-size", type=int,   default=64)
    ap.add_argument("--lr",         type=float, default=3e-4)
    ap.add_argument("--val-frac",   type=float, default=0.20)
    ap.add_argument("--seed",       type=int,   default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*55}")
    print(f"  Pion vs Positron  |  3-channel waveform CNN")
    print(f"  Device : {device}")
    print(f"{'='*55}")

    # ── Data ───────────────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    X, y = load_dataset(args.n_events)
    print(f"  Total: {len(y)}  "
          f"(positron={int(y.sum())}, pion={int((1-y).sum())})")
    print(f"  Input shape: {X.shape}  "
          f"[{N_BINS*TIME_PER_BIN:.0f} ns window]")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=args.val_frac,
        random_state=args.seed, stratify=y
    )

    def make_loader(Xa, ya, shuffle):
        ds = TensorDataset(torch.from_numpy(Xa), torch.from_numpy(ya))
        return DataLoader(ds, batch_size=args.batch_size,
                          shuffle=shuffle, num_workers=2, pin_memory=True)

    train_loader = make_loader(X_tr,  y_tr,  shuffle=True)
    val_loader   = make_loader(X_val, y_val, shuffle=False)

    # ── Model ──────────────────────────────────────────────────────────────
    print("\n[2] Building model...")
    model     = PIDNet().to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    print(model)

    # Class-balanced BCE
    pos_weight = torch.tensor([(1 - y).sum() / y.sum()]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ── Training loop ──────────────────────────────────────────────────────
    print(f"\n[3] Training {args.epochs} epochs...")
    history  = dict(tl=[], vl=[], ta=[], va=[])
    best_auc = 0.0

    for epoch in range(1, args.epochs + 1):
        tl, ta = run_epoch(model, train_loader, criterion,
                           optimizer, device, train=True)
        vl, va = run_epoch(model, val_loader,   criterion,
                           optimizer, device, train=False)
        scheduler.step()

        # Quick AUC on val
        model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb.to(device))
                all_scores.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(yb.numpy())
        auc = roc_auc_score(all_labels, all_scores)

        history["tl"].append(tl); history["vl"].append(vl)
        history["ta"].append(ta); history["va"].append(va)

        if auc > best_auc:
            best_auc = auc
            torch.save({"model_state": model.state_dict(),
                        "epoch": int(epoch),
                        "auc": float(auc)
                        }, os.path.join(args.outdir, "best_model.pt"))
            flag = " ← best"
        else:
            flag = ""

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"  Ep {epoch:3d}/{args.epochs}  "
                  f"loss {tl:.4f}/{vl:.4f}  "
                  f"acc {ta:.3f}/{va:.3f}  "
                  f"AUC {auc:.4f}{flag}")

    # ── Final eval with best model ─────────────────────────────────────────
    print(f"\n[4] Evaluating best model (AUC={best_auc:.4f})...")
    # ckpt = torch.load(os.path.join(args.outdir, "best_model.pt"),
    #                   map_location=device)
    ckpt = torch.load(os.path.join(args.outdir, "best_model.pt"),
                      map_location=device,weights_only=False,)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_scores, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb.to(device))
            all_scores.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(yb.numpy())

    scores = np.array(all_scores)
    labels = np.array(all_labels)
    auc    = roc_auc_score(labels, scores)
    acc    = ((scores > 0.5) == labels).mean()

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n[5] Saving plots...")
    plot_loss_accuracy(history, args.outdir)
    plot_scores(scores, labels, args.outdir)
    plot_roc(scores, labels, auc, args.outdir)

    # ── Interpretability: which channel matters most? ─────────────────────
    print("\n[6] Running channel-importance ranking...")
    channel_importance_ranking(
        model=model,
        X_val=X_val,
        y_val=y_val,
        device=device,
        outdir=args.outdir,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    summary = (f"Best epoch : {ckpt['epoch']}\n"
               f"Val AUC    : {auc:.4f}\n"
               f"Val Acc    : {acc:.4f}\n"
               f"N train    : {len(y_tr)}\n"
               f"N val      : {len(y_val)}\n"
               f"Params     : {n_params:,}\n")
    print("\n" + summary)
    with open(os.path.join(args.outdir, "run_summary.txt"), "w") as fh:
        fh.write(summary)

    print(f"[DONE]  outputs → {args.outdir}/")


if __name__ == "__main__":
    main()