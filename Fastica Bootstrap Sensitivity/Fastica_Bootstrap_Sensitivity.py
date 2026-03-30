#!/usr/bin/env python3
# fastica_bootstrap_grid.py
# Runs FastICA + blocks bootstrap for multiple configurations (BLOCK_SIZE, N_BOOT)
# Saved: comparative JSON + PNG results
import os
import json
import numpy as np
from scipy.io import loadmat
from scipy.stats import kurtosis
from scipy.signal import welch, hilbert, butter, filtfilt
from sklearn.decomposition import FastICA
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
import warnings
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# -------------------------
# User settings
# -------------------------
MAT_FILES = ["ch1.mat", "ch2.mat", "ch3.mat"]
FS = 50000
WINDOW_N = 5000
OUTDIR = "Fastica Bootstrap Results"
os.makedirs(OUTDIR, exist_ok=True)

# Test grid (editable)
test_configs = [
    {"BLOCK_SIZE": 500,  "N_BOOT": 1000},
    {"BLOCK_SIZE": 1000, "N_BOOT": 1000},
    {"BLOCK_SIZE": 1500, "N_BOOT": 1000},
    {"BLOCK_SIZE": 1000, "N_BOOT": 2000},
    {"BLOCK_SIZE": 2500, "N_BOOT": 1000},
]

RANDOM_STATE = 0

# -------------------------
# Utility functions
# -------------------------
def load_mat_signal(path):
    d = loadmat(path)
    keys = [k for k in d.keys() if not k.startswith("__")]
    # Current priorities
    for pref in ["data", "signal", "x", "y", "acc"]:
        if pref in d:
            return np.ravel(d[pref])
    if len(keys) == 0:
        raise ValueError(f"Aucune variable trouvée dans {path}")
    return np.ravel(d[keys[0]])

def apply_fastica(X, n_components=3, random_state=0):
    ica = FastICA(n_components=n_components, fun='exp', algorithm='parallel',
                  tol=1e-4, max_iter=1000, random_state=random_state)
    Y = ica.fit_transform(X.T).T
    try:
        whitened = ica.transform(X.T).T
    except Exception:
        whitened = Y.copy()
    return Y, whitened, ica

def spectral_corr(a, b, fs):
    fa, Pa = welch(a, fs=fs, nperseg=2048)
    fb, Pb = welch(b, fs=fs, nperseg=2048)
    if np.std(Pa) < 1e-12 or np.std(Pb) < 1e-12:
        return 0.0
    return np.corrcoef(Pa, Pb)[0,1]

def match_sources(S_true, Y_est, fs):
    n = S_true.shape[0]
    score = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            p = pearsonr(S_true[i], Y_est[j])[0] if np.std(S_true[i])>0 and np.std(Y_est[j])>0 else 0.0
            sc = spectral_corr(S_true[i], Y_est[j], fs)
            score[i, j] = 0.5 * abs(p) + 0.5 * abs(sc)
    row_ind, col_ind = linear_sum_assignment(-score)
    return row_ind, col_ind, score

def compute_kri(ku_mix, ku_sep):
    ku_mix = np.array(ku_mix)
    ku_sep = np.array(ku_sep)
    eps = 1e-12
    rel_err = np.abs(ku_sep - ku_mix) / (np.abs(ku_mix) + eps)
    kri = 100.0 * (1.0 - np.mean(rel_err))
    return kri

def spectral_kurtosis_score(x, fs, n_bands=32):
    freqs = np.geomspace(20, fs/2, n_bands)
    kurt_vals = []
    for i in range(len(freqs)-1):
        f1, f2 = freqs[i], freqs[i+1]
        nyq = fs/2
        low = max(f1/nyq, 1e-6)
        high = min(f2/nyq, 0.999)
        if low >= high:
            continue
        b, a = butter(3, [low, high], btype='band')
        try:
            xb = filtfilt(b, a, x)
        except Exception:
            xb = np.convolve(x, b, mode='same')
        env = np.abs(hilbert(xb))
        kurt_vals.append(kurtosis(env, fisher=False))
    if len(kurt_vals) == 0:
        return 0.0, np.array([])
    return float(np.max(kurt_vals)), np.array(kurt_vals)

# -------------------------
# Loading Signals
# -------------------------
print("Loading files .mat...")
signals = []
for f in MAT_FILES:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Fichier introuvable: {f}")
    s = load_mat_signal(f)
    signals.append(s)
minlen = min(len(s) for s in signals)
if minlen < WINDOW_N:
    raise ValueError(f"The signals have fewer than {WINDOW_N} samples. Adjust WINDOW_N or provide longer signals.")
S_true = np.vstack([s[:WINDOW_N] for s in signals])
S_true = S_true - np.mean(S_true, axis=1, keepdims=True)
X = S_true.copy()
print(f"Signals loaded shape {S_true.shape}")

# Single-point calculation (FastICA on the window)
Y, whitened, ica_model = apply_fastica(X, n_components=3, random_state=RANDOM_STATE)
row, col, score_mat = match_sources(S_true, Y, FS)
ku_mix = kurtosis(S_true, axis=1, fisher=False)
ku_sep = kurtosis(Y, axis=1, fisher=False)[col]
KRI_point = compute_kri(ku_mix, ku_sep)
PCC_point = np.corrcoef(ku_mix, ku_sep)[0,1]
RMSE_point = float(np.sqrt(np.mean((ku_sep - ku_mix)**2)))

# SK on mixed channels (single value)
sk_vals = []
for ch in range(X.shape[0]):
    sk_val, _ = spectral_kurtosis_score(X[ch], FS, n_bands=32)
    sk_vals.append(sk_val)
SK_max_mixed = float(max(sk_vals))

# -------------------------
# Bootstrap grid loop
# -------------------------
grid_results = {}
for cfg in test_configs:
    BLOCK_SIZE = int(cfg["BLOCK_SIZE"])
    N_BOOT = int(cfg["N_BOOT"])
    print(f"Running config BLOCK_SIZE={BLOCK_SIZE}, N_BOOT={N_BOOT} ...")
    rng = np.random.default_rng(RANDOM_STATE)
    kri_boot = []; pcc_boot = []; rmse_boot = []
    n_blocks = WINDOW_N // BLOCK_SIZE
    use_blocks = n_blocks >= 2
    block_indices = [np.arange(i*BLOCK_SIZE, (i+1)*BLOCK_SIZE) for i in range(n_blocks)] if use_blocks else None

    for _ in trange(N_BOOT, desc=f"Bootstrap L={BLOCK_SIZE}"):
        if use_blocks:
            chosen = rng.integers(0, n_blocks, size=n_blocks)
            idx = np.concatenate([block_indices[c] for c in chosen])
        else:
            idx = rng.integers(0, WINDOW_N, size=WINDOW_N)
        S_b = S_true[:, idx]
        X_b = X[:, idx]
        try:
            Y_b, _, _ = apply_fastica(X_b, n_components=3, random_state=RANDOM_STATE)
        except Exception:
            continue
        _, col_b, _ = match_sources(S_b, Y_b, FS)
        ku_mix_b = kurtosis(S_b, axis=1, fisher=False)
        ku_sep_b = kurtosis(Y_b, axis=1, fisher=False)[col_b]
        kri_b = compute_kri(ku_mix_b, ku_sep_b)
        pcc_b = np.corrcoef(ku_mix_b, ku_sep_b)[0,1]
        rmse_b = float(np.sqrt(np.mean((ku_sep_b - ku_mix_b)**2)))
        kri_boot.append(kri_b); pcc_boot.append(float(pcc_b)); rmse_boot.append(rmse_b)

    def ci95(arr):
        arr = np.array(arr)
        if arr.size == 0:
            return (np.nan, np.nan, np.nan)
        return (float(np.percentile(arr,50)), float(np.percentile(arr,2.5)), float(np.percentile(arr,97.5)))

    kri_med, kri_lo, kri_hi = ci95(kri_boot)
    pcc_med, pcc_lo, pcc_hi = ci95(pcc_boot)
    rmse_med, rmse_lo, rmse_hi = ci95(rmse_boot)

    key = f"L{BLOCK_SIZE}_B{N_BOOT}"
    grid_results[key] = {
        "BLOCK_SIZE": BLOCK_SIZE,
        "N_BOOT": N_BOOT,
        "kri_point": None,  # point estimate recomputed below if desired
        "kri_boot": {"median": kri_med, "ci95_lo": kri_lo, "ci95_hi": kri_hi},
        "pcc_boot": {"median": pcc_med, "ci95_lo": pcc_lo, "ci95_hi": pcc_hi},
        "rmse_boot": {"median": rmse_med, "ci95_lo": rmse_lo, "ci95_hi": rmse_hi},
        "n_blocks": n_blocks,
        "n_successful_boot": len(kri_boot)
    }

# -------------------------
# JSON backup summary
# -------------------------
summary = {
    "files": MAT_FILES,
    "fs": FS,
    "window_N": WINDOW_N,
    "point_metrics": {
        "KRI_point": float(KRI_point),
        "PCC_point": float(PCC_point),
        "RMSE_point": float(RMSE_point),
        "SK_max_mixed": SK_max_mixed,
        "match": {"row_indices": list(map(int,row.tolist())), "col_indices": list(map(int,col.tolist()))}
    },
    "grid_results": grid_results
}
with open(os.path.join(OUTDIR, "grid_results_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# -------------------------
# Comparative figures
# -------------------------
# Prepare arrays for plotting
labels = []
kri_meds = []; kri_lo = []; kri_hi = []
pcc_meds = []; pcc_lo = []; pcc_hi = []
rmse_meds = []; rmse_lo = []; rmse_hi = []

for k, v in grid_results.items():
    labels.append(k)
    kri_meds.append(v["kri_boot"]["median"]); kri_lo.append(v["kri_boot"]["ci95_lo"]); kri_hi.append(v["kri_boot"]["ci95_hi"])
    pcc_meds.append(v["pcc_boot"]["median"]); pcc_lo.append(v["pcc_boot"]["ci95_lo"]); pcc_hi.append(v["pcc_boot"]["ci95_hi"])
    rmse_meds.append(v["rmse_boot"]["median"]); rmse_lo.append(v["rmse_boot"]["ci95_lo"]); rmse_hi.append(v["rmse_boot"]["ci95_hi"])

x = np.arange(len(labels))
plt.figure(figsize=(10,6))
plt.errorbar(x, kri_meds, yerr=[np.array(kri_meds)-np.array(kri_lo), np.array(kri_hi)-np.array(kri_meds)],
             fmt='o', capsize=5, label='KRI (%)', color='C0')
plt.axhline(y=KRI_point, color='C0', linestyle='--', alpha=0.5)
plt.xticks(x, labels, rotation=30)
plt.ylabel("KRI (%)")
plt.title("Comparison of KRI (median + IC95) for different block sizes / B")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "kri_grid_comparison.png"), dpi=200)
plt.close()

plt.figure(figsize=(10,6))
plt.errorbar(x, pcc_meds, yerr=[np.array(pcc_meds)-np.array(pcc_lo), np.array(pcc_hi)-np.array(pcc_meds)],
             fmt='o', capsize=5, label='PCC', color='C1')
plt.axhline(y=PCC_point, color='C1', linestyle='--', alpha=0.5)
plt.xticks(x, labels, rotation=30)
plt.ylabel("PCC (median + IC95)")
plt.title("Comparison of PCC (median + IC95) for different block sizes / B")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "pcc_grid_comparison.png"), dpi=200)
plt.close()

plt.figure(figsize=(10,6))
plt.errorbar(x, rmse_meds, yerr=[np.array(rmse_meds)-np.array(rmse_lo), np.array(rmse_hi)-np.array(rmse_meds)],
             fmt='o', capsize=5, label='RMSE', color='C2')
plt.axhline(y=RMSE_point, color='C2', linestyle='--', alpha=0.5)
plt.xticks(x, labels, rotation=30)
plt.ylabel("RMSE (median + IC95)")
plt.title("Comparison of RMSE (median + IC95) for different block sizes / B")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "rmse_grid_comparison.png"), dpi=200)
plt.close()

# Summary heatmap (KRI medians)
plt.figure(figsize=(6,4))
sns.heatmap(np.array(kri_meds).reshape(1,-1), annot=True, fmt=".2f", cmap="viridis", xticklabels=labels, yticklabels=["KRI_med"])
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "kri_median_heatmap.png"), dpi=200)
plt.close()

print(f"Finished. Results saved in: {OUTDIR}")
print("Key files: grid_results_summary.json, kri_grid_comparison.png, pcc_grid_comparison.png, rmse_grid_comparison.png, kri_median_heatmap.png")
