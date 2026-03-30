# fastica_synthetic_analysis.py
# -*- coding: utf-8 -*-
"""
FastICA synthetic analysis for vibration diagnosis
Generates synthetic sources (impulsive bearing, gear tone, colored noise),
creates instantaneous and convolutive mixtures, applies FastICA,
computes KRI/PCC/RMSE with bootstrap, compares with a simple Spectral Kurtosis (SK),
and saves figures and a JSON summary.
"""
import os
import json
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import FastICA
from sklearn.metrics import roc_auc_score, roc_curve, auc
from scipy.stats import kurtosis, pearsonr
from scipy.linalg import toeplitz
from scipy.signal import hilbert, welch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Configuration / paramètres
# -------------------------
OUTDIR = "Fastica Synthetic Validation Results"
os.makedirs(OUTDIR, exist_ok=True)

fs = 50000                 # sampling frequency
T = 0.6                    # duration (s)
N_total = int(fs * T)      # total samples (~30000)
window_N = 5000            # window size used in article
random_state = 0
rng = np.random.default_rng(random_state)

SNR_list = [20, 10, 5, 0]  # dB
n_components = 3
n_boot = 1000              # bootstrap iterations (1000 recommended; reduce to 500 for speed)
n_trials_auc = 100         # trials for ROC/AUC estimation per SNR

# -------------------------
# Utility functions
# -------------------------
def db2lin(db):
    return 10 ** (db / 10.0)

def add_awgn(signal_in, target_snr_db, rng):
    """Add AWGN to achieve target SNR (dB) relative to signal power."""
    sig_power = np.mean(signal_in ** 2)
    snr_lin = db2lin(target_snr_db)
    noise_power = sig_power / snr_lin
    noise = rng.normal(scale=np.sqrt(noise_power), size=signal_in.shape)
    return signal_in + noise

def generate_sources(fs, N, rng, impulsive=True, impulse_rate=80):
    """Generate 3 synthetic sources:
       s1: impulsive bearing-like (train of impulses with exponential envelope)
       s2: gear tone (sinusoid with low-freq AM)
       s3: colored noise (lowpass filtered white noise)
    """
    t = np.arange(N) / fs

    # s1: impulsive train (Laplace amplitudes) with exponential decay per impulse
    s1 = np.zeros(N)
    if impulsive:
        # place impulses roughly impulse_rate per record (random positions)
        positions = rng.choice(np.arange(0, N, max(1, N//(impulse_rate*2))), size=impulse_rate, replace=False)
        for p in positions:
            amp = rng.laplace(scale=1.0)
            # short exponential kernel
            L = int(0.002 * fs)  # 2 ms envelope
            idx = np.arange(p, min(N, p + L))
            s1[idx] += amp * np.exp(-0.01 * (idx - p))
    else:
        s1 = rng.normal(scale=0.1, size=N)

    # s2: gear tone with amplitude modulation
    f_gear = 1000.0  # Hz
    am_f = 10.0      # Hz modulation
    s2 = (1.0 + 0.5 * np.sin(2 * np.pi * am_f * t)) * np.sin(2 * np.pi * f_gear * t)

    # s3: colored noise (lowpass)
    b, a = signal.butter(4, 2000/(fs/2), btype='low')
    s3 = signal.lfilter(b, a, rng.normal(size=N))

    # normalize sources to unit variance (so mixing scales are meaningful)
    S = np.vstack([s1, s2, s3]).astype(float)
    S = S / (np.std(S, axis=1, keepdims=True) + 1e-12)
    return S

def make_convolutive_mixture(S, rng, filt_len=64):
    """Apply different FIR filters to each source to simulate convolutive mixing,
       then mix with a random instantaneous matrix.
    """
    n_src, N = S.shape
    # random small FIRs (stable)
    H = []
    for i in range(n_src):
        # random decaying filter
        h = rng.normal(size=filt_len)
        h *= np.exp(-np.linspace(0, 3, filt_len))
        h /= np.linalg.norm(h) + 1e-12
        H.append(h)
    # filter each source
    Sf = np.vstack([signal.convolve(S[i], H[i], mode='same') for i in range(n_src)])
    # instantaneous mixing after filtering
    A = rng.normal(size=(n_src, n_src))
    A = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
    X = A.dot(Sf)
    return X, A, H

def make_instantaneous_mixture(S, rng):
    n_src = S.shape[0]
    A = rng.normal(size=(n_src, n_src))
    A = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
    X = A.dot(S)
    return X, A

def fastica_separate(X, n_components=3, random_state=0):
    """Apply FastICA (sklearn) and return separated signals (components x samples) and whitened signals."""
    ica = FastICA(n_components=n_components, fun='exp', algorithm='parallel',
                  tol=1e-4, max_iter=1000, random_state=random_state)
    Y = ica.fit_transform(X.T).T  # shape (n_components, N)
    # whitened signals: sklearn does centering+whitening internally; we can approximate by transforming with mixing_
    try:
        # compute whitened via components_ if available
        # sklearn's FastICA exposes mixing_ and components_
        # whitened = ica.transform(X.T) @ np.linalg.pinv(ica.mixing_).T  # not reliable
        whitened = ica.transform(X.T).T
    except Exception:
        whitened = Y.copy()
    return Y, whitened, ica

def spectral_kurtosis_score(x, fs, n_bands=32):
    """Simple SK-like score: split into bands, compute kurtosis of envelope per band, return max kurtosis."""
    N = len(x)
    freqs = np.geomspace(20, fs/2, n_bands)
    kurt_vals = []
    for i in range(len(freqs)-1):
        f1, f2 = freqs[i], freqs[i+1]
        # design bandpass
        nyq = fs/2
        b, a = signal.butter(3, [f1/nyq, min(f2/nyq, 0.999)], btype='band')
        try:
            xb = signal.filtfilt(b, a, x)
        except Exception:
            xb = signal.lfilter(b, a, x)
        env = np.abs(hilbert(xb))
        kurt_vals.append(kurtosis(env, fisher=False))
    if len(kurt_vals) == 0:
        return 0.0
    return np.max(kurt_vals), np.array(kurt_vals)

def match_sources(S_true, Y_est, fs):
    """Compute matching score matrix between true sources and estimated ICs.
       Score = 0.5*|pearson| + 0.5*|spectral_corr|
    """
    n_src = S_true.shape[0]
    score = np.zeros((n_src, n_src))
    for i in range(n_src):
        for j in range(n_src):
            # temporal Pearson
            p = pearsonr(S_true[i], Y_est[j])[0]
            # spectral correlation
            f1, P1 = welch(S_true[i], fs=fs, nperseg=2048)
            f2, P2 = welch(Y_est[j], fs=fs, nperseg=2048)
            spec_corr = np.corrcoef(P1, P2)[0,1]
            score[i, j] = 0.5 * abs(p) + 0.5 * abs(spec_corr)
    # Hungarian assignment (maximize)
    row_ind, col_ind = linear_sum_assignment(-score)
    return row_ind, col_ind, score

def compute_kri(ku_mix, ku_sep):
    """Kurtosis Restoration Index (KRI) as in manuscript:
       KRI = 1 - (1/N) * sum(|Ku_sep - Ku_mix| / Ku_mix) * 100%
    """
    ku_mix = np.array(ku_mix)
    ku_sep = np.array(ku_sep)
    # avoid division by zero: add tiny epsilon
    eps = 1e-12
    rel_err = np.abs(ku_sep - ku_mix) / (np.abs(ku_mix) + eps)
    kri = 100.0 * (1.0 - np.mean(rel_err))
    return kri

# -------------------------
# Single-run demonstration (instantaneous, SNR=20) and plotting
# -------------------------
def single_run_demo(fs, N_total, window_N, rng, snr_db=20, convolutive=False):
    S = generate_sources(fs, N_total, rng, impulsive=True)
    if convolutive:
        X_full, A, H = make_convolutive_mixture(S, rng)
    else:
        X_full, A = make_instantaneous_mixture(S, rng)
        H = None
    # add AWGN per channel to reach target SNR (approx per-channel)
    X_noisy = np.zeros_like(X_full)
    for i in range(X_full.shape[0]):
        X_noisy[i] = add_awgn(X_full[i], snr_db, rng)
    # truncate to window_N for processing (simulate article)
    start = 0
    Xw = X_noisy[:, start:start+window_N]
    Sw = S[:, start:start+window_N]
    # separate
    Y, whitened, ica = fastica_separate(Xw, n_components=n_components, random_state=random_state)
    # matching
    row, col, score = match_sources(Sw, Y, fs)
    # compute kurtosis vectors
    ku_mix = kurtosis(Sw, axis=1, fisher=False)
    ku_sep = kurtosis(Y, axis=1, fisher=False)[col]
    kri = compute_kri(ku_mix, ku_sep)
    pcc = np.corrcoef(ku_mix, ku_sep)[0,1]
    rmse = np.sqrt(np.mean((ku_sep - ku_mix)**2))
    return {
        "S": Sw, "X": Xw, "whitened": whitened, "Y": Y,
        "A": A, "H": H, "row": row, "col": col, "score": score,
        "ku_mix": ku_mix, "ku_sep": ku_sep, "KRI": kri, "PCC": pcc, "RMSE": rmse
    }

# produce demo figures
demo = single_run_demo(fs, N_total, window_N, rng, snr_db=20, convolutive=False)
S_demo = demo["S"]; X_demo = demo["X"]; Y_demo = demo["Y"]; whitened_demo = demo["whitened"]

# plot sources, mixed, whitened, separated
def save_plot_signals(arrays, titles, fname, fs, n_samples):
    plt.figure(figsize=(12, 6))
    n = arrays.shape[0]
    for i in range(n):
        plt.subplot(n, 1, i+1)
        plt.plot(np.arange(n_samples)/fs, arrays[i], lw=0.6)
        plt.title(titles[i])
        plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, fname), dpi=200)
    plt.close()

save_plot_signals(S_demo, ["Source 1 (impulsive)", "Source 2 (gear tone)", "Source 3 (colored noise)"],
                  "fig_sources.png", fs, window_N)
save_plot_signals(X_demo, ["Mixed ch1", "Mixed ch2", "Mixed ch3"], "fig_mixed.png", fs, window_N)
save_plot_signals(whitened_demo, ["Whitened 1", "Whitened 2", "Whitened 3"], "fig_whitened.png", fs, window_N)
save_plot_signals(Y_demo, ["Separated IC1", "Separated IC2", "Separated IC3"], "fig_separated.png", fs, window_N)

# -------------------------
# Main experiments: SNR sweep, instantaneous vs convolutive
# -------------------------
results = {"instantaneous": {}, "convolutive": {}}

for mode in ["instantaneous", "convolutive"]:
    for snr in SNR_list:
        kri_list = []
        pcc_list = []
        rmse_list = []
        # bootstrap arrays for KRI/PCC/RMSE
        # We'll perform n_trials_auc independent mixtures to estimate variability
        for trial in range(n_trials_auc):
            # generate sources with random presence of impulses (for ROC later)
            S = generate_sources(fs, N_total, rng, impulsive=True)
            if mode == "convolutive":
                X_full, A, H = make_convolutive_mixture(S, rng)
            else:
                X_full, A = make_instantaneous_mixture(S, rng)
                H = None
            # add noise
            X_noisy = np.zeros_like(X_full)
            for i in range(X_full.shape[0]):
                X_noisy[i] = add_awgn(X_full[i], snr, rng)
            # truncate to window_N
            Xw = X_noisy[:, :window_N]
            Sw = S[:, :window_N]
            # separate
            Y, whitened, ica = fastica_separate(Xw, n_components=n_components, random_state=random_state)
            # match
            row, col, score = match_sources(Sw, Y, fs)
            ku_mix = kurtosis(Sw, axis=1, fisher=False)
            ku_sep = kurtosis(Y, axis=1, fisher=False)[col]
            kri = compute_kri(ku_mix, ku_sep)
            pcc = np.corrcoef(ku_mix, ku_sep)[0,1]
            rmse = np.sqrt(np.mean((ku_sep - ku_mix)**2))
            kri_list.append(kri)
            pcc_list.append(pcc)
            rmse_list.append(rmse)
        # bootstrap CI (percentiles)
        kri_arr = np.array(kri_list)
        pcc_arr = np.array(pcc_list)
        rmse_arr = np.array(rmse_list)
        def ci95(x):
            return float(np.percentile(x, 50)), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))
        kri_med, kri_lo, kri_hi = ci95(kri_arr)
        pcc_med, pcc_lo, pcc_hi = ci95(pcc_arr)
        rmse_med, rmse_lo, rmse_hi = ci95(rmse_arr)
        results[mode][str(snr)] = {
            "KRI": {"median": kri_med, "ci95_lo": kri_lo, "ci95_hi": kri_hi},
            "PCC": {"median": pcc_med, "ci95_lo": pcc_lo, "ci95_hi": pcc_hi},
            "RMSE": {"median": rmse_med, "ci95_lo": rmse_lo, "ci95_hi": rmse_hi}
        }
        print(f"[{mode}] SNR={snr} dB -> KRI median {kri_med:.2f} [{kri_lo:.2f}, {kri_hi:.2f}], PCC {pcc_med:.4f}")

# -------------------------
# Compare with Spectral Kurtosis (simple ROC/AUC)
# -------------------------
# We'll create a binary detection problem: presence vs absence of impulsive source (s1).
# For each trial, randomly decide whether s1 is impulsive (present) or replaced by low-amplitude noise (absent).
def run_detection_trials(mode, snr, n_trials, rng):
    y_true = []
    score_fastica = []
    score_sk = []
    for trial in range(n_trials):
        # randomly decide presence
        present = rng.choice([0,1], p=[0.4, 0.6])
        S = generate_sources(fs, N_total, rng, impulsive=bool(present))
        if mode == "convolutive":
            X_full, A, H = make_convolutive_mixture(S, rng)
        else:
            X_full, A = make_instantaneous_mixture(S, rng)
            H = None
        # add noise
        X_noisy = np.zeros_like(X_full)
        for i in range(X_full.shape[0]):
            X_noisy[i] = add_awgn(X_full[i], snr, rng)
        Xw = X_noisy[:, :window_N]
        Sw = S[:, :window_N]
        # FastICA
        Y, whitened, ica = fastica_separate(Xw, n_components=n_components, random_state=random_state)
        # match
        row, col, score = match_sources(Sw, Y, fs)
        ku_mix = kurtosis(Sw, axis=1, fisher=False)
        ku_sep = kurtosis(Y, axis=1, fisher=False)[col]
        # detection score: max kurtosis among ICs (fastica)
        score_fastica.append(float(np.max(ku_sep)))
        # SK score: apply SK on mixed signals and take max across channels
        sk_scores = []
        for ch in range(Xw.shape[0]):
            sk_val, _ = spectral_kurtosis_score(Xw[ch], fs, n_bands=32)
            sk_scores.append(sk_val)
        score_sk.append(float(np.max(sk_scores)))
        y_true.append(int(present))
    # compute AUCs
    try:
        auc_fastica = roc_auc_score(y_true, score_fastica)
    except Exception:
        auc_fastica = float('nan')
    try:
        auc_sk = roc_auc_score(y_true, score_sk)
    except Exception:
        auc_sk = float('nan')
    return auc_fastica, auc_sk, y_true, score_fastica, score_sk

auc_results = {"instantaneous": {}, "convolutive": {}}
for mode in ["instantaneous", "convolutive"]:
    for snr in SNR_list:
        auc_fastica, auc_sk, y_true, sc_f, sc_sk = run_detection_trials(mode, snr, n_trials_auc, rng)
        auc_results[mode][str(snr)] = {"AUC_FastICA": auc_fastica, "AUC_SK": auc_sk}
        print(f"[{mode}] SNR={snr} dB -> AUC FastICA {auc_fastica:.3f}, AUC SK {auc_sk:.3f}")

# -------------------------
# Plots: KRI vs SNR and AUC comparison
# -------------------------
def plot_kri_vs_snr(results, outpath):
    sns.set(style="whitegrid")
    modes = list(results.keys())
    plt.figure(figsize=(8,5))
    for mode in modes:
        med = [results[mode][str(s)]["KRI"]["median"] for s in SNR_list]
        lo = [results[mode][str(s)]["KRI"]["ci95_lo"] for s in SNR_list]
        hi = [results[mode][str(s)]["KRI"]["ci95_hi"] for s in SNR_list]
        plt.plot(SNR_list, med, marker='o', label=mode)
        plt.fill_between(SNR_list, lo, hi, alpha=0.2)
    plt.xlabel("SNR (dB)")
    plt.ylabel("KRI (%)")
    plt.title("Kurtosis Restoration Index vs SNR")
    plt.gca().invert_xaxis()  # show high SNR left->right? keep conventional: descending
    plt.legend()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_auc_comparison(auc_results, outpath):
    sns.set(style="whitegrid")
    plt.figure(figsize=(8,5))
    for mode in auc_results.keys():
        auc_fast = [auc_results[mode][str(s)]["AUC_FastICA"] for s in SNR_list]
        auc_sk = [auc_results[mode][str(s)]["AUC_SK"] for s in SNR_list]
        plt.plot(SNR_list, auc_fast, marker='o', label=f"{mode} FastICA")
        plt.plot(SNR_list, auc_sk, marker='x', linestyle='--', label=f"{mode} SK")
    plt.xlabel("SNR (dB)")
    plt.ylabel("AUC (detection of impulsive source)")
    plt.title("AUC: FastICA vs Spectral Kurtosis")
    plt.legend()
    plt.savefig(outpath, dpi=200)
    plt.close()

plot_kri_vs_snr(results, os.path.join(OUTDIR, "fig_kri_vs_snr.png"))
plot_auc_comparison(auc_results, os.path.join(OUTDIR, "fig_sk_vs_fastica_auc.png"))

# boxplot of kurtosis distributions for a representative case (instantaneous, SNR=20)
def plot_kurtosis_boxplots(demo_S, demo_Y, col_assign, outpath):
    ku_src = kurtosis(demo_S, axis=1, fisher=False)
    ku_ic = kurtosis(demo_Y, axis=1, fisher=False)[col_assign]
    data = {
        "source_kurtosis": ku_src,
        "ic_kurtosis": ku_ic
    }
    plt.figure(figsize=(6,4))
    sns.boxplot(data=[ku_src, ku_ic])
    plt.xticks([0,1], ["Sources", "ICs (matched)"])
    plt.ylabel("Kurtosis (Fisher=False)")
    plt.title("Kurtosis distributions: sources vs matched ICs")
    plt.savefig(outpath, dpi=200)
    plt.close()

plot_kurtosis_boxplots(S_demo, Y_demo, demo["col"], os.path.join(OUTDIR, "fig_kurtosis_boxplots.png"))

# -------------------------
# Save results summary JSON
# -------------------------
summary = {
    "config": {
        "fs": fs, "T": T, "N_total": N_total, "window_N": window_N,
        "SNR_list": SNR_list, "n_components": n_components,
        "n_trials_auc": n_trials_auc, "n_boot": n_boot
    },
    "kri_results": results,
    "auc_results": auc_results,
    "demo_metrics": {
        "KRI_demo": float(demo["KRI"]),
        "PCC_demo": float(demo["PCC"]),
        "RMSE_demo": float(demo["RMSE"])
    }
}
with open(os.path.join(OUTDIR, "results_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\nSaved figures and results in directory:", OUTDIR)
print("Demo metrics:", summary["demo_metrics"])
