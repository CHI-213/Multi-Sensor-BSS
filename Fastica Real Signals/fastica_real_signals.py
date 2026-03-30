# fastica_from_mat.py
# -*- coding: utf-8 -*-
"""
Traitement FastICA pour trois signaux réels stockés en fichiers MATLAB .mat
- Lecture de 3 fichiers .mat (variable unique ou première variable)
- Prétraitement (centrage)
- FastICA (sklearn) fun='exp' algorithm='parallel'
- Appariement automatique source <-> IC (Pearson + corrélation spectrale)
- Calcul KRI, PCC, RMSE
- Bootstrap par blocs pour IC 95%
- Comparaison simple avec Spectral Kurtosis (SK)
- Sauvegarde figures PNG et results_summary.json
"""
import os
import json
import numpy as np
from scipy.io import loadmat
import scipy.signal as signal
from scipy.signal import welch, hilbert
from scipy.stats import kurtosis, pearsonr
from sklearn.decomposition import FastICA
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Configuration utilisateur
# -------------------------
MAT_FILES = ["ch1.mat", "ch2.mat", "ch3.mat"]  # modifier si nécessaire
FS = 50000                # fréquence d'échantillonnage en Hz, adapter si besoin
WINDOW_N = 5000           # taille de fenêtre (comme dans l'article)
OUTDIR = "Fastica Real Signals Results"
os.makedirs(OUTDIR, exist_ok=True)

N_BOOT = 1000             # bootstrap iterations (réduire si lent)
BLOCK_SIZE = 1000         # taille de bloc pour bootstrap par blocs
RANDOM_STATE = 0

# -------------------------
# Fonctions utilitaires
# -------------------------
def load_mat_signal(path):
    """Charge un fichier .mat et retourne le premier vecteur 1D trouvé.
       Si le fichier contient une variable nommée 'data' ou 'signal', on la privilégie.
    """
    d = loadmat(path)
    # supprimer clés internes
    keys = [k for k in d.keys() if not k.startswith("__")]
    # priorités
    for pref in ["data", "signal", "x", "y", "acc"]:
        if pref in d:
            arr = d[pref]
            return np.ravel(arr)
    # sinon prendre la première variable non interne
    if len(keys) == 0:
        raise ValueError(f"Aucune variable trouvée dans {path}")
    arr = d[keys[0]]
    return np.ravel(arr)

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
            p = pearsonr(S_true[i], Y_est[j])[0]
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
        b, a = signal.butter(3, [low, high], btype='band')
        try:
            xb = signal.filtfilt(b, a, x)
        except Exception:
            xb = signal.lfilter(b, a, x)
        env = np.abs(hilbert(xb))
        kurt_vals.append(kurtosis(env, fisher=False))
    if len(kurt_vals) == 0:
        return 0.0, np.array([])
    return float(np.max(kurt_vals)), np.array(kurt_vals)

# -------------------------
# Chargement des signaux .mat
# -------------------------
print("Chargement des fichiers .mat...")
signals = []
for f in MAT_FILES:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Fichier introuvable: {f}")
    s = load_mat_signal(f)
    signals.append(s)
minlen = min(len(s) for s in signals)
if minlen < WINDOW_N:
    raise ValueError(f"Les signaux ont moins de {WINDOW_N} échantillons. Ajustez WINDOW_N ou fournissez des signaux plus longs.")
# tronquer et empiler
S_true = np.vstack([s[:WINDOW_N] for s in signals])
# centrer
S_true = S_true - np.mean(S_true, axis=1, keepdims=True)
print(f"Signaux chargés et centrés shape {S_true.shape}")

# Pour comparaison, considérer les canaux d'entrée comme 'mixés' X
X = S_true.copy()

# -------------------------
# FastICA
# -------------------------
print("Application de FastICA...")
Y, whitened, ica_model = apply_fastica(X, n_components=3, random_state=RANDOM_STATE)

# -------------------------
# Appariement et métriques
# -------------------------
print("Appariement automatique et calcul des métriques...")
row, col, score_mat = match_sources(S_true, Y, FS)
ku_mix = kurtosis(S_true, axis=1, fisher=False)
ku_sep = kurtosis(Y, axis=1, fisher=False)[col]
KRI_val = compute_kri(ku_mix, ku_sep)
PCC_val = np.corrcoef(ku_mix, ku_sep)[0,1]
RMSE_val = np.sqrt(np.mean((ku_sep - ku_mix)**2))
print(f"KRI = {KRI_val:.2f} %, PCC = {PCC_val:.4f}, RMSE = {RMSE_val:.4f}")

# -------------------------
# Bootstrap par blocs
# -------------------------
print(f"Bootstrap par blocs n={N_BOOT} taille bloc {BLOCK_SIZE}...")
rng = np.random.default_rng(RANDOM_STATE)
kri_boot = []; pcc_boot = []; rmse_boot = []
n_blocks = WINDOW_N // BLOCK_SIZE
use_blocks = n_blocks >= 2
block_indices = [np.arange(i*BLOCK_SIZE, (i+1)*BLOCK_SIZE) for i in range(n_blocks)] if use_blocks else None

for _ in trange(N_BOOT, desc="Bootstrap"):
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
    rmse_b = np.sqrt(np.mean((ku_sep_b - ku_mix_b)**2))
    kri_boot.append(kri_b); pcc_boot.append(pcc_b); rmse_boot.append(rmse_b)

def ci95(arr):
    arr = np.array(arr)
    if arr.size == 0:
        return (np.nan, np.nan, np.nan)
    return (float(np.percentile(arr,50)), float(np.percentile(arr,2.5)), float(np.percentile(arr,97.5)))

kri_med, kri_lo, kri_hi = ci95(kri_boot)
pcc_med, pcc_lo, pcc_hi = ci95(pcc_boot)
rmse_med, rmse_lo, rmse_hi = ci95(rmse_boot)

print("Bootstrap terminé.")
print(f"KRI median {kri_med:.2f} [{kri_lo:.2f}, {kri_hi:.2f}]")
print(f"PCC median {pcc_med:.4f} [{pcc_lo:.4f}, {pcc_hi:.4f}]")
print(f"RMSE median {rmse_med:.4f} [{rmse_lo:.4f}, {rmse_hi:.4f}]")

# -------------------------
# Spectral Kurtosis simple sur canaux mixtes
# -------------------------
print("Calcul SK simple sur canaux mixtes...")
sk_vals = []
for ch in range(X.shape[0]):
    sk_val, sk_vec = spectral_kurtosis_score(X[ch], FS, n_bands=32)
    sk_vals.append(sk_val)
sk_max = max(sk_vals)

# -------------------------
# Figures
# -------------------------
sns.set(style="whitegrid")
time = np.arange(WINDOW_N) / FS

def save_timeseries(mat, titles, fname):
    plt.figure(figsize=(10, 6))
    n = mat.shape[0]
    for i in range(n):
        plt.subplot(n,1,i+1)
        plt.plot(time, mat[i], lw=0.6)
        plt.title(titles[i])
        plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, fname), dpi=200)
    plt.close()

save_timeseries(S_true, ["Canal 1", "Canal 2", "Canal 3"], "sources.png")
save_timeseries(X, ["Mix ch1", "Mix ch2", "Mix ch3"], "mixed.png")
save_timeseries(whitened, ["Whitened 1", "Whitened 2", "Whitened 3"], "whitened.png")
save_timeseries(Y, ["IC 1", "IC 2", "IC 3"], "separated.png")

# boxplot kurtosis
ku_src = kurtosis(S_true, axis=1, fisher=False)
ku_ic = kurtosis(Y, axis=1, fisher=False)[col]
plt.figure(figsize=(6,4))
sns.boxplot(data=[ku_src, ku_ic])
plt.xticks([0,1], ["Sources", "ICs (appariés)"])
plt.ylabel("Kurtosis (Fisher=False)")
plt.title("Kurtosis: sources vs ICs")
plt.savefig(os.path.join(OUTDIR, "kurtosis_boxplot.png"), dpi=200)
plt.close()

# heatmap matrice score d'appariement
plt.figure(figsize=(6,5))
sns.heatmap(score_mat, annot=True, fmt=".3f", cmap="viridis")
plt.xlabel("IC index")
plt.ylabel("Source index")
plt.title("Matrice de score d'appariement")
plt.savefig(os.path.join(OUTDIR, "match_score_matrix.png"), dpi=200)
plt.close()

# -------------------------
# Sauvegarde résumé JSON
# -------------------------
summary = {
    "files": MAT_FILES,
    "fs": FS,
    "window_N": WINDOW_N,
    "metrics": {
        "KRI_point": float(KRI_val),
        "PCC_point": float(PCC_val),
        "RMSE_point": float(RMSE_val),
        "KRI_bootstrap": {"median": kri_med, "ci95_lo": kri_lo, "ci95_hi": kri_hi},
        "PCC_bootstrap": {"median": pcc_med, "ci95_lo": pcc_lo, "ci95_hi": pcc_hi},
        "RMSE_bootstrap": {"median": rmse_med, "ci95_lo": rmse_lo, "ci95_hi": rmse_hi},
        "SK_max_mixed": float(sk_max)
    },
    "match": {
        "row_indices": list(map(int, row.tolist())),
        "col_indices": list(map(int, col.tolist()))
    }
}
with open(os.path.join(OUTDIR, "results_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"Résultats et figures sauvegardés dans le dossier: {OUTDIR}")
print("Fichiers clés: sources.png, mixed.png, whitened.png, separated.png, kurtosis_boxplot.png, match_score_matrix.png, results_summary.json")

# -------------------------
# Interprétation succincte
# -------------------------
print("\n--- Interprétation rapide ---")
print(f"KRI = {KRI_val:.2f}% (bootstrap médiane {kri_med:.2f}% [{kri_lo:.2f},{kri_hi:.2f}])")
print(f"PCC = {PCC_val:.4f} (bootstrap médiane {pcc_med:.4f})")
print("Si KRI élevé et PCC proche de 1, FastICA préserve la structure impulsive. Sinon, tester plusieurs fenêtres, SNR, ou mélange convolutif.")
