#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iva_en.py

Multichannel IVA for three MATLAB files:
    ch1.mat, ch2.mat, ch3.mat

Requirements:
- The three .mat files must be located in the same directory as this script.
- The results are saved in a folder named:
      "IVA Results"

Contents of "IVA Results" :
- Figures/  -> PNG figures
- Tables/   -> CSV tables
- iva_numeric_results.json -> complete numerical values for comparison

Important note:
- IVA is a multichannel source separation method, not a mode decomposition
  like VMD/MVMD/MEMD.
- This script implements a frequency-domain AuxIVA-type version (determined case),
  adapted to 3 simultaneously measured signals.

Dependencies:
    pip install numpy scipy pandas matplotlib
"""

import os
import json
import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import stft, istft
from scipy.stats import kurtosis, skew


# ============================================================
# USER PARAMETERS
# ============================================================
FS = 50000  # Hz
INPUT_FILES = ["ch1.mat", "ch2.mat", "ch3.mat"]

MAIN_OUTPUT_DIRNAME = "IVA Results"
FIGURES_DIRNAME = "Figures"
TABLES_DIRNAME = "Tables"

SAVE_FULL_SIGNALS_IN_JSON = True
SAVE_DEMIXING_MATRICES_IN_JSON = True

# STFT parameters
N_FFT = 1024
N_OVERLAP = 768
WINDOW = "hann"

# IVA parameters
IVA_N_ITER = 20
REF_CHANNEL = 0
EPS = 1e-10


# ============================================================
# GENERAL UTILITIES
# ============================================================
def get_script_directory():
    return os.path.dirname(os.path.abspath(__file__))


def ensure_directories(base_dir):
    main_out = os.path.join(base_dir, MAIN_OUTPUT_DIRNAME)
    figures_out = os.path.join(main_out, FIGURES_DIRNAME)
    tables_out = os.path.join(main_out, TABLES_DIRNAME)

    os.makedirs(main_out, exist_ok=True)
    os.makedirs(figures_out, exist_ok=True)
    os.makedirs(tables_out, exist_ok=True)

    return main_out, figures_out, tables_out


def find_file_case_insensitive(base_dir, filename):
    exact = os.path.join(base_dir, filename)
    if os.path.exists(exact):
        return exact

    target_lower = filename.lower()
    for f in os.listdir(base_dir):
        if f.lower() == target_lower:
            return os.path.join(base_dir, f)

    raise FileNotFoundError(
        f"The file '{filename}' could not be found in the directory: {base_dir}"
    )


def load_signal_from_mat(filepath):
    """
    Loads a 1D signal from a .mat file.
    Priority is given to the variable 'y'; otherwise, the first numeric vector variable is used.
    """
    data = sio.loadmat(filepath)
    excluded = {"__header__", "__version__", "__globals__"}

    if "y" in data:
        x = np.squeeze(np.asarray(data["y"], dtype=float))
        if x.ndim == 1 and x.size > 10:
            return x, "y"

    for key, value in data.items():
        if key in excluded:
            continue
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
            arr = np.squeeze(np.asarray(value, dtype=float))
            if arr.ndim == 1 and arr.size > 10:
                return arr, key

    raise ValueError(
        f"No usable numeric vector was found in: {filepath}"
    )


def preprocess_signal(x):
    x = np.asarray(x, dtype=float).flatten()
    x = x - np.mean(x)
    std = np.std(x)
    if std < 1e-12:
        raise ValueError("Nearly constant signal: normalization is not possible.")
    return x / std


def to_serializable(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj


# ============================================================
# MULTICHANNEL STFT / ISTFT
# ============================================================
def multichannel_stft(X, fs, n_fft, n_overlap, window):
    """
    X : shape (C, N)
    Returns:
    - f: frequency axis
    - t: STFT time axis
    - X_stft : shape (F, T, C)
    """
    C, _ = X.shape
    specs = []
    f = t = None

    for c in range(C):
        f, t, Z = stft(
            X[c, :],
            fs=fs,
            window=window,
            nperseg=n_fft,
            noverlap=n_overlap,
            boundary="zeros",
            padded=True,
            return_onesided=True,
        )
        specs.append(Z)

    X_stft = np.stack(specs, axis=2)  # (F, T, C)
    return f, t, X_stft


def reconstruct_sources_from_stft(Y, fs, n_fft, n_overlap, window, target_len):
    """
    Y : shape (F, T, N)
    Returns:
    - S : shape (N, target_len)
    """
    sources = []

    for n in range(Y.shape[2]):
        _, x = istft(
            Y[:, :, n],
            fs=fs,
            window=window,
            nperseg=n_fft,
            noverlap=n_overlap,
            input_onesided=True,
            boundary=True,
        )

        if len(x) < target_len:
            x = np.pad(x, (0, target_len - len(x)))
        else:
            x = x[:target_len]

        sources.append(np.real(x))

    return np.asarray(sources)


# ============================================================
# IVA (determined AuxIVA)
# ============================================================
def auxiva_laplace(X, n_iter=20, ref_channel=0, eps=1e-10):
    """
    Frequency-domain AuxIVA-type IVA (Laplace prior), determined case.

    Parameters
    ----------
    X : ndarray, shape (F, T, M)
        STFT of the observations, M microphones/sensors.
    n_iter : int
        Number of iterations.
    ref_channel : int
        Reference channel for back-projection.
    eps : float
        Numerical regularization.

    Returns
    ------
    Y_bp : ndarray, shape (F, T, N)
        Separated sources in the STFT domain after back-projection.
    W : ndarray, shape (F, N, M)
        Frequency-domain demixing matrices.
    history : list[dict]
        Simple convergence history.
    """
    F, T, M = X.shape
    N = M  # determined case

    W = np.tile(np.eye(N, dtype=np.complex128)[None, :, :], (F, 1, 1))
    Y = np.einsum("fns,fts->ftn", W, X)

    history = []
    eye_m = np.eye(M, dtype=np.complex128)
    unit = np.eye(N, dtype=np.complex128)

    for it in range(n_iter):
        # r(t, n) = frequency norm of source n at frame t
        power = np.sum(np.abs(Y) ** 2, axis=0)   # (T, N)
        r = np.sqrt(np.maximum(power, eps))

        for n in range(N):
            g = 1.0 / np.maximum(r[:, n], eps)   # (T,)

            for f in range(F):
                Xf = X[f, :, :]  # (T, M)

                # V_{f,n}
                V = (Xf.conj().T * g) @ Xf / T
                V += eps * eye_m

                WV = W[f] @ V
                try:
                    w = np.linalg.solve(WV, unit[:, n])
                except np.linalg.LinAlgError:
                    w = np.linalg.pinv(WV) @ unit[:, n]

                denom = np.sqrt(max(np.real(np.conj(w).T @ V @ w), eps))
                w = w / denom

                # store w^H in row n of W[f]
                W[f, n, :] = np.conj(w)

        Y = np.einsum("fns,fts->ftn", W, X)

        # Simple convergence index: sum of the off-diagonal terms of W W^H
        offdiag_sum = 0.0
        for f in range(F):
            G = W[f] @ W[f].conj().T
            offdiag_sum += np.sum(np.abs(G - np.diag(np.diag(G))))

        history.append({
            "iteration": it + 1,
            "demixing_offdiag_sum": float(offdiag_sum)
        })

    # Back-projection to resolve the scale ambiguity
    Y_bp = Y.copy()
    for f in range(F):
        try:
            A = np.linalg.inv(W[f])
        except np.linalg.LinAlgError:
            A = np.linalg.pinv(W[f])

        for n in range(N):
            Y_bp[f, :, n] *= A[ref_channel, n]

    return Y_bp, W, history


# ============================================================
# METRICS / TABLES
# ============================================================
def compute_signal_features(signals, fs, labels, signal_type):
    """
    signals : shape (n_signals, N)
    """
    rows = []
    N = signals.shape[1]
    freq_axis = np.fft.rfftfreq(N, d=1.0 / fs)

    for i, x in enumerate(signals):
        rms = np.sqrt(np.mean(x ** 2))
        peak = np.max(np.abs(x))
        crest = peak / rms if rms > 1e-12 else np.nan
        ku = kurtosis(x, fisher=False, bias=False)
        sk = skew(x, bias=False)

        spectrum = np.abs(np.fft.rfft(x))
        dominant_freq = freq_axis[np.argmax(spectrum)]

        rows.append({
            "SignalType": signal_type,
            "Label": labels[i],
            "RMS": rms,
            "Kurtosis": ku,
            "Skewness": sk,
            "CrestFactor": crest,
            "DominantFreq_Hz": dominant_freq
        })

    return pd.DataFrame(rows)


def compute_energy_percentages(signals, labels):
    energies = np.sum(signals ** 2, axis=1)
    total = np.sum(energies)

    if total <= 1e-18:
        energy_pct = np.zeros_like(energies)
    else:
        energy_pct = 100.0 * energies / total

    rows = []
    for i, label in enumerate(labels):
        rows.append({
            "Label": label,
            "EnergyPercent": energy_pct[i]
        })

    return pd.DataFrame(rows)


def compute_correlation_matrix(signals, labels):
    corr = np.corrcoef(signals)
    return pd.DataFrame(corr, index=labels, columns=labels)


def save_signals_csv(tables_dir, filename, time_s, signals, labels):
    data = {"time_s": time_s}
    for i, label in enumerate(labels):
        data[label] = signals[i, :]

    out_csv = os.path.join(tables_dir, filename)
    pd.DataFrame(data).to_csv(out_csv, index=False)
    return out_csv


def save_correlation_csv(tables_dir, corr_df):
    out_csv = os.path.join(tables_dir, "IVA_source_correlations.csv")
    corr_df.to_csv(out_csv, index=True)
    return out_csv


# ============================================================
# FIGURES
# ============================================================
def save_raw_signals_figure(figures_dir, X, fs, channel_names):
    C, N = X.shape
    t = np.arange(N) / fs

    fig, axes = plt.subplots(C, 1, figsize=(14, 2.8 * C), sharex=True)
    if C == 1:
        axes = [axes]

    for c in range(C):
        axes[c].plot(t, X[c], linewidth=1.0)
        axes[c].set_title(f"Original signal - {channel_names[c]}")
        axes[c].set_ylabel("Amplitude")
        axes[c].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "01_raw_signals.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_sources_figure(figures_dir, S, fs, source_labels):
    C, N = S.shape
    t = np.arange(N) / fs

    fig, axes = plt.subplots(C, 1, figsize=(14, 2.8 * C), sharex=True)
    if C == 1:
        axes = [axes]

    for c in range(C):
        axes[c].plot(t, S[c], linewidth=1.0)
        axes[c].set_title(f"Separated source - {source_labels[c]}")
        axes[c].set_ylabel("Amplitude")
        axes[c].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "02_separated_sources.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_sources_fft_figure(figures_dir, S, fs, source_labels):
    C, N = S.shape
    freq_axis = np.fft.rfftfreq(N, d=1.0 / fs)

    fig, axes = plt.subplots(C, 1, figsize=(14, 2.8 * C), sharex=True)
    if C == 1:
        axes = [axes]

    for c in range(C):
        spectrum = np.abs(np.fft.rfft(S[c]))
        axes[c].plot(freq_axis, spectrum, linewidth=1.0)
        axes[c].set_title(f"FFT of separated source - {source_labels[c]}")
        axes[c].set_ylabel("Amplitude")
        axes[c].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frequency (Hz)")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "03_separated_sources_fft.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_spectrograms(figures_dir, Y, f_axis, t_axis, source_labels):
    F, T, N = Y.shape

    for n in range(N):
        mag_db = 20.0 * np.log10(np.maximum(np.abs(Y[:, :, n]), 1e-12))

        fig = plt.figure(figsize=(12, 4.5))
        plt.pcolormesh(t_axis, f_axis, mag_db, shading="gouraud")
        plt.title(f"Spectrogramme - {source_labels[n]}")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(label="Amplitude (dB)")
        plt.tight_layout()
        fig.savefig(
            os.path.join(figures_dir, f"04_spectrogram_{source_labels[n]}.png"),
            dpi=180,
            bbox_inches="tight"
        )
        plt.close(fig)


def save_correlation_heatmap(figures_dir, corr_df):
    fig = plt.figure(figsize=(6, 5))
    plt.imshow(corr_df.values, aspect="auto")
    plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr_df.index)), corr_df.index)

    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            plt.text(j, i, f"{corr_df.values[i, j]:.3f}", ha="center", va="center")

    plt.title("Correlation between separated sources")
    plt.colorbar()
    plt.tight_layout()
    fig.savefig(
        os.path.join(figures_dir, "05_source_correlation_heatmap.png"),
        dpi=180,
        bbox_inches="tight"
    )
    plt.close(fig)


def save_energy_barplot(figures_dir, energy_df):
    fig = plt.figure(figsize=(8, 4.5))
    plt.bar(energy_df["Label"], energy_df["EnergyPercent"])
    plt.title("Energy percentage of the separated sources")
    plt.xlabel("Source")
    plt.ylabel("Energy (%)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(
        os.path.join(figures_dir, "06_source_energy_barplot.png"),
        dpi=180,
        bbox_inches="tight"
    )
    plt.close(fig)


def save_iva_convergence(figures_dir, history):
    if not history:
        return

    x = [item["iteration"] for item in history]
    y = [item["demixing_offdiag_sum"] for item in history]

    fig = plt.figure(figsize=(10, 4.5))
    plt.plot(x, y, marker="o")
    plt.title("IVA convergence (demixing index)")
    plt.xlabel("Iteration")
    plt.ylabel("Off-diagonal sum of W W^H")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(
        os.path.join(figures_dir, "07_iva_convergence.png"),
        dpi=180,
        bbox_inches="tight"
    )
    plt.close(fig)


# ============================================================
# COMPLETE NUMERICAL JSON
# ============================================================
def build_json_payload(
    input_file_info,
    X,
    S,
    f_axis,
    t_axis,
    W,
    history,
    features_df,
    energy_df,
    corr_df,
    time_s,
    params
):
    payload = {
        "metadata": {
            "method": "IVA (AuxIVA, frequency-domain, determined case)",
            "sampling_frequency_hz": FS,
            "n_channels": int(X.shape[0]),
            "signal_length_samples": int(X.shape[1]),
            "n_sources": int(S.shape[0]),
            "input_files": input_file_info
        },
        "parameters": params,
        "stft_frequency_axis_hz": f_axis,
        "stft_time_axis_s": t_axis,
        "iva_history": history,
        "features_table": features_df,
        "energy_table": energy_df,
        "source_correlation_matrix": corr_df,
        "time_s": time_s
    }

    if SAVE_FULL_SIGNALS_IN_JSON:
        payload["preprocessed_signals"] = X
        payload["separated_sources"] = S

    if SAVE_DEMIXING_MATRICES_IN_JSON:
        payload["demixing_matrices_real"] = np.real(W)
        payload["demixing_matrices_imag"] = np.imag(W)

    return to_serializable(payload)


# ============================================================
# MAIN PROGRAM
# ============================================================
def main():
    script_dir = get_script_directory()
    main_out, figures_out, tables_out = ensure_directories(script_dir)

    print("=" * 70)
    print("IVA multichannel (AuxIVA)")
    print("Script directory:", script_dir)
    print("Output directory:", main_out)
    print("=" * 70)

    # Loading the 3 signals
    signals = []
    input_file_info = []

    for requested_name in INPUT_FILES:
        filepath = find_file_case_insensitive(script_dir, requested_name)
        signal_raw, variable_name = load_signal_from_mat(filepath)
        signal_preprocessed = preprocess_signal(signal_raw)

        signals.append(signal_preprocessed)

        input_file_info.append({
            "requested_filename": requested_name,
            "resolved_filename": os.path.basename(filepath),
            "variable_name": variable_name,
            "channel_name": os.path.splitext(os.path.basename(filepath))[0],
            "n_samples": int(len(signal_raw))
        })

        print(f"Loaded: {os.path.basename(filepath)} | variable: {variable_name} | N = {len(signal_raw)}")

    # Matching lengths if necessary
    min_len = min(len(s) for s in signals)
    if len(set(len(s) for s in signals)) != 1:
        print(f"Different lengths detected. Truncating to {min_len} samples.")
    signals = [s[:min_len] for s in signals]

    # Multichannel stacking: shape (C, N)
    X = np.vstack(signals)
    channel_names = [item["channel_name"] for item in input_file_info]
    source_labels = [f"source_{i + 1}" for i in range(X.shape[0])]

    print("Multichannel data dimensions:", X.shape)

    # Temps
    time_s = np.arange(X.shape[1]) / FS

    # Raw signal figures
    save_raw_signals_figure(figures_out, X, FS, channel_names)

    # STFT
    print("Calcul STFT...")
    f_axis, t_axis, X_stft = multichannel_stft(X, FS, N_FFT, N_OVERLAP, WINDOW)

    # IVA
    print("Starting IVA...")
    Y, W, history = auxiva_laplace(
        X_stft,
        n_iter=IVA_N_ITER,
        ref_channel=REF_CHANNEL,
        eps=EPS
    )
    print("IVA completed.")

    # Reconstruction temporelle
    S = reconstruct_sources_from_stft(
        Y,
        FS,
        N_FFT,
        N_OVERLAP,
        WINDOW,
        X.shape[1]
    )

    print("Separated source dimensions:", S.shape)

    # Features / correlations / energies
    features_df = compute_signal_features(S, FS, source_labels, "SeparatedSource")
    energy_df = compute_energy_percentages(S, source_labels)
    corr_df = compute_correlation_matrix(S, source_labels)

    # CSV table exports
    features_csv = os.path.join(tables_out, "IVA_features.csv")
    energy_csv = os.path.join(tables_out, "IVA_energy_percentages.csv")

    features_df.to_csv(features_csv, index=False)
    energy_df.to_csv(energy_csv, index=False)
    save_correlation_csv(tables_out, corr_df)

    save_signals_csv(tables_out, "IVA_separated_sources.csv", time_s, S, source_labels)

    # Figures
    save_sources_figure(figures_out, S, FS, source_labels)
    save_sources_fft_figure(figures_out, S, FS, source_labels)
    save_spectrograms(figures_out, Y, f_axis, t_axis, source_labels)
    save_correlation_heatmap(figures_out, corr_df)
    save_energy_barplot(figures_out, energy_df)
    save_iva_convergence(figures_out, history)

    # Complete numerical JSON
    params = {
        "FS": FS,
        "N_FFT": N_FFT,
        "N_OVERLAP": N_OVERLAP,
        "WINDOW": WINDOW,
        "IVA_N_ITER": IVA_N_ITER,
        "REF_CHANNEL": REF_CHANNEL,
        "EPS": EPS
    }

    json_payload = build_json_payload(
        input_file_info=input_file_info,
        X=X,
        S=S,
        f_axis=f_axis,
        t_axis=t_axis,
        W=W,
        history=history,
        features_df=features_df,
        energy_df=energy_df,
        corr_df=corr_df,
        time_s=time_s,
        params=params
    )

    json_path = os.path.join(main_out, "iva_numeric_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    # Text summary
    summary_txt = os.path.join(main_out, "README_results.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("IVA Results\n")
        f.write("====================\n\n")
        f.write("Method: Frequency-domain AuxIVA-type IVA (determined case)\n")
        f.write("- Multichannel source separation\n")
        f.write("- STFT multi-channel\n")
        f.write("- Frequency-domain demixing\n")
        f.write("- Back-projection onto the reference channel\n\n")

        f.write("Input files:\n")
        for item in input_file_info:
            f.write(
                f"- {item['resolved_filename']} | variable: {item['variable_name']} | "
                f"N = {item['n_samples']}\n"
            )

        f.write("\nIVA/STFT parameters:\n")
        for k_param, v_param in params.items():
            f.write(f"- {k_param} = {v_param}\n")

        f.write("\nContents:\n")
        f.write("- Figures/ : PNG figures\n")
        f.write("- Tables/ : CSV tables\n")
        f.write("- iva_numeric_results.json : complete numerical values\n")

    print("=" * 70)
    print("Processing completed successfully.")
    print("Main table:", features_csv)
    print("Energy table:", energy_csv)
    print("Numerical JSON:", json_path)
    print("Figures:", figures_out)
    print("Tables:", tables_out)
    print("=" * 70)


if __name__ == "__main__":
    main()
