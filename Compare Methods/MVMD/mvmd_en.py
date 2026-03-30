#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mvmd_en.py

Multichannel MVMD for three MATLAB files:
    ch1.mat, ch2.mat, ch3.mat

Requirements:
- The three .mat files must be located in the same directory as this script.
- The results are saved in a folder named:
      "MVMD Results"

Contents of "MVMD Results" :
- Figures/  -> PNG figures
- Tables/   -> CSV tables
- mvmd_numeric_results.json -> complete numerical values for comparison

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
from scipy.stats import kurtosis, skew


# ============================================================
# USER PARAMETERS
# ============================================================
FS = 50000  # Hz

INPUT_FILES = ["ch1.mat", "ch2.mat", "ch3.mat"]

# MVMD parameters
ALPHA = 5000.0      # bandwidth penalty
TAU = 0.0           # dual-ascent step
K = 5               # number of modes
DC = 0              # 1 => first mode forced to 0 Hz
INIT = 1            # 0: omega=0, 1: uniform, 2: random
TOL = 1e-7          # convergence tolerance
N_ITER = 500        # maximum number of iterations

# Saving
MAIN_OUTPUT_DIRNAME = "MVMD Results"
FIGURES_DIRNAME = "Figures"
TABLES_DIRNAME = "Tables"

# Full JSON export (may be large)
SAVE_FULL_SIGNALS_IN_JSON = True

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
    """
    Robust conversion for JSON.
    """
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
# MVMD UTILITIES
# ============================================================
def mirror_extend_multichannel(X):
    """
    X : shape (C, N)
    Returns a mirror extension to reduce edge effects.
    """
    return np.concatenate([np.flip(X, axis=1), X, np.flip(X, axis=1)], axis=1)


def analytic_positive_spectrum(X):
    """
    X : shape (C, N)
    Returns the analytic spectrum on a channel-by-channel basis.
    """
    C, N = X.shape
    Xf = np.fft.fft(X, axis=1)
    H = np.zeros(N)

    if N % 2 == 0:
        H[0] = 1.0
        H[N // 2] = 1.0
        H[1:N // 2] = 2.0
    else:
        H[0] = 1.0
        H[1:(N + 1) // 2] = 2.0

    return Xf * H[None, :]


def mvmd(X, alpha=2000.0, tau=0.0, K=5, DC=0, init=1, tol=1e-7, N_iter=500):
    """
    Practical MVMD implementation for multichannel data.

    Parameters
    ----------
    X : ndarray, shape (C, N)
        C channels, N samples.

    Returns
    ------
    u : ndarray, shape (K, C, N)
        Time-domain modes.
    omega : ndarray, shape (n_iter_effectif, K)
        Evolution of the common center frequencies (normalized).
    """
    X = np.asarray(X, dtype=float)
    C, N0 = X.shape

    # Mirror extension
    X_ext = mirror_extend_multichannel(X)
    _, T = X_ext.shape

    # Normalized frequencies centered at 0
    freqs = np.arange(0, T) / T - 0.5
    freqs = np.fft.fftshift(freqs)

    # Analytic spectrum
    f_hat = analytic_positive_spectrum(X_ext)
    f_hat = np.fft.fftshift(f_hat, axes=1)

    # Initializations
    u_hat = np.zeros((K, C, T), dtype=complex)
    lambda_hat = np.zeros((C, T), dtype=complex)
    omega = np.zeros((N_iter, K), dtype=float)

    if init == 1:
        omega[0, :] = np.linspace(0.0, 0.5, K, endpoint=False)
    elif init == 2:
        rng = np.random.default_rng(0)
        omega[0, :] = np.sort(rng.uniform(0.0, 0.5, K))
    else:
        omega[0, :] = 0.0

    if DC and K > 0:
        omega[0, 0] = 0.0

    uDiff = tol + np.finfo(float).eps
    n = 0

    while (uDiff > tol) and (n < N_iter - 1):
        u_hat_prev = np.copy(u_hat)
        sum_uk = np.sum(u_hat, axis=0)  # shape (C, T)

        for k in range(K):
            sum_uk -= u_hat[k]

            denom = 1.0 + 2.0 * alpha * (freqs - omega[n, k]) ** 2
            for c in range(C):
                u_hat[k, c, :] = (
                    f_hat[c, :] - sum_uk[c, :] - lambda_hat[c, :] / 2.0
                ) / denom

            if not (DC and k == 0):
                num = 0.0
                den = 0.0
                pos = freqs > 0
                for c in range(C):
                    uk2 = np.abs(u_hat[k, c, pos]) ** 2
                    num += np.sum(freqs[pos] * uk2)
                    den += np.sum(uk2)
                omega[n + 1, k] = num / den if den > 1e-18 else omega[n, k]
            else:
                omega[n + 1, k] = 0.0

            sum_uk += u_hat[k]

        residual = np.sum(u_hat, axis=0) - f_hat
        lambda_hat = lambda_hat + tau * residual

        uDiff = np.sum(np.abs(u_hat - u_hat_prev) ** 2) / (u_hat.size + 1e-18)
        n += 1

    omega = omega[:n + 1, :]

    # Return to the time domain
    u_hat = np.fft.ifftshift(u_hat, axes=2)
    u = np.fft.ifft(u_hat, axis=2).real

    # Suppression extension miroir
    start = N0
    end = 2 * N0
    u = u[:, :, start:end]

    return u, omega


# ============================================================
# METRICS / TABLES
# ============================================================
def compute_features_per_mode_channel(u, fs, channel_names):
    """
    u : shape (K, C, N)
    Returnsne un DataFrame des indicateurs by mode and by channel.
    """
    K_, C_, N = u.shape
    freq_axis = np.fft.rfftfreq(N, d=1.0 / fs)
    rows = []

    for k in range(K_):
        for c in range(C_):
            x = u[k, c, :]
            rms = np.sqrt(np.mean(x ** 2))
            peak = np.max(np.abs(x))
            crest = peak / rms if rms > 1e-12 else np.nan
            ku = kurtosis(x, fisher=False, bias=False)
            sk = skew(x, bias=False)

            spectrum = np.abs(np.fft.rfft(x))
            dominant_freq = freq_axis[np.argmax(spectrum)]

            rows.append({
                "Mode": k + 1,
                "Channel": channel_names[c],
                "RMS": rms,
                "Kurtosis": ku,
                "Skewness": sk,
                "CrestFactor": crest,
                "DominantFreq_Hz": dominant_freq
            })

    return pd.DataFrame(rows)


def save_modes_csv_per_channel(tables_dir, u, time_s, channel_names):
    """
    Saves a CSV table for each channel containing all modes.
    """
    K_, C_, N = u.shape
    for c in range(C_):
        data = {"time_s": time_s}
        for k in range(K_):
            data[f"mode_{k + 1}"] = u[k, c, :]
        df = pd.DataFrame(data)
        out_csv = os.path.join(tables_dir, f"{channel_names[c]}_MVMD_modes.csv")
        df.to_csv(out_csv, index=False)


def save_omega_csv(tables_dir, omega):
    df = pd.DataFrame(omega, columns=[f"mode_{k + 1}" for k in range(omega.shape[1])])
    df.insert(0, "iteration", np.arange(1, len(df) + 1))
    df.to_csv(os.path.join(tables_dir, "MVMD_center_frequencies.csv"), index=False)


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
    fig.savefig(os.path.join(figures_dir, "01_raw_signals.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_modes_figures(figures_dir, u, fs, channel_names):
    K_, C_, N = u.shape
    t = np.arange(N) / fs

    for c in range(C_):
        fig, axes = plt.subplots(K_, 1, figsize=(14, 2.6 * K_), sharex=True)
        if K_ == 1:
            axes = [axes]

        for k in range(K_):
            axes[k].plot(t, u[k, c, :], linewidth=1.0)
            axes[k].set_title(f"{channel_names[c]} - Mode {k + 1}")
            axes[k].set_ylabel("Amplitude")
            axes[k].grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time (s)")
        fig.tight_layout()
        fig.savefig(
            os.path.join(figures_dir, f"02_modes_{channel_names[c]}.png"),
            dpi=200,
            bbox_inches="tight"
        )
        plt.close(fig)


def save_mode_spectra_figures(figures_dir, u, fs, channel_names):
    K_, C_, N = u.shape
    freq_axis = np.fft.rfftfreq(N, d=1.0 / fs)

    for c in range(C_):
        fig, axes = plt.subplots(K_, 1, figsize=(14, 2.6 * K_), sharex=True)
        if K_ == 1:
            axes = [axes]

        for k in range(K_):
            spectrum = np.abs(np.fft.rfft(u[k, c, :]))
            axes[k].plot(freq_axis, spectrum, linewidth=1.0)
            axes[k].set_title(f"{channel_names[c]} - FFT Mode {k + 1}")
            axes[k].set_ylabel("Amplitude")
            axes[k].grid(True, alpha=0.3)

        axes[-1].set_xlabel("Frequency (Hz)")
        fig.tight_layout()
        fig.savefig(
            os.path.join(figures_dir, f"03_fft_modes_{channel_names[c]}.png"),
            dpi=200,
            bbox_inches="tight"
        )
        plt.close(fig)


def save_omega_figure(figures_dir, omega):
    fig = plt.figure(figsize=(12, 4.5))
    for k in range(omega.shape[1]):
        plt.plot(omega[:, k], label=f"Mode {k + 1}")
    plt.title("Evolution of the center frequencies communes (MVMD)")
    plt.xlabel("Iteration")
    plt.ylabel("Normalized frequency")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "04_center_frequencies_evolution.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_feature_barplots(figures_dir, features_df):
    metrics = ["RMS", "Kurtosis", "Skewness", "CrestFactor", "DominantFreq_Hz"]

    for metric in metrics:
        pivot = features_df.pivot(index="Mode", columns="Channel", values=metric)

        fig = plt.figure(figsize=(10, 4.5))
        pivot.plot(kind="bar", ax=plt.gca())
        plt.title(f"{metric} by mode and by channel")
        plt.xlabel("Mode")
        plt.ylabel(metric)
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(
            os.path.join(figures_dir, f"05_{metric}_barplot.png"),
            dpi=200,
            bbox_inches="tight"
        )
        plt.close(fig)


# ============================================================
# COMPLETE NUMERICAL JSON
# ============================================================
def build_json_payload(
    input_file_info,
    X,
    u,
    omega,
    time_s,
    features_df,
    fs,
    params
):
    payload = {
        "metadata": {
            "method": "MVMD",
            "sampling_frequency_hz": fs,
            "n_channels": int(X.shape[0]),
            "signal_length_samples": int(X.shape[1]),
            "n_modes": int(u.shape[0]),
            "input_files": input_file_info
        },
        "parameters": params,
        "common_center_frequencies": omega,
        "features_table": features_df,
        "channel_names": [item["channel_name"] for item in input_file_info],
        "time_s": time_s
    }

    if SAVE_FULL_SIGNALS_IN_JSON:
        payload["preprocessed_signals"] = X
        payload["modes"] = u

    return to_serializable(payload)


# ============================================================
# MAIN PROGRAM
# ============================================================
def main():
    script_dir = get_script_directory()
    main_out, figures_out, tables_out = ensure_directories(script_dir)

    print("=" * 70)
    print("MVMD multichannel")
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

    print("Multichannel data dimensions:", X.shape)

    # Temps
    time_s = np.arange(X.shape[1]) / FS

    # MVMD decomposition
    print("Starting MVMD decomposition...")
    u, omega = mvmd(
        X,
        alpha=ALPHA,
        tau=TAU,
        K=K,
        DC=DC,
        init=INIT,
        tol=TOL,
        N_iter=N_ITER
    )
    print("Decomposition completed.")
    print("Mode dimensions:", u.shape)

    # Features
    features_df = compute_features_per_mode_channel(u, FS, channel_names)

    # Saving tables CSV
    features_csv = os.path.join(tables_out, "MVMD_features.csv")
    features_df.to_csv(features_csv, index=False)

    save_modes_csv_per_channel(tables_out, u, time_s, channel_names)
    save_omega_csv(tables_out, omega)

    # Saving figures
    save_raw_signals_figure(figures_out, X, FS, channel_names)
    save_modes_figures(figures_out, u, FS, channel_names)
    save_mode_spectra_figures(figures_out, u, FS, channel_names)
    save_omega_figure(figures_out, omega)
    save_feature_barplots(figures_out, features_df)

    # Complete numerical JSON
    params = {
        "ALPHA": ALPHA,
        "TAU": TAU,
        "K": K,
        "DC": DC,
        "INIT": INIT,
        "TOL": TOL,
        "N_ITER": N_ITER
    }

    json_payload = build_json_payload(
        input_file_info=input_file_info,
        X=X,
        u=u,
        omega=omega,
        time_s=time_s,
        features_df=features_df,
        fs=FS,
        params=params
    )

    json_path = os.path.join(main_out, "mvmd_numeric_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    # Brief text summary
    summary_txt = os.path.join(main_out, "README_results.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("MVMD Results\n")
        f.write("====================\n\n")
        f.write("Input files:\n")
        for item in input_file_info:
            f.write(
                f"- {item['resolved_filename']} | variable: {item['variable_name']} | "
                f"N = {item['n_samples']}\n"
            )
        f.write("\nMVMD parameters:\n")
        for k_param, v_param in params.items():
            f.write(f"- {k_param} = {v_param}\n")
        f.write("\nContents:\n")
        f.write("- Figures/ : PNG figures\n")
        f.write("- Tables/ : CSV tables\n")
        f.write("- mvmd_numeric_results.json : complete numerical values\n")

    print("=" * 70)
    print("Processing completed successfully.")
    print("Main table:", features_csv)
    print("Numerical JSON:", json_path)
    print("Figures:", figures_out)
    print("Tables:", tables_out)
    print("=" * 70)


if __name__ == "__main__":
    main()
