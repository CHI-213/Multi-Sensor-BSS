#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fastica_en.py

Multichannel FastICA for three MATLAB files:
    ch1.mat, ch2.mat, ch3.mat

Requirements:
- The three .mat files must be located in the same directory as this script.
- The results are saved in a folder named:
      "FastICA Results"

Contents of "FastICA Results" :
- Figures/  -> PNG figures
- Tables/   -> CSV tables
- fastica_numeric_results.json -> complete numerical values for comparison

Important note:
- FastICA is a multichannel source separation method for an
  instantaneous linear mixing model.
- This script applies FastICA to the three synchronous signals and then exports
  the independent components, the reconstruction, the mixing /
  demixing matrices, as well as several useful indicators for comparison with IVA,
  MVMD, MEMD, NA-MEMD, VMD, and SK.

Dependencies:
    pip install numpy scipy pandas matplotlib scikit-learn
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
from sklearn.decomposition import FastICA


# ============================================================
# USER PARAMETERS
# ============================================================
FS = 50000  # Hz
INPUT_FILES = ["ch1.mat", "ch2.mat", "ch3.mat"]

MAIN_OUTPUT_DIRNAME = "FastICA Results"
FIGURES_DIRNAME = "Figures"
TABLES_DIRNAME = "Tables"

SAVE_FULL_SIGNALS_IN_JSON = True
SAVE_MATRICES_IN_JSON = True

# FastICA parameters
N_COMPONENTS = 3
ALGORITHM = "deflation"          # "parallel" ou "deflation"
FUN = "exp"                 # "logcosh", "exp = gaus", "cube"
MAX_ITER = 1000
TOL = 1e-4
WHITEN = "unit-variance"
RANDOM_STATE = 42

# Options d'analyse
SAVE_SCATTER_PLOTS = True


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
# FASTICA / METRICS
# ============================================================
def run_fastica(X, n_components=3, algorithm="parallel", fun="logcosh",
                max_iter=1000, tol=1e-4, whiten="unit-variance",
                random_state=42):
    """
    X : shape (C, N)
    Returns:
    - S: separated sources, shape (n_components, N)
    - X_recon: reconstruction, shape (C, N)
    - model: trained FastICA object
    """
    X_samples = X.T  # shape (N, C)

    model = FastICA(
        n_components=n_components,
        algorithm=algorithm,
        fun=fun,
        max_iter=max_iter,
        tol=tol,
        whiten=whiten,
        random_state=random_state,
    )

    S_samples = model.fit_transform(X_samples)   # (N, n_components)
    X_recon_samples = model.inverse_transform(S_samples)

    S = S_samples.T
    X_recon = X_recon_samples.T
    return S, X_recon, model


def compute_signal_features(signals, fs, labels, signal_type):
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
        energy = np.sum(x ** 2)

        rows.append({
            "SignalType": signal_type,
            "Label": labels[i],
            "RMS": rms,
            "Kurtosis": ku,
            "Skewness": sk,
            "CrestFactor": crest,
            "DominantFreq_Hz": dominant_freq,
            "Energy": energy
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


def compute_reconstruction_metrics(X, X_recon, channel_names):
    residual = X - X_recon
    rows = []

    for i, name in enumerate(channel_names):
        rmse = np.sqrt(np.mean((X[i] - X_recon[i]) ** 2))
        snr_db = 10.0 * np.log10(
            (np.sum(X[i] ** 2) + 1e-18) / (np.sum((X[i] - X_recon[i]) ** 2) + 1e-18)
        )
        rows.append({
            "Channel": name,
            "ReconstructionRMSE": rmse,
            "ReconstructionSNR_dB": snr_db
        })

    return pd.DataFrame(rows), residual


def compute_matching_matrix(X, S):
    """
    Simple absolute-correlation matrix between measured channels and separated sources.
    """
    C = X.shape[0]
    N = S.shape[0]
    M = np.zeros((C, N), dtype=float)

    for i in range(C):
        for j in range(N):
            corr = np.corrcoef(X[i], S[j])[0, 1]
            if np.isnan(corr):
                corr = 0.0
            M[i, j] = abs(corr)
    return M


def compute_source_spectra(S, fs):
    N = S.shape[1]
    freq_axis = np.fft.rfftfreq(N, d=1.0 / fs)
    spectra = np.abs(np.fft.rfft(S, axis=1))
    return freq_axis, spectra


def save_signals_csv(tables_dir, filename, time_s, signals, labels):
    data = {"time_s": time_s}
    for i, label in enumerate(labels):
        data[label] = signals[i, :]

    out_csv = os.path.join(tables_dir, filename)
    pd.DataFrame(data).to_csv(out_csv, index=False)
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
        axes[c].set_title(f"Independent component - {source_labels[c]}")
        axes[c].set_ylabel("Amplitude")
        axes[c].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "02_independent_components.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_reconstruction_figure(figures_dir, X, X_recon, residual, fs, channel_names):
    C, N = X.shape
    t = np.arange(N) / fs

    for c in range(C):
        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

        axes[0].plot(t, X[c], linewidth=1.0)
        axes[0].set_title(f"{channel_names[c]} - Original signal")
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, X_recon[c], linewidth=1.0)
        axes[1].set_title(f"{channel_names[c]} - Reconstruction FastICA")
        axes[1].set_ylabel("Amplitude")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(t, residual[c], linewidth=1.0)
        axes[2].set_title(f"{channel_names[c]} - Residual")
        axes[2].set_ylabel("Amplitude")
        axes[2].set_xlabel("Time (s)")
        axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(
            os.path.join(figures_dir, f"03_reconstruction_{channel_names[c]}.png"),
            dpi=180,
            bbox_inches="tight"
        )
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
        axes[c].set_title(f"FFT of independent component - {source_labels[c]}")
        axes[c].set_ylabel("Amplitude")
        axes[c].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frequency (Hz)")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "04_independent_components_fft.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_correlation_heatmap(figures_dir, corr_df, filename, title):
    fig = plt.figure(figsize=(6.5, 5.5))
    plt.imshow(corr_df.values, aspect="auto")
    plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr_df.index)), corr_df.index)

    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            plt.text(j, i, f"{corr_df.values[i, j]:.3f}", ha="center", va="center")

    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, filename), dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_matching_heatmap(figures_dir, matching_df):
    fig = plt.figure(figsize=(6.5, 5.0))
    plt.imshow(matching_df.values, aspect="auto")
    plt.xticks(range(len(matching_df.columns)), matching_df.columns, rotation=45, ha="right")
    plt.yticks(range(len(matching_df.index)), matching_df.index)

    for i in range(matching_df.shape[0]):
        for j in range(matching_df.shape[1]):
            plt.text(j, i, f"{matching_df.values[i, j]:.3f}", ha="center", va="center")

    plt.title("Absolute correlation |measurement vs. independent component|")
    plt.colorbar()
    plt.tight_layout()
    fig.savefig(
        os.path.join(figures_dir, "07_matching_matrix.png"),
        dpi=180,
        bbox_inches="tight"
    )
    plt.close(fig)


def save_energy_barplot(figures_dir, energy_df):
    fig = plt.figure(figsize=(8, 4.5))
    plt.bar(energy_df["Label"], energy_df["EnergyPercent"])
    plt.title("Energy percentage of the independent components")
    plt.xlabel("Component")
    plt.ylabel("Energy (%)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(
        os.path.join(figures_dir, "08_source_energy_barplot.png"),
        dpi=180,
        bbox_inches="tight"
    )
    plt.close(fig)


def save_feature_barplots(figures_dir, features_df):
    metrics = ["RMS", "Kurtosis", "Skewness", "CrestFactor", "DominantFreq_Hz"]

    for metric in metrics:
        fig = plt.figure(figsize=(8, 4.5))
        plt.bar(features_df["Label"], features_df[metric])
        plt.title(f"{metric} of the independent components")
        plt.xlabel("Component")
        plt.ylabel(metric)
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(
            os.path.join(figures_dir, f"09_{metric}_barplot.png"),
            dpi=180,
            bbox_inches="tight"
        )
        plt.close(fig)


def save_reconstruction_barplot(figures_dir, recon_df):
    metrics = ["ReconstructionRMSE", "ReconstructionSNR_dB"]

    for metric in metrics:
        fig = plt.figure(figsize=(8, 4.5))
        plt.bar(recon_df["Channel"], recon_df[metric])
        plt.title(f"{metric} by channel")
        plt.xlabel("Channel")
        plt.ylabel(metric)
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(
            os.path.join(figures_dir, f"10_{metric}_barplot.png"),
            dpi=180,
            bbox_inches="tight"
        )
        plt.close(fig)


def save_scatter_plots(figures_dir, S, source_labels):
    n = S.shape[0]
    if n < 2:
        return

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))

    for i, j in pairs:
        fig = plt.figure(figsize=(5.5, 5.5))
        plt.scatter(S[i], S[j], s=2, alpha=0.4)
        plt.xlabel(source_labels[i])
        plt.ylabel(source_labels[j])
        plt.title(f"Scatter plot: {source_labels[i]} vs {source_labels[j]}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(
            os.path.join(figures_dir, f"11_scatter_{source_labels[i]}_{source_labels[j]}.png"),
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
    X_recon,
    residual,
    source_features_df,
    energy_df,
    corr_sources_df,
    corr_measured_df,
    matching_df,
    recon_df,
    freq_axis,
    spectra,
    model,
    time_s,
    params
):
    payload = {
        "metadata": {
            "method": "FastICA",
            "sampling_frequency_hz": FS,
            "n_channels": int(X.shape[0]),
            "signal_length_samples": int(X.shape[1]),
            "n_components": int(S.shape[0]),
            "input_files": input_file_info
        },
        "parameters": params,
        "source_features_table": source_features_df,
        "energy_table": energy_df,
        "source_correlation_matrix": corr_sources_df,
        "measured_signal_correlation_matrix": corr_measured_df,
        "matching_matrix_abs_correlation": matching_df,
        "reconstruction_table": recon_df,
        "source_frequency_axis_hz": freq_axis,
        "time_s": time_s,
        "fastica_info": {
            "n_iter_": int(getattr(model, "n_iter_", -1))
        }
    }

    if SAVE_FULL_SIGNALS_IN_JSON:
        payload["preprocessed_signals"] = X
        payload["independent_components"] = S
        payload["reconstruction"] = X_recon
        payload["residual"] = residual
        payload["source_spectra"] = spectra

    if SAVE_MATRICES_IN_JSON:
        payload["mixing_matrix"] = getattr(model, "mixing_", None)
        payload["components_matrix"] = getattr(model, "components_", None)
        payload["mean_vector"] = getattr(model, "mean_", None)
        payload["whitening_matrix"] = getattr(model, "whitening_", None)

    return to_serializable(payload)


# ============================================================
# MAIN PROGRAM
# ============================================================
def main():
    script_dir = get_script_directory()
    main_out, figures_out, tables_out = ensure_directories(script_dir)

    print("=" * 70)
    print("FastICA multichannel")
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

    # Truncation if needed
    min_len = min(len(s) for s in signals)
    if len(set(len(s) for s in signals)) != 1:
        print(f"Different lengths detected. Truncating to {min_len} samples.")
    signals = [s[:min_len] for s in signals]

    X = np.vstack(signals)  # shape (3, N)
    channel_names = [item["channel_name"] for item in input_file_info]
    source_labels = [f"IC_{i + 1}" for i in range(N_COMPONENTS)]
    time_s = np.arange(X.shape[1]) / FS

    print("Multichannel data dimensions:", X.shape)

    # Raw signal figure
    save_raw_signals_figure(figures_out, X, FS, channel_names)

    # FastICA
    print("Starting FastICA separation...")
    S, X_recon, model = run_fastica(
        X,
        n_components=N_COMPONENTS,
        algorithm=ALGORITHM,
        fun=FUN,
        max_iter=MAX_ITER,
        tol=TOL,
        whiten=WHITEN,
        random_state=RANDOM_STATE
    )
    print("FastICA completed.")
    print("Independent component dimensions:", S.shape)

    # Analyses
    source_features_df = compute_signal_features(S, FS, source_labels, "IndependentComponent")
    energy_df = compute_energy_percentages(S, source_labels)
    corr_sources_df = compute_correlation_matrix(S, source_labels)
    corr_measured_df = compute_correlation_matrix(X, channel_names)
    recon_df, residual = compute_reconstruction_metrics(X, X_recon, channel_names)

    matching = compute_matching_matrix(X, S)
    matching_df = pd.DataFrame(
        matching,
        index=channel_names,
        columns=source_labels
    )

    freq_axis, spectra = compute_source_spectra(S, FS)

    # CSV export
    source_features_df.to_csv(os.path.join(tables_out, "FastICA_features.csv"), index=False)
    energy_df.to_csv(os.path.join(tables_out, "FastICA_energy_percentages.csv"), index=False)
    corr_sources_df.to_csv(os.path.join(tables_out, "FastICA_source_correlations.csv"), index=True)
    corr_measured_df.to_csv(os.path.join(tables_out, "FastICA_measured_correlations.csv"), index=True)
    matching_df.to_csv(os.path.join(tables_out, "FastICA_matching_matrix.csv"), index=True)
    recon_df.to_csv(os.path.join(tables_out, "FastICA_reconstruction_metrics.csv"), index=False)

    save_signals_csv(tables_out, "FastICA_independent_components.csv", time_s, S, source_labels)
    save_signals_csv(tables_out, "FastICA_reconstruction.csv", time_s, X_recon, channel_names)
    save_signals_csv(tables_out, "FastICA_residual.csv", time_s, residual, channel_names)

    spec_data = {"Frequency_Hz": freq_axis}
    for i, label in enumerate(source_labels):
        spec_data[label] = spectra[i, :]
    pd.DataFrame(spec_data).to_csv(
        os.path.join(tables_out, "FastICA_source_spectra.csv"),
        index=False
    )

    # Matrices
    if hasattr(model, "mixing_") and model.mixing_ is not None:
        pd.DataFrame(
            model.mixing_,
            index=channel_names,
            columns=source_labels
        ).to_csv(os.path.join(tables_out, "FastICA_mixing_matrix.csv"), index=True)

    if hasattr(model, "components_") and model.components_ is not None:
        pd.DataFrame(
            model.components_,
            index=source_labels,
            columns=channel_names
        ).to_csv(os.path.join(tables_out, "FastICA_demixing_matrix.csv"), index=True)

    # Figures
    save_sources_figure(figures_out, S, FS, source_labels)
    save_reconstruction_figure(figures_out, X, X_recon, residual, FS, channel_names)
    save_sources_fft_figure(figures_out, S, FS, source_labels)
    save_correlation_heatmap(
        figures_out,
        corr_sources_df,
        "05_source_correlation_heatmap.png",
        "Correlation between independent components"
    )
    save_correlation_heatmap(
        figures_out,
        corr_measured_df,
        "06_measured_correlation_heatmap.png",
        "Correlation between measured signals"
    )
    save_matching_heatmap(figures_out, matching_df)
    save_energy_barplot(figures_out, energy_df)
    save_feature_barplots(figures_out, source_features_df)
    save_reconstruction_barplot(figures_out, recon_df)

    if SAVE_SCATTER_PLOTS:
        save_scatter_plots(figures_out, S, source_labels)

    # JSON
    params = {
        "FS": FS,
        "N_COMPONENTS": N_COMPONENTS,
        "ALGORITHM": ALGORITHM,
        "FUN": FUN,
        "MAX_ITER": MAX_ITER,
        "TOL": TOL,
        "WHITEN": WHITEN,
        "RANDOM_STATE": RANDOM_STATE
    }

    json_payload = build_json_payload(
        input_file_info=input_file_info,
        X=X,
        S=S,
        X_recon=X_recon,
        residual=residual,
        source_features_df=source_features_df,
        energy_df=energy_df,
        corr_sources_df=corr_sources_df,
        corr_measured_df=corr_measured_df,
        matching_df=matching_df,
        recon_df=recon_df,
        freq_axis=freq_axis,
        spectra=spectra,
        model=model,
        time_s=time_s,
        params=params
    )

    json_path = os.path.join(main_out, "fastica_numeric_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    # Text summary
    summary_txt = os.path.join(main_out, "README_results.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("FastICA Results\n")
        f.write("====================\n\n")
        f.write("Method: FastICA\n")
        f.write("- Multichannel source separation under the assumption of an instantaneous linear mixing model\n")
        f.write("- Export of the independent components, reconstruction, and residual\n")
        f.write("- Export of the mixing / demixing matrices when available\n")
        f.write("- Figures, CSV tables, and complete numerical JSON\n\n")

        f.write("Input files:\n")
        for item in input_file_info:
            f.write(
                f"- {item['resolved_filename']} | variable: {item['variable_name']} | "
                f"N = {item['n_samples']}\n"
            )

        f.write("\nFastICA parameters:\n")
        for k_param, v_param in params.items():
            f.write(f"- {k_param} = {v_param}\n")

        f.write("\nContents:\n")
        f.write("- Figures/ : PNG figures\n")
        f.write("- Tables/ : CSV tables\n")
        f.write("- fastica_numeric_results.json : complete numerical values\n")

    print("=" * 70)
    print("Processing completed successfully.")
    print("Main table:", os.path.join(tables_out, "FastICA_features.csv"))
    print("Numerical JSON:", json_path)
    print("Figures:", figures_out)
    print("Tables:", tables_out)
    print("=" * 70)


if __name__ == "__main__":
    main()
