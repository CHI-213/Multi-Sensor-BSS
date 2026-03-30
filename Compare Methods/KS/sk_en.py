#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sk_en.py

Spectral Kurtosis (SK) for three MATLAB files:
    ch1.mat, ch2.mat, ch3.mat

Requirements:
- The three .mat files must be located in the same directory as this script.
- The results are saved in a folder named:
      "SK Results"

Contents of "SK Results" :
- Figures/  -> PNG figures
- Tables/   -> CSV tables
- sk_numeric_results.json -> complete numerical values for comparison

Methodological note:
- Classical Spectral Kurtosis is treated here on a channel-by-channel basis.
- The script estimates an SK curve based on the STFT spectrogram:
      SK(f) = kurtosis_t(|STFT(f,t)|^2)
- It then automatically selects an "optimal" band around the SK peak,
  applies band-pass filtering, computes the Hilbert envelope, and computes the
  envelope spectrum.

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

from scipy.signal import stft, butter, sosfiltfilt, hilbert
from scipy.stats import kurtosis, skew


# ============================================================
# USER PARAMETERS
# ============================================================
FS = 50000  # Hz
INPUT_FILES = ["ch1.mat", "ch2.mat", "ch3.mat"]

MAIN_OUTPUT_DIRNAME = "SK Results"
FIGURES_DIRNAME = "Figures"
TABLES_DIRNAME = "Tables"

SAVE_FULL_SIGNALS_IN_JSON = True

# STFT / SK parameters
N_PERSEG = 1024
N_OVERLAP = 768
WINDOW = "hann"
SK_SMOOTH_BINS = 5            # light smoothing of the SK curve
BAND_THRESHOLD_RATIO = 0.50   # relative threshold around the peak (50%)
MIN_BAND_BINS = 4             # minimum band width in number of bins

# Filtering parameters
FILTER_ORDER = 4

# Envelope spectrum
ENVELOPE_MAX_FREQ_HZ = 1000.0


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


def moving_average(x, w):
    if w <= 1:
        return x.copy()
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(x, kernel, mode="same")


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
# SK / OPTIMAL BAND / ENVELOPE
# ============================================================
def compute_sk_curve(x, fs, nperseg, noverlap, window, smooth_bins=1):
    """
    Practical estimation of the Spectral Kurtosis curve from the STFT spectrogram:
        SK(f) = kurtosis_t(|STFT(f,t)|^2)
    """
    f, t, Z = stft(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary="zeros",
        padded=True,
        return_onesided=True
    )

    power = np.abs(Z) ** 2
    sk = kurtosis(power, axis=1, fisher=True, bias=False)
    sk = np.nan_to_num(sk, nan=0.0, posinf=0.0, neginf=0.0)

    if smooth_bins > 1:
        sk_smooth = moving_average(sk, smooth_bins)
    else:
        sk_smooth = sk.copy()

    return f, t, Z, power, sk, sk_smooth


def select_optimal_band(f_axis, sk_curve, threshold_ratio=0.5, min_band_bins=4):
    """
    Automatic selection of a band around the SK maximum.
    - The global peak of positive SK is selected.
    - The band is extended left/right as long as SK >= threshold_ratio * peak.
    - A minimum width is enforced.
    """
    sk_pos = np.maximum(sk_curve, 0.0)

    if np.all(sk_pos <= 0):
        peak_idx = int(np.argmax(sk_curve))
        left = max(0, peak_idx - min_band_bins // 2)
        right = min(len(f_axis) - 1, peak_idx + min_band_bins // 2)
    else:
        peak_idx = int(np.argmax(sk_pos))
        peak_value = float(sk_pos[peak_idx])
        threshold = threshold_ratio * peak_value

        left = peak_idx
        while left > 0 and sk_pos[left - 1] >= threshold:
            left -= 1

        right = peak_idx
        while right < len(f_axis) - 1 and sk_pos[right + 1] >= threshold:
            right += 1

        width = right - left + 1
        if width < min_band_bins:
            extra = (min_band_bins - width + 1) // 2
            left = max(0, left - extra)
            right = min(len(f_axis) - 1, right + extra)

    # Ensure correct ordering
    if right <= left:
        right = min(len(f_axis) - 1, left + 1)

    return {
        "peak_index": int(peak_idx),
        "peak_frequency_hz": float(f_axis[peak_idx]),
        "peak_sk_value": float(sk_curve[peak_idx]),
        "band_left_index": int(left),
        "band_right_index": int(right),
        "f_low_hz": float(f_axis[left]),
        "f_center_hz": float(f_axis[peak_idx]),
        "f_high_hz": float(f_axis[right]),
        "bandwidth_hz": float(f_axis[right] - f_axis[left])
    }


def safe_bandpass_filter(x, fs, f_low, f_high, order=4):
    """
    Robust band-pass filtering with safeguarded bounds.
    """
    nyq = 0.5 * fs
    low = max(float(f_low), 1.0)
    high = min(float(f_high), nyq - 1.0)

    if high <= low:
        # Minimal widening
        center = 0.5 * (low + high)
        low = max(center - 50.0, 1.0)
        high = min(center + 50.0, nyq - 1.0)

    if high <= low:
        raise ValueError("Invalid filtering band after safeguarding.")

    sos = butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    y = sosfiltfilt(sos, x)
    return y, low, high


def compute_envelope_and_spectrum(x_band, fs, max_freq_hz=1000.0):
    analytic = hilbert(x_band)
    envelope = np.abs(analytic)
    envelope = envelope - np.mean(envelope)

    N = len(envelope)
    freq_axis = np.fft.rfftfreq(N, d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(envelope))

    mask = freq_axis <= max_freq_hz
    freq_axis_lim = freq_axis[mask]
    spectrum_lim = spectrum[mask]

    if len(spectrum_lim) > 0:
        dom_idx = int(np.argmax(spectrum_lim))
        dom_freq = float(freq_axis_lim[dom_idx])
        dom_amp = float(spectrum_lim[dom_idx])
    else:
        dom_idx = 0
        dom_freq = 0.0
        dom_amp = 0.0

    return envelope, freq_axis_lim, spectrum_lim, dom_freq, dom_amp


def compute_basic_features(x, fs):
    x = np.asarray(x, dtype=float)
    N = len(x)
    freq_axis = np.fft.rfftfreq(N, d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(x))

    return {
        "RMS": float(np.sqrt(np.mean(x ** 2))),
        "Kurtosis": float(kurtosis(x, fisher=False, bias=False)),
        "Skewness": float(skew(x, bias=False)),
        "CrestFactor": float(np.max(np.abs(x)) / (np.sqrt(np.mean(x ** 2)) + 1e-18)),
        "DominantFreq_Hz": float(freq_axis[int(np.argmax(spectrum))])
    }


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


def save_sk_curve_figure(figures_dir, channel_name, f_axis, sk_raw, sk_smooth, band_info):
    fig = plt.figure(figsize=(12, 4.5))
    plt.plot(f_axis, sk_raw, linewidth=1.0, alpha=0.5, label="SK brute")
    plt.plot(f_axis, sk_smooth, linewidth=1.5, label="Smoothed SK")

    plt.axvspan(
        band_info["f_low_hz"],
        band_info["f_high_hz"],
        alpha=0.2,
        label="Selected band"
    )
    plt.axvline(band_info["f_center_hz"], linestyle="--", linewidth=1.2, label="Pic SK")

    plt.title(f"Spectral Kurtosis - {channel_name}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("SK")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(
        os.path.join(figures_dir, f"02_SK_curve_{channel_name}.png"),
        dpi=180,
        bbox_inches="tight"
    )
    plt.close(fig)


def save_band_signal_and_envelope_figure(figures_dir, channel_name, x_band, envelope, fs):
    N = len(x_band)
    t = np.arange(N) / fs

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    axes[0].plot(t, x_band, linewidth=1.0)
    axes[0].set_title(f"{channel_name} - Signal filtered in the optimal band")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, envelope, linewidth=1.0)
    axes[1].set_title(f"{channel_name} - Enveloppe de Hilbert")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figures_dir, f"03_band_and_envelope_{channel_name}.png"),
        dpi=180,
        bbox_inches="tight"
    )
    plt.close(fig)


def save_envelope_spectrum_figure(figures_dir, channel_name, f_env, spec_env, dom_freq):
    fig = plt.figure(figsize=(12, 4.5))
    plt.plot(f_env, spec_env, linewidth=1.0)
    plt.axvline(dom_freq, linestyle="--", linewidth=1.2, label=f"Peak = {dom_freq:.2f} Hz")
    plt.title(f"{channel_name} - Envelope spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(
        os.path.join(figures_dir, f"04_envelope_spectrum_{channel_name}.png"),
        dpi=180,
        bbox_inches="tight"
    )
    plt.close(fig)


def save_summary_barplots(figures_dir, summary_df):
    metrics = [
        "SK_peak_value",
        "CenterFreq_Hz",
        "Bandwidth_Hz",
        "FilteredKurtosis",
        "EnvelopeDominantFreq_Hz"
    ]

    for metric in metrics:
        fig = plt.figure(figsize=(9, 4.5))
        plt.bar(summary_df["Channel"], summary_df[metric])
        plt.title(f"{metric} by channel")
        plt.xlabel("Channel")
        plt.ylabel(metric)
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(
            os.path.join(figures_dir, f"05_{metric}_barplot.png"),
            dpi=180,
            bbox_inches="tight"
        )
        plt.close(fig)


# ============================================================
# COMPLETE NUMERICAL JSON
# ============================================================
def build_json_payload(input_file_info, results_per_channel, summary_df, fs, params):
    payload = {
        "metadata": {
            "method": "Spectral Kurtosis (STFT-based estimator, per channel)",
            "sampling_frequency_hz": fs,
            "n_channels": len(results_per_channel),
            "input_files": input_file_info
        },
        "parameters": params,
        "summary_table": summary_df,
        "channels": {}
    }

    for channel_name, result in results_per_channel.items():
        channel_payload = {
            "band_info": result["band_info"],
            "basic_features_original": result["basic_features_original"],
            "basic_features_filtered": result["basic_features_filtered"],
            "envelope_dominant_frequency_hz": result["envelope_dominant_frequency_hz"],
            "envelope_dominant_amplitude": result["envelope_dominant_amplitude"],
            "sk_frequency_axis_hz": result["f_axis"],
            "sk_curve_raw": result["sk_raw"],
            "sk_curve_smooth": result["sk_smooth"],
            "envelope_frequency_axis_hz": result["f_env"],
            "envelope_spectrum": result["spec_env"]
        }

        if SAVE_FULL_SIGNALS_IN_JSON:
            channel_payload["signal_preprocessed"] = result["signal_preprocessed"]
            channel_payload["signal_bandpassed"] = result["signal_bandpassed"]
            channel_payload["envelope"] = result["envelope"]

        payload["channels"][channel_name] = channel_payload

    return to_serializable(payload)


# ============================================================
# MAIN PROGRAM
# ============================================================
def main():
    script_dir = get_script_directory()
    main_out, figures_out, tables_out = ensure_directories(script_dir)

    print("=" * 70)
    print("Spectral Kurtosis (SK) par channel")
    print("Script directory:", script_dir)
    print("Output directory:", main_out)
    print("=" * 70)

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

    min_len = min(len(s) for s in signals)
    if len(set(len(s) for s in signals)) != 1:
        print(f"Different lengths detected. Truncating to {min_len} samples.")
    signals = [s[:min_len] for s in signals]

    X = np.vstack(signals)
    channel_names = [item["channel_name"] for item in input_file_info]

    save_raw_signals_figure(figures_out, X, FS, channel_names)

    results_per_channel = {}
    summary_rows = []

    for c, channel_name in enumerate(channel_names):
        print(f"Processing SK: {channel_name}")

        x = X[c, :]

        f_axis, t_axis, Z, power, sk_raw, sk_smooth = compute_sk_curve(
            x,
            FS,
            N_PERSEG,
            N_OVERLAP,
            WINDOW,
            smooth_bins=SK_SMOOTH_BINS
        )

        band_info = select_optimal_band(
            f_axis,
            sk_smooth,
            threshold_ratio=BAND_THRESHOLD_RATIO,
            min_band_bins=MIN_BAND_BINS
        )

        x_band, f_low_used, f_high_used = safe_bandpass_filter(
            x,
            FS,
            band_info["f_low_hz"],
            band_info["f_high_hz"],
            order=FILTER_ORDER
        )
        band_info["f_low_used_hz"] = float(f_low_used)
        band_info["f_high_used_hz"] = float(f_high_used)

        envelope, f_env, spec_env, env_dom_freq, env_dom_amp = compute_envelope_and_spectrum(
            x_band,
            FS,
            max_freq_hz=ENVELOPE_MAX_FREQ_HZ
        )

        basic_features_original = compute_basic_features(x, FS)
        basic_features_filtered = compute_basic_features(x_band, FS)

        results_per_channel[channel_name] = {
            "signal_preprocessed": x,
            "signal_bandpassed": x_band,
            "envelope": envelope,
            "f_axis": f_axis,
            "sk_raw": sk_raw,
            "sk_smooth": sk_smooth,
            "band_info": band_info,
            "basic_features_original": basic_features_original,
            "basic_features_filtered": basic_features_filtered,
            "f_env": f_env,
            "spec_env": spec_env,
            "envelope_dominant_frequency_hz": env_dom_freq,
            "envelope_dominant_amplitude": env_dom_amp
        }

        # Detailed SK table for each channel
        sk_df = pd.DataFrame({
            "Frequency_Hz": f_axis,
            "SK_raw": sk_raw,
            "SK_smooth": sk_smooth
        })
        sk_df.to_csv(
            os.path.join(tables_out, f"{channel_name}_SK_curve.csv"),
            index=False
        )

        # Envelope-spectrum table for each channel
        env_df = pd.DataFrame({
            "Frequency_Hz": f_env,
            "EnvelopeSpectrum": spec_env
        })
        env_df.to_csv(
            os.path.join(tables_out, f"{channel_name}_Envelope_spectrum.csv"),
            index=False
        )

        # Band-passed signal + envelope table
        time_s = np.arange(len(x)) / FS
        sig_df = pd.DataFrame({
            "time_s": time_s,
            "signal_preprocessed": x,
            "signal_bandpassed": x_band,
            "envelope": envelope
        })
        sig_df.to_csv(
            os.path.join(tables_out, f"{channel_name}_Bandpassed_and_envelope.csv"),
            index=False
        )

        # Figures
        save_sk_curve_figure(figures_out, channel_name, f_axis, sk_raw, sk_smooth, band_info)
        save_band_signal_and_envelope_figure(figures_out, channel_name, x_band, envelope, FS)
        save_envelope_spectrum_figure(figures_out, channel_name, f_env, spec_env, env_dom_freq)

        summary_rows.append({
            "Channel": channel_name,
            "SK_peak_value": band_info["peak_sk_value"],
            "CenterFreq_Hz": band_info["f_center_hz"],
            "LowFreq_Hz": band_info["f_low_hz"],
            "HighFreq_Hz": band_info["f_high_hz"],
            "Bandwidth_Hz": band_info["bandwidth_hz"],
            "FilteredKurtosis": basic_features_filtered["Kurtosis"],
            "FilteredRMS": basic_features_filtered["RMS"],
            "EnvelopeDominantFreq_Hz": env_dom_freq,
            "EnvelopeDominantAmplitude": env_dom_amp
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(tables_out, "SK_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    save_summary_barplots(figures_out, summary_df)

    params = {
        "FS": FS,
        "N_PERSEG": N_PERSEG,
        "N_OVERLAP": N_OVERLAP,
        "WINDOW": WINDOW,
        "SK_SMOOTH_BINS": SK_SMOOTH_BINS,
        "BAND_THRESHOLD_RATIO": BAND_THRESHOLD_RATIO,
        "MIN_BAND_BINS": MIN_BAND_BINS,
        "FILTER_ORDER": FILTER_ORDER,
        "ENVELOPE_MAX_FREQ_HZ": ENVELOPE_MAX_FREQ_HZ
    }

    json_payload = build_json_payload(
        input_file_info=input_file_info,
        results_per_channel=results_per_channel,
        summary_df=summary_df,
        fs=FS,
        params=params
    )

    json_path = os.path.join(main_out, "sk_numeric_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    summary_txt = os.path.join(main_out, "README_results.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("SK Results\n")
        f.write("====================\n\n")
        f.write("Method: STFT-based Spectral Kurtosis (channel-by-channel processing)\n")
        f.write("- Estimation de la courbe SK(f)\n")
        f.write("- Automatic band selection around the SK peak\n")
        f.write("- Filtrage passe-bande\n")
        f.write("- Hilbert envelope and envelope spectrum\n\n")

        f.write("Input files:\n")
        for item in input_file_info:
            f.write(
                f"- {item['resolved_filename']} | variable: {item['variable_name']} | "
                f"N = {item['n_samples']}\n"
            )

        f.write("\nSK parameters:\n")
        for k_param, v_param in params.items():
            f.write(f"- {k_param} = {v_param}\n")

        f.write("\nContents:\n")
        f.write("- Figures/ : PNG figures\n")
        f.write("- Tables/ : CSV tables\n")
        f.write("- sk_numeric_results.json : complete numerical values\n")

    print("=" * 70)
    print("Processing completed successfully.")
    print("Main table:", summary_csv)
    print("Numerical JSON:", json_path)
    print("Figures:", figures_out)
    print("Tables:", tables_out)
    print("=" * 70)


if __name__ == "__main__":
    main()
