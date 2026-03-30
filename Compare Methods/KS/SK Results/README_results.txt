SK Results
====================

Method: STFT-based Spectral Kurtosis (channel-by-channel processing)
- Estimation de la courbe SK(f)
- Automatic band selection around the SK peak
- Filtrage passe-bande
- Hilbert envelope and envelope spectrum

Input files:
- ch1.mat | variable: y | N = 29981
- ch2.mat | variable: y | N = 29981
- ch3.mat | variable: y | N = 29981

SK parameters:
- FS = 50000
- N_PERSEG = 1024
- N_OVERLAP = 768
- WINDOW = hann
- SK_SMOOTH_BINS = 5
- BAND_THRESHOLD_RATIO = 0.5
- MIN_BAND_BINS = 4
- FILTER_ORDER = 4
- ENVELOPE_MAX_FREQ_HZ = 1000.0

Contents:
- Figures/ : PNG figures
- Tables/ : CSV tables
- sk_numeric_results.json : complete numerical values
