IVA Results
====================

Method: Frequency-domain AuxIVA-type IVA (determined case)
- Multichannel source separation
- STFT multi-channel
- Frequency-domain demixing
- Back-projection onto the reference channel

Input files:
- ch1.mat | variable: y | N = 29981
- ch2.mat | variable: y | N = 29981
- ch3.mat | variable: y | N = 29981

IVA/STFT parameters:
- FS = 50000
- N_FFT = 1024
- N_OVERLAP = 768
- WINDOW = hann
- IVA_N_ITER = 20
- REF_CHANNEL = 0
- EPS = 1e-10

Contents:
- Figures/ : PNG figures
- Tables/ : CSV tables
- iva_numeric_results.json : complete numerical values
