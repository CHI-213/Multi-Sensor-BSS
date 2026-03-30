VMD Results
====================

Method: Classical VMD (channel-by-channel processing)
- Decomposition of each channel into K modes
- Computation of statistical and frequency-domain indicators for each mode
- Signal reconstruction and residual
- Export of figures, CSV tables, and complete numerical JSON

Input files:
- ch1.mat | variable: y | N = 29981
- ch2.mat | variable: y | N = 29981
- ch3.mat | variable: y | N = 29981

VMD parameters:
- FS = 50000
- ALPHA = 2000.0
- TAU = 0.0
- K = 5
- DC = 0
- INIT = 1
- TOL = 1e-07

Contents:
- Figures/ : PNG figures
- Tables/ : CSV tables
- vmd_numeric_results.json : complete numerical values
