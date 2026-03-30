FastICA Results
====================

Method: FastICA
- Multichannel source separation under the assumption of an instantaneous linear mixing model
- Export of the independent components, reconstruction, and residual
- Export of the mixing / demixing matrices when available
- Figures, CSV tables, and complete numerical JSON

Input files:
- ch1.mat | variable: y | N = 29981
- ch2.mat | variable: y | N = 29981
- ch3.mat | variable: y | N = 29981

FastICA parameters:
- FS = 50000
- N_COMPONENTS = 3
- ALGORITHM = deflation
- FUN = exp
- MAX_ITER = 1000
- TOL = 0.0001
- WHITEN = unit-variance
- RANDOM_STATE = 42

Contents:
- Figures/ : PNG figures
- Tables/ : CSV tables
- fastica_numeric_results.json : complete numerical values
