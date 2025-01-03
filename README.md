# Estimating the Joint Distribution of GRE Verbal, Quantitative, and Writing Scores

This repository contains a few different utilities for estimating distributions
of GRE Scores, and estimating joint percentiles. This is a special case of a
general question about simulating multivariate data based on knowledge of each
variable's marginal distribution and the matrix of the variables.

## Contents of the Repository

### Documentation from ETS

1. `gre-guide-table-1a.pdf` contains score percentiles for the three sections of
the GRE. These apply to GREs taken from 2020-2023. It also contains data on the
mean and standard deviation of scores for each of the three sections. Downloaded
from `https://www.ets.org/pdfs/gre/gre-guide-table-1a.pdf` December 17, 2024.
2. `vr-qr-scores.csv` is a `.csv` file containing the data from table 1b of
`gre-guide-table-1a.pdf`.
3. `aw-scores.csv` is a `.csv` file containing the data from table 1c of
`gre-guide-table-1a.pdf`.

### libraries.py

* `libraries.py` includes code to import the following libraries:
    1. numpy
    2. pandas
    3. math
    4. importlib.util

### main.py

* `main.py` is a driver script that uses the library to simulate 50,000 GRE
scores conforming to the percentiles and inter-section correlations provided
by ETS in `gre-guide-table-1a.pdf`.

### corr_sim.py

* `corr_sim.py` is a module that contains three functions. For complete
documentation see the docstrings therein.
    1. `corr_sim.percentiles_to_probabilities()` is a function that takes a vector
    of outcomes, a vector of associated percentile ranks, and converts the
    percentiles to probabilities.
    2. `corr_sim.corr_sim()` is a function that takes two vectors and orders them
    so the correlation between them is approximately (within a specified
    tolerance) equal to a target value.
    3. `corr_sim.multi_corr_sim()` is a function that takes two ordered vectors and a
    third unordered vector as an input. It then orders the third vector so its
    correlation with each of the other two is approximately (within a specified
    tolerance) equal to two given targets.
