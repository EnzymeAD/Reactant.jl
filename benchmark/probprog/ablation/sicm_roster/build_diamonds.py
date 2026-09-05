#!/usr/bin/env python3
"""Build models/real_data/diamonds.json from the local posteriordb checkout.

Source: posteriordb posterior `diamonds-diamonds`, data file
`posterior_database/data/data/diamonds.json.zip` (brms 2.10.0 log-log price
regression on the ggplot2 diamonds data). The file is copied verbatim: N, K, Y,
X and prior_only, with a `_provenance` string prepended. Nothing is subsampled
and nothing is simulated; the centering of X's non-intercept columns is done by
`models/diamonds_regression.py` at load time, mirroring the Stan file's
`transformed data` block.

Usage:  python3 build_diamonds.py
"""
import json
import os
import zipfile

import numpy as np

SRC = ("/mnt/sbrantq/probprog/posteriordb/posterior_database/data/data/"
       "diamonds.json.zip")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "pplbench", "pplbench", "models", "real_data",
                   "diamonds.json")

PROVENANCE = (
    "brms `diamonds` log-log price regression on the ggplot2 diamonds data. "
    "Source: posteriordb posterior `diamonds-diamonds`, data file "
    "posterior_database/data/data/diamonds.json.zip, model file "
    "posterior_database/models/stan/diamonds.stan (generated with brms 2.10.0). "
    "Copied verbatim: N=5000 diamonds, response Y = log(price), and the brms "
    "population-level design matrix X (5000 x 25) whose first column is the "
    "intercept column of ones and whose remaining 24 columns are log(carat), "
    "spline/polynomial terms in the 4C grades and the physical dimensions. "
    "Y and X are the recorded regression variables, not simulated. "
    "`prior_only` is carried through from the source file (0 = fit the "
    "likelihood). Retrieved from the local posteriordb checkout at "
    "/mnt/sbrantq/probprog/posteriordb."
)


def main():
    with zipfile.ZipFile(SRC) as z:
        name = [n for n in z.namelist() if n.endswith(".json")][0]
        raw = json.loads(z.read(name))

    N = int(raw["N"])
    K = int(raw["K"])
    X = np.asarray(raw["X"], dtype=np.float64)
    Y = np.asarray(raw["Y"], dtype=np.float64)
    assert X.shape == (N, K), X.shape
    assert Y.shape == (N,), Y.shape
    assert np.allclose(X[:, 0], 1.0), "column 0 must be the intercept column"

    out = {
        "_provenance": PROVENANCE,
        "N": N,
        "K": K,
        "prior_only": int(raw["prior_only"]),
        "Y": [float(v) for v in Y],
        "X": [[float(v) for v in row] for row in X],
    }

    path = os.path.abspath(OUT)
    with open(path, "w") as f:
        json.dump(out, f)
    print("wrote", path, "(%.1f MB)" % (os.path.getsize(path) / 1e6))

    chk = json.load(open(path))
    assert np.allclose(np.asarray(chk["X"]), X)
    assert np.allclose(np.asarray(chk["Y"]), Y)
    print("roundtrip verified: N=%d K=%d" % (N, K))


if __name__ == "__main__":
    main()
