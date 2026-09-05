#!/usr/bin/env python3
"""Downstream-cascade evidence for the SICM ablation roster.

The LAPACK view (`analyze_ir.py lapack`) shows the factorization leaving the
leapfrog step. This one shows what happens to the dense elementwise algebra
around it.

If only the Cholesky rewrite fired, the density would still have to build the
per-sample factor `sigma * L` inside the leapfrog step (an n x n elementwise
multiply at depth 3) and solve against that per-sample matrix.
TriangularSolveScale removes that by rewriting solve(s*L, b) -> (1/s)*solve(L, b).
LogMultiply would remove the per-step log of the scaled diagonal by rewriting
log(s * diag(L)) -> log s + log diag(L), leaving the invariant sum of logs
outside the loop; whether it does so observably is what the `log` column tests.

Counts are by `stablehlo.while` nesting depth, using the same depth walker as
analyze_ir.py; depth 3 is the leapfrog body.

Usage: analyze_cascade.py            (all four roster models)
       analyze_cascade.py <model> <n>
"""
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_ir import _walk  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = [('car_spatial_scotland', 56), ('car_spatial_nyc', 250),
         ('hierarchical_mvn', 60), ('lmm_radon', 250)]


def run(model, n):
    print("########", model, "n=%d" % n)
    nn = 'tensor<%dx%dxf64>' % (n, n)
    nv = ('tensor<%dxf64>' % n, 'tensor<%dx1xf64>' % n, 'tensor<1x%dxf64>' % n)
    for arm in ('nosicm', 'sicm'):
        mm = collections.Counter()
        lg = collections.Counter()
        path = os.path.join(HERE, 'hlo', '%s_%s.mlir' % (model, arm))
        for depth, line in _walk(path):
            if re.search(r'stablehlo\.(multiply|divide)\b', line) and nn in line:
                mm[depth] += 1
            if re.search(r'stablehlo\.log\b', line) and any(t in line for t in nv):
                lg[depth] += 1
        print("  arm %-7s  %s multiply/divide by depth: %s"
              % (arm, nn, dict(sorted(mm.items())) or 'none'))
        print("            log of an n-vector by depth:  %s"
              % (dict(sorted(lg.items())) or 'none'))


if __name__ == "__main__":
    if len(sys.argv) == 3:
        run(sys.argv[1], int(sys.argv[2]))
    else:
        for m, n in CASES:
            run(m, n)
