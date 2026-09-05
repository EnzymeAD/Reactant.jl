#!/usr/bin/env python3
"""Structural IR evidence for the SICM ablation roster.

Two views of a saved dump, both classified by `stablehlo.while` nesting depth
so that "the factorization left the sampling loop" is checkable rather than
asserted:

  lapack   final IR (`*_pre_xla_compile.mlir`). Classifies the
           `stablehlo.custom_call @enzymexla_compile_cpu` LAPACK sites by
           result signature:
             (tensor<NxNxf64>, tensor<i64>)  -> potrf  (factor + info)
             tensor<NxMxf64>                 -> trsm   (triangular solve)
             (tensor<NxNxf64>, tensor<Nxf64>, tensor<i64>) -> syevd
           and reports each by (kind, N, depth). N distinguishes the model's
           own covariance (n x n) from the NUTS dense mass matrix
           (position_size x position_size), which both arms carry.

  hlo      pre-lowering StableHLO (`*_post_impulse_pm.mlir`). Counts
           stablehlo.cholesky / triangular_solve / dot_general and
           enzymexla.syevd by depth, which is where the SICM rewrites are
           still visible as StableHLO ops.

The interesting depth is "inside the NUTS sampling while" (depth >= 1). A
factorization at depth 0 has been hoisted out of sampling entirely.

Usage:
  analyze_ir.py lapack <file.mlir> [<file.mlir> ...]
  analyze_ir.py hlo    <file.mlir> [<file.mlir> ...]
  analyze_ir.py json   <label>=<lapack.mlir> ...      (machine-readable)
"""
import collections
import json
import re
import sys

TENSOR_RE = re.compile(r"tensor<([0-9x]*)f64>")


def _walk(path):
    """Yield (depth, line) with depth = number of enclosing while-body regions.

    Copied from ablation/pre_lowering/analyze_ir.py so both ablations classify
    by identical code. MLIR pretty-prints `stablehlo.while(...)` then
    `cond { ... } do { ... }` on following lines, so brace-depth tracking of
    the op line alone is wrong; track region open/close markers instead.
    """
    region_is_while = []
    pending_while = False
    for raw in open(path, errors="replace"):
        line = raw.strip()
        if line.startswith("}") and region_is_while:
            region_is_while.pop()
        yield sum(region_is_while), line
        if "stablehlo.while" in line:
            pending_while = True
        if line.endswith("{"):
            is_body = ("do {" in line or line.endswith("cond {")) and pending_while
            region_is_while.append(is_body)
            if "do {" in line:
                pending_while = False


def _result_sig(line):
    m = re.search(r"->\s*(\(.*?\)|tensor<[^>]*>)\s*(loc\(.*\))?\s*$", line)
    return m.group(1) if m else None


def _dims(t):
    m = re.match(r"tensor<([^>]*)>", t)
    if not m:
        return []
    parts = m.group(1).split("x")[:-1]
    try:
        return [int(p) for p in parts]
    except ValueError:
        return []


def lapack_sites(path):
    """-> Counter[(kind, size, depth)] and a list of (depth, kind, size, sig)."""
    counts = collections.Counter()
    for depth, line in _walk(path):
        if "custom_call @enzymexla_compile_cpu" not in line:
            continue
        sig = _result_sig(line)
        if sig is None:
            counts[("unclassified", 0, depth)] += 1
            continue
        tensors = re.findall(r"tensor<[^>]*>", sig)
        dims = [_dims(t) for t in tensors]
        if len(tensors) >= 3 and "i64" in sig and any(len(d) == 1 for d in dims):
            kind, size = "syevd", (dims[0][0] if dims[0] else 0)
        elif len(tensors) == 2 and "i64" in sig:
            kind, size = "potrf", (dims[0][0] if dims[0] else 0)
        elif len(tensors) == 1:
            d = dims[0]
            kind = "trsm"
            size = d[0] if d else 0
        else:
            kind, size = "other[" + sig[:40] + "]", 0
        counts[(kind, size, depth)] += 1
    return counts


def hlo_sites(path):
    counts = collections.Counter()
    pats = [
        ("stablehlo.cholesky", re.compile(r"stablehlo\.cholesky")),
        ("stablehlo.triangular_solve", re.compile(r"stablehlo\.triangular_solve")),
        ("enzymexla.syevd", re.compile(r"enzymexla\.syevd")),
        ("stablehlo.dot_general", re.compile(r"stablehlo\.dot_general")),
        ("stablehlo.sqrt", re.compile(r"stablehlo\.sqrt")),
    ]
    for depth, line in _walk(path):
        for name, rx in pats:
            if rx.search(line):
                sig = _result_sig(line) or ""
                t = re.findall(r"tensor<[^>]*>", sig)
                d = _dims(t[0]) if t else []
                size = "x".join(str(x) for x in d) if d else "scalar"
                counts[(name, size, depth)] += 1
    return counts


def report(counts, title):
    print("==", title)
    if not counts:
        print("   (no sites)")
        return
    for (kind, size, depth) in sorted(counts, key=lambda k: (str(k[0]), str(k[1]), k[2])):
        print("   %-28s size=%-10s depth=%d  x%d"
              % (kind, size, depth, counts[(kind, size, depth)]))


def main():
    mode = sys.argv[1]
    if mode == "json":
        out = {}
        for spec in sys.argv[2:]:
            label, path = spec.split("=", 1)
            c = lapack_sites(path)
            out[label] = {"%s|%s|%s" % k: v for k, v in c.items()}
        print(json.dumps(out, indent=2))
        return
    fn = {"lapack": lapack_sites, "hlo": hlo_sites}[mode]
    for p in sys.argv[2:]:
        report(fn(p), p.split("/")[-1])


if __name__ == "__main__":
    main()
