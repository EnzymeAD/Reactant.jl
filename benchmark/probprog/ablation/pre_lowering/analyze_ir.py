#!/usr/bin/env python3
"""Structural IR evidence for the tab:prelowering ablation.

Classifies ops in a saved final-HLO dump by stablehlo.while nesting depth, so
that "the loop-invariant op left the innermost model loop" is checkable rather
than asserted. Depth conventions for these harnesses (max nesting 4):

  depth 0  top level (the InitHMC gradient evaluation)
  depth 1  inside the NUTS sampling while (and, at depth 1, the InitHMC
           gradient's own model time loop)
  depth 3  per leapfrog step, outside the model time loop
  depth 4  the innermost model time loop

Three analyses:

  dots   FLOP-classify stablehlo.dot_general. FLOPs = numel(result) *
         prod(sizes of the LHS contracting dims); shapes alone are not enough
         because `(DxD, DxD) -> f64` is a Frobenius inner product (O(D^2)),
         not a matmul. This is the Cubic Kernel evidence.
  elt    Count D x D elementwise/shape ops by depth. This is the Projected
         Dynamics evidence (its invariant K = theta*A + I is elementwise).
  potrf  Classify enzymexla_compile_cpu LAPACK custom_calls by result
         signature ((DxD, i64) = potrf, Dx1 = trsm). This is the Heat
         Equation evidence, matching heat_inverse_RESULTS_2026-08-21.md.

Usage:
  python3 analyze_ir.py dots  <D> <file.mlir> ...
  python3 analyze_ir.py elt   <D> <file.mlir> ...
  python3 analyze_ir.py potrf <D> <file.mlir> ...
"""
import re
import sys
import collections

SIG_RE = re.compile(r":\s*\((tensor<[^)]*)\)\s*->\s*(tensor<[^>]*>|\S+)\s*$")
CDIM_RE = re.compile(r"contracting_dims\s*=\s*\[([0-9,\s]*)\]\s*x\s*\[([0-9,\s]*)\]")


def _dims(tensor_str):
    m = re.match(r"tensor<([^>]*)>", tensor_str)
    if not m:
        return None
    try:
        return [int(p) for p in m.group(1).split("x")[:-1]]
    except ValueError:
        return None


def _walk(path):
    """Yield (depth, stripped_line) with depth = enclosing while-body regions.

    MLIR pretty-prints `stablehlo.while(...)` then `cond { ... } do { ... }` on
    following lines, so brace-depth tracking of the op line alone is wrong;
    track region open/close via the `cond {` / `} do {` / `}` markers instead.
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


def _flops(line):
    m = SIG_RE.search(line)
    cm = CDIM_RE.search(line)
    if not m or not cm:
        return None, None
    operands = re.findall(r"tensor<[^>]*>", m.group(1))
    lhs = _dims(operands[0]) if operands else None
    res = _dims(m.group(2)) if m.group(2).startswith("tensor<") else []
    if lhs is None or res is None:
        return None, None
    n_out = 1
    for d in res:
        n_out *= d
    n_c = 1
    for i in (int(x) for x in cm.group(1).split(",") if x.strip()):
        if i < len(lhs):
            n_c *= lhs[i]
    sig = "%s x %s -> %s" % (operands[0],
                             operands[1] if len(operands) > 1 else "?", m.group(2))
    return n_out * n_c, sig


def dots(path, D):
    cubic_thresh = D ** 3 / 2.0
    by_depth = collections.Counter()
    detail = collections.Counter()
    for depth, line in _walk(path):
        if "stablehlo.dot_general" not in line:
            continue
        f, sig = _flops(line)
        kind = "unknown" if f is None else ("O(D^3)" if f >= cubic_thresh else "O(<=D^2)")
        by_depth[(depth, kind)] += 1
        detail[(depth, kind, sig, f)] += 1
    cub = {d: n for (d, k), n in by_depth.items() if k == "O(D^3)"}
    print("   O(D^3) dot_generals by while-depth:",
          ", ".join("depth%d=%d" % (d, cub[d]) for d in sorted(cub)) or "none")
    for k in sorted(detail, key=lambda x: (x[0], -(x[3] or 0))):
        print("   depth=%d %-9s %-58s flops=%.3g x%d"
              % (k[0], k[1], k[2], k[3] or -1, detail[k]))


def elt(path, D):
    rx = re.compile(r"stablehlo\.(multiply|add|subtract|negate|transpose|reshape|"
                    r"broadcast_in_dim)[^:]*: .*tensor<%dx%dxf64>$" % (D, D))
    d = collections.Counter()
    for depth, line in _walk(path):
        if rx.search(line):
            op = line.split("=")[1].strip().split()[0] if "=" in line else line.split()[0]
            d[(depth, op)] += 1
    for k in sorted(d):
        print("   depth=%d %-28s %d" % (k[0], k[1], d[k]))


def potrf(path, D):
    potrf_res = re.compile(r"->\s*\(tensor<%dx%dxf64>,\s*tensor<i64>\)" % (D, D))
    trsm_res = re.compile(r"->\s*tensor<%dx1xf64>" % D)
    d = collections.Counter()
    for depth, line in _walk(path):
        if "enzymexla_compile_cpu" not in line:
            continue
        if potrf_res.search(line):
            kind = "potrf(DxD)"
        elif trsm_res.search(line):
            kind = "trsm(Dx1)"
        else:
            m = re.search(r"->\s*(\(.*\)|tensor<[^>]*>)\s*$", line)
            kind = "other:" + (m.group(1)[:40] if m else "?")
        d[(depth, kind)] += 1
    for k in sorted(d):
        print("   depth=%d %-40s %d" % (k[0], k[1], d[k]))


if __name__ == "__main__":
    mode, D = sys.argv[1], int(sys.argv[2])
    fn = {"dots": dots, "elt": elt, "potrf": potrf}[mode]
    for p in sys.argv[3:]:
        print("==", p.split("/")[-1], "(D=%d, mode=%s)" % (D, mode))
        fn(p, D)
