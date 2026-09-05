#!/usr/bin/env python3
"""Build the STAR (space-time autoregression) real-data instance.

Two real inputs, both public and re-downloadable:

  W  US state contiguity for the 48 contiguous states + DC (N=49), derived
     deterministically from the Census Bureau county adjacency file
     https://www2.census.gov/geo/docs/reference/county_adjacency.txt :
     two states are neighbours iff some county of one is listed adjacent to
     some county of the other. This reproduces the classic `spdep` usa48
     contiguity graph from a primary source rather than shipping an R object.

  Y  CDC ILINet weekly state-level influenza-like-illness rates, retrieved
     through the Delphi Epidata API mirror of CDC FluView
     (https://api.delphi.cmu.edu/epidata/fluview/), field `ili` (percent of
     outpatient visits for ILI). Log-transformed and per-state centred, which
     is the standard preprocessing for STAR/Cliff-Ord estimation.

Output: models/real_data/star_ili_states.json with a _provenance block in the
house style (cf. nyc_traffic_submap.json).

Usage:  python3 build_star_ili.py [--start 201540] [--weeks 104]
"""
import argparse
import json
import math
import os
import urllib.request

COUNTY_ADJ_URL = "https://www2.census.gov/geo/docs/reference/county_adjacency.txt"
EPIDATA_URL = "https://api.delphi.cmu.edu/epidata/fluview/"

# 48 contiguous states + DC. Excludes AK, HI and all territories: the
# contiguity graph is only defined for the contiguous land mass, and ILINet
# reports AK/HI as isolates.
STATES = [
    "al", "az", "ar", "ca", "co", "ct", "de", "dc", "fl", "ga", "id", "il",
    "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo",
    "mt", "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok", "or",
    "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi",
    "wy",
]
FIPS_TO_ABBR = {
    "01": "al", "04": "az", "05": "ar", "06": "ca", "08": "co", "09": "ct",
    "10": "de", "11": "dc", "12": "fl", "13": "ga", "16": "id", "17": "il",
    "18": "in", "19": "ia", "20": "ks", "21": "ky", "22": "la", "23": "me",
    "24": "md", "25": "ma", "26": "mi", "27": "mn", "28": "ms", "29": "mo",
    "30": "mt", "31": "ne", "32": "nv", "33": "nh", "34": "nj", "35": "nm",
    "36": "ny", "37": "nc", "38": "nd", "39": "oh", "40": "ok", "41": "or",
    "42": "pa", "44": "ri", "45": "sc", "46": "sd", "47": "tn", "48": "tx",
    "49": "ut", "50": "vt", "51": "va", "53": "wa", "54": "wv", "55": "wi",
    "56": "wy",
}


def fetch(url, dest):
    if os.path.exists(dest):
        print("  cached: %s" % dest)
        return dest
    print("  downloading %s" % url)
    urllib.request.urlretrieve(url, dest)
    return dest


def epiweek_range(start, weeks):
    """Advance `weeks` epiweeks from `start` (YYYYWW), 52 weeks per year.

    ILINet seasons are indexed by MMWR week. Treating every year as 52 weeks
    skips the occasional week 53; the API simply returns no row for a skipped
    week and the caller drops incomplete weeks, so the series stays contiguous
    in the rows that survive.
    """
    out = []
    year, wk = divmod(start, 100)
    for _ in range(weeks):
        out.append(year * 100 + wk)
        wk += 1
        if wk > 52:
            wk = 1
            year += 1
    return out


def build_adjacency(path):
    """State contiguity from the county adjacency file (latin-1 encoded)."""
    neighbours = {s: set() for s in STATES}
    cur_state = None
    with open(path, encoding="latin-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            own_fips, nbr_fips = parts[1].strip('" '), parts[3].strip('" ')
            if own_fips:
                cur_state = FIPS_TO_ABBR.get(own_fips[:2])
            nbr_state = FIPS_TO_ABBR.get(nbr_fips[:2])
            if cur_state and nbr_state and cur_state != nbr_state:
                neighbours[cur_state].add(nbr_state)
                neighbours[nbr_state].add(cur_state)
    return neighbours


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=201540)
    ap.add_argument("--weeks", type=int, default=104)
    ap.add_argument("--cache", default="/tmp/sbrantq-star")
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)

    print("[1/3] CDC ILINet state series via Delphi Epidata")
    weeks = epiweek_range(args.start, args.weeks)
    series, missing = {}, []
    for s in STATES:
        cache = os.path.join(args.cache, "ili_%s_%d_%d.json" % (s, args.start, args.weeks))
        if not os.path.exists(cache):
            url = "%s?regions=%s&epiweeks=%d-%d" % (EPIDATA_URL, s, weeks[0], weeks[-1])
            urllib.request.urlretrieve(url, cache)
        d = json.load(open(cache))
        if d.get("result") != 1:
            # Florida does not report state-level ILINet to CDC FluView; the API
            # answers "no results". Drop such states rather than imputing.
            missing.append(s)
            continue
        series[s] = {r["epiweek"]: r["ili"] for r in d["epidata"] if r["ili"] is not None}
    # Node-selection rule: fix the observation window first, then keep the
    # states whose series covers it completely. The alternative (fix the state
    # set, truncate the window to the intersection) throws away far more data:
    # for the 201540 window a single late-starting state (Louisiana, first
    # reported week 201640) would halve T from 104 to 52.
    partial = [s for s in series if any(w not in series[s] for w in weeks)]
    nodes = [s for s in STATES if s in series and s not in partial]
    dropped_partial = {s: sum(1 for w in weeks if w in series[s]) for s in partial}
    print("  no ILINet series at all: %s" % (", ".join(missing) or "none"))
    print("  incomplete over the window (dropped): %s"
          % (", ".join("%s(%d/%d wk)" % (s, n, len(weeks))
                       for s, n in sorted(dropped_partial.items())) or "none"))
    complete = list(weeks)
    print("  kept %d states x %d weeks (%d..%d)"
          % (len(nodes), len(complete), complete[0], complete[-1]))
    assert all(w in series[s] for s in nodes for w in complete)

    print("[2/3] state contiguity from Census county adjacency")
    adj_path = fetch(COUNTY_ADJ_URL, os.path.join(args.cache, "county_adjacency.txt"))
    nbrs = build_adjacency(adj_path)
    # Restrict the contiguity graph to the induced subgraph on the reporting set.
    idx = {s: i for i, s in enumerate(nodes)}
    N = len(nodes)
    W = [[0.0] * N for _ in range(N)]
    for s in nodes:
        for t in nbrs[s]:
            if t in idx:
                W[idx[s]][idx[t]] = 1.0
    deg = [sum(r) for r in W]
    isolates = [nodes[i] for i, d_ in enumerate(deg) if d_ == 0]
    assert not isolates, "isolate state(s) after restriction: %s" % isolates
    assert all(W[i][j] == W[j][i] for i in range(N) for j in range(N)), "asymmetric W"
    n_edges = int(sum(deg) / 2)
    print("  N=%d, undirected edges=%d, degree min/mean/max = %d/%.2f/%d"
          % (N, n_edges, min(deg), sum(deg) / N, max(deg)))
    print("  degree-1 nodes: %s"
          % ", ".join(nodes[i] for i, d_ in enumerate(deg) if d_ == 1) or "none")
    # Row-standardised weights: the STAR/Cliff-Ord convention, and it bounds the
    # spectral radius of W at 1 so the stationary region for rho is (-1, 1).
    Wrs = [[w / deg[i] for w in W[i]] for i in range(N)]

    print("[3/3] log-transform + per-state centring")
    # log(ili + 0.1): ILI is a percentage that can legitimately touch 0 in
    # low-incidence state-weeks; the offset is the smallest reported increment.
    raw = [[series[s][w] for s in nodes] for w in complete]       # T x N
    logy = [[math.log(v + 0.1) for v in row] for row in raw]
    means = [sum(logy[t][i] for t in range(len(logy))) / len(logy) for i in range(N)]
    Y = [[logy[t][i] - means[i] for i in range(N)] for t in range(len(logy))]

    out = {
        "_provenance": (
            "STAR (space-time autoregression) instance on real US state data. "
            "W: contiguity of the ILINet-reporting subset of the 48 contiguous "
            "states + DC (N=%d, %d undirected edges; dropped %s), derived "
            "deterministically from the US Census Bureau county "
            "adjacency file %s -- two states are neighbours iff some county of one "
            "is listed adjacent to some county of the other; W is then "
            "row-standardised (STAR/Cliff-Ord convention, bounds the spectral "
            "radius at 1). Y: CDC ILINet weekly state-level influenza-like-illness "
            "rates (percent of outpatient visits), retrieved from the Delphi "
            "Epidata mirror of CDC FluView %s field `ili`, epiweeks %d..%d "
            "(T=%d complete weeks); transformed as log(ili + 0.1) and centred per "
            "state. Both W and Y are real observed data; nothing is simulated. "
            "Built by benchmark/probprog/ablation/pre_lowering/build_star_ili.py."
            % (N, n_edges,
               "; ".join(filter(None, [
                   "no state-level ILINet series: " + ", ".join(missing) if missing else "",
                   "series incomplete over the window: " + ", ".join(sorted(dropped_partial))
                   if dropped_partial else ""])) or "nothing",
               COUNTY_ADJ_URL, EPIDATA_URL, complete[0], complete[-1], len(complete))),
        "N": N, "T": len(complete),
        "states": nodes, "dropped_no_series": missing,
        "dropped_incomplete": dropped_partial, "epiweeks": complete,
        "n_edges": n_edges, "degree": deg,
        "W_binary": W, "W": Wrs, "Y": Y,
        "Y_raw_ili": raw, "log_offset": 0.1, "state_means_log": means,
    }
    here = os.path.dirname(os.path.abspath(__file__))
    dest = os.path.normpath(os.path.join(
        here, "..", "..", "pplbench", "pplbench", "models", "real_data",
        "star_ili_states.json"))
    with open(dest, "w") as f:
        json.dump(out, f)
    print("  wrote %s (N=%d, T=%d)" % (dest, N, len(complete)))


if __name__ == "__main__":
    main()
