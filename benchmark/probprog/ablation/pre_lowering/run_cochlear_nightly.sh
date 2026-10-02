#!/usr/bin/env bash
# Multi-night driver for the two cochlear figure-9 cells (m1_r, m4_r).
#
# Each night, inside the 00:00-07:00 window:
#   1. read the expected binary md5 from $MD5_FILE (the orchestrator's final-binary
#      record) and require the built libReactantExtra.so to match it;
#   2. if that md5 has not been attested yet, dump the pre-XLA IR of the four
#      cochlear programs (both cases x both configurations) CAPPED and compile-only,
#      and compare it with the reference dumps (ir/binary_3af76ebb) under the
#      normalisation of ir_identity_compare_2026-09-23.py. Identical -> reuse the
#      saved adapted states. Different -> re-mint both adapted states (mode warmup,
#      capped) before measuring;
#   3. run run_prelowering_cochlear.sh (the v5 gate plus window, pinned-cell and md5
#      gates) on the cells still missing for this md5;
#   4. accept a case when both arms' cell JSONs exist, the gate did not give up on
#      it, and the arms-equivalence gate holds (identical acceptance and divergence
#      decisions). Stop once both cases are accepted on the current md5.
#
# Policy (user decision 2026-10-01 22:20): NO load gates; the batch script
# run_prelowering_cochlear_locked.sh starts at 00:00:30 whatever the ambient,
# serialised with the other all-core night jobs by flock on
# /tmp/sbrantq-rt/uncapped_night.lock, ambient recorded per cell. Acceptance =
# rc 0 for both arms + arms-equivalence gate. Supersedes the 20:25 loosened gate.
# Usage (from anywhere): bash run_cochlear_nightly.sh   (runs until done)
set -u
R=/mnt4/sbrantq/probprog/oopsla/Reactant.jl
E=/mnt4/sbrantq/probprog/oopsla/evidence/fig9_public_row_2026-09-30
P=$R/benchmark/probprog/ablation/pre_lowering
SO=$R/deps/ReactantExtra/bazel-bin/libReactantExtra.so
MD5_FILE=${MD5_FILE:-/mnt4/sbrantq/probprog/oopsla/evidence/FINAL_BINARY_MD5.txt}
LOG=$E/logs/nightly_driver.log
VERIFIED=$E/ir/attested_md5.txt
CASES="m1_r m4_r"
IDC=/mnt4/sbrantq/probprog/oopsla/evidence/rebase_2026-09-22/drivers/ir_identity_compare_2026-09-23.py
touch "$VERIFIED"
note () { echo "$(date -Is) $*" >> "$LOG"; }

capped_julia () {   # $1 case, rest: harness args; output to stdout
  local c="$1"; shift
  ( cd "$R" && COCHLEAR_CASE="$c" JULIA_CI=true CUDA_VISIBLE_DEVICES="" timeout 5000 \
      taskset -c 88-95,184-191 env OPENBLAS_NUM_THREADS=16 \
      julia -t 16 --project=benchmark/probprog "$P/cochlear_diffusion.jl" "$@" )
}

attest () {   # $1 md5 -> 0 identical, 1 different, 2 failure
  local md5="$1" short="${1:0:8}" d="$E/ir/binary_${1:0:8}" c a rc
  mkdir -p "$d"
  for c in $CASES; do for a in A B; do
    capped_julia "$c" "$a" ir > "$d/ir_${c}_arm$a.log" 2>&1; rc=$?
    [ $rc -ne 0 ] && { note "IR dump failed $c $a rc=$rc"; return 2; }
    cp "$P/logs/hlo/cochlear_${c}_arm$a.mlir" "$d/"
  done; done
  python3 - "$E/ir/binary_3af76ebb" "$d" "$IDC" > "$d/IDENTITY.txt" <<'EOF'
import sys, importlib.util
ref, new, idc = sys.argv[1:4]
spec = importlib.util.spec_from_file_location("idc", idc); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ok = True
for c in ("m1_r", "m4_r"):
    for a in "AB":
        f = f"cochlear_{c}_arm{a}.mlir"
        same = m.norm(f"{ref}/{f}") == m.norm(f"{new}/{f}")
        ok &= same
        print(f, "IDENTICAL" if same else "DIFFERENT", "(vs 3af76ebb; loc/#loc/comments/backend_config normalised)")
print("ALL_IDENTICAL" if ok else "SOME_DIFFERENT")
EOF
  grep -q ALL_IDENTICAL "$d/IDENTITY.txt" && return 0 || return 1
}

remint () {
  local c
  for c in $CASES; do
    capped_julia "$c" B warmup > "$E/logs/warmup_${c}_remint_$(date +%F).log" 2>&1 || { note "re-mint $c failed"; return 1; }
    cp "$P/adapted_states/cochlear_$c.json" "$E/cells/adapted_state_cochlear_${c}_remint_$(date +%F).json"
  done
}

accepted () {   # $1 case, $2 md5 -> 0 if an accepted pair exists for this md5
  local c="$1" short="${2:0:8}" L
  for L in "$E"/logs/uncapped_${short}*; do
    [ -f "$L/cells/cochlear_${c}_armA.json" ] && [ -f "$L/cells/cochlear_${c}_armB.json" ] || continue
    [ "$(grep -A0 -E "^--- cell cochlear_${c}_[AB] " -A1 "$L/batch_meta.txt" 2>/dev/null | grep -c '^    rc=0 ')" -ge 2 ] || continue
    grep -q "so md5:   $2" "$L/batch_meta.txt" 2>/dev/null || continue
    python3 - "$L/cells" "$c" <<'EOF' && return 0
import json, sys
d, c = sys.argv[1:3]
a = json.load(open(f"{d}/cochlear_{c}_armA.json")); b = json.load(open(f"{d}/cochlear_{c}_armB.json"))
sys.exit(0 if a["accepted"] == b["accepted"] and a["diverging"] == b["diverging"] else 1)
EOF
  done
  return 1
}

note "nightly driver start (pid $$)"
while :; do
  hour=$((10#$(date +%H)))
  if [ "$hour" -ge 7 ]; then
    wake=$(date -d 'tomorrow 00:00:30' +%s); note "outside window; sleeping until $(date -d @$wake -Is)"
    sleep $(( wake - $(date +%s) )); continue
  fi
  MD5="$(awk '{print $1}' "$MD5_FILE")"
  ACT="$(md5sum "$SO" | awk '{print $1}')"
  if [ "$ACT" != "$MD5" ]; then note "built binary $ACT != final record $MD5; retry in 10 min"; sleep 600; continue; fi
  if ! grep -q "^$MD5" "$VERIFIED"; then
    note "attesting IR on $MD5 (capped, compile only)"
    attest "$MD5"; rc=$?
    if [ $rc -eq 0 ]; then echo "$MD5 IR-identical to 3af76ebb; adapted states reused $(date -Is)" >> "$VERIFIED"; note "IR identical on $MD5"
    elif [ $rc -eq 1 ]; then note "IR differs on $MD5; re-minting adapted states"; remint || { sleep 600; continue; }
         echo "$MD5 IR differs from 3af76ebb; adapted states re-minted $(date -Is)" >> "$VERIFIED"
    else sleep 600; continue; fi
  fi
  TODO=""
  for c in $CASES; do accepted "$c" "$MD5" || TODO="$TODO cochlear_$c"; done
  if [ -z "$TODO" ]; then note "both cases accepted on $MD5; done"; exit 0; fi
  L="$E/logs/uncapped_${MD5:0:8}_$(date +%F_%H%M)"; mkdir -p "$L"
  note "launching batch for:$TODO -> $L"
  ( cd "$R" && PRELOWER_EXPECTED_MD5="$MD5" PRELOWER_CELLS="${TODO# }" PRELOWER_LOGDIR="$L" \
      bash "$P/run_prelowering_cochlear_locked.sh" > "$L/driver.log" 2>&1 ); rc=$?
  note "batch rc=$rc ($L)"
  [ $rc -eq 2 ] && sleep 600
done
