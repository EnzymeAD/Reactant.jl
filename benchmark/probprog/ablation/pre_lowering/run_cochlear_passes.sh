#!/usr/bin/env bash
# Repeat passes of both cochlear figure-9 cases under the 2026-10-01 22:20 policy
# (run_prelowering_cochlear_locked.sh: no load gates, shared flock, window, md5).
# Usage: run_cochlear_passes.sh <pass name> [<pass name> ...]   e.g. pass3
# Each pass runs in its own night window (00:00-07:00), at most one pass per night.
# The expected md5 is read from FINAL_BINARY_MD5.txt at each launch; a new md5 is
# attested first (capped, compile-only IR of the four programs against 3af76ebb).
# If the attestation reports a difference the driver STOPS (exit 5): no re-mint.
# Output: logs/uncapped_<md5:0:8>_<pass>. Pass 1 (uncapped_4e437776_2026-10-02_0000) is never touched.
set -u
R=/mnt4/sbrantq/probprog/oopsla/Reactant.jl
E=/mnt4/sbrantq/probprog/oopsla/evidence/fig9_public_row_2026-09-30
P=$R/benchmark/probprog/ablation/pre_lowering
MD5_FILE=/mnt4/sbrantq/probprog/oopsla/evidence/FINAL_BINARY_MD5.txt
CASES="m1_r m4_r"
IDC=/mnt4/sbrantq/probprog/oopsla/evidence/rebase_2026-09-22/drivers/ir_identity_compare_2026-09-23.py
LOG=$E/logs/passes_driver.log
note () { echo "$(date -Is) $*" >> "$LOG"; }
eval "$(sed -n '/^capped_julia ()/,/^}/p;/^attest ()/,/^}/p' "$P/run_cochlear_nightly.sh")"
landed () { [ "$(grep -c '^    rc=0 ' "$1/batch_meta.txt" 2>/dev/null)" -ge 4 ]; }
last_night="$(date -d '-7 hours' +%F)"   # pass 2 already ran in the current night
for pass in "$@"; do
  while :; do
    night="$(date -d '-7 hours' +%F)"
    if [ "$((10#$(date +%H)))" -ge 7 ] || [ "$night" = "$last_night" ]; then
      wake=$(date -d 'tomorrow 00:00:30' +%s)
      note "$pass: waiting for the next window, sleeping until $(date -d @$wake -Is)"
      sleep $(( wake - $(date +%s) )); continue
    fi
    MD5="$(awk '{print $1}' "$MD5_FILE")"
    ACT="$(md5sum "$R/deps/ReactantExtra/bazel-bin/libReactantExtra.so" | awk '{print $1}')"
    if [ "$ACT" != "$MD5" ]; then note "$pass: built $ACT != record $MD5; retry in 10 min"; sleep 600; continue; fi
    if ! grep -q "^$MD5" "$E/ir/attested_md5.txt"; then
      note "$pass: attesting IR on $MD5 (capped, compile only)"
      attest "$MD5"; rc=$?
      if [ $rc -eq 0 ]; then echo "$MD5 IR-identical to 3af76ebb; adapted states reused $(date -Is)" >> "$E/ir/attested_md5.txt"; note "$pass: IR identical on $MD5"
      else note "$pass: attestation rc=$rc on $MD5 (1 = IR differs, 2 = dump failed); STOPPING without re-mint"; exit 5; fi
    fi
    L=$E/logs/uncapped_${MD5:0:8}_$pass
    [ -d "$L" ] && ! landed "$L" && mv "$L" "${L}_incomplete_$(date +%m%d%H%M)"
    mkdir -p "$L"; note "$pass: launching -> $L"
    ( cd "$R" && PRELOWER_EXPECTED_MD5="$MD5" PRELOWER_CELLS="cochlear_m1_r cochlear_m4_r" PRELOWER_LOGDIR="$L" \
        bash "$P/run_prelowering_cochlear_locked.sh" > "$L/driver.log" 2>&1 ); rc=$?
    note "$pass: batch rc=$rc"
    if landed "$L"; then last_night="$night"; break; fi
  done
done
note "passes $* landed"
