#!/bin/bash
# Uncapped cells of the real-data rows campaign (evidence
# oopsla/evidence/real_data_rows_2026-09-29/): Diamonds N=5000 under the figure-10
# protocol, then three SICM-on-VI repetitions of the Scotland Leroux CAR row.
# Run alone, after 00:00, on the measurement binary (EXPECT_MD5).
#
# Start gate before every cell (REFREEZE D-5/D-6): procs_running < 8 and
# external (non-$USER) CPU < 700%, both held 30 s, no bazel build of any user,
# no other benchmark process of this user. Acceptance: the drivers' strict
# instantaneous limit, external CPU < 1000% on every 5 s sample.
# Inputs: EXPECT_MD5 (required), EV (evidence dir), STAGES (default
# "diamonds vi vi_blas1"), VI_REPS, PINNED_PAUSE. Outputs: $EV/logs/uncapped_driver.log,
# $EV/diamonds_5000/ and logs_frozen/diamonds_5000_2026-09-29/ (run_frozen_batch.sh),
# $EV/vi_car_leroux/run<r>/ and openblas1_run<r>/ (svi_sicm_real_data_benchmark.jl).
set -uo pipefail
B="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$B/../.." && pwd)"
SO="$ROOT/deps/ReactantExtra/bazel-bin/libReactantExtra.so"
EV="${EV:-/mnt4/sbrantq/probprog/oopsla/evidence/real_data_rows_2026-09-29}"
EXPECT_MD5="${EXPECT_MD5:?set EXPECT_MD5}"
STAGES="${STAGES:-diamonds vi vi_blas1}"
ME="$(id -un)"
LOG="$EV/logs/uncapped_driver.log"
mkdir -p "$EV/logs"
note() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
external_cpu() { ps -eo user,pcpu --no-headers | awk -v me="$ME" '$1 != me { s += $2 } END { printf "%.0f", s+0 }'; }
procs_running() { awk '/^procs_running/ {print $2}' /proc/stat; }
others() { pgrep -u "$ME" -f "harness.jl|pplbench|refreeze_cell|run_frozen_batch|svi_sicm_benchmark|svi_sicm_real_data" | grep -v -w "$$" | wc -l; }
check_md5() {
  local m; m="$(md5sum "$SO" | cut -d' ' -f1)"
  [ "$m" = "$EXPECT_MD5" ] || { note "STOP: md5 $m != $EXPECT_MD5"; exit 3; }
  note "md5 $m"
}
# Pinned SlicStan cells never run concurrently with an uncapped cell: this
# driver holds the pinned driver's pause file from its first gate to its end,
# and waits for any pinned cell in flight (others() counts refreeze_cell).
# Uncapped cells start only between 00:00 and 07:00; outside that window the
# pause is released so the pinned queue can proceed.
PINNED_PAUSE="${PINNED_PAUSE:-}"
in_window() { local h; h=$((10#$(date +%H))); [ "$h" -lt 7 ]; }
pregate() {
  local held=0
  while ! in_window; do
    [ -n "$PINNED_PAUSE" ] && rm -f "$PINNED_PAUSE"
    sleep 60
  done
  [ -n "$PINNED_PAUSE" ] && touch "$PINNED_PAUSE"
  while :; do
    if ! in_window; then pregate; return 0; fi
    if [ "$(procs_running)" -lt 8 ] && [ "$(external_cpu)" -lt 700 ] && [ "$(others)" -eq 0 ] \
       && ! /mnt4/sbrantq/probprog/oopsla/tools/bazel_building.sh; then
      held=$((held + 5)); [ "$held" -ge 30 ] && return 0
    else
      held=0
    fi
    sleep 5
  done
}

for st in $STAGES; do
case "$st" in
diamonds)
  check_md5; pregate; note "START diamonds ext=$(external_cpu)% run=$(procs_running)"
  FROZEN_CELLS="diamonds:sicm/diamonds_regression:0.05:10" FROZEN_START_EXT_CPU=700 \
    bash "$B/ablation/sicm_roster/run_frozen_batch.sh" diamonds_5000_2026-09-29 \
    > "$EV/diamonds_5000/driver.log" 2>&1
  note "END diamonds rc=$? md5=$(md5sum "$SO" | cut -d' ' -f1)"
  ;;
vi)
  for r in ${VI_REPS:-1 2 3}; do
    mkdir -p "$EV/vi_car_leroux/run$r"
    check_md5; pregate; note "START vi run$r ext=$(external_cpu)% run=$(procs_running)"
    LOAD_GATE=12 START_EXT_CPU=700 LOAD_MAX=12 EXT_CPU_LIMIT=1000 STRICT=1 \
    ABORT_SAMPLES=24 MAX_RETRIES=100 \
      "$B/refreeze_cell.sh" "vi_car_leroux_run$r" "$EV/vi_car_leroux" 2h none -- \
      "cd $ROOT && CUDA_VISIBLE_DEVICES= D3_OUT=$EV/vi_car_leroux/run$r julia --project=test test/probprog/svi_sicm_real_data_benchmark.jl" \
      >> "$EV/vi_car_leroux/driver.log" 2>&1
    note "END vi run$r rc=$? md5=$(md5sum "$SO" | cut -d' ' -f1)"
  done
  ;;
vi_blas1)
  # Sensitivity check, not the protocol: the same cell with OpenBLAS held to one
  # thread, to separate the LAPACK threading overhead on 56x56 factorizations
  # from the compiled per-step cost.
  for r in ${VI_REPS:-1 2 3}; do
    mkdir -p "$EV/vi_car_leroux/openblas1_run$r"
    check_md5; pregate; note "START vi openblas1 run$r ext=$(external_cpu)% run=$(procs_running)"
    LOAD_GATE=12 START_EXT_CPU=700 LOAD_MAX=12 EXT_CPU_LIMIT=1000 STRICT=1 \
    ABORT_SAMPLES=24 MAX_RETRIES=100 \
      "$B/refreeze_cell.sh" "vi_car_leroux_openblas1_run$r" "$EV/vi_car_leroux" 2h none -- \
      "cd $ROOT && CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 SVI_REAL_DATA_SKIP_IR=1 D3_OUT=$EV/vi_car_leroux/openblas1_run$r julia --project=test test/probprog/svi_sicm_real_data_benchmark.jl" \
      >> "$EV/vi_car_leroux/driver.log" 2>&1
    note "END vi openblas1 run$r rc=$? md5=$(md5sum "$SO" | cut -d' ' -f1)"
  done
  ;;
esac
done
note "all stages done"
[ -n "$PINNED_PAUSE" ] && rm -f "$PINNED_PAUSE"
