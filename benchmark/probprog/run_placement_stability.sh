#!/bin/bash
# Placement study for the two rows whose SICM ratio is bimodal across
# repetitions (2026-09-24): SICM on VI (scale-family MVN, n=200) and the
# figure 10 Shorebird PGLS row. Each cell runs the SAME harness command the
# campaign used; only the placement changes.
#
#   U    uncapped default (the campaign's regime), control
#   P8   taskset to one 8-core CCD with its SMT siblings (16 hardware threads)
#   P1S  taskset to one physical core plus its SMT sibling (2 hardware threads)
#   T1   uncapped placement, XLA_FLAGS=--xla_cpu_multi_thread_eigen=false and
#        julia -t 1 (this XLA rejects --intra_op_parallelism_threads, so the
#        Eigen flag and the Julia thread count are the only pool controls)
#   P8B, P1SB  P8 and P1S with OPENBLAS_NUM_THREADS set to the number of
#        hardware threads in the set (16, 2). The LAPACK/BLAS custom calls bind
#        to Julia's libblastrampoline (OpenBLAS), whose pool is 96 threads on
#        this machine whatever the affinity mask (BLAS.get_num_threads() = 96
#        under taskset too), so a bare taskset oversubscribes it: the literal P8
#        cell measured 157 ms per SICM-off step against 1 ms uncapped.
#   P8S, P1SS  P8 and P1S with OPENBLAS_NUM_THREADS=1. With a BLAS pool as
#        large as the pinned set (P8B) the SVI SICM-off step still ran 9x
#        slower than uncapped (8.7 ms), the OpenBLAS workers contending with
#        XLA's own pool for the same 16 hardware threads.
#
# Every cell goes through refreeze_cell.sh (unchanged) with the strict gate:
# start when external (non-$USER) CPU < 700% and ambient load1 < 12 and no
# bazel build is active; any 5 s sample at or above 1000% external CPU aborts
# the attempt at once (ABORT_SAMPLES=1) and fails it (STRICT=1), and the cell
# is retried in the next quiet window. Load samples are kept per attempt.
#
#   run_placement_stability.sh            # full interleaved sequence
#   ROWS="pgls" CONFIGS="U T1" REPS="1 2" run_placement_stability.sh
#
# Output: oopsla/evidence/rebase_2026-09-22/stable_statistic/<row>/<config>/rep<N>/
set -uo pipefail
BENCH="/mnt4/sbrantq/probprog/oopsla/Reactant.jl/benchmark/probprog"
ROOT="/mnt4/sbrantq/probprog/oopsla/Reactant.jl"
SO="$ROOT/deps/ReactantExtra/bazel-bin/libReactantExtra.so"
WANT_MD5="b4264c67692ea3a57e067467289b9837"
EV="/mnt4/sbrantq/probprog/oopsla/evidence/rebase_2026-09-22/stable_statistic"
HERE="$BENCH/ablation/sicm_roster"
CCD_CPUS="${CCD_CPUS:-80-87,176-183}"   # CCD 10 of the 7995WX and its SMT siblings
CORE_PAIR="${CORE_PAIR:-84,180}"        # physical core 84 and its SMT sibling
ROWS="${ROWS:-svi pgls}"
CONFIGS="${CONFIGS:-U P8 P1S T1}"
REPS="${REPS:-1 2 3 4}"
mkdir -p "$EV"
DRV="$EV/driver.log"
note() { echo "[$(date '+%F %T')] $*" | tee -a "$DRV"; }

GATE_ENV=(LOAD_GATE=12 START_EXT_CPU=700 LOAD_MAX=1000 EXT_CPU_LIMIT=1000
          STRICT=1 ABORT_SAMPLES=1 MAX_RETRIES=100)

check_md5() {
  local m; m="$(md5sum "$SO" | cut -d' ' -f1)"
  if [ "$m" != "$WANT_MD5" ]; then note "!! binary md5 $m != $WANT_MD5, stopping"; exit 2; fi
  note "   md5 $m ok"
}

wait_no_sibling() {
  # No other benchmark of this user (the campaign drivers and harnesses).
  while pgrep -u "$(id -un)" -f "harness.jl|python -m pplbench|svi_sicm_benchmark.jl|mh_sicm_benchmark.jl" > /dev/null; do
    note "   another benchmark of this user is running; waiting 60 s"
    sleep 60
  done
}

placement() {  # -> CORE, ENVX, JT_SVI, JT_PGLS
  case "$1" in
    U)   CORE=none;          ENVX="";  JT_SVI="";      JT_PGLS="-t auto" ;;
    P8)  CORE="$CCD_CPUS";   ENVX="";  JT_SVI="";      JT_PGLS="-t auto" ;;
    P1S) CORE="$CORE_PAIR";  ENVX="";  JT_SVI="";      JT_PGLS="-t auto" ;;
    P8B) CORE="$CCD_CPUS";   ENVX="OPENBLAS_NUM_THREADS=16";  JT_SVI="";  JT_PGLS="-t auto" ;;
    P1SB) CORE="$CORE_PAIR"; ENVX="OPENBLAS_NUM_THREADS=2";   JT_SVI="";  JT_PGLS="-t auto" ;;
    P8S) CORE="$CCD_CPUS";   ENVX="OPENBLAS_NUM_THREADS=1";   JT_SVI="";  JT_PGLS="-t auto" ;;
    P1SS) CORE="$CORE_PAIR"; ENVX="OPENBLAS_NUM_THREADS=1";   JT_SVI="";  JT_PGLS="-t auto" ;;
    T1)  CORE=none; ENVX="XLA_FLAGS=--xla_cpu_multi_thread_eigen=false"
         JT_SVI="-t 1"; JT_PGLS="-t 1" ;;
    *) note "unknown config $1"; exit 1 ;;
  esac
}

run_cell() {  # label outdir timeout core cmd
  local label="$1" out="$2" tmo="$3" core="$4" cmd="$5"
  wait_no_sibling
  check_md5
  note "   cell $label core=$core: $cmd"
  env "${GATE_ENV[@]}" "$BENCH/refreeze_cell.sh" "$label" "$out" "$tmo" "$core" -- "$cmd" \
    >> "$out/driver.log" 2>&1
  local rc=$?
  note "   cell $label rc=$rc $(grep '^verdict=' "$out/cells/$label.gate.txt" 2>/dev/null)"
  check_md5
  return $rc
}

note "placement study start: rows=[$ROWS] configs=[$CONFIGS] reps=[$REPS] ccd=$CCD_CPUS pair=$CORE_PAIR"
for rep in $REPS; do
  for row in $ROWS; do
    for cfg in $CONFIGS; do
      placement "$cfg"
      out="$EV/$row/$cfg/rep$rep"
      mkdir -p "$out"
      note "== $row $cfg rep$rep"
      if [ "$row" = "svi" ]; then
        [ -f "$out/svi_benchmark.json" ] && { note "   done already"; continue; }
        run_cell svi "$out" 2h "$CORE" \
          "cd $ROOT && CUDA_VISIBLE_DEVICES= $ENVX D3_OUT=$out julia $JT_SVI --project=test test/probprog/svi_sicm_benchmark.jl"
      else
        for arm in sicm nosicm; do
          [ -f "$out/pgls_$arm.out.json" ] && { note "   $arm done already"; continue; }
          flag=""; [ "$arm" = "nosicm" ] && flag="--disable-sicm"
          run_cell "pgls_$arm" "$out" 1h "$CORE" \
            "cd $BENCH && CUDA_VISIBLE_DEVICES= JULIA_CI=true $ENVX julia $JT_PGLS --project=. harness.jl --model sicm/pgls --data $HERE/data/pgls.json --output $out/pgls_$arm.out.json --num-warmup 0 --num-samples 20 --seed 1396133 --step-size 0.05 --max-tree-depth 10 --load-state $HERE/adapted_states/pgls.json --repeat 6 $flag"
        done
      fi
    done
  done
done
note "placement study end"
