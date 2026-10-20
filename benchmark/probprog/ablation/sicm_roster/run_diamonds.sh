#!/bin/bash
# Diamonds row (GaussianSufficientStats) driver.
#
# Separate from run_batch.sh on purpose: that script owns the four-cell figure
# batch and is used by a concurrent session, so this one is additive and touches
# none of its state. The timed stage below is deliberately the same recipe
# (uncapped --threads all, md5 and loadavg recorded before and after) so the
# diamonds cell is comparable to the other figure rows.
#
# Stages, run one at a time and checked in between:
#   smoke   20/10 both arms, functional only, no timing claimed
#   ir      6/3 single arm each, dumps the module for the IR evidence gate
#   timed   500/200/2, the figure row
#
# Usage:  ./run_diamonds.sh <smoke|ir|timed> [tag]
#
# The timed stage refuses to start if another pplbench run is active, because a
# concurrent uncapped run would contaminate both. Override only if you are sure
# (DIAMONDS_FORCE=1), which you should not normally be.
set -uo pipefail

STAGE="${1:-}"
TAG="${2:-diamonds1}"
BENCH_DIR="/mnt4/sbrantq/probprog/oopsla/Reactant.jl/benchmark/probprog"
SO="/mnt4/sbrantq/probprog/oopsla/Reactant.jl/deps/ReactantExtra/bazel-bin/libReactantExtra.so"
LOG_DIR="$BENCH_DIR/ablation/sicm_roster/logs_diamonds/$TAG"
HLO_DIR="$BENCH_DIR/ablation/sicm_roster/hlo"
META="$LOG_DIR/meta.txt"

if [ -z "$STAGE" ]; then
    echo "usage: $0 <smoke|ir|timed> [tag]" >&2
    exit 2
fi
mkdir -p "$LOG_DIR" "$HLO_DIR"
cd "$BENCH_DIR" || exit 1

note() { echo "$*" | tee -a "$META"; }

others_running() {
    # Real pplbench/harness processes, excluding this script and its children.
    pgrep -af "pplbench|harness\.jl" 2>/dev/null \
        | grep -v "run_diamonds.sh" \
        | grep -vE "^$$ " \
        | grep -v "claude-shell\|snapshot-zsh\|CODEX_COMPANION" \
        | grep -q . && return 0
    return 1
}

note "=== diamonds stage=$STAGE tag=$TAG"
note "start: $(date -Is)"
note "binary: $SO"
note "md5 before: $(md5sum "$SO" | cut -d' ' -f1)"
note "so mtime: $(stat -c %y "$SO")"
note "loadavg before: $(cat /proc/loadavg)"

case "$STAGE" in
  smoke)
    CUDA_VISIBLE_DEVICES="" timeout 7200 \
        script -q -c "./run_pplbench.sh --threads all probe/sicm_smoke_diamonds" \
        "$LOG_DIR/smoke.log"
    note "smoke exit: $(grep -o 'COMMAND_EXIT_CODE=\"[0-9]*\"' "$LOG_DIR/smoke.log" | tail -1)"
    note "smoke outdir: $(grep -m1 'Output dir:' "$LOG_DIR/smoke.log" | sed 's/.*Output dir: //' | tr -d '\r')"
    ;;

  ir)
    for arm in sicm nosicm; do
      log="$LOG_DIR/ir_${arm}.log"
      CUDA_VISIBLE_DEVICES="" PPLBENCH_DUMP_MLIR_ALWAYS=1 timeout 7200 \
          script -q -c "./run_pplbench.sh --threads all --dump-mlir probe/sicm_ir_diamonds_${arm}" \
          "$log"
      outdir=$(grep -m1 'Output dir:' "$log" | sed 's/.*Output dir: //' | tr -d '\r')
      note "ir $arm exit: $(grep -o 'COMMAND_EXIT_CODE=\"[0-9]*\"' "$log" | tail -1)  outdir=$outdir"
      src=$(ls "$outdir"/mlir_dumps/*pre_xla_compile.mlir 2>/dev/null | head -1)
      src2=$(ls "$outdir"/mlir_dumps/*post_impulse_pm.mlir 2>/dev/null | head -1)
      [ -n "$src2" ] && sed -n '1,2p' "$src2" > "$HLO_DIR/diamonds_${arm}_pipeline.txt"
      if [ -n "$src" ]; then
        cp "$src" "$HLO_DIR/diamonds_${arm}.mlir"
        note "  kept $(du -h "$HLO_DIR/diamonds_${arm}.mlir" | cut -f1) -> hlo/diamonds_${arm}.mlir"
      else
        note "  !! no pre_xla_compile dump found in $outdir"
      fi
      rm -rf "$outdir/mlir_dumps"
    done
    ;;

  timed)
    if [ "${DIAMONDS_FORCE:-0}" != "1" ] && others_running; then
        note "REFUSING: another pplbench/harness run is active; a concurrent"
        note "uncapped run would contaminate both. Wait, or set DIAMONDS_FORCE=1."
        pgrep -af "pplbench|harness\.jl" | grep -v run_diamonds.sh | head -5 | tee -a "$META"
        exit 3
    fi
    CUDA_VISIBLE_DEVICES="" timeout 21600 \
        script -q -c "./run_pplbench.sh --threads all sicm_ablation/diamonds" \
        "$LOG_DIR/timed.log"
    note "timed exit: $(grep -o 'COMMAND_EXIT_CODE=\"[0-9]*\"' "$LOG_DIR/timed.log" | tail -1)"
    note "timed outdir: $(grep -m1 'Output dir:' "$LOG_DIR/timed.log" | sed 's/.*Output dir: //' | tr -d '\r')"
    ;;

  *)
    echo "unknown stage: $STAGE" >&2
    exit 2
    ;;
esac

note "end: $(date -Is)"
note "md5 after: $(md5sum "$SO" | cut -d' ' -f1)"
note "loadavg after: $(cat /proc/loadavg)"
