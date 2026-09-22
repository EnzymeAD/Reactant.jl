#!/usr/bin/env bash
# Frozen-adapted-state MEASURED batch, v5 (2026-08-25).
#
# Supersedes v4. v4's protocol was correct and its adapted states are reused
# verbatim here, but three of its twelve cells were contaminated by an external
# job that the sibling detector could not see: it matched only
# `pplbench|harness.jl`, and it sampled load only BEFORE and AFTER each cell.
# advdiff_B ran while load1 reached 240 and came out 4x slow, which inverted the
# Advection-Diffusion row to a physically impossible 0.61x even though the two
# arms were verified trajectory-identical at three separate seeds.
#
# What is new here:
#   * EXTERNAL-load monitoring DURING every cell. Raw load1 cannot be used as
#     the contamination test because the heavy cells legitimately drive it to
#     ~70 by themselves. Instead we sample, every 5 s, the summed %CPU of every
#     process NOT owned by this user, and record the maximum. A cell whose
#     external CPU ever exceeds EXT_CPU_LIMIT is marked CONTAMINATED and
#     retried, up to MAX_RETRIES times.
#   * The warmup phase is NOT repeated: adapted_states/*.json from v4 are
#     deterministic given (seed, warmup length) and independent of machine load,
#     and are reused so the measured state is identical across batches.
#
# Run from the Reactant.jl root:
#   bash benchmark/probprog/ablation/pre_lowering/run_prelowering_batch_v5.sh
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
DIR="$ROOT/benchmark/probprog/ablation/pre_lowering"
LOGS="${PRELOWER_LOGDIR:-$DIR/logs}"
SO="$ROOT/deps/ReactantExtra/bazel-bin/libReactantExtra.so"
META="$LOGS/batch_meta.txt"
LOAD_GATE="${PRELOWER_LOAD_GATE:-8}"
EXT_CPU_LIMIT="${PRELOWER_EXT_CPU_LIMIT:-1000}"
START_EXT_CPU="${PRELOWER_START_EXT_CPU:-300}"   # stricter gate to START a cell  # percent. Baseline external load on
                                                 # this shared box is 400-550% (4-5 cores);
                                                 # 1000% is well clear of it and far below
                                                 # the excursion that corrupted v4.
MAX_RETRIES="${PRELOWER_MAX_RETRIES:-40}"
ABORT_SAMPLES="${PRELOWER_ABORT_SAMPLES:-24}"   # 24 x 5 s = 2 min of bad conditions aborts the cell
ME="$(id -un)"

mkdir -p "$LOGS" "$LOGS/cells" "$LOGS/hlo"

SO_MD5="$(md5sum "$SO" | awk '{print $1}')"
{
  echo "batch start: $(date -Is)"
  echo "protocol: frozen-adapted-state, MEASURED phase only (adapted states reused from v4)"
  echo "so md5:   $SO_MD5"
  echo "loadavg:  $(cat /proc/loadavg)"
  echo "nproc:    $(nproc)"
  echo "gate: load1 < $LOAD_GATE, no pplbench/harness.jl, NO ceiling"
  echo "contamination test: max external (non-$ME) CPU during cell < ${EXT_CPU_LIMIT}%; retries $MAX_RETRIES"
} > "$META"

kill_tree () {
  local p="$1" k
  for k in $(ps --ppid "$p" -o pid= 2>/dev/null); do kill_tree "$k"; done
  kill -TERM "$p" 2>/dev/null
}

# load1 is a ONE-MINUTE exponential average: after a burst that drove it to 250
# it needs four to five minutes of idle to fall below 8, which is longer than
# this machine's quiet windows, so a raw-load1 gate would never open even on an
# idle machine. procs_running from /proc/stat is the same quantity (how many
# tasks are runnable right now) without the lag, and the external-CPU sum is
# already instantaneous. The gate is therefore: external CPU below the limit
# AND fewer than LOAD_GATE runnable tasks, both holding for QUIET_HOLD seconds.
procs_running () { awk '/^procs_running/ {print $2}' /proc/stat; }

sibling_count () {
  pgrep -af 'pplbench|harness\.jl' 2>/dev/null | grep -v 'zsh -c' | grep -cv pgrep || true
}

# Summed %CPU of every process NOT owned by this user. Immune to our own load.
external_cpu () {
  ps -eo user,pcpu --no-headers 2>/dev/null \
    | awk -v me="$ME" '$1 != me { s += $2 } END { printf "%.0f", s+0 }'
}

QUIET_HOLD="${PRELOWER_QUIET_HOLD:-30}"
wait_for_quiet () {
  local waited=0 busy load1 ext run held=0
  while :; do
    busy="$(sibling_count)"
    load1="$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)"
    ext="$(external_cpu)"
    run="$(procs_running)"
    if [ "$busy" -eq 0 ] && [ "$run" -lt "$LOAD_GATE" ] && [ "$ext" -lt "$START_EXT_CPU" ]; then
      held=$((held + 5))
      if [ "$held" -ge "$QUIET_HOLD" ]; then
        [ "$waited" -gt 0 ] && \
          echo "    (waited ${waited}s; procs_running=$run load1=$load1 ext=${ext}% sibling=$busy)" >> "$META"
        return 0
      fi
      sleep 5; waited=$((waited + 5)); continue
    fi
    held=0
    [ $((waited % 600)) -lt 30 ] && \
      echo "    waiting: ${waited}s, sibling=$busy procs_running=$run load1=$load1 ext=${ext}% $(date -Is)" >> "$META"
    sleep 30
    waited=$((waited + 30))
  done
}

run_cell () {   # $1 = tag, rest = julia script + args
  local tag="$1"; shift
  local attempt=1
  while :; do
    wait_for_quiet
    local extfile="$LOGS/.ext_$tag"
    echo 0 > "$extfile"
    local csv="$LOGS/${tag}.samples.csv"; echo "epoch,load1,extcpu" > "$csv"
    # Background external-CPU + load sampler for the duration of the cell.
    ( while :; do
        e="$(external_cpu)"
        m="$(cat "$extfile" 2>/dev/null || echo 0)"
        [ "$e" -gt "$m" ] && echo "$e" > "$extfile"
        echo "$(date +%s),$(cut -d' ' -f1 /proc/loadavg),$e" >> "$csv"
        sleep 5
      done ) &
    local sampler=$!

    echo "--- cell $tag (attempt $attempt)  load_before=$(cut -d' ' -f1-3 /proc/loadavg)  ext_before=$(external_cpu)%  $(date -Is)" >> "$META"
    ( cd "$ROOT" && timeout 7200 script -q -c \
        "JULIA_CI=true CUDA_VISIBLE_DEVICES='' julia --project=benchmark/probprog $*" \
        "$LOGS/${tag}.log" > /dev/null 2>&1 ) &
    local cellpid=$!
    # Abort as soon as conditions are sustained-bad; retry in the next quiet
    # window rather than measure through an external burst.
    ( bad=0
      while kill -0 "$cellpid" 2>/dev/null; do
        sleep 5
        e="$(external_cpu)"; l1="$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)"
        # External CPU ONLY. Raw load1 is not a contamination test in the
        # uncapped regime: a heavy ablation cell drives load1 to 30-70 by
        # itself through multithreaded LAPACK. Gating on it aborted healthy
        # cells (heat_s3_B was killed four times at external CPU 597-689%,
        # well inside the 1000% limit, because its own load1 reached 38-41).
        if [ "$e" -ge "$EXT_CPU_LIMIT" ]; then
          bad=$((bad + 1))
        else
          bad=0
        fi
        if [ "$bad" -ge "$ABORT_SAMPLES" ]; then
          echo "    WATCHDOG $tag: external CPU=${e}% load1=$l1 for $((bad * 5))s; aborting attempt $attempt" >> "$META"
          kill_tree "$cellpid"
          exit 1
        fi
      done ) &
    local wd=$!
    wait "$cellpid"
    local rc=$?
    kill "$wd" 2>/dev/null; wait "$wd" 2>/dev/null
    kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
    local ext_max; ext_max="$(cat "$extfile" 2>/dev/null || echo 0)"
    rm -f "$extfile"

    local now_md5; now_md5="$(md5sum "$SO" | awk '{print $1}')"
    [ "$now_md5" != "$SO_MD5" ] && \
      echo "!! BINARY CHANGED during $tag: $SO_MD5 -> $now_md5" >> "$META"
    local sib_after; sib_after="$(sibling_count)"

    local load_max; load_max="$(awk -F, 'NR>1 && $2+0>m {m=$2+0} END {printf "%.2f", m+0}' "$csv")"
    echo "    rc=$rc  md5=$now_md5  ext_max=${ext_max}%  load1_max=${load_max}  sibling_after=$sib_after  load_after=$(cut -d' ' -f1-3 /proc/loadavg)  $(date -Is)" >> "$META"
    grep -E "Compile:|Median|Geomean|All:|verification|frozen-adapted|step=" "$LOGS/${tag}.log" >> "$META"

    if [ "$rc" -eq 0 ] && [ "$ext_max" -lt "$EXT_CPU_LIMIT" ] && [ "$sib_after" -eq 0 ]; then
      return 0
    fi
    echo "!! CELL $tag CONTAMINATED (ext_max=${ext_max}% limit=${EXT_CPU_LIMIT}% rc=$rc sibling_after=$sib_after)" >> "$META"
    if [ "$attempt" -ge "$MAX_RETRIES" ]; then
      echo "!! GIVING UP on $tag after $attempt attempts; cell must be treated as INVALID" >> "$META"
      return 1
    fi
    attempt=$((attempt + 1))
    echo "    retrying $tag (attempt $attempt)" >> "$META"
  done
}

S="benchmark/probprog/ablation/pre_lowering"

CELLS="${PRELOWER_CELLS:-heat_s1 heat_s3 advdiff star}"
for m in $CELLS; do
  case "$m" in
    heat_s1)  run_cell heat_s1_B "$S/heat_inverse.jl B 1 bench"
              run_cell heat_s1_A "$S/heat_inverse.jl A 1 bench" ;;
    heat_s3)  run_cell heat_s3_B "$S/heat_inverse.jl B 3 bench"
              run_cell heat_s3_A "$S/heat_inverse.jl A 3 bench" ;;
    advdiff)  run_cell advdiff_B "$S/advection_diffusion.jl B 1 bench"
              run_cell advdiff_A "$S/advection_diffusion.jl A 1 bench" ;;
    star)     run_cell star_B    "$S/star_ili.jl B bench"
              run_cell star_A    "$S/star_ili.jl A bench" ;;
    # Frozen-adapted warmup cells (run once; the states are then reused like the
    # v4 ones). Only needed the first time a model gets an adapted state.
    cubic_s1_w)  run_cell cubic_s1_warmup  "$S/cubic_kernel_dynamics.jl B small warmup" ;;
    cubic_s2_w)  run_cell cubic_s2_warmup  "$S/cubic_kernel_dynamics.jl B large warmup" ;;
    projected_w) run_cell projected_warmup "$S/projected_dynamics.jl B warmup" ;;
    cubic_s1) run_cell cubic_s1_B  "$S/cubic_kernel_dynamics.jl B small bench"
              run_cell cubic_s1_A  "$S/cubic_kernel_dynamics.jl A small bench" ;;
    cubic_s2) run_cell cubic_s2_B  "$S/cubic_kernel_dynamics.jl B large bench"
              run_cell cubic_s2_A  "$S/cubic_kernel_dynamics.jl A large bench" ;;
    projected) run_cell projected_B "$S/projected_dynamics.jl B bench"
               run_cell projected_A "$S/projected_dynamics.jl A bench" ;;
    # Problem-size sweep, 2026-09-10. Heat scale 6 = D=1000 T=100, scale 7 =
    # D=300 T=400; advection-diffusion scale 3 = D=500 T=250, scale 4 = D=1000
    # T=500 (T raised with D to hold the explicit-advection CFL below 1; see
    # advection_diffusion_data.jl).
    heat_s6_w)   run_cell heat_s6_warmup   "$S/heat_inverse.jl B 6 warmup" ;;
    heat_s7_w)   run_cell heat_s7_warmup   "$S/heat_inverse.jl B 7 warmup" ;;
    advdiff_s3_w) run_cell advdiff_s3_warmup "$S/advection_diffusion.jl B 3 warmup" ;;
    advdiff_s4_w) run_cell advdiff_s4_warmup "$S/advection_diffusion.jl B 4 warmup" ;;
    heat_s6)  run_cell heat_s6_B "$S/heat_inverse.jl B 6 bench"
              run_cell heat_s6_A "$S/heat_inverse.jl A 6 bench" ;;
    heat_s7)  run_cell heat_s7_B "$S/heat_inverse.jl B 7 bench"
              run_cell heat_s7_A "$S/heat_inverse.jl A 7 bench" ;;
    advdiff_s3) run_cell advdiff_s3_B "$S/advection_diffusion.jl B 3 bench"
                run_cell advdiff_s3_A "$S/advection_diffusion.jl A 3 bench" ;;
    advdiff_s4) run_cell advdiff_s4_B "$S/advection_diffusion.jl B 4 bench"
                run_cell advdiff_s4_A "$S/advection_diffusion.jl A 4 bench" ;;
  esac
done

{
  echo "batch end: $(date -Is)"
  echo "so md5:   $(md5sum "$SO" | awk '{print $1}')"
} >> "$META"
echo "done"
