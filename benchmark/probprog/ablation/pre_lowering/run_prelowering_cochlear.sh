#!/usr/bin/env bash
# Uncapped frozen-adapted-state batch for the cochlear-aqueduct diffusion row
# (published model and real data, Alghamdi et al. JARO 2026). The gate, the
# external-CPU contamination test, the watchdog and the retry logic are copied
# verbatim from run_prelowering_batch_v5.sh (the batch the plotted figure-9 rows
# came from). Added gates, per the 2026-09-30 brief:
#   * start a cell only between 00:00 and 07:00 local time;
#   * never start while another agent's pinned timed cell runs
#     (`pgrep -af "julia -t 1 --project"`, output printed to the meta file).
#
# Run from the Reactant.jl root:
#   PRELOWER_LOGDIR=<evidence>/logs/uncapped bash benchmark/probprog/ablation/pre_lowering/run_prelowering_cochlear.sh
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

# Library-load workaround (2026-09-30 19:00): a bazel fetch in the ReactantExtra
# workspace at 18:43 re-created external/cuda_cudnn and cuda_nccl without their
# lib/ directories, so the frozen GPU-enabled binary (md5 unchanged) no longer
# resolves libcudnn*.so.9 / libnccl.so.2 through its RUNPATH. These libraries are
# only loaded, never used, by the CPU cells. When (and only when) they are
# unresolvable, point the loader at the same sonames from the Reactant_jll
# artifact in the Julia depot. Recorded in the meta file.
CUDALIB_FALLBACK="${PRELOWER_CUDALIB_FALLBACK:-/tmp/sbrantq-fig9/cudalibs}"
if ldd "$SO" 2>/dev/null | grep -q "not found"; then
  export LD_LIBRARY_PATH="$CUDALIB_FALLBACK${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  LDNOTE="LD_LIBRARY_PATH=$LD_LIBRARY_PATH (cudnn/nccl unresolvable via RUNPATH; fallback -> $(readlink -f "$CUDALIB_FALLBACK/libcudnn.so.9"))"
else
  LDNOTE="all shared libraries resolve via RUNPATH"
fi

SO_MD5="$(md5sum "$SO" | awk '{print $1}')"
EXPECTED_MD5="${PRELOWER_EXPECTED_MD5:-3af76ebbb5b87dfb3820c41647a700ea}"
if [ "$SO_MD5" != "$EXPECTED_MD5" ]; then
  echo "!! binary md5 $SO_MD5 != expected $EXPECTED_MD5; not measuring" | tee -a "$LOGS/batch_meta.txt"
  exit 2
fi
{
  echo "batch start: $(date -Is)"
  echo "protocol: frozen-adapted-state, MEASURED phase only (cochlear adapted state, 200-iteration warmup in arm B)"
  echo "so md5:   $SO_MD5"
  echo "loadavg:  $(cat /proc/loadavg)"
  echo "nproc:    $(nproc)"
  echo "libs:     $LDNOTE"
  echo "gate: start when procs_running < $LOAD_GATE and external CPU < ${START_EXT_CPU}% (held ${QUIET_HOLD:-30}s), no pplbench/harness.jl sibling, no pinned julia -t 1 cell, 00:00-07:00, NO ceiling"
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
    local hour pinned
    hour=$((10#$(date +%H)))
    pinned="$(pgrep -af 'julia -t 1 --project' 2>/dev/null | grep -v pgrep || true)"
    if [ "$hour" -ge 7 ]; then
      echo "    OUTSIDE 00:00-07:00 window at $(date -Is); stopping batch" >> "$META"
      exit 3
    fi
    if [ -n "$pinned" ]; then
      echo "    pinned cell present $(date -Is):" >> "$META"
      echo "$pinned" | sed 's/^/      /' >> "$META"
      held=0; sleep 30; waited=$((waited + 30)); continue
    fi
    if [ "$busy" -eq 0 ] && [ "$run" -lt "$LOAD_GATE" ] && [ "$ext" -lt "$START_EXT_CPU" ] && ! /mnt4/sbrantq/probprog/oopsla/tools/bazel_building.sh; then
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
        "COCHLEAR_CASE=${COCHLEAR_CASE:-m1_r} JULIA_CI=true CUDA_VISIBLE_DEVICES='' julia --project=benchmark/probprog $*" \
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

CELLS="${PRELOWER_CELLS:-cochlear_m1_r}"
for m in $CELLS; do
  case "$m" in
    cochlear_*)
      c="${m#cochlear_}"
      COCHLEAR_CASE="$c" run_cell "${m}_B" "$S/cochlear_diffusion.jl B bench"
      cp -a "$DIR/logs/cells/cochlear_${c}_armB.json" "$LOGS/cells/" 2>/dev/null
      COCHLEAR_CASE="$c" run_cell "${m}_A" "$S/cochlear_diffusion.jl A bench"
      cp -a "$DIR/logs/cells/cochlear_${c}_armA.json" "$LOGS/cells/" 2>/dev/null ;;
  esac
done

{
  echo "batch end: $(date -Is)"
  echo "so md5:   $(md5sum "$SO" | awk '{print $1}')"
} >> "$META"
echo "done"
