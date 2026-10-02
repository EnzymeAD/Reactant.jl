#!/bin/bash
# Run ONE measurement cell under the re-freeze machine-load protocol.
#
#   refreeze_cell.sh <label> <logdir> <timeout> <core|none> -- <command ...>
#
# Per attempt:
#   * wait until the ambient load (load1 minus this campaign's own runnable
#     processes) is below LOAD_GATE and external (non-$USER) CPU is below
#     START_EXT_CPU,
#   * pin to <core> with taskset when a core is given ("none" = the command
#     pins itself, e.g. run_pplbench.sh via PPLBENCH_TASKSET_CPUS),
#   * sample load1, external CPU and our own runnable count every 5 s into
#     <logdir>/cells/<label>.samples.csv,
#   * record the cell's OWN cpu seconds with /usr/bin/time: on a single-core
#     pinned cell, cpu/wall is the direct test of whether the core was stolen,
#   * ABORT the attempt as soon as bad conditions are sustained for
#     ABORT_SAMPLES samples (default 24 = 2 min), and retry in the next quiet
#     window. A cell measured through an external burst is wrong, not merely
#     flagged: on this machine a pinned cell keeps 99% of its core under
#     external load 200 and still runs 2x slower, because the other busy cores
#     take the memory bandwidth and the boost headroom, and the slowdown is not
#     uniform across systems.
#
# An attempt is accepted when it exited 0, its ambient load1 five-minute
# rolling mean stayed at or below LOAD_MAX, and its external-CPU five-minute
# rolling mean stayed below EXT_CPU_LIMIT. The instantaneous maxima and the
# fraction of samples over the limit are recorded either way.
#
# Env: LOAD_GATE (24) START_EXT_CPU (1000) LOAD_MAX (24) EXT_CPU_LIMIT (1000)
#      ABORT_SAMPLES (24) MAX_RETRIES (100)
set -uo pipefail
LABEL="$1"; LOGDIR="$2"; TMO="$3"; CORE="$4"; shift 4
[ "${1:-}" = "--" ] && shift

ME="$(id -un)"
LOAD_GATE="${LOAD_GATE:-24}"
START_EXT_CPU="${START_EXT_CPU:-${EXT_CPU_LIMIT:-1000}}"
LOAD_MAX="${LOAD_MAX:-24}"
EXT_CPU_LIMIT="${EXT_CPU_LIMIT:-1000}"
ABORT_SAMPLES="${ABORT_SAMPLES:-24}"
MAX_RETRIES="${MAX_RETRIES:-100}"
SO="$(cd "$(dirname "$0")" && pwd)/../../deps/ReactantExtra/bazel-bin/libReactantExtra.so"
mkdir -p "$LOGDIR/cells"
GATE="$LOGDIR/cells/${LABEL}.gate.txt"

external_cpu() {
  ps -eo user,pcpu --no-headers 2>/dev/null \
    | awk -v me="$ME" '$1 != me { s += $2 } END { printf "%.0f", s+0 }'
}
load1() { cut -d' ' -f1 /proc/loadavg; }
our_running() { ps -u "$ME" -o stat= --no-headers 2>/dev/null | grep -c '^R' || true; }
ambient_load1() { awk -v a="$(load1)" -v b="$(our_running)" 'BEGIN{printf "%.2f",(a-b<0?0:a-b)}'; }
kill_tree() {
  local p="$1" k
  for k in $(ps --ppid "$p" -o pid= 2>/dev/null); do kill_tree "$k"; done
  kill -TERM "$p" 2>/dev/null
}

ATTEMPT=1
while :; do
  SAMPLES="$LOGDIR/cells/${LABEL}.samples.csv"
  TIMEFILE="$LOGDIR/cells/${LABEL}.time.txt"
  [ "$ATTEMPT" -gt 1 ] && {
    SAMPLES="$LOGDIR/cells/${LABEL}.attempt${ATTEMPT}.samples.csv"
    TIMEFILE="$LOGDIR/cells/${LABEL}.attempt${ATTEMPT}.time.txt"; }

  waited=0
  while :; do
    l="$(ambient_load1 | cut -d. -f1)"; e="$(external_cpu)"
    [ "$l" -lt "$LOAD_GATE" ] && [ "$e" -lt "$START_EXT_CPU" ] && ! /mnt4/sbrantq/probprog/oopsla/tools/bazel_building.sh && break
    [ $((waited % 900)) -eq 0 ] && \
      echo "[$(date '+%F %T')] $LABEL waiting ${waited}s (attempt $ATTEMPT): ambient load1=$l external CPU=${e}%"
    sleep 30; waited=$((waited + 30))
  done

  echo "epoch,load1,extcpu,our_running,ambient_load1" > "$SAMPLES"
  ( while :; do
      l="$(load1)"; o="$(our_running)"
      echo "$(date +%s),$l,$(external_cpu),$o,$(awk -v a="$l" -v b="$o" 'BEGIN{printf "%.2f",(a-b<0?0:a-b)}')" >> "$SAMPLES"
      sleep 5
    done ) &
  SAMPLER=$!

  T0=$(date +%s)
  MD5_BEFORE="$(md5sum "$SO" 2>/dev/null | awk '{print $1}')"
  TASKSET=()
  [ "$CORE" != "none" ] && TASKSET=(taskset -c "$CORE")
  echo "[$(date '+%F %T')] START $LABEL core=$CORE attempt=$ATTEMPT: $*"

  /usr/bin/time -f "cell_elapsed_s=%e cell_user_s=%U cell_sys_s=%S cell_cpu_pct=%P" \
    -o "$TIMEFILE" \
    timeout "$TMO" "${TASKSET[@]}" script -q -e -c "$*" "$LOGDIR/${LABEL}.log" > /dev/null 2>&1 &
  CELLPID=$!

  ABORTFLAG="$LOGDIR/cells/.${LABEL}.aborted"
  rm -f "$ABORTFLAG"
  ( bad=0
    while kill -0 "$CELLPID" 2>/dev/null; do
      sleep 5
      a="$(ambient_load1)"; e="$(external_cpu)"
      if awk -v a="$a" -v m="$LOAD_MAX" -v e="$e" -v l="$EXT_CPU_LIMIT" \
             'BEGIN{exit !(a>m || e>=l)}'; then
        bad=$((bad + 1))
      else
        bad=0
      fi
      if [ "$bad" -ge "$ABORT_SAMPLES" ]; then
        echo "[$(date '+%F %T')]   WATCHDOG $LABEL: ambient load1=$a external CPU=${e}% for $((bad * 5))s; aborting attempt $ATTEMPT"
        touch "$ABORTFLAG"
        kill_tree "$CELLPID"
        exit 1
      fi
    done ) &
  WATCHDOG=$!
  wait "$CELLPID"
  RC=$?
  kill "$WATCHDOG" 2>/dev/null; wait "$WATCHDOG" 2>/dev/null
  T1=$(date +%s)
  kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null
  MD5_AFTER="$(md5sum "$SO" 2>/dev/null | awk '{print $1}')"

  read -r LMAX EMAX LMEAN EMEAN AMAX ESUS ASUS EFRAC N <<< "$(awk -F, '
      NR>1 { if($2+0>l)l=$2+0; if($3+0>e)e=$3+0; if($5+0>a)a=$5+0
             sl+=$2; se+=$3; n++; v[n]=$3+0; w[n]=$5+0 }
      END {
        W = (n < 60 ? n : 60); beste = 0; besta = 0; over = 0
        if (n > 0) {
          re = 0; ra = 0
          for (i = 1; i <= n; i++) {
            re += v[i]; ra += w[i]
            if (i > W) { re -= v[i-W]; ra -= w[i-W] }
            if (i >= W) { if (re/W > beste) beste = re/W; if (ra/W > besta) besta = ra/W }
            if (v[i] >= LIM) over++
          }
        }
        printf "%.2f %.0f %.2f %.0f %.2f %.0f %.2f %.4f %d",
               l+0, e+0, (n?sl/n:0), (n?se/n:0), a+0, beste+0, besta+0,
               (n?over/n:0), n+0
      }' LIM="$EXT_CPU_LIMIT" "$SAMPLES")"

  VERDICT=CLEAN
  [ -f "$ABORTFLAG" ] && VERDICT="ABORTED(sustained contamination)"
  [ "$VERDICT" = "CLEAN" ] && [ "$RC" -ne 0 ] && VERDICT="FAILED(rc=$RC)"
  [ "$VERDICT" = "CLEAN" ] && awk -v l="$ASUS" -v m="$LOAD_MAX" 'BEGIN{exit !(l>m)}' \
    && VERDICT="CONTAMINATED(ambient load1 5-min mean $ASUS)"
  [ "$VERDICT" = "CLEAN" ] && awk -v e="$ESUS" -v m="$EXT_CPU_LIMIT" 'BEGIN{exit !(e>=m)}' \
    && VERDICT="CONTAMINATED(external CPU 5-min mean $ESUS)"
  # STRICT=1 (2026-09-22): instantaneous acceptance, any 5 s sample at or over
  # EXT_CPU_LIMIT fails the attempt.
  [ "$VERDICT" = "CLEAN" ] && [ "${STRICT:-0}" = "1" ] && [ "$EMAX" -ge "$EXT_CPU_LIMIT" ] \
    && VERDICT="CONTAMINATED(strict: external CPU max $EMAX)"
  [ "$MD5_BEFORE" != "$MD5_AFTER" ] && VERDICT="BINARY_CHANGED"

  {
    echo "label=$LABEL"
    echo "cmd=$*"
    echo "core=$CORE attempt=$ATTEMPT"
    echo "rc=$RC t_start=$T0 t_end=$T1 dur_s=$((T1-T0))"
    echo "load1_max=$LMAX load1_mean=$LMEAN ambient_load1_max=$AMAX ambient_load1_sustained_5min=$ASUS"
    echo "extcpu_max=$EMAX extcpu_mean=$EMEAN extcpu_sustained_5min=$ESUS extcpu_frac_over_limit=$EFRAC"
    echo "n_samples=$N"
    echo "so_md5_before=$MD5_BEFORE so_md5_after=$MD5_AFTER"
    cat "$TIMEFILE" 2>/dev/null
    echo "gates: LOAD_GATE=$LOAD_GATE START_EXT_CPU=$START_EXT_CPU LOAD_MAX=$LOAD_MAX EXT_CPU_LIMIT=$EXT_CPU_LIMIT ABORT_SAMPLES=$ABORT_SAMPLES"
    echo "verdict=$VERDICT"
  } > "$GATE"

  echo "[$(date '+%F %T')] END   $LABEL: $VERDICT rc=$RC dur=$((T1-T0))s load1_max=$LMAX ambient_sustained=$ASUS extcpu_max=$EMAX extcpu_sustained=$ESUS"
  rm -f "$ABORTFLAG"
  [ "$VERDICT" = "CLEAN" ] && exit 0
  [ "$VERDICT" = "BINARY_CHANGED" ] && exit 2
  ATTEMPT=$((ATTEMPT + 1))
  if [ "$ATTEMPT" -gt "$MAX_RETRIES" ]; then
    echo "[$(date '+%F %T')] GIVING UP on $LABEL after $MAX_RETRIES attempts"
    exit 1
  fi
  sleep 30
done
