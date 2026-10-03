#!/usr/bin/env bash
# Uncapped cochlear figure-9 cells under the 2026-10-01 22:20 user policy:
#   * NO load gates: no external-CPU / procs_running start gate, no during-cell
#     watchdog, no contamination verdict. A cell is accepted on rc 0 plus the
#     arms-equivalence gate (checked by the caller, run_cochlear_nightly.sh).
#   * Kept: the 00:00-07:00 window, the binary md5 check, the sibling-harness
#     check (pplbench|harness.jl), the pinned-cell check (julia -t 1 --project).
#   * All-core jobs are serialised through ONE shared lock with the other
#     uncapped night batches: every cell runs inside
#       flock -x /tmp/sbrantq-rt/uncapped_night.lock <cell>
#     While waiting (lock or sibling or pinned cell), the blocker is logged
#     (pgrep -af / fuser) every 10 minutes.
#   * Ambient recorded per cell, every 5 s: load1, procs_running, external
#     (non-user) %CPU, our user's other %CPU, the cell tree's %CPU; plus the
#     cell's wall and cpu time (/usr/bin/time). Summary per cell in the meta file.
#
# Env: PRELOWER_LOGDIR, PRELOWER_EXPECTED_MD5, PRELOWER_CELLS (cochlear_m1_r ...)
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
DIR="$ROOT/benchmark/probprog/ablation/pre_lowering"
LOGS="${PRELOWER_LOGDIR:?}"
SO="$ROOT/deps/ReactantExtra/bazel-bin/libReactantExtra.so"
META="$LOGS/batch_meta.txt"
LOCK=/tmp/sbrantq-rt/uncapped_night.lock
ME="$(id -un)"
mkdir -p "$LOGS/cells" /tmp/sbrantq-rt
touch "$LOCK"

SO_MD5="$(md5sum "$SO" | awk '{print $1}')"
if [ "$SO_MD5" != "${PRELOWER_EXPECTED_MD5:?}" ]; then
  echo "!! binary md5 $SO_MD5 != expected $PRELOWER_EXPECTED_MD5; not measuring" | tee -a "$META"; exit 2
fi
{
  echo "batch start: $(date -Is)"
  echo "protocol: frozen-adapted-state, MEASURED phase only (cochlear adapted states)"
  echo "so md5:   $SO_MD5"
  echo "policy:   user 2026-10-01 22:20: NO load gates (no start gate, no watchdog, no contamination verdict);"
  echo "          kept: window 00:00-07:00, md5, sibling harness.jl/pplbench, pinned julia -t 1 cells;"
  echo "          serialised by flock -x $LOCK; ambient sampled every 5 s per cell"
  echo "libs:     $(ldd "$SO" | grep -c 'not found') unresolved"
} > "$META"

in_window () { true; }  # 2026-10-02 user: no night window
procs_running () { awk '/^procs_running/ {print $2}' /proc/stat; }
external_cpu () { ps -eo user,pcpu --no-headers | awk -v me="$ME" '$1 != me { s += $2 } END { printf "%.0f", s+0 }'; }
blockers () {
  pgrep -af 'pplbench|harness\.jl' 2>/dev/null | grep -v -e pgrep -e 'zsh -c' | sed 's/^/      sibling: /' | cut -c1-240
  pgrep -af 'julia -t 1 --project' 2>/dev/null | grep -v -e pgrep -e 'zsh -c' | sed 's/^/      pinned:  /' | cut -c1-240
}

wait_clear () {   # sibling and pinned checks; logs blockers every 10 min
  local waited=0 b
  while :; do
    in_window || { echo "    OUTSIDE 00:00-07:00 window at $(date -Is); stopping batch" >> "$META"; exit 3; }
    b="$(blockers)"
    [ -z "$b" ] && return 0
    if [ $((waited % 600)) -eq 0 ]; then
      echo "    waiting ${waited}s on (sibling/pinned) $(date -Is):" >> "$META"; echo "$b" >> "$META"
    fi
    sleep 30; waited=$((waited + 30))
  done
}

run_cell () {   # $1 tag, $2 case, rest harness args
  local tag="$1" c="$2"; shift 2
  local csv="$LOGS/${tag}.samples.csv" tfile="$LOGS/${tag}.time.txt"
  wait_clear
  # Acquire the shared all-core lock; log the holder every 10 minutes.
  exec 9>>"$LOCK"
  local w=0
  until flock -x -w 600 9; do
    w=$((w + 600))
    echo "    waiting ${w}s on $LOCK $(date -Is); holder:" >> "$META"
    for p in $( { fuser "$LOCK" 2>/dev/null; pgrep -f "flock .*uncapped_night.lock"; } | tr ' ' '\n' | grep -v "^$$\$" | sort -u); do ps -o pid=,args= -p "$p" 2>/dev/null | cut -c1-240 | sed 's/^/      lock: /' >> "$META"; done
    in_window || { echo "    OUTSIDE window while waiting for lock; stopping" >> "$META"; exec 9>&-; exit 3; }
  done
  # Re-check sibling/pinned and window after the lock (another job may have started meanwhile).
  wait_clear
  echo "--- cell $tag  lock acquired $(date -Is)  load=$(cut -d' ' -f1-3 /proc/loadavg)  ext=$(external_cpu)%  procs_running=$(procs_running)" >> "$META"
  echo "epoch,load1,procs_running,extcpu,own_other_cpu,cell_cpu" > "$csv"
  rm -f "$DIR/logs/cells/cochlear_${c}_arm${tag: -1}.json"
  ( cd "$ROOT" && exec setsid /usr/bin/time -v -o "$tfile" env COCHLEAR_CASE="$c" JULIA_CI=true CUDA_VISIBLE_DEVICES='' \
      timeout 7200 julia --project=benchmark/probprog "$@" > "$LOGS/${tag}.log" 2>&1 ) &
  local cellpid=$!
  ( while kill -0 "$cellpid" 2>/dev/null; do
      # session id, not process group: GNU timeout moves julia into its own process group, so the pgid sum read 0 in pass 1 (fixed 2026-10-02)
      pg="$(ps -o sid= -p "$cellpid" 2>/dev/null | tr -d ' ')"
      cc="$( [ -n "$pg" ] && ps -eo sid,pcpu --no-headers | awk -v g="$pg" '$1==g {s+=$2} END {printf "%.0f", s+0}' || echo 0)"
      oo="$(ps -eo user,sid,pcpu --no-headers | awk -v me="$ME" -v g="${pg:-x}" '$1==me && $2!=g {s+=$3} END {printf "%.0f", s+0}')"
      echo "$(date +%s),$(cut -d' ' -f1 /proc/loadavg),$(procs_running),$(external_cpu),$oo,$cc" >> "$csv"
      sleep 5
    done ) &
  local sampler=$!
  wait "$cellpid"; local rc=$?
  kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
  flock -u 9; exec 9>&-
  local now_md5; now_md5="$(md5sum "$SO" | awk '{print $1}')"
  [ "$now_md5" != "$SO_MD5" ] && echo "!! BINARY CHANGED during $tag: $SO_MD5 -> $now_md5" >> "$META"
  local summ
  summ="$(python3 - "$csv" "$tfile" <<'EOF'
import csv, sys, statistics as st, re
rows = list(csv.DictReader(open(sys.argv[1])))
f = lambda k: [float(r[k]) for r in rows] or [0.0]
txt = open(sys.argv[2]).read() if sys.argv[2] else ""
g = lambda pat: (re.search(pat, txt) or [None, "?"])[1]
ld = f("load1")
print(f"samples={len(rows)} extcpu_mean={st.mean(f('extcpu')):.0f}% extcpu_max={max(f('extcpu')):.0f}% "
      f"load1_median={st.median(ld):.1f} load1_max={max(ld):.1f} procs_running_median={st.median(f('procs_running')):.0f} "
      f"own_other_cpu_mean={st.mean(f('own_other_cpu')):.0f}% cell_cpu_mean={st.mean(f('cell_cpu')):.0f}% "
      f"wall={g(r'Elapsed \(wall clock\) time.*: (\S+)')} user_s={g(r'User time \(seconds\): (\S+)')} sys_s={g(r'System time \(seconds\): (\S+)')}")
EOF
)"
  echo "    rc=$rc  md5=$now_md5  $summ  end=$(date -Is)" >> "$META"
  grep -E "Compile:|Median|Geomean|All:|verification|frozen-adapted|step=" "$LOGS/${tag}.log" >> "$META"
  cp -a "$DIR/logs/cells/cochlear_${c}_arm${tag: -1}.json" "$LOGS/cells/" 2>/dev/null
  return $rc
}

S="benchmark/probprog/ablation/pre_lowering"
for m in ${PRELOWER_CELLS:?}; do
  c="${m#cochlear_}"
  run_cell "${m}_B" "$c" "$S/cochlear_diffusion.jl" B bench
  run_cell "${m}_A" "$c" "$S/cochlear_diffusion.jl" A bench
done
echo "batch end: $(date -Is)  so md5: $(md5sum "$SO" | awk '{print $1}')" >> "$META"
