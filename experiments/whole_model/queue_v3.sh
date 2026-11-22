#!/usr/bin/env bash
# Stage queue, several workers per host (successor of queue_v2.sh).
#
#   bash queue_v3.sh <gpu>          one worker bound to physical GPU <gpu>
#   bash queue_v3.sh --dry-run      show GPU state and the next runnable stage, start nothing
#
# Workers share plan.tsv and state/.  A worker claims a stage by creating the directory
# state/<name>.running (atomic), so two workers never run the same stage.  If a worker dies,
# remove its state/<name>.running by hand.
#
# As in queue_v2.sh:
#   * a stage starts only after the worker's GPU showed no compute process and at most 8 MiB
#     in use for IDLE_CHECKS consecutive checks, IDLE_SLEEP seconds apart;
#   * while a stage runs the GPU is checked every 10 s, and if another compute process
#     appears on it the stage is stopped at once and re-queued;
#   * plan.tsv is re-read before every stage; line format (tab separated):
#         name <TAB> pykey <TAB> deps(comma list or -) <TAB> script and arguments
#     the stage runs "$PY_<pykey> <script and arguments>" here and writes results/<name>.json;
#   * state/<name>.done holds the receipt status; a dependency counts only if it is tested_pass;
#   * a file named STOP in this directory ends every worker.
set -u
RUN="$(cd "$(dirname "$0")" && pwd)"; cd "$RUN"
# shellcheck disable=SC1091
source ./host.env
IDLE_CHECKS=${IDLE_CHECKS:-5}; IDLE_SLEEP=${IDLE_SLEEP:-60}
MAX_FAIL=${MAX_FAIL:-2}; MAX_YIELD=${MAX_YIELD:-8}
mkdir -p results traces logs state cache
ARG=${1:-}
case "$ARG" in --dry-run|"") ;; *) GPUS="$ARG";; esac
TAG=${GPUS// /_}
say(){ echo "[$(date -Is)] [gpu $TAG] $*" >&2; }

gpu_idle(){
  local used apps
  used=$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  apps=$(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^$/d')
  [ -n "$used" ] && [ "$used" -le 8 ] && [ -z "$apps" ]
}

LAST_GPU=""; LAST_END=0
pick_gpu(){
  local -A ok=(); local g now i
  now=$(date +%s)
  if [ -n "$LAST_GPU" ] && [ $((now - LAST_END)) -le 45 ]; then
    for i in 1 2 3 4; do
      if gpu_idle "$LAST_GPU"; then sleep 5; if gpu_idle "$LAST_GPU"; then echo "$LAST_GPU"; return 0; fi; fi
      sleep 5
    done
  fi
  while true; do
    [ -e STOP ] && return 1
    for g in $GPUS; do
      if gpu_idle "$g"; then ok[$g]=$(( ${ok[$g]:-0} + 1 )); else ok[$g]=0; fi
      if [ "${ok[$g]}" -ge "$IDLE_CHECKS" ]; then echo "$g"; return 0; fi
    done
    sleep "$IDLE_SLEEP"
  done
}

foreign_on_gpu(){   # $1 = GPU, $2 = our stage pid (also its process group)
  local p g
  for p in $(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sed '/^$/d'); do
    [ "$p" = "$2" ] && continue
    g=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' ')
    [ "$g" = "$2" ] && continue
    echo "$p"; return 0
  done
  return 1
}

dep_state(){   # ok | wait | bad
  local d s
  [ "$1" = "-" ] && { echo ok; return; }
  for d in ${1//,/ }; do
    if [ -e "state/$d.failed" ] || [ -e "state/$d.blocked" ]; then echo bad; return; fi
    if [ -e "state/$d.done" ]; then
      s=$(cat "state/$d.done")
      [ "$s" = "tested_pass" ] || { echo bad; return; }
    else echo wait; return; fi
  done
  echo ok
}

next_stage(){   # prints "name<TAB>pykey<TAB>cmd" of the first stage that is runnable and unclaimed
  local name pykey deps cmd st
  while IFS=$'\t' read -r name pykey deps cmd; do
    [ -z "${name:-}" ] && continue
    case "$name" in \#*) continue;; esac
    [ -e "state/$name.done" ] || [ -e "state/$name.failed" ] || [ -e "state/$name.blocked" ] || [ -d "state/$name.running" ] && continue
    st=$(dep_state "$deps")
    if [ "$st" = "bad" ]; then echo "dependency not passed: $deps" > "state/$name.blocked"; say "BLOCKED $name ($deps)"; continue; fi
    [ "$st" = "wait" ] && continue
    printf '%s\t%s\t%s\n' "$name" "$pykey" "$cmd"; return 0
  done < plan.tsv
  return 1
}

receipt_status(){
  "$PY_formal" - "$1" <<'PY' 2>/dev/null || echo "no_receipt"
import json, sys
d = json.load(open(sys.argv[1])); r = d.get("rows")
if r is None: print(d.get("status", "unknown"))
else: print("tested_pass" if r and all(x.get("status") == "tested_pass" for x in r) else "tested_invalid")
PY
}

if [ "$ARG" = "--dry-run" ]; then
  for g in $GPUS; do if gpu_idle "$g"; then echo "GPU $g: idle now"; else echo "GPU $g: in use"; fi; done
  echo "first runnable stage:"; next_stage || echo "(none)"
  exit 0
fi

say "queue_v3 worker started on $(hostname -s), pid $$"
idle_note=0
while true; do
  [ -e STOP ] && { say "STOP file found, exiting"; exit 0; }
  if ! line=$(next_stage); then
    [ $idle_note -eq 0 ] && say "no runnable stage in plan.tsv; waiting"
    idle_note=1; rm -f "CURRENT_STAGE.$TAG"; sleep 30; continue
  fi
  idle_note=0
  gpu=$(pick_gpu) || { say "stopped while waiting for the GPU"; exit 0; }
  line=$(next_stage) || continue
  IFS=$'\t' read -r name pykey cmd <<< "$line"
  mkdir "state/$name.running" 2>/dev/null || continue          # another worker claimed it
  pyvar="PY_$pykey"; PY="${!pyvar:-}"
  if [ -z "$PY" ] || [ ! -x "$PY" ]; then echo "no interpreter for key $pykey" > "state/$name.failed"; rmdir "state/$name.running"; say "FAILED $name: no interpreter $pykey"; continue; fi
  export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1
  export TRITON_CACHE_DIR="$RUN/cache/$pykey/gpu$gpu/triton" TORCHINDUCTOR_CACHE_DIR="$RUN/cache/$pykey/gpu$gpu/inductor"
  mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
  say "START $name with $pykey"
  echo "$name gpu=$gpu started=$(date -Is)" > "CURRENT_STAGE.$TAG"
  # shellcheck disable=SC2086
  setsid "$PY" $cmd > "logs/$name.log" 2>&1 &
  pid=$!
  yielded=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 10
    if other=$(foreign_on_gpu "$gpu" "$pid"); then
      say "YIELD $name: compute process $other appeared on GPU $gpu; stopping our stage"
      kill -TERM -- "-$pid" 2>/dev/null; sleep 5; kill -KILL -- "-$pid" 2>/dev/null
      yielded=1; break
    fi
    if [ -e STOP ]; then kill -TERM -- "-$pid" 2>/dev/null; sleep 3; kill -KILL -- "-$pid" 2>/dev/null; rmdir "state/$name.running" 2>/dev/null; say "STOP during $name"; exit 0; fi
  done
  wait "$pid" 2>/dev/null; rc=$?
  LAST_GPU=$gpu; LAST_END=$(date +%s)
  if [ $yielded -eq 1 ]; then
    n=$(( $(cat "state/$name.yields" 2>/dev/null || echo 0) + 1 )); echo "$n" > "state/$name.yields"
    [ -e "results/$name.json" ] && mv "results/$name.json" "results/$name.yielded$n.json"
    [ "$n" -ge "$MAX_YIELD" ] && { echo "yielded $n times" > "state/$name.failed"; say "FAILED $name: yielded $n times"; }
    LAST_GPU=""
  elif [ $rc -eq 0 ]; then
    st=$(receipt_status "results/$name.json"); echo "$st" > "state/$name.done"
    say "DONE $name rc=0 status=$st"
  else
    n=$(( $(cat "state/$name.fails" 2>/dev/null || echo 0) + 1 )); echo "$n" > "state/$name.fails"
    cp "logs/$name.log" "logs/$name.fail$n.log" 2>/dev/null
    say "ERROR $name rc=$rc attempt $n: $(tail -n 1 "logs/$name.log" | cut -c1-200)"
    [ "$n" -ge "$MAX_FAIL" ] && { echo "rc=$rc after $n attempts" > "state/$name.failed"; say "FAILED $name after $n attempts"; }
  fi
  rmdir "state/$name.running" 2>/dev/null
  rm -f "CURRENT_STAGE.$TAG"
done
