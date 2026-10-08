#!/bin/bash
# Laptop-side watchdog for RunPod sweeps: results never live only on a pod.
#
# Every INTERVAL seconds, for every pod listed in a .runpod_pod_<name> file:
#   * rsync outputs_v2/ and outputs_small_v2/ back to this machine (only new
#     files transfer), and log how many COMPLETE runs (model.pkl) are local;
#   * when the pod's sweep has ended (no tmux session): drop half-written runs
#     and relaunch its command (.runpod_pod_<name>.job) ONCE -- finished runs
#     are skipped, so this only retries failed/missing ones; when that retry
#     ends too, pull a final time and DELETE the pod so it stops billing;
#   * if the account balance is below MIN_BALANCE: pull everything once more,
#     then STOP every pod and exit. An exhausted balance makes RunPod delete
#     pods together with their disks, so we stop them first.
#
# Run detached so it survives the terminal / Claude session, and keep the
# laptop from sleeping while it runs:
#   setsid nohup systemd-inhibit --what=sleep:idle --why="runpod sync" \
#       bash scripts/runpod_watch.sh >> runpod_watch.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
INTERVAL=${INTERVAL:-1800}
MIN_BALANCE=${MIN_BALANCE:-3}

pods() {  # pod-id files only, not their .job / .retried companions
  for f in .runpod_pod_*; do
    case $f in *.job|*.retried|*.finished) continue ;; esac
    [ -s "$f" ] && echo "$f"
  done
}

balance() {
  runpodctl user 2>/dev/null | python3 -c \
    "import json,sys; print(json.load(sys.stdin)['clientBalance'])" 2>/dev/null
}

SYNC_DIRS=${SYNC_DIRS:-outputs_v2 outputs_small_v2}  # first one must exist on the pod

pull_one() {
  local first=1 dir
  for dir in $SYNC_DIRS; do
    if POD_FILE=$1 timeout 1200 bash scripts/runpod_job.sh pull "$dir" > /dev/null 2>&1; then :
    elif [ $first = 1 ]; then echo "  pull $dir from $1 FAILED"; return 1
    fi  # later dirs may not exist on every pod; that is not an error
    first=0
  done
}

pull_all() { for f in $(pods); do pull_one "$f"; done; }

local_counts() {
  local out=""
  for ds in scania churn_lastfm_months big_rw nasa; do
    out="$out $ds=$(find outputs_v2/*/$ds -name model.pkl 2>/dev/null | wc -l)"
  done
  echo "$out small=$(find outputs_small_v2 -name model.pkl 2>/dev/null | wc -l)"
}

remote() { POD_FILE=$1 timeout 120 bash scripts/runpod_job.sh ssh 2>/dev/null | tail -1; }

finish_if_done() {
  for f in $(pods); do
    alive=$(remote "$f" <<< 'tmux has-session -t job 2>/dev/null && echo ALIVE || echo DONE')
    [ "$alive" = DONE ] || continue       # running, or unreachable: leave it
    if [ ! -e "$f.retried" ] && [ -s "$f.job" ]; then
      # a run killed between results_test.json and model.pkl looks finished
      remote "$f" <<< 'cd /workspace/deeptcsr; for r in $(find outputs_v2 outputs_small_v2 -name results_test.json 2>/dev/null); do d=$(dirname $r); [ -f $d/model.pkl ] || rm -f $r; done; echo ok' > /dev/null
      POD_FILE=$f bash scripts/runpod_job.sh run "$(cat "$f.job")" > /dev/null 2>&1
      touch "$f.retried"
      echo "$(date '+%F %T') $f: sweep ended; relaunched once to retry failed runs"
    elif [ -e "$f.retried" ]; then
      if pull_one "$f"; then
        POD_FILE=$f bash scripts/runpod_job.sh destroy > /dev/null 2>&1
        echo "$(date '+%F %T') $f: retry pass ended; final pull done, pod DELETED"
        mv "$f" "$f.finished"; rm -f "$f.retried"
      fi
    fi
  done
}

while true; do
  [ -n "$(pods)" ] || { echo "$(date '+%F %T') no pods listed; exiting"; exit 0; }
  pull_all
  bal=$(balance)
  echo "$(date '+%F %T') synced | local complete:$(local_counts) | balance \$${bal:-?}"
  if [ -n "$bal" ] && python3 -c "import sys; sys.exit(not float('$bal') < $MIN_BALANCE)"; then
    echo "$(date '+%F %T') balance below \$$MIN_BALANCE: final pull, then stopping pods"
    pull_all
    for f in $(pods); do
      id=$(cat "$f"); [ -n "$id" ] && runpodctl pod stop "$id" > /dev/null 2>&1 \
        && echo "  stopped $id ($f)"
    done
    echo "$(date '+%F %T') local complete:$(local_counts). Watchdog exiting."
    exit 0
  fi
  finish_if_done
  sleep "$INTERVAL"
done
