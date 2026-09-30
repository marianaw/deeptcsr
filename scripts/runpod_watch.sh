#!/bin/bash
# Laptop-side watchdog for RunPod sweeps: results never live only on a pod.
#
# Every INTERVAL seconds, for every pod listed in .runpod_pod_* files:
#   * rsync outputs_v2/ and outputs_small_v2/ back to this machine (only new
#     files transfer), and log how many COMPLETE runs (model.pkl) are local;
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

balance() {
  runpodctl user 2>/dev/null | python3 -c \
    "import json,sys; print(json.load(sys.stdin)['clientBalance'])" 2>/dev/null
}

pull_all() {
  for f in .runpod_pod_*; do
    [ -s "$f" ] || continue
    POD_FILE=$f timeout 1200 bash scripts/runpod_job.sh pull outputs_v2 \
      > /dev/null 2>&1 || echo "  pull outputs_v2 from $f FAILED"
    # only the pod running PBC2/AIDS has this dir; absence is not an error
    POD_FILE=$f timeout 1200 bash scripts/runpod_job.sh pull outputs_small_v2 \
      > /dev/null 2>&1 || true
  done
}

local_counts() {
  local out=""
  for ds in scania churn_lastfm_months big_rw nasa; do
    out="$out $ds=$(find outputs_v2/*/$ds -name model.pkl 2>/dev/null | wc -l)"
  done
  echo "$out small=$(find outputs_small_v2 -name model.pkl 2>/dev/null | wc -l)"
}

while true; do
  ls .runpod_pod_* > /dev/null 2>&1 || { echo "$(date '+%F %T') no pods listed; exiting"; exit 0; }
  pull_all
  bal=$(balance)
  echo "$(date '+%F %T') synced | local complete:$(local_counts) | balance \$${bal:-?}"
  if [ -n "$bal" ] && python3 -c "import sys; sys.exit(not float('$bal') < $MIN_BALANCE)"; then
    echo "$(date '+%F %T') balance below \$$MIN_BALANCE: final pull, then stopping pods"
    pull_all
    for f in .runpod_pod_*; do
      id=$(cat "$f"); [ -n "$id" ] && runpodctl pod stop "$id" > /dev/null 2>&1 \
        && echo "  stopped $id ($f)"
    done
    echo "$(date '+%F %T') local complete:$(local_counts). Watchdog exiting."
    exit 0
  fi
  sleep "$INTERVAL"
done
