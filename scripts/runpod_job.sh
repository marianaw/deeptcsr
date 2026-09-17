#!/bin/zsh
# Minimal CLI for running deeptcsr jobs on RunPod.
#
# Requires: runpodctl (v2, `brew install runpod/runpodctl/runpodctl`) with
# RUNPOD_API_KEY set (or saved via `runpodctl doctor`), rsync, an ssh key
# in ~/.ssh (its .pub is injected into the pod).
#
# Typical session:
#   scripts/runpod_job.sh deploy            # create cheapest CPU pod, wait for ssh
#   scripts/runpod_job.sh push              # rsync code + needed datasets
#   scripts/runpod_job.sh setup             # apt deps + uv sync on the pod
#   scripts/runpod_job.sh run "NWORKERS=6 zsh scripts/run_inc_large.sh"
#   scripts/runpod_job.sh status            # progress: result count + log tail
#   scripts/runpod_job.sh pull              # rsync outputs/ back (merges locally)
#   scripts/runpod_job.sh destroy           # DELETE the pod (stops billing)
#
# The active pod id is kept in .runpod_pod (override with POD_ID env var).
# `deploy` flags:  --gpu "NVIDIA GeForce RTX 4090"  for a GPU pod instead of CPU
#                  --instance <id>  CPU instance type (default cpu5c-16-32)
set -e
cd "$(dirname "$0")/.."

POD_FILE=.runpod_pod
IMAGE_CPU="runpod/base:0.6.2-cpu"
IMAGE_GPU="runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04"
REMOTE_DIR=/workspace/deeptcsr
DATA_DIR=/workspace/SurvanData
LOCAL_DATA=../SurvanData

pod_id() { echo "${POD_ID:-$(cat $POD_FILE 2>/dev/null)}"; }

ssh_info() {
  # -> "user@host port". Prefers direct TCP (fast rsync), falls back to proxy.
  runpodctl ssh info "$(pod_id)" -o json | python3 -c "
import json, sys
d = json.load(sys.stdin)
def walk(o):
    if isinstance(o, dict):
        yield o
        for v in o.values(): yield from walk(v)
    elif isinstance(o, list):
        for v in o: yield from walk(v)
best = None
for o in walk(d):
    ip = o.get('ip') or o.get('publicIp') or o.get('host')
    port = o.get('publicPort') or o.get('port')
    if ip and port and str(o.get('privatePort', 22)) == '22':
        best = (o.get('user', 'root'), ip, port)
print(f'{best[0]}@{best[1]} {best[2]}' if best else '', end='')
"
}

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

resolve_ssh() {
  read -r TARGET PORT <<< "$(ssh_info)"
  if [[ -z "$TARGET" || -z "$PORT" ]]; then
    echo "no ssh info for pod $(pod_id)"; exit 1
  fi
}

remote() {
  resolve_ssh
  ssh ${=SSH_OPTS} -p "$PORT" "$TARGET" "$@"
}

cmd_deploy() {
  local gpu=""
  while [[ $# -gt 0 ]]; do case "$1" in
    --gpu) gpu="$2"; shift 2 ;;
    *) echo "unknown flag $1"; exit 1 ;;
  esac; done

  local pubkey
  pubkey=$(cat ~/.ssh/id_*.pub 2>/dev/null | head -1)
  if [[ -z "$pubkey" ]]; then echo "no ssh public key in ~/.ssh"; exit 1; fi
  local envjson
  envjson=$(python3 -c "import json,sys; print(json.dumps({'PUBLIC_KEY': sys.argv[1]}))" "$pubkey")

  local out
  if [[ -n "$gpu" ]]; then
    out=$(runpodctl pod create --name deeptcsr-job --image "$IMAGE_GPU" \
          --gpu-id "$gpu" --ports "22/tcp" --env "$envjson" \
          --container-disk-in-gb 30 --volume-in-gb 40 -o json)
  else
    out=$(runpodctl pod create --name deeptcsr-job --compute-type cpu \
          --image "$IMAGE_CPU" --ports "22/tcp" \
          --env "$envjson" --container-disk-in-gb 30 --volume-in-gb 40 -o json)
  fi
  echo "$out"
  echo "$out" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d.get('id') or d.get('podId') or '', end='')" > $POD_FILE
  echo "pod id: $(cat $POD_FILE) (saved to $POD_FILE)"

  echo "waiting for ssh..."
  for i in $(seq 1 60); do
    if [[ -n "$(ssh_info)" ]] && remote true 2>/dev/null; then
      echo "ssh ready: $(ssh_info)"; return
    fi
    sleep 10
  done
  echo "timed out waiting for ssh; check: runpodctl pod get $(pod_id)"; exit 1
}

cmd_push() {
  resolve_ssh
  remote "mkdir -p $REMOTE_DIR $DATA_DIR"
  rsync -rltz --progress -e "ssh $SSH_OPTS -p $PORT" \
      --exclude .git --exclude .venv --exclude __pycache__ --exclude 'outputs*' \
      --exclude data --exclude results --exclude '.runpod_pod' \
      ./ "$TARGET:$REMOTE_DIR/"
  (cd $LOCAL_DATA && rsync -rltzR --progress -e "ssh $SSH_OPTS -p $PORT" \
      NASA.h5 bigrw-seqs.pkl mimic_iv/mimic.h5 \
      lastfm-dataset-1K/surv_logs_last.csv lastfm-dataset-1K/events.csv \
      "$TARGET:$DATA_DIR/")
}

cmd_setup() {
  remote "set -e
    command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux rsync git curl)
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH=\$HOME/.local/bin:\$PATH
    cd $REMOTE_DIR && uv sync
    uv run python -c 'import jax; print(\"jax ok:\", jax.__version__, jax.devices())'"
}

cmd_run() {
  local job="$*"
  if [[ -z "$job" ]]; then echo "usage: runpod_job.sh run '<command>'"; exit 1; fi
  remote "cd $REMOTE_DIR && export PATH=\$HOME/.local/bin:\$PATH && \
    tmux kill-session -t job 2>/dev/null; \
    tmux new-session -d -s job \"$job > job.log 2>&1\""
  echo "launched in tmux session 'job'. Use: runpod_job.sh status"
}

cmd_status() {
  remote "cd $REMOTE_DIR 2>/dev/null && \
    echo \"tmux: \$(tmux list-sessions 2>/dev/null || echo 'no session (job finished or not started)')\" && \
    echo \"results: \$(find outputs -name results_test.json 2>/dev/null | wc -l)\" && \
    echo '--- job.log tail ---' && tail -5 job.log 2>/dev/null"
}

cmd_pull() {
  resolve_ssh
  rsync -rltz --progress -e "ssh $SSH_OPTS -p $PORT" \
      "$TARGET:$REMOTE_DIR/outputs/" ./outputs/
  echo "merged into ./outputs/"
}

cmd_ssh() {
  resolve_ssh
  ssh ${=SSH_OPTS} -p "$PORT" "$TARGET"
}

cmd_destroy() {
  runpodctl pod delete "$(pod_id)" && rm -f $POD_FILE
  echo "pod deleted."
}

case "$1" in
  deploy) shift; cmd_deploy "$@" ;;
  push) cmd_push ;;
  setup) cmd_setup ;;
  run) shift; cmd_run "$@" ;;
  status) cmd_status ;;
  pull) cmd_pull ;;
  ssh) cmd_ssh ;;
  destroy) cmd_destroy ;;
  *) grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -20 ;;
esac
