#!/bin/bash
# Large benchmark, tuning grid on seeds 0..9 (NASA, Scania, LastFM, LargeRW).
#
#   arms per dataset/seed:
#     cox          LH baseline
#     inc_tc_cox   LH Inc-TCSR, tau = 1, lambda in {0.1, 0.5, 0.95}
#     tc_cox       LH D-TCSR,   tau in $TAUS x same lambdas
#     ddh, inc_tc_ddh, tc_ddh   the same for Dynamic-DeepHit
#   = 74 runs per (dataset, seed), 2,960 in total.
#
# Every arm: up to EPOCHS epochs with early stopping on validation loss.
# Architectures: transformer 64x2 (LH), GRU+attention 128 (DDH), selected by
# scripts/run_arch_pilot.sh.
#
# Env: DATA_ROOT, OUTPUT_DIR, NWORKERS (light datasets), NWORKERS_NASA,
#      NWORKERS_BIGRW, EPOCHS, ONLY=<dataset> to restrict to one dataset.
set -e
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
# one thread per run; parallelism comes from running several runs at once
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false ${XLA_FLAGS:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

DATA_ROOT=${DATA_ROOT:-data}
OUTPUT_DIR=${OUTPUT_DIR:-outputs}
NWORKERS=${NWORKERS:-8}
NWORKERS_NASA=${NWORKERS_NASA:-4}    # NASA (horizon 363) uses the most memory
NWORKERS_BIGRW=${NWORKERS_BIGRW:-4}
EPOCHS=${EPOCHS:-1000}
export TAUS=${TAUS:-0.001,0.003,0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99}
COX_ARCH="backbone.kwargs.hidden_size=64 backbone.kwargs.num_layers=2"
DDH_ARCH="backbone.kwargs.hidden_size=128"

gen_cmds() {  # $1 = light | nasa | big_rw
python3 - "$1" "$COX_ARCH" "$DDH_ARCH" <<'EOF'
import os, sys
which, cox_arch, ddh_arch = sys.argv[1:4]
DATASETS = {"light": ["scania", "churn_lastfm_months"],
            "nasa": ["nasa"], "big_rw": ["big_rw"]}[which]
LAMBDAS = [0.1, 0.5, 0.95]
TAUS = [float(t) for t in os.environ["TAUS"].split(",")]

def emit(ds, seed, algo, arch, **ov):
    ovs = " ".join(f"algorithm.{k}={v}" for k, v in ov.items())
    print(f"uv run python scripts/run.py dataset={ds} algorithm={algo} "
          f"{arch} {ovs} seed={seed}".rstrip())

for ds in DATASETS:
    for seed in range(10):
        emit(ds, seed, "cox", cox_arch)
        emit(ds, seed, "ddh", ddh_arch)
        for lam in LAMBDAS:
            emit(ds, seed, "tc_cox", cox_arch, name="inc_tc_cox", lambda_=lam, target_lr=1.0)
            emit(ds, seed, "tc_ddh", ddh_arch, name="inc_tc_ddh", lambda_=lam, target_lr=1.0)
            for tau in TAUS:
                emit(ds, seed, "tc_cox", cox_arch, lambda_=lam, target_lr=tau)
                emit(ds, seed, "tc_ddh", ddh_arch, lambda_=lam, target_lr=tau)
EOF
}

run_phase() {  # $1 = command list, $2 = workers; finished runs are skipped by run.py
  local cmd
  while IFS= read -r cmd; do
    [ -z "$cmd" ] && continue
    while [ "$(jobs -rp | wc -l)" -ge "$2" ]; do sleep 1; done
    ( eval "$cmd dataset.num_epochs=$EPOCHS data_root=$DATA_ROOT \
            output_dir=$OUTPUT_DIR" >/dev/null 2>&1 \
        || echo "FAILED: $cmd" ) &
  done <<< "$1"
  wait
}

only() { grep -E "dataset=(${ONLY:-[a-z_]+}) " || true; }
for grp in light nasa big_rw; do
  case $grp in light) w=$NWORKERS ;; nasa) w=$NWORKERS_NASA ;; *) w=$NWORKERS_BIGRW ;; esac
  cmds=$(gen_cmds $grp | only)
  echo "$grp: $(echo "$cmds" | grep -c . || true) runs at $w workers"
  run_phase "$cmds" "$w"
done
echo "done: $(find $OUTPUT_DIR -name results_test.json | wc -l) results"
