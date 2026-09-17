#!/bin/bash
# Large-benchmark sweeps for the TMLR submission, all with current code:
#
#   1. Inc-TCSR (tau=1.0):  inc_tc_cox (transformer), inc_tc_ddh (gru_attn)
#        lambda_ in {0.1, 0.5, 0.8, 0.95}
#   2. Cox-family recompute (legacy results predate the IBS off-by-one fix):
#        cox baseline (no TC), tc_cox D-TCSR grid
#        lambda_ in {0.1, 0.5, 0.8, 0.95} x target_lr in {0.05, 0.1, 0.25}
#
# Datasets: nasa, mimic, churn_lastfm_months, big_rw; seeds 0..10.
# Hyper-parameters are selected on validation per metric at aggregation time.
# 924 runs total; already-finished runs are skipped (results_test.json check).
#
# big_rw target construction peaks around 12 GB RAM per process, so big_rw
# runs execute in a separate phase at NWORKERS_BIGRW (default 2) after the
# other datasets run at NWORKERS (default 6).
#
# Env: DATA_ROOT (default /workspace/SurvanData), NWORKERS, NWORKERS_BIGRW,
#      OUTPUT_DIR (default outputs).
set -e
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH
# Allow many processes to share one GPU (no-op on CPU-only jaxlib).
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
# Per-process CPU thread cap. Needed inside containers that report the
# host's core count (e.g. 64) while the cgroup grants far fewer vCPUs:
# without it every jax process sizes its thread pool to the host count and
# the box thrashes. Set SINGLE_THREAD=0 on a machine whose reported cores
# are real (a laptop), where letting each run use a few threads is faster.
if [[ "${SINGLE_THREAD:-1}" == "1" ]]; then
  export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false ${XLA_FLAGS:-}"
  export OMP_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
fi

# Unbuffered stdout: otherwise Python block-buffers progress into a 8 KB
# buffer when redirected to a file, so a running job looks frozen.
export PYTHONUNBUFFERED=1

DATA_ROOT=${DATA_ROOT:-/workspace/SurvanData}
NWORKERS=${NWORKERS:-6}
NWORKERS_BIGRW=${NWORKERS_BIGRW:-2}
OUTPUT_DIR=${OUTPUT_DIR:-outputs}

gen_cmds() {  # $1 = "main" (all but big_rw) or "big_rw"
python3 - "$1" <<'EOF'
import sys
which = sys.argv[1]
DATASETS = ["big_rw"] if which == "big_rw" else ["nasa", "mimic", "churn_lastfm_months"]
LAMBDAS = [0.1, 0.5, 0.8, 0.95]
TAUS = [0.05, 0.1, 0.25]
SEEDS = range(11)

def emit(ds, seed, algo, **ov):
    ovs = " ".join(f"algorithm.{k}={v}" for k, v in ov.items())
    print(f"uv run python scripts/run.py dataset={ds} algorithm={algo} "
          f"{ovs} seed={seed}".rstrip())

for ds in DATASETS:
    for seed in SEEDS:
        emit(ds, seed, "cox")                                 # baseline
        for lam in LAMBDAS:
            emit(ds, seed, "tc_cox", name="inc_tc_cox",       # Inc-TCSR Cox
                 lambda_=lam, target_lr=1.0)
            emit(ds, seed, "tc_ddh", name="inc_tc_ddh",       # Inc-TCSR DDH
                 lambda_=lam, target_lr=1.0)
            for tau in TAUS:                                  # D-TCSR Cox
                emit(ds, seed, "tc_cox", lambda_=lam, target_lr=tau)
EOF
}

run_phase() {  # $1 = command list (one per line), $2 = workers
  # Hand-rolled pool rather than `xargs -P -I{}`: BSD/macOS xargs refuses
  # these command lines ("cannot be assembled, too long"), and `-d` is
  # GNU-only. `jobs -rp` polling works on bash 3.2 (macOS) and bash 5.
  local cmd
  while IFS= read -r cmd; do
    [ -z "$cmd" ] && continue
    while [ "$(jobs -rp | wc -l)" -ge "$2" ]; do sleep 1; done
    ( eval "$cmd data_root=$DATA_ROOT output_dir=$OUTPUT_DIR" >/dev/null 2>&1 \
        || echo "FAILED: $cmd" ) &
  done <<< "$1"
  wait
}

main_cmds=$(gen_cmds main)
bigrw_cmds=$(gen_cmds big_rw)
echo "phase 1: $(echo "$main_cmds" | wc -l) runs at $NWORKERS workers"
run_phase "$main_cmds" "$NWORKERS"
echo "phase 2 (big_rw): $(echo "$bigrw_cmds" | wc -l) runs at $NWORKERS_BIGRW workers"
run_phase "$bigrw_cmds" "$NWORKERS_BIGRW"
echo "done: $(find $OUTPUT_DIR \( -path '*inc_tc*' -o -path '*tc_cox*' -o -path '*/cox/*' \) -name results_test.json | wc -l) results"
