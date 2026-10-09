#!/bin/bash
# Tau ablation (stabilizing effect of the target network): runs every pending
# (horizon, tau, run), then aggregates into $ABL_OUT (see scripts/tau_ablation.py).
#   HORIZONS="30 50 100" NWORKERS=8 bash scripts/run_tau_ablation.sh
cd "$(dirname "$0")/.."
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false" OMP_NUM_THREADS=1
uv run python scripts/tau_ablation.py list --horizons ${HORIZONS:-30 50 100} | \
  xargs -P ${NWORKERS:-8} -L 1 sh -c 'uv run python scripts/tau_ablation.py run --horizon $0 --tau $1 --run $2 > /dev/null 2>&1 || echo "FAILED H=$0 tau=$1 run=$2"'
echo "done: $(find ${ABL_OUT:-results/tau_ablation_main} -name 'run_*.npz' | wc -l) runs"
uv run python scripts/tau_ablation.py aggregate
