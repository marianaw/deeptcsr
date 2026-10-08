#!/bin/bash
# Run the pending tau-ablation runs for the given horizons, then exit.
#   HORIZONS="30 50" NWORKERS=28 bash scripts/run_tau_ablation.sh
cd "$(dirname "$0")/.."
taskset -cp 0-$(( ${PIN_CPUS:-32} - 1 )) $$ >/dev/null 2>&1 || true  # see run_final_sweep.sh
export UV_NO_SYNC=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false" OMP_NUM_THREADS=1
uv run python scripts/tau_ablation.py list --horizons ${HORIZONS:-30 50 100} | \
  xargs -P ${NWORKERS:-8} -L 1 sh -c 'uv run python scripts/tau_ablation.py run --horizon $0 --tau $1 --run $2 > /dev/null 2>&1 || echo "FAILED H=$0 tau=$1 run=$2"'
echo "done: $(find results/tau_ablation_es -name 'run_*.npz' | wc -l) runs"
