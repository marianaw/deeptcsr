#!/bin/bash
# Extend the cross-validation grid with tau = 0.95 and 0.99.
#
# These are ordinary grid members: run across ALL lambdas on the 10 TUNING
# seeds (0..9), exactly like the taus already in the suite. Selection then
# re-runs over the extended grid; any config that newly wins gets the 20
# extra seeds afterwards via scripts/run_seed_extension.sh.
#
# Deliberately NOT a "curve" sweep -- filling the tau curve to 30 seeds at a
# pinned lambda is a separate diagnostic and is not mixed in here.
#
#   2 taus x 3 lambdas x 4 datasets x 2 families x 10 seeds = 480 runs
set -e
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1
if [[ "${SINGLE_THREAD:-1}" == "1" ]]; then
  export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false ${XLA_FLAGS:-}"
  export OMP_NUM_THREADS=1; export OPENBLAS_NUM_THREADS=1
fi
DATA_ROOT=${DATA_ROOT:-/workspace/SurvanData}
OUTPUT_DIR=${OUTPUT_DIR:-outputs_final}
EPOCHS=${EPOCHS:-1000}
NWORKERS=${NWORKERS:-12}; NWORKERS_NASA=${NWORKERS_NASA:-6}; NWORKERS_BIGRW=${NWORKERS_BIGRW:-5}
COX_H=${COX_H:-64}; COX_L=${COX_L:-2}; DDH_H=${DDH_H:-128}

gen() {
python3 - "$1" "$COX_H" "$COX_L" "$DDH_H" <<'EOF'
import sys
which, ch, cl, dh = sys.argv[1:5]
G = {"light": ["churn_lastfm_months","scania"], "nasa": ["nasa"], "big_rw": ["big_rw"]}
for algo in ("tc_cox", "tc_ddh"):
    arch = (f"backbone.kwargs.hidden_size={dh}" if "ddh" in algo else
            f"backbone.kwargs.hidden_size={ch} backbone.kwargs.num_layers={cl}")
    for ds in G[which]:
        for tau in (0.95, 0.99):
            for lam in (0.1, 0.5, 0.95):
                for seed in range(10):          # tuning seeds only
                    print(f"uv run python scripts/run.py dataset={ds} "
                          f"algorithm={algo} {arch} algorithm.lambda_={lam} "
                          f"algorithm.target_lr={tau} seed={seed}")
EOF
}
run_phase() {
  local cmd
  while IFS= read -r cmd; do
    [ -z "$cmd" ] && continue
    while [ "$(jobs -rp | wc -l)" -ge "$2" ]; do sleep 1; done
    ( eval "$cmd dataset.num_epochs=$EPOCHS data_root=$DATA_ROOT output_dir=$OUTPUT_DIR" \
        >/dev/null 2>&1 || echo "FAILED: $cmd" ) &
  done <<< "$1"; wait
}
for grp in light nasa big_rw; do
  case $grp in light) w=$NWORKERS ;; nasa) w=$NWORKERS_NASA ;; *) w=$NWORKERS_BIGRW ;; esac
  c=$(gen $grp); echo "phase $grp: $(echo "$c" | wc -l) runs at $w workers"
  run_phase "$c" "$w"
done
echo "done: $(find $OUTPUT_DIR -name results_test.json | wc -l) results"
