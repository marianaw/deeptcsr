#!/bin/bash
# Architecture pilot -- deliberately small.
#
# Two representative datasets span the scale range, so the grid is NOT
# multiplied by the dataset count:
#   nasa    n=200,  horizon=363 -> few sequences, long horizon
#   scania  n=5000, horizon=100 -> many sequences, 105 features, 90% censored
# One winner per family is then applied to EVERY dataset and EVERY arm, so the
# method comparison stays controlled by construction.
#
# Grid (hidden_size=16 / num_layers=1 dropped: the existing 924-run sweep
# already evidences it, and NASA sat at chance there). The h16/l1 reference
# under this same 1000-epoch protocol comes from the timing benchmark.
#   transformer (Cox family): (64,2), (128,2), (128,4)
#   gru_attn    (DDH family): 64, 128
# 2 datasets x 5 configs x 2 seeds = 20 runs.
#
# backbone.name labels the run directory, +backbone.arch selects the factory;
# without that split the configs would overwrite each other's results.
set -e
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1
if [[ "${SINGLE_THREAD:-1}" == "1" ]]; then
  export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false ${XLA_FLAGS:-}"
  export OMP_NUM_THREADS=1
fi

DATA_ROOT=${DATA_ROOT:-/workspace/SurvanData}
OUTPUT_DIR=${OUTPUT_DIR:-outputs_arch}
NWORKERS=${NWORKERS:-16}
EPOCHS=${EPOCHS:-1000}
SEEDS=${SEEDS:-"0 1"}
LAM=0.1; TAU=0.05          # same corner as the timing benchmark

cmds=$(python3 - "$SEEDS" <<'EOF'
import sys
seeds = sys.argv[1].split()
DS = ["nasa", "scania"]
cox = [(64, 2), (128, 2), (128, 4)]
ddh = [64, 128]
for ds in DS:
    for s in seeds:
        for h, l in cox:
            print(f"uv run python scripts/run.py dataset={ds} algorithm=tc_cox "
                  f"backbone.kwargs.hidden_size={h} backbone.kwargs.num_layers={l} "
                  f"backbone.name=transformer_h{h}_l{l} +backbone.arch=transformer seed={s}")
        for h in ddh:
            print(f"uv run python scripts/run.py dataset={ds} algorithm=tc_ddh "
                  f"backbone.kwargs.hidden_size={h} "
                  f"backbone.name=gru_attn_h{h} +backbone.arch=gru_attn seed={s}")
EOF
)

echo "pilot: $(echo "$cmds" | wc -l) runs at $NWORKERS workers, cap ${EPOCHS} epochs"
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$NWORKERS" ]; do sleep 1; done
  ( start=$(date +%s)
    eval "$cmd algorithm.lambda_=$LAM algorithm.target_lr=$TAU \
          dataset.num_epochs=$EPOCHS data_root=$DATA_ROOT \
          output_dir=$OUTPUT_DIR" >/dev/null 2>&1 \
      && echo "OK $(( $(date +%s)-start ))s $cmd" \
      || echo "FAILED $cmd" ) &
done <<< "$cmds"
wait
echo "pilot done: $(find $OUTPUT_DIR -name results_test.json | wc -l) results"
