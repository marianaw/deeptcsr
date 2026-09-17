#!/bin/bash
# FINAL large-benchmark sweep for the TMLR submission.
#
# Every arm is recomputed with current code under ONE protocol, so nothing
# mixes legacy and new numbers (the legacy DDH runs scaled epochs with tau,
# `num_epochs = 50 * int(1/target_lr)`, which confounded D-TCSR with a larger
# training budget; see scripts/large_data_table.py).
#
#   arms per dataset/seed:
#     cox          baseline, no temporal consistency
#     inc_tc_cox   Inc-TCSR, tau=1.0,  lambda in {0.1,0.5,0.95}
#     tc_cox       D-TCSR,   tau in {0.05,0.1,0.25} x same lambdas
#     ddh          baseline
#     inc_tc_ddh   Inc-TCSR, tau=1.0
#     tc_ddh       D-TCSR
#   = 26 runs per (dataset, seed)
#
# Datasets: nasa, scania, churn_lastfm_months, big_rw  (MIMIC excluded: only
# 12 events / 0.03 per horizon bin, per-seed C-index spanned 0.0-1.0).
# Seeds 0..9.  26 x 4 x 10 = 1040 runs.
#
# Budget: EPOCHS cap (default 1000) + early stopping on validation loss, the
# SAME for every arm. The old 100-epoch cap truncated NASA (wanted 438) and
# LastFM (185) while big_rw converged at 46, and it penalised small tau more
# than tau=1, biasing against D-TCSR.
#
# Architecture: one backbone per family for ALL datasets and arms, chosen by
# scripts/run_arch_pilot.sh on validation (see results/arch_pilot/).
#
# Output goes to OUTPUT_DIR (default outputs_final) -- deliberately NOT the
# old `outputs/`, whose runs used the 100-epoch protocol and must not be
# pooled with these.
#
# Env: DATA_ROOT, OUTPUT_DIR, NWORKERS, NWORKERS_BIGRW, EPOCHS,
#      COX_H, COX_L (transformer hidden/layers), DDH_H (gru_attn hidden).
set -e
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1
if [[ "${SINGLE_THREAD:-1}" == "1" ]]; then
  export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false ${XLA_FLAGS:-}"
  export OMP_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
fi

DATA_ROOT=${DATA_ROOT:-/workspace/SurvanData}
OUTPUT_DIR=${OUTPUT_DIR:-outputs_final}
NWORKERS=${NWORKERS:-28}
NWORKERS_BIGRW=${NWORKERS_BIGRW:-12}
EPOCHS=${EPOCHS:-1000}
COX_H=${COX_H:-128}; COX_L=${COX_L:-2}; DDH_H=${DDH_H:-64}

COX_ARCH="backbone.kwargs.hidden_size=$COX_H backbone.kwargs.num_layers=$COX_L"
DDH_ARCH="backbone.kwargs.hidden_size=$DDH_H"

gen_cmds() {  # $1 = "main" (all but big_rw) or "big_rw"
python3 - "$1" "$COX_ARCH" "$DDH_ARCH" <<'EOF'
import sys
which, cox_arch, ddh_arch = sys.argv[1], sys.argv[2], sys.argv[3]
DATASETS = ["big_rw"] if which == "big_rw" else ["nasa", "scania",
                                                 "churn_lastfm_months"]
LAMBDAS = [0.1, 0.5, 0.95]   # 0.8 dropped to cut the sweep by a quarter
TAUS = [0.05, 0.1, 0.25]

def emit(ds, seed, algo, arch, **ov):
    ovs = " ".join(f"algorithm.{k}={v}" for k, v in ov.items())
    print(f"uv run python scripts/run.py dataset={ds} algorithm={algo} "
          f"{arch} {ovs} seed={seed}".rstrip())

for ds in DATASETS:
    for seed in range(10):
        emit(ds, seed, "cox", cox_arch)                      # Cox baseline
        emit(ds, seed, "ddh", ddh_arch)                      # DDH baseline
        for lam in LAMBDAS:
            emit(ds, seed, "tc_cox", cox_arch, name="inc_tc_cox",
                 lambda_=lam, target_lr=1.0)
            emit(ds, seed, "tc_ddh", ddh_arch, name="inc_tc_ddh",
                 lambda_=lam, target_lr=1.0)
            for tau in TAUS:
                emit(ds, seed, "tc_cox", cox_arch, lambda_=lam, target_lr=tau)
                emit(ds, seed, "tc_ddh", ddh_arch, lambda_=lam, target_lr=tau)
EOF
}

run_phase() {  # $1 = command list, $2 = workers
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

main_cmds=$(gen_cmds main)
bigrw_cmds=$(gen_cmds big_rw)
echo "arch: transformer h=$COX_H l=$COX_L | gru_attn h=$DDH_H | epochs<=$EPOCHS"
echo "phase 1: $(echo "$main_cmds" | wc -l) runs at $NWORKERS workers"
run_phase "$main_cmds" "$NWORKERS"
echo "phase 2 (big_rw): $(echo "$bigrw_cmds" | wc -l) runs at $NWORKERS_BIGRW workers"
run_phase "$bigrw_cmds" "$NWORKERS_BIGRW"
echo "done: $(find $OUTPUT_DIR -name results_test.json | wc -l) results"
