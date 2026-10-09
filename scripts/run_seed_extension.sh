#!/bin/bash
# Seed extension: seeds 10..29 for the configurations frozen by
# scripts/select_configs.py (results/final/selected_configs.csv, chosen on the
# mean validation score over seeds 0..9). No tuning happens here.
#
# Env: DATA_ROOT, OUTPUT_DIR, NWORKERS, NWORKERS_NASA, NWORKERS_BIGRW, EPOCHS,
#      ONLY=<dataset>, DRY_RUN=1 to only count pending runs.
set -e
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false ${XLA_FLAGS:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

DATA_ROOT=${DATA_ROOT:-data}
OUTPUT_DIR=${OUTPUT_DIR:-outputs}
EPOCHS=${EPOCHS:-1000}
SEEDS=${SEEDS:-"10 29"}
NWORKERS=${NWORKERS:-8}
NWORKERS_NASA=${NWORKERS_NASA:-4}
NWORKERS_BIGRW=${NWORKERS_BIGRW:-4}

gen() {  # $1 = group -> lines of "<run_dir>\t<command>"
python3 - "$1" "$SEEDS" 64 2 128 "$OUTPUT_DIR" <<'EOF'
import csv, re, sys
which, seeds, ch, cl, dh, outdir = sys.argv[1:7]


def landmark_of(ds, _cache={}):
    """Read the landmark flag straight from the YAML -- this generator runs
    under bare python3, which has no pyyaml."""
    if ds not in _cache:
        txt = open(f"configs/dataset/{ds}.yaml").read()
        m = re.search(r"^landmark:\s*(\w+)", txt, re.M)
        _cache[ds] = m.group(1).capitalize() if m else "True"
    return _cache[ds]
lo, hi = (int(x) for x in seeds.split())
GROUPS = {"light": {"scania", "churn_lastfm_months"},
          "nasa": {"nasa"}, "big_rw": {"big_rw"}}
want = GROUPS[which]
for r in csv.DictReader(open("results/final/selected_configs.csv")):
    if r["dataset"] not in want:
        continue
    algo, lam, tau = r["algorithm"], float(r["lambda_"]), float(r["target_lr"])
    ddh = "ddh" in algo
    arch = (f"backbone.kwargs.hidden_size={dh}" if ddh else
            f"backbone.kwargs.hidden_size={ch} backbone.kwargs.num_layers={cl}")
    # inc_tc_* are tc_* with target_lr=1.0 and a renamed output directory
    base = algo.replace("inc_", "") if algo.startswith("inc_tc") else algo
    name = f" algorithm.name={algo}" if algo.startswith("inc_tc") else ""
    ov = "" if algo in ("cox", "ddh") else \
         f" algorithm.lambda_={lam} algorithm.target_lr={tau}"
    bb = "gru_attn" if ddh else "transformer"
    for s in range(lo, hi + 1):
        rd = (f"{outdir}/{algo}/{r['dataset']}/{bb}/lambda_{lam}/"
              f"target_lr_{tau}/landmark_{landmark_of(r['dataset'])}/seed_{s}")
        cmd = (f"uv run python scripts/run.py dataset={r['dataset']} "
               f"algorithm={base}{name} {arch}{ov} seed={s}")
        print(f"{rd}\t{cmd}")
EOF
}

run_phase() {  # finished runs are skipped with a file test
  local line dir cmd
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    dir=${line%%$'\t'*}; cmd=${line#*$'\t'}
    [ -f "$dir/results_test.json" ] && continue
    while [ "$(jobs -rp | wc -l)" -ge "$2" ]; do sleep 1; done
    ( eval "$cmd dataset.num_epochs=$EPOCHS data_root=$DATA_ROOT \
            output_dir=$OUTPUT_DIR" >/dev/null 2>&1 || echo "FAILED: $cmd" ) &
  done <<< "$1"
  wait
}

for grp in light nasa big_rw; do
  case $grp in light) w=$NWORKERS ;; nasa) w=$NWORKERS_NASA ;; *) w=$NWORKERS_BIGRW ;; esac
  c=$(gen $grp | grep -E "dataset=(${ONLY:-[a-z_]+}) " || true)
  todo=0
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    [ -f "${line%%$'\t'*}/results_test.json" ] || todo=$((todo+1))
  done <<< "$c"
  echo "$grp: $todo runs to do (workers=$w)"
  [ "${DRY_RUN:-0}" = "1" ] || run_phase "$c" "$w"
done
[ "${DRY_RUN:-0}" = "1" ] && exit 0
echo "done: $(find $OUTPUT_DIR -name results_test.json | wc -l) results"
