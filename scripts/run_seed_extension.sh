#!/bin/bash
# Seed extension: 20 more seeds (10..29) at the FROZEN winning configs.
#
# Configs come from results/final/selected_configs.csv, chosen once on mean
# validation across seeds 0..9 (scripts/select_configs.py). No hyper-parameter
# search happens here -- that is the point: with the config fixed, the error
# bars reflect model/split variance rather than selection variance.
#
# Phased by memory weight: the container is capped at 64 GB (cgroup) even
# though `free` reports the host's RAM, and each run needs ~2.8 GB.
set -e
cd "$(dirname "$0")/.."
# XLA sizes its thread pools by the CPUs it can see; on a big shared host
# (256 cores) that is ~900 threads per run and exhausts the pod's pids limit.
# Children inherit this shell's affinity; the pod's CPU quota is time-based.
taskset -cp 0-$(( ${PIN_CPUS:-32} - 1 )) $$ >/dev/null 2>&1 || true
export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1
if [[ "${SINGLE_THREAD:-1}" == "1" ]]; then
  export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false ${XLA_FLAGS:-}"
  export OMP_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
fi

DATA_ROOT=${DATA_ROOT:-/workspace/SurvanData}
OUTPUT_DIR=${OUTPUT_DIR:-outputs_final}
EPOCHS=${EPOCHS:-1000}
SEEDS=${SEEDS:-"10 29"}
NWORKERS=${NWORKERS:-12}
NWORKERS_NASA=${NWORKERS_NASA:-6}
NWORKERS_BIGRW=${NWORKERS_BIGRW:-5}
COX_H=${COX_H:-64}; COX_L=${COX_L:-2}; DDH_H=${DDH_H:-128}

gen() {  # $1 = group -> lines of "<run_dir>\t<command>"
python3 - "$1" "$SEEDS" "$COX_H" "$COX_L" "$DDH_H" "$OUTPUT_DIR" <<'EOF'
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

run_phase() {
  # Finished runs are skipped with a FILE TEST, not by launching python and
  # letting run.py print "skip". Re-enumerating ~800 completed runs at ~2 s of
  # interpreter startup each added over an hour of pod time per resume.
  local line dir cmd skipped=0 launched=0
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    dir=${line%%$'\t'*}; cmd=${line#*$'\t'}
    if [ -f "$dir/results_test.json" ]; then skipped=$((skipped+1)); continue; fi
    while [ "$(jobs -rp | wc -l)" -ge "$2" ]; do sleep 1; done
    launched=$((launched+1))
    ( eval "$cmd dataset.num_epochs=$EPOCHS data_root=$DATA_ROOT \
            output_dir=$OUTPUT_DIR" >/dev/null 2>&1 || echo "FAILED: $cmd" ) &
  done <<< "$1"
  wait
  echo "   (skipped $skipped already-finished, launched $launched)"
}

# DRY_RUN=1 reports what WOULD run without launching anything -- use it to
# cost a sweep before paying for a pod.
total_todo=0
for grp in light nasa big_rw; do
  case $grp in
    light) w=$NWORKERS ;; nasa) w=$NWORKERS_NASA ;; *) w=$NWORKERS_BIGRW ;;
  esac
  c=$(gen $grp)
  todo=0; have=0
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    d=${line%%$'\t'*}
    if [ -f "$d/results_test.json" ]; then have=$((have+1)); else todo=$((todo+1)); fi
  done <<< "$c"
  total_todo=$((total_todo+todo))
  echo "phase $grp: $todo to run, $have already done (workers=$w)"
  [ "${DRY_RUN:-0}" = "1" ] || run_phase "$c" "$w"
done
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY RUN: $total_todo runs would execute"; exit 0
fi
echo "done: $(find $OUTPUT_DIR -name results_test.json | wc -l) results"
