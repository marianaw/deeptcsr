"""Explore raw SCANIA Component X training trajectories.

The script reads only ``vehicle_id`` and ``time_step`` from the large readout
file, streams it in chunks, and writes per-vehicle summaries, a fixed-seed
5,000-vehicle sample, summary CSV/JSON files, and diagnostic PNGs.

Run from the repository root with::

    .venv/bin/python scripts/analyze_scania.py

The default inputs are the version-3 files downloaded from the Swedish
National Data Service into ``data/scania_component_x_v3``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SOURCE_VERSION = 3
SOURCE_DOI = "10.5878/bnh5-ka77"
SOURCE_CATALOG = "https://researchdata.se/en/catalogue/dataset/2024-34/3"
SOURCE_BASE = "https://api.researchdata.se/dataset/2024-34/3/file/data"
QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99, 1.0)
QUANTILE_NAMES = ("min", "p25", "median", "p75", "p90", "p95", "p99", "max")
DEFAULT_SEED = 20250913


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_schema(path: Path, sample_rows: int = 1_000) -> dict:
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    sample = pd.read_csv(path, nrows=sample_rows)
    return {
        "n_columns": len(columns),
        "columns": columns,
        "sample_dtypes": {column: str(dtype) for column, dtype in sample.dtypes.items()},
        "sample_rows": int(len(sample)),
    }


def describe(values: pd.Series | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {name: None for name in (*QUANTILE_NAMES, "mean", "std")}
    quantiles = np.quantile(array, QUANTILES)
    result = {name: float(value) for name, value in zip(QUANTILE_NAMES, quantiles)}
    result["mean"] = float(np.mean(array))
    result["std"] = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    return result


def ks_statistic(left: pd.Series, right: pd.Series) -> float:
    """Two-sample empirical KS distance, using NumPy only."""
    a = np.sort(np.asarray(left, dtype=float))
    b = np.sort(np.asarray(right, dtype=float))
    points = np.sort(np.concatenate((a, b)))
    cdf_a = np.searchsorted(a, points, side="right") / len(a)
    cdf_b = np.searchsorted(b, points, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def sequence_band(n: int) -> str:
    if n == 1:
        return "1"
    if n <= 4:
        return "2-4"
    if n <= 9:
        return "5-9"
    if n <= 19:
        return "10-19"
    if n <= 49:
        return "20-49"
    if n <= 99:
        return "50-99"
    if n <= 199:
        return "100-199"
    return "200+"


SEQUENCE_BANDS = ("1", "2-4", "5-9", "10-19", "20-49", "50-99", "100-199", "200+")


def stream_readouts(path: Path, chunksize: int) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Aggregate readout counts/times and positive consecutive gaps."""
    counts: dict[int, int] = {}
    first: dict[int, float] = {}
    last: dict[int, float] = {}
    previous: dict[int, float] = {}
    closed: set[int] = set()
    gap_values: list[float] = []
    last_vehicle: int | None = None
    total_rows = 0
    missing_vehicle = 0
    missing_time = 0
    nonmonotonic = 0
    duplicate_time = 0
    interleaved_vehicle = 0

    for chunk in pd.read_csv(
        path,
        usecols=["vehicle_id", "time_step"],
        dtype={"vehicle_id": "Int64", "time_step": "float64"},
        chunksize=chunksize,
    ):
        total_rows += len(chunk)
        missing_vehicle += int(chunk["vehicle_id"].isna().sum())
        missing_time += int(chunk["time_step"].isna().sum())
        for vehicle_value, time_value in zip(chunk["vehicle_id"], chunk["time_step"]):
            if pd.isna(vehicle_value):
                continue
            vehicle = int(vehicle_value)
            if pd.isna(time_value):
                continue
            time_step = float(time_value)
            if vehicle in previous:
                if last_vehicle != vehicle:
                    if vehicle in closed:
                        interleaved_vehicle += 1
                delta = time_step - previous[vehicle]
                if delta < 0:
                    nonmonotonic += 1
                elif delta == 0:
                    duplicate_time += 1
                else:
                    gap_values.append(delta)
            if last_vehicle is not None and vehicle != last_vehicle:
                closed.add(last_vehicle)
            last_vehicle = vehicle
            counts[vehicle] = counts.get(vehicle, 0) + 1
            first[vehicle] = min(time_step, first.get(vehicle, time_step))
            last[vehicle] = max(time_step, last.get(vehicle, time_step))
            previous[vehicle] = time_step

    observations = pd.DataFrame({
        "vehicle_id": sorted(counts),
        "n_observations": [counts[vehicle] for vehicle in sorted(counts)],
        "first_observed_time_step": [first[vehicle] for vehicle in sorted(counts)],
        "last_observed_time_step": [last[vehicle] for vehicle in sorted(counts)],
    })
    observations["observed_time_span"] = (
        observations["last_observed_time_step"]
        - observations["first_observed_time_step"]
    )
    quality = {
        "readout_rows": total_rows,
        "readout_subjects": len(observations),
        "missing_vehicle_id": missing_vehicle,
        "missing_time_step": missing_time,
        "time_step_order_decreases": nonmonotonic,
        "consecutive_duplicate_time_steps": duplicate_time,
        "reappearing_interleaved_vehicles": interleaved_vehicle,
        "gap_count": len(gap_values),
    }
    return observations, np.asarray(gap_values, dtype=float), quality


def validate_and_join(observations: pd.DataFrame, tte_path: Path) -> tuple[pd.DataFrame, dict]:
    tte = pd.read_csv(
        tte_path,
        dtype={
            "vehicle_id": "Int64",
            "length_of_study_time_step": "float64",
            "in_study_repair": "Int64",
        },
    )
    required = {"vehicle_id", "length_of_study_time_step", "in_study_repair"}
    missing_columns = sorted(required - set(tte.columns))
    if missing_columns:
        raise ValueError(f"train_tte.csv is missing columns: {missing_columns}")
    if tte["vehicle_id"].isna().any() or tte["length_of_study_time_step"].isna().any():
        raise ValueError("train_tte.csv contains missing vehicle IDs or study times")
    if tte["in_study_repair"].isna().any():
        raise ValueError("train_tte.csv contains missing event indicators")
    if not set(tte["in_study_repair"].astype(int).unique()).issubset({0, 1}):
        raise ValueError("in_study_repair must contain only 0 and 1")

    tte["vehicle_id"] = tte["vehicle_id"].astype(int)
    tte["in_study_repair"] = tte["in_study_repair"].astype(int)
    tte_subjects = set(tte["vehicle_id"])
    readout_subjects = set(observations["vehicle_id"])
    quality = {
        "tte_rows": int(len(tte)),
        "tte_subjects": int(tte["vehicle_id"].nunique()),
        "tte_duplicate_vehicle_ids": int(tte["vehicle_id"].duplicated().sum()),
        "readout_ids_missing_from_tte": len(readout_subjects - tte_subjects),
        "tte_ids_missing_from_readouts": len(tte_subjects - readout_subjects),
    }
    if quality["tte_duplicate_vehicle_ids"]:
        raise ValueError("train_tte.csv is not one row per vehicle")

    metrics = observations.merge(tte, on="vehicle_id", how="outer", indicator=True, validate="one_to_one")
    if (metrics["_merge"] != "both").any():
        raise ValueError("Vehicle IDs do not match between the two training files")
    metrics = metrics.drop(columns="_merge").sort_values("vehicle_id").reset_index(drop=True)
    metrics["sequence_length_band"] = metrics["n_observations"].map(sequence_band)
    metrics["study_time_minus_last_observed"] = (
        metrics["length_of_study_time_step"] - metrics["last_observed_time_step"]
    )
    metrics["observed_span_fraction_of_study_time"] = (
        metrics["observed_time_span"] / metrics["length_of_study_time_step"]
    )
    quality["joined_subjects"] = int(len(metrics))
    quality["event_indicator_values"] = sorted(metrics["in_study_repair"].unique().tolist())
    quality["study_end_before_or_at_last_observed_count"] = int(
        (metrics["study_time_minus_last_observed"] <= 0).sum()
    )
    return metrics, quality


def population_summary(frame: pd.DataFrame) -> dict:
    n_subjects = len(frame)
    events = int(frame["in_study_repair"].sum())
    censored = n_subjects - events
    return {
        "n_subjects": int(n_subjects),
        "n_readouts": int(frame["n_observations"].sum()),
        "events": events,
        "censored": censored,
        "event_rate": events / n_subjects,
        "censoring_rate": censored / n_subjects,
        "sequence_length": describe(frame["n_observations"]),
        "observed_time_span": describe(frame["observed_time_span"]),
        "length_of_study_time_step": describe(frame["length_of_study_time_step"]),
        "first_observed_time_step": describe(frame["first_observed_time_step"]),
        "last_observed_time_step": describe(frame["last_observed_time_step"]),
        "study_time_minus_last_observed": describe(frame["study_time_minus_last_observed"]),
    }


def length_stratification(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for band in SEQUENCE_BANDS:
        group = frame[frame["sequence_length_band"] == band]
        n = len(group)
        events = int(group["in_study_repair"].sum())
        rows.append({
            "sequence_length_band": band,
            "n_subjects": n,
            "events": events,
            "censored": n - events,
            "event_rate": events / n if n else None,
            "censoring_rate": (n - events) / n if n else None,
        })
    return rows


def gap_summary(gaps: np.ndarray) -> dict:
    result = describe(gaps)
    result["n_gaps"] = int(len(gaps))
    result["positive_gap_fraction"] = float(np.mean(gaps > 0)) if len(gaps) else None
    return result


def summary_rows(full: pd.DataFrame, sample: pd.DataFrame) -> list[dict]:
    rows = []
    metrics = {
        "n_observations": "sequence_length",
        "observed_time_span": "observed_time_span",
        "length_of_study_time_step": "length_of_study_time_step",
        "first_observed_time_step": "first_observed_time_step",
        "last_observed_time_step": "last_observed_time_step",
        "study_time_minus_last_observed": "study_time_minus_last_observed",
    }
    for population, frame in (("full_training", full), ("subsample_5000", sample)):
        pop = population_summary(frame)
        for name in ("n_subjects", "n_readouts", "events", "censored", "event_rate", "censoring_rate"):
            rows.append({"population": population, "metric": name, "statistic": "value", "value": pop[name]})
        for column, metric in metrics.items():
            for statistic, value in describe(frame[column]).items():
                rows.append({"population": population, "metric": metric, "statistic": statistic, "value": value})
    return rows


def comparison(full: pd.DataFrame, sample: pd.DataFrame) -> dict:
    result = {
        "censoring_rate": {
            "full": population_summary(full)["censoring_rate"],
            "sample": population_summary(sample)["censoring_rate"],
        },
        "metrics": {},
    }
    result["censoring_rate"]["absolute_difference"] = abs(
        result["censoring_rate"]["sample"] - result["censoring_rate"]["full"]
    )
    for column in ("n_observations", "length_of_study_time_step", "observed_time_span", "study_time_minus_last_observed"):
        full_stats = describe(full[column])
        sample_stats = describe(sample[column])
        result["metrics"][column] = {
            "ks_statistic": ks_statistic(full[column], sample[column]),
            "full": full_stats,
            "sample": sample_stats,
            "quantile_absolute_differences": {
                name: abs(sample_stats[name] - full_stats[name])
                for name in QUANTILE_NAMES
            },
        }
    return result


def annotate_bars(ax, bars, counts, n):
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{count:,}\n{count / n:.1%}", ha="center", va="bottom", fontsize=9)


def make_figures(frame: pd.DataFrame, gaps: np.ndarray, out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    counts = [int((frame["in_study_repair"] == 0).sum()), int((frame["in_study_repair"] == 1).sum())]
    n = len(frame)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(frame["n_observations"], bins=50, color="#4472C4", edgecolor="white")
    ax.set(xlabel="Number of longitudinal observations", ylabel="Vehicles", title="Sequence length per vehicle")
    fig.tight_layout()
    path = out / "hist_observations_per_vehicle.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path.name)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(frame["length_of_study_time_step"], bins=50, color="#70AD47", edgecolor="white")
    ax.set(xlabel="length_of_study_time_step", ylabel="Vehicles", title="Elapsed study time per vehicle")
    fig.tight_layout()
    path = out / "hist_length_of_study_time_step.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path.name)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    labels = ["Censored (0)", "Event (1)"]
    bars = axes[0].bar(labels, counts, color=["#A5A5A5", "#ED7D31"])
    annotate_bars(axes[0], bars, counts, n)
    axes[0].set(ylabel="Vehicles", title="Counts")
    bars = axes[1].bar(labels, np.asarray(counts) / n, color=["#A5A5A5", "#ED7D31"])
    axes[1].set_ylim(0, max(np.asarray(counts) / n) * 1.25)
    axes[1].set(ylabel="Proportion", title="Proportions")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    for bar, count in zip(bars, counts):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{count / n:.1%}",
                     ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path = out / "event_vs_censored.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path.name)

    gap_limit = float(np.quantile(gaps, 0.995)) if len(gaps) else 0.0
    shown_gaps = gaps[gaps <= gap_limit] if len(gaps) and gap_limit > 0 else gaps
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if len(shown_gaps):
        ax.hist(shown_gaps, bins=50, color="#8064A2", edgecolor="white")
    ax.set(xlabel="Gap between consecutive readouts (time_step)", ylabel="Gaps",
           title="Inter-observation gaps (up to 99.5th percentile)")
    fig.tight_layout()
    path = out / "hist_inter_observation_gaps.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path.name)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].hist(frame["n_observations"], bins=50, color="#4472C4", edgecolor="white")
    axes[0, 0].set(xlabel="Number of observations", ylabel="Vehicles", title="Sequence length")
    axes[0, 1].hist(frame["length_of_study_time_step"], bins=50, color="#70AD47", edgecolor="white")
    axes[0, 1].set(xlabel="Study time_step", ylabel="Vehicles", title="Elapsed study time")
    bars = axes[1, 0].bar(labels, counts, color=["#A5A5A5", "#ED7D31"])
    annotate_bars(axes[1, 0], bars, counts, n)
    axes[1, 0].set(ylabel="Vehicles", title="Event / censoring")
    if len(shown_gaps):
        axes[1, 1].hist(shown_gaps, bins=50, color="#8064A2", edgecolor="white")
    axes[1, 1].set(xlabel="Gap (time_step)", ylabel="Gaps", title="Readout gaps (≤99.5th percentile)")
    fig.tight_layout()
    path = out / "scania_diagnostics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path.name)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readouts", type=Path, default=Path("data/scania_component_x_v3/train_operational_readouts.csv"))
    parser.add_argument("--tte", type=Path, default=Path("data/scania_component_x_v3/train_tte.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/scania_component_x"))
    parser.add_argument("--sample-size", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--chunksize", type=int, default=250_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.readouts.exists() or not args.tte.exists():
        raise FileNotFoundError("Download the version-3 training files before running this script")
    args.out.mkdir(parents=True, exist_ok=True)

    readout_schema = file_schema(args.readouts)
    tte_schema = file_schema(args.tte)
    if not {"vehicle_id", "time_step"}.issubset(readout_schema["columns"]):
        raise ValueError("train_operational_readouts.csv needs vehicle_id and time_step")

    observations, gaps, readout_quality = stream_readouts(args.readouts, args.chunksize)
    metrics, tte_quality = validate_and_join(observations, args.tte)
    if args.sample_size > len(metrics):
        raise ValueError(f"sample-size {args.sample_size} exceeds {len(metrics)} vehicles")
    rng = np.random.default_rng(args.seed)
    sampled_ids = np.sort(rng.choice(metrics["vehicle_id"].to_numpy(), size=args.sample_size, replace=False))
    sample = metrics[metrics["vehicle_id"].isin(sampled_ids)].copy().sort_values("vehicle_id")
    if len(sample) != args.sample_size:
        raise AssertionError("fixed-size vehicle sample was not created")

    metrics.to_csv(args.out / "vehicle_summary.csv", index=False)
    sample.to_csv(args.out / "subsample_vehicle_summary.csv", index=False)
    pd.DataFrame({"vehicle_id": sampled_ids}).to_csv(args.out / "subsample_vehicle_ids.csv", index=False)

    full = population_summary(metrics)
    sample_summary = population_summary(sample)
    stratified = length_stratification(metrics)
    pd.DataFrame(stratified).to_csv(args.out / "event_by_sequence_length.csv", index=False)
    pd.DataFrame(summary_rows(metrics, sample)).to_csv(args.out / "summary_statistics.csv", index=False)

    source_files = {
        "train_operational_readouts.csv": {
            "path": str(args.readouts),
            "download_url": f"{SOURCE_BASE}?filePath=train_operational_readouts.csv",
            "bytes": args.readouts.stat().st_size,
            "sha256": sha256(args.readouts),
            "schema": readout_schema,
        },
        "train_tte.csv": {
            "path": str(args.tte),
            "download_url": f"{SOURCE_BASE}?filePath=train_tte.csv",
            "bytes": args.tte.stat().st_size,
            "sha256": sha256(args.tte),
            "schema": tte_schema,
        },
    }
    summary = {
        "source": {
            "dataset": "SCANIA Component X Dataset",
            "version": SOURCE_VERSION,
            "doi": SOURCE_DOI,
            "catalog_url": SOURCE_CATALOG,
            "files": source_files,
        },
        "full_training": full,
        "subsample": {
            "seed": args.seed,
            "sample_size": args.sample_size,
            "sampling": "uniform without replacement over vehicles; event/censoring classes not balanced",
            "summary": sample_summary,
            "vehicle_ids_file": "subsample_vehicle_ids.csv",
        },
        "representativeness": comparison(metrics, sample),
        "event_censoring_by_sequence_length": stratified,
        "inter_observation_gap": gap_summary(gaps),
        "quality_checks": {**readout_quality, **tte_quality},
        "interpretation": {
            "time_step": "raw operational time_step; values were not discretized or resampled",
            "observed_time_span": "last_observed_time_step - first_observed_time_step, using min/max per vehicle",
            "in_study_repair": "1 = repair/failure during the study window; 0 = right-censored",
            "gap_definition": "positive differences between consecutive time_step values in the stored vehicle/time order",
            "study_time_minus_last_observed": "length_of_study_time_step - last_observed_time_step; positive values mean the final readout precedes the study endpoint",
        },
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    figure_names = make_figures(metrics, gaps, args.out)
    summary["figures"] = figure_names
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps({
        "subjects": full["n_subjects"],
        "readouts": full["n_readouts"],
        "events": full["events"],
        "censored": full["censored"],
        "event_rate": full["event_rate"],
        "sample_size": sample_summary["n_subjects"],
        "out": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
