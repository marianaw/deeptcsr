"""Build the LastFM churn dataset from the Last.fm 1K listening logs.

Input: userid-timestamp-artid-artname-traid-traname.tsv from
http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html (Celma, 2010).
Output (in --out, the dataset's data_path): surv_logs_last.csv (one row per
user and active week: listen_count, unique_artists, artist_entropy,
repeat_ratio, novelty and their week-to-week differences) and events.csv
(time = number of active weeks; a user is censored if active in the week of
2009-05-05, the end of the collection period, otherwise churned).

    uv run python scripts/prepare_lastfm.py --raw path/to/userid-timestamp-artid-artname-traid-traname.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEATS = ["listen_count", "unique_artists", "artist_entropy", "repeat_ratio", "novelty"]
END_WEEK = pd.Period("2009-05-05", "W-MON")  # week of the last log entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/lastfm-dataset-1K"))
    a = ap.parse_args()

    df = pd.read_csv(a.raw, sep="\t", header=None, usecols=[0, 1, 3, 5],
                     names=["user", "timestamp", "artist_id", "artist", "track_id", "song"],
                     encoding="utf-8", encoding_errors="replace", on_bad_lines="skip")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["week"] = df["timestamp"].dt.tz_localize(None).dt.to_period("W-MON")
    df = df.sort_values(["user", "timestamp"], kind="stable")

    # n-th play of a song by this user: repeat if n > 0, novelty 1 / (1 + n)
    n = df.groupby(["user", "song"], dropna=False, sort=False).cumcount()
    df["repeat"], df["novelty"] = n > 0, 1.0 / (1.0 + n)

    g = df.groupby(["user", "week"])
    weekly = pd.DataFrame({"listen_count": g.size(), "unique_artists": g["artist"].nunique(),
                           "repeat_ratio": g["repeat"].mean(), "novelty": g["novelty"].mean()})
    cnt = df.groupby(["user", "week", "artist"]).size()
    p = cnt / cnt.groupby(level=[0, 1]).transform("sum")
    weekly["artist_entropy"] = (-p * np.log(p)).groupby(level=[0, 1]).sum()
    weekly["artist_entropy"] = weekly["artist_entropy"].fillna(0.0)
    weekly = weekly.reset_index().rename(columns={"user": "userid"})
    weekly = weekly.sort_values(["userid", "week"]).reset_index(drop=True)
    for c in FEATS:
        weekly[f"delta_{c}"] = weekly.groupby("userid")[c].diff().fillna(0)
    weekly["week_idx"] = weekly.groupby("userid").cumcount()

    weekly["in_last_week"] = weekly["week"] == END_WEEK
    events = weekly.groupby("userid").agg(time=("week", "size"),
                                          censored=("in_last_week", "any"))

    cols = ["userid"] + FEATS + [f"delta_{c}" for c in FEATS] + ["week_idx"]
    a.out.mkdir(parents=True, exist_ok=True)
    weekly[cols].to_csv(a.out / "surv_logs_last.csv", index=True)
    events[["time", "censored"]].to_csv(a.out / "events.csv", index=False)
    print(f"{len(events)} users, {len(weekly)} user-weeks, "
          f"churn rate {(~events.censored).mean():.2%}")


if __name__ == "__main__":
    main()
