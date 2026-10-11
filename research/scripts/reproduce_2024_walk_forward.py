from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr
from dotenv import load_dotenv

from nse.config import MODEL
from nse.models.ranker import NSERanker
from nse.models.walk_forward import _generate_slices, _slice_df


OUT = Path("research/outputs/reproduction")
OUT.mkdir(parents=True, exist_ok=True)
SNAPSHOT = OUT / "features_daily_snapshot.csv"
CHUNK_DIR = OUT / "feature_chunks"
CHUNK_DIR.mkdir(parents=True, exist_ok=True)


def month_starts(start: date, end: date) -> list[date]:
    months = []
    cur = date(start.year, start.month, 1)
    while cur <= end:
        months.append(cur)
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)
    return months


def next_month(d: date) -> date:
    if d.month == 12:
        return date(d.year + 1, 1, 1)
    return date(d.year, d.month + 1, 1)


def connect_pg():
    load_dotenv(".env")
    import os

    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        connect_timeout=30,
        keepalives=1,
        keepalives_idle=10,
        keepalives_interval=5,
        keepalives_count=3,
    )


def export_feature_chunk(start: date, end: date, path: Path) -> None:
    query = f"""
    COPY (
      SELECT *
      FROM features_daily
      WHERE date >= DATE '{start.isoformat()}'
        AND date < DATE '{end.isoformat()}'
      ORDER BY date, ticker
    ) TO STDOUT WITH CSV HEADER
    """
    conn = connect_pg()
    try:
        with conn, conn.cursor() as cur, path.open("w", encoding="utf-8") as f:
            cur.execute("SET statement_timeout TO '120s'")
            cur.copy_expert(query, f)
    finally:
        conn.close()


def load_features_sql() -> pd.DataFrame:
    if SNAPSHOT.exists():
        df = pd.read_csv(SNAPSHOT)
        normalize_booleans(df)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values(["date", "ticker"]).reset_index(drop=True)

    starts = month_starts(date(2019, 1, 1), date(2024, 12, 1))
    for start in starts:
        end = next_month(start)
        path = CHUNK_DIR / f"features_{start:%Y_%m}.csv"
        if path.exists() and path.stat().st_size > 100:
            continue
        last_exc: Exception | None = None
        for attempt in range(1, 5):
            try:
                print(f"exporting {start:%Y-%m} attempt {attempt}", flush=True)
                export_feature_chunk(start, end, path)
                rows = sum(1 for _ in path.open("r", encoding="utf-8")) - 1
                print(f"  wrote {rows} rows", flush=True)
                break
            except Exception as exc:
                last_exc = exc
                print(
                    f"  failed {start:%Y-%m}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                if path.exists():
                    path.unlink()
                time.sleep(5 * attempt)
        else:
            raise RuntimeError(f"Could not export features for {start:%Y-%m}") from last_exc

    frames = [pd.read_csv(CHUNK_DIR / f"features_{start:%Y_%m}.csv") for start in starts]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(SNAPSHOT, index=False)
    normalize_booleans(df)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def normalize_booleans(df: pd.DataFrame) -> None:
    for col in ["dq_price_ok", "dq_features_ok", "dq_labels_ok", "dq_tradable"]:
        if col in df.columns:
            df[col] = df[col].map(
                {
                    True: True,
                    False: False,
                    "t": True,
                    "f": False,
                    "true": True,
                    "false": False,
                    "True": True,
                    "False": False,
                    1: True,
                    0: False,
                }
            ).fillna(False)


def eval_predictions(df: pd.DataFrame, label: str) -> tuple[float, float, pd.DataFrame]:
    rows = []
    for dt, grp in df.groupby("date"):
        grp = grp.dropna(subset=["score", label])
        if len(grp) < 3:
            continue
        ic, pval = spearmanr(grp["score"], grp[label])
        best = grp.sort_values("score", ascending=False).iloc[0]
        rows.append(
            {
                "date": pd.Timestamp(dt).date().isoformat(),
                "ic": float(ic) if not np.isnan(ic) else np.nan,
                "pvalue": float(pval) if not np.isnan(pval) else np.nan,
                "top1_ticker": best["ticker"],
                "top1_label": float(best[label]),
                "top1_hit": int(best[label] > 0),
                "n": int(len(grp)),
            }
        )
    daily = pd.DataFrame(rows)
    mean_ic = float(daily["ic"].mean()) if not daily.empty else np.nan
    hit_rate = float(daily["top1_hit"].mean()) if not daily.empty else np.nan
    return mean_ic, hit_rate, daily


def main() -> None:
    label = MODEL["primary_label"]
    features = load_features_sql()
    if features.empty:
        raise SystemExit("No features loaded")

    summary_rows = []
    all_daily = []
    all_preds = []

    for slc in _generate_slices(features, 12):
        slice_id = slc.test_start.strftime("%Y-%m")
        train_df = _slice_df(features, slc.train_start, slc.train_end).dropna(subset=[label])
        val_df = _slice_df(features, slc.val_start, slc.val_end).dropna(subset=[label])
        test_df = _slice_df(features, slc.test_start, slc.test_end).dropna(subset=[label])

        ranker = NSERanker(run_id=f"research_{slice_id}")
        ranker.fit(train_df, val_df if not val_df.empty else None)

        val_scores = ranker.predict(val_df)
        test_scores = ranker.predict(test_df)
        val_scored = val_df.merge(val_scores, on=["date", "ticker"], how="inner")
        test_scored = test_df.merge(test_scores, on=["date", "ticker"], how="inner")

        val_ic, val_hit, val_daily = eval_predictions(val_scored, label)
        test_ic, test_hit, test_daily = eval_predictions(test_scored, label)
        test_daily["slice_id"] = slice_id
        val_daily["slice_id"] = slice_id
        val_daily["split"] = "validation"
        test_daily["split"] = "test"
        all_daily.extend([val_daily, test_daily])

        test_scored["slice_id"] = slice_id
        test_scored["split"] = "test"
        val_scored["slice_id"] = slice_id
        val_scored["split"] = "validation"
        keep_cols = [
            "slice_id",
            "split",
            "date",
            "ticker",
            "raw_score",
            "score",
            "top_bucket_prob",
            label,
            "label_binary",
            "label_top_bucket",
        ]
        all_preds.extend([val_scored[[c for c in keep_cols if c in val_scored.columns]],
                          test_scored[[c for c in keep_cols if c in test_scored.columns]]])

        summary_rows.append(
            {
                "slice_id": slice_id,
                "train_start": slc.train_start.isoformat(),
                "train_end": slc.train_end.isoformat(),
                "val_start": slc.val_start.isoformat(),
                "val_end": slc.val_end.isoformat(),
                "test_start": slc.test_start.isoformat(),
                "test_end": slc.test_end.isoformat(),
                "val_ic": val_ic,
                "test_ic": test_ic,
                "val_hit_rate": val_hit,
                "test_hit_rate": test_hit,
                "n_train": int(len(train_df)),
                "n_val": int(len(val_df)),
                "n_test": int(len(test_df)),
                "n_features": int(len(ranker.feature_cols)),
            }
        )

        ranker.feature_importance().to_csv(OUT / f"feature_importance_{slice_id}.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    daily = pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame()
    preds = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()

    summary.to_csv(OUT / "walk_forward_2024_summary.csv", index=False)
    daily.to_csv(OUT / "walk_forward_2024_daily_ic.csv", index=False)
    preds.to_csv(OUT / "walk_forward_2024_predictions.csv", index=False)

    manifest = {
        "label": label,
        "model_config": MODEL,
        "feature_rows": int(len(features)),
        "date_min": str(features["date"].min()),
        "date_max": str(features["date"].max()),
        "summary": {
            "mean_val_ic": float(summary["val_ic"].mean()),
            "mean_test_ic": float(summary["test_ic"].mean()),
            "positive_test_months": int((summary["test_ic"] > 0).sum()),
            "n_months": int(len(summary)),
            "mean_test_hit_rate": float(summary["test_hit_rate"].mean()),
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
