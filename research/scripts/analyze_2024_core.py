from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nse.config import MODEL

OUT = Path("research/outputs")
REPRO = OUT / "reproduction"
BASE = OUT / "baselines"
DAILY = OUT / "daily_ic"
HIT = OUT / "hit_rate"
TABLES = Path("research/tables")
for p in [BASE, DAILY, HIT, TABLES]:
    p.mkdir(parents=True, exist_ok=True)

LABEL = MODEL["primary_label"]
EXCLUDE_PREFIXES = ("label_", "dq_", "id", "date", "ticker")


def feature_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if any(c.startswith(p) for p in EXCLUDE_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def daily_metrics(scored: pd.DataFrame, model: str) -> tuple[pd.DataFrame, dict]:
    rows = []
    for dt, grp in scored.groupby("date"):
        grp = grp.dropna(subset=["score", LABEL])
        if len(grp) < 3:
            continue
        ic, pvalue = spearmanr(grp["score"], grp[LABEL])
        ordered = grp.sort_values("score", ascending=False)
        top1 = ordered.iloc[0]
        top2 = ordered.head(2)
        top3 = ordered.head(3)
        rows.append(
            {
                "model": model,
                "date": pd.Timestamp(dt).date().isoformat(),
                "month": pd.Timestamp(dt).strftime("%Y-%m"),
                "ic": float(ic) if not np.isnan(ic) else np.nan,
                "pvalue": float(pvalue) if not np.isnan(pvalue) else np.nan,
                "n": int(len(grp)),
                "top1_hit": int(top1[LABEL] > 0),
                "top2_hit": int(top2[LABEL].mean() > 0),
                "top3_hit": int(top3[LABEL].mean() > 0),
                "top1_label": float(top1[LABEL]),
                "top2_label": float(top2[LABEL].mean()),
                "top3_label": float(top3[LABEL].mean()),
            }
        )
    daily = pd.DataFrame(rows)
    ic = daily["ic"].dropna()
    monthly = daily.groupby("month")["ic"].mean()
    summary = {
        "model": model,
        "mean_daily_ic": ic.mean(),
        "median_daily_ic": ic.median(),
        "ic_std": ic.std(ddof=1),
        "icir": ic.mean() / ic.std(ddof=1) if ic.std(ddof=1) else np.nan,
        "frac_daily_ic_positive": (ic > 0).mean(),
        "mean_monthly_ic": monthly.mean(),
        "positive_months": int((monthly > 0).sum()),
        "n_months": int(monthly.notna().sum()),
        "top1_hit_rate": daily["top1_hit"].mean(),
        "top2_hit_rate": daily["top2_hit"].mean(),
        "top3_hit_rate": daily["top3_hit"].mean(),
        "n_days": int(len(daily)),
    }
    return daily, summary


def newey_west_mean_test(x: pd.Series, lag: int = 3) -> dict:
    vals = x.dropna().astype(float).to_numpy()
    n = len(vals)
    mean = vals.mean()
    centered = vals - mean
    gamma0 = np.dot(centered, centered) / n
    lrvar = gamma0
    for k in range(1, min(lag, n - 1) + 1):
        gamma = np.dot(centered[k:], centered[:-k]) / n
        weight = 1 - k / (lag + 1)
        lrvar += 2 * weight * gamma
    se = np.sqrt(lrvar / n)
    t_stat = mean / se if se > 0 else np.nan
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1)) if se > 0 else np.nan
    return {
        "n_days": n,
        "mean_ic": mean,
        "median_ic": np.median(vals),
        "ic_std": vals.std(ddof=1),
        "icir": mean / vals.std(ddof=1),
        "hac_lag": lag,
        "hac_se": se,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
        "t_stat": t_stat,
        "p_value": p_value,
        "frac_positive": (vals > 0).mean(),
    }


def main() -> None:
    features = pd.read_csv(REPRO / "features_daily_snapshot.csv", parse_dates=["date"])
    slices = pd.read_csv(REPRO / "walk_forward_2024_summary.csv")
    lgbm_preds = pd.read_csv(REPRO / "walk_forward_2024_predictions.csv", parse_dates=["date"])
    lgbm_test = lgbm_preds[lgbm_preds["split"] == "test"].copy()
    lgbm_test = lgbm_test.rename(columns={"score": "score"})

    all_daily = []
    summaries = []

    daily, summary = daily_metrics(lgbm_test, "LightGBM")
    all_daily.append(daily)
    summaries.append(summary)

    rng = np.random.default_rng(42)
    simple_specs = {
        "Random ranking": lambda df: rng.random(len(df)),
        "1-day reversal": lambda df: -df["ret_1d"],
        "3-day reversal": lambda df: -df["ret_3d"],
        "5-day momentum": lambda df: df["ret_5d"],
        "20-day momentum": lambda df: df["ret_20d"],
    }

    model_specs = {
        "Linear regression": lambda: make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), LinearRegression()
        ),
        "Ridge regression": lambda: make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)
        ),
        "Random Forest": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    }

    scored_frames = []
    for _, slc in slices.iterrows():
        test = features[
            (features["date"] >= pd.Timestamp(slc["test_start"]))
            & (features["date"] <= pd.Timestamp(slc["test_end"]))
        ].dropna(subset=[LABEL]).copy()
        train = features[
            (features["date"] >= pd.Timestamp(slc["train_start"]))
            & (features["date"] <= pd.Timestamp(slc["train_end"]))
        ].dropna(subset=[LABEL]).copy()

        for name, fn in simple_specs.items():
            scored = test[["date", "ticker", LABEL]].copy()
            scored["score"] = fn(test)
            scored["model"] = name
            scored_frames.append(scored)

        cols = feature_cols(train)
        X_train = train[cols]
        y_train = train[LABEL]
        X_test = test[cols]
        for name, factory in model_specs.items():
            model = factory()
            model.fit(X_train, y_train)
            scored = test[["date", "ticker", LABEL]].copy()
            scored["score"] = model.predict(X_test)
            scored["model"] = name
            scored_frames.append(scored)

    baseline_scored = pd.concat(scored_frames, ignore_index=True)
    baseline_scored.to_csv(BASE / "baseline_predictions_2024.csv", index=False)

    for name, grp in baseline_scored.groupby("model"):
        daily, summary = daily_metrics(grp, name)
        all_daily.append(daily)
        summaries.append(summary)

    daily_all = pd.concat(all_daily, ignore_index=True)
    summary_df = pd.DataFrame(summaries).sort_values("mean_daily_ic", ascending=False)
    daily_all.to_csv(DAILY / "daily_ic_all_models_2024.csv", index=False)
    summary_df.to_csv(BASE / "baseline_comparison_2024.csv", index=False)

    lightgbm_daily = daily_all[daily_all["model"] == "LightGBM"]
    inference = pd.DataFrame([{"model": "LightGBM", **newey_west_mean_test(lightgbm_daily["ic"], lag=3)}])
    inference.to_csv(DAILY / "lightgbm_hac_inference_2024.csv", index=False)

    unconditional = features[
        (features["date"] >= "2024-01-01")
        & (features["date"] <= "2024-12-31")
    ].dropna(subset=[LABEL])
    hit = pd.DataFrame(
        [
            {
                "metric": "unconditional_security_day_positive_rate",
                "count": int((unconditional[LABEL] > 0).sum()),
                "n": int(len(unconditional)),
                "rate": float((unconditional[LABEL] > 0).mean()),
            },
            {
                "metric": "lightgbm_top1_positive_rate",
                "count": int(lightgbm_daily["top1_hit"].sum()),
                "n": int(len(lightgbm_daily)),
                "rate": float(lightgbm_daily["top1_hit"].mean()),
            },
            {
                "metric": "lightgbm_top2_equal_weight_positive_rate",
                "count": int(lightgbm_daily["top2_hit"].sum()),
                "n": int(len(lightgbm_daily)),
                "rate": float(lightgbm_daily["top2_hit"].mean()),
            },
            {
                "metric": "lightgbm_top3_equal_weight_positive_rate",
                "count": int(lightgbm_daily["top3_hit"].sum()),
                "n": int(len(lightgbm_daily)),
                "rate": float(lightgbm_daily["top3_hit"].mean()),
            },
        ]
    )
    hit.to_csv(HIT / "hit_rate_comparison_2024.csv", index=False)

    # Compact LaTeX tables.
    summary_df.to_latex(
        TABLES / "baseline_comparison_2024.tex",
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
    inference.to_latex(
        TABLES / "lightgbm_hac_inference_2024.tex",
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
    hit.to_latex(
        TABLES / "hit_rate_comparison_2024.tex",
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )

    print(json.dumps({
        "best_by_mean_daily_ic": summary_df.iloc[0].to_dict(),
        "lightgbm_inference": inference.iloc[0].to_dict(),
        "hit_rates": hit.to_dict(orient="records"),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
