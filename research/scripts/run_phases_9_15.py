from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from scipy import stats
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=stats.ConstantInputWarning if hasattr(stats, "ConstantInputWarning") else Warning)

ROOT = Path("research")
OUT = ROOT / "outputs" / "phases_9_15"
REPRO = ROOT / "outputs" / "reproduction"
OUT.mkdir(parents=True, exist_ok=True)

LABEL = "label_3d_net_alpha"
EMBEDDED_LABEL_COST_BPS = 400
HOLD_DAYS = 3
PERIODS_PER_YEAR = 252

EXCLUDE_PREFIXES = ("label_", "dq_", "id", "date", "ticker")

LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 10,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "n_estimators": 250,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}


@dataclass(frozen=True)
class Slice:
    slice_id: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


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


def read_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    normalize_booleans(df)
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def feature_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if any(col.startswith(prefix) for prefix in EXCLUDE_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def daily_ic(scored: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for dt, grp in scored.groupby("date"):
        grp = grp.dropna(subset=["score", label])
        if len(grp) < 3:
            continue
        scores = grp["score"].to_numpy(dtype=float)
        labels = grp[label].to_numpy(dtype=float)
        ic, pvalue = spearmanr(scores, labels)
        order = np.argsort(-scores)
        top1_label = labels[order[:1]].mean()
        top2_label = labels[order[:2]].mean()
        top3_label = labels[order[:3]].mean()
        rows.append(
            {
                "date": pd.Timestamp(dt).date().isoformat(),
                "month": pd.Timestamp(dt).strftime("%Y-%m"),
                "ic": float(ic) if np.isfinite(ic) else np.nan,
                "pvalue": float(pvalue) if np.isfinite(pvalue) else np.nan,
                "n": int(len(grp)),
                "top1_label": float(top1_label),
                "top1_hit": int(top1_label > 0),
                "top2_label": float(top2_label),
                "top2_hit": int(top2_label > 0),
                "top3_label": float(top3_label),
                "top3_hit": int(top3_label > 0),
            }
        )
    return pd.DataFrame(rows)


def summarize_daily(daily: pd.DataFrame, model: str, label: str = "ic") -> dict:
    vals = daily[label].dropna().astype(float)
    monthly = daily.groupby("month")[label].mean() if "month" in daily else pd.Series(dtype=float)
    return {
        "model": model,
        "n_days": int(len(vals)),
        "mean_daily_ic": float(vals.mean()) if len(vals) else np.nan,
        "median_daily_ic": float(vals.median()) if len(vals) else np.nan,
        "ic_std": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
        "icir": float(vals.mean() / vals.std(ddof=1)) if len(vals) > 1 and vals.std(ddof=1) else np.nan,
        "frac_daily_ic_positive": float((vals > 0).mean()) if len(vals) else np.nan,
        "mean_monthly_ic": float(monthly.mean()) if len(monthly) else np.nan,
        "positive_months": int((monthly > 0).sum()) if len(monthly) else 0,
        "n_months": int(monthly.notna().sum()) if len(monthly) else 0,
        "top1_hit_rate": float(daily["top1_hit"].mean()) if "top1_hit" in daily else np.nan,
        "top2_hit_rate": float(daily["top2_hit"].mean()) if "top2_hit" in daily else np.nan,
        "top3_hit_rate": float(daily["top3_hit"].mean()) if "top3_hit" in daily else np.nan,
    }


def hac_mean_test(x: pd.Series, lag: int = 3) -> dict:
    vals = x.dropna().astype(float).to_numpy()
    n = len(vals)
    if n == 0:
        return {"n_days": 0}
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
        "n_days": int(n),
        "mean": float(mean),
        "median": float(np.median(vals)),
        "std": float(vals.std(ddof=1)) if n > 1 else np.nan,
        "hac_lag": lag,
        "hac_se": float(se),
        "ci95_low": float(mean - 1.96 * se),
        "ci95_high": float(mean + 1.96 * se),
        "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
        "frac_positive": float((vals > 0).mean()),
    }


def month_slices(start: str, end: str, train_start: str = "2019-01-01", rolling_years: int | None = None) -> list[Slice]:
    months = pd.date_range(start=start, end=end, freq="MS")
    slices = []
    for current in months:
        test_end = current + pd.offsets.MonthEnd(0)
        val_end = current - pd.Timedelta(days=1)
        val_start = current - pd.DateOffset(months=3)
        if rolling_years is None:
            t_start = pd.Timestamp(train_start)
        else:
            t_start = current - pd.DateOffset(years=rolling_years)
        slices.append(
            Slice(
                slice_id=current.strftime("%Y-%m"),
                train_start=t_start,
                train_end=val_start - pd.Timedelta(days=1),
                val_start=val_start,
                val_end=val_end,
                test_start=current,
                test_end=test_end,
            )
        )
    return slices


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
    label: str,
    seed: int = 42,
    model_kind: str = "lightgbm",
) -> pd.DataFrame:
    train = train.dropna(subset=[label]).copy()
    test = test.dropna(subset=[label]).copy()
    cols = [c for c in cols if c in train.columns and c in test.columns]
    if not cols or len(train) < 50 or test.empty:
        return pd.DataFrame(columns=["date", "ticker", label, "score", "raw_score"])

    X_train = train[cols]
    y_train = train[label]
    X_test = test[cols]

    if model_kind == "random_forest":
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=10,
                random_state=seed,
                n_jobs=-1,
            ),
        )
    else:
        params = {**LGB_PARAMS, "random_state": seed, "seed": seed}
        model = make_pipeline(SimpleImputer(strategy="median"), lgb.LGBMRegressor(**params))

    model.fit(X_train, y_train)
    out = test[["date", "ticker", label]].copy()
    out["raw_score"] = model.predict(X_test)
    out["score"] = out.groupby("date")["raw_score"].transform(
        lambda s: (s - s.min()) / (s.max() - s.min() + 1e-10)
    )
    return out


def run_walk_forward_experiment(
    features: pd.DataFrame,
    slices: list[Slice],
    cols: list[str],
    label: str,
    experiment: str,
    seed: int = 42,
    model_kind: str = "lightgbm",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pred_frames = []
    daily_frames = []
    for slc in slices:
        train = features[
            (features["date"] >= slc.train_start) & (features["date"] <= slc.train_end)
        ]
        test = features[
            (features["date"] >= slc.test_start) & (features["date"] <= slc.test_end)
        ]
        pred = fit_predict(train, test, cols, label, seed=seed, model_kind=model_kind)
        if pred.empty:
            continue
        pred["slice_id"] = slc.slice_id
        pred["experiment"] = experiment
        pred_frames.append(pred)
        dly = daily_ic(pred, label)
        dly["slice_id"] = slc.slice_id
        dly["experiment"] = experiment
        daily_frames.append(dly)
    preds = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    summary = summarize_daily(daily, experiment)
    summary["label"] = label
    summary["model_kind"] = model_kind
    summary["n_features"] = len(cols)
    return preds, daily, summary


def max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    equity = (1 + returns.fillna(0)).cumprod()
    dd = equity / equity.cummax() - 1
    return float(dd.min())


def portfolio_metrics(daily_returns: pd.Series, sleeve_returns: pd.DataFrame) -> dict:
    rets = daily_returns.dropna()
    if rets.empty:
        return {}
    equity = (1 + rets).cumprod()
    total = equity.iloc[-1] - 1
    ann = (1 + total) ** (PERIODS_PER_YEAR / len(rets)) - 1
    vol = rets.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)
    sharpe = ann / vol if vol and np.isfinite(vol) else np.nan
    mdd = max_drawdown(rets)
    return {
        "n_days": int(len(rets)),
        "total_return": float(total),
        "annualised_return": float(ann),
        "annualised_volatility": float(vol),
        "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "max_drawdown": float(mdd),
        "calmar": float(ann / abs(mdd)) if mdd < 0 else np.nan,
        "daily_hit_rate": float((rets > 0).mean()),
        "n_sleeves": int(len(sleeve_returns)),
        "sleeve_hit_rate": float((sleeve_returns["net_3d_return"] > 0).mean()) if len(sleeve_returns) else np.nan,
        "mean_sleeve_net_3d_return": float(sleeve_returns["net_3d_return"].mean()) if len(sleeve_returns) else np.nan,
        "median_sleeve_net_3d_return": float(sleeve_returns["net_3d_return"].median()) if len(sleeve_returns) else np.nan,
        "viable_trade_fraction": float((sleeve_returns["gross_3d_return"] > sleeve_returns["cost_bps"] / 10000).mean()) if len(sleeve_returns) else np.nan,
        "mean_turnover_names": float(sleeve_returns["turnover_names"].mean()) if len(sleeve_returns) else np.nan,
    }


def simulate_overlapping_portfolio(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    top_k: int,
    cost_bps: int,
    score_threshold: float | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    scored = predictions[predictions["split"].eq("test")].copy() if "split" in predictions else predictions.copy()
    scored["date"] = pd.to_datetime(scored["date"])
    realized = features[["date", "ticker", LABEL]].copy()
    realized["gross_3d_return"] = realized[LABEL] + EMBEDDED_LABEL_COST_BPS / 10000
    scored = scored.merge(realized, on=["date", "ticker"], how="inner")

    dates = sorted(scored["date"].unique())
    all_dates = pd.Index(dates)
    daily_contrib: dict[pd.Timestamp, list[float]] = {pd.Timestamp(d): [] for d in all_dates}
    sleeves = []
    prev_names: set[str] = set()

    for i, dt in enumerate(dates):
        day = scored[scored["date"].eq(dt)].dropna(subset=["score", "gross_3d_return"])
        if score_threshold is not None:
            day = day[day["score"] >= score_threshold]
        selected = day.sort_values("score", ascending=False).head(top_k)
        if selected.empty:
            continue
        names = set(selected["ticker"].astype(str))
        turnover_names = len(names.symmetric_difference(prev_names)) / max(top_k, 1)
        prev_names = names
        gross = float(selected["gross_3d_return"].mean())
        net = gross - cost_bps / 10000
        daily_equiv = (1 + net) ** (1 / HOLD_DAYS) - 1 if net > -1 else -1.0
        for j in range(HOLD_DAYS):
            if i + j < len(dates):
                daily_contrib[pd.Timestamp(dates[i + j])].append(daily_equiv)
        sleeves.append(
            {
                "signal_date": pd.Timestamp(dt).date().isoformat(),
                "top_k": top_k,
                "cost_bps": cost_bps,
                "tickers": ",".join(selected["ticker"].astype(str).tolist()),
                "gross_3d_return": gross,
                "net_3d_return": net,
                "daily_equivalent_return": daily_equiv,
                "mean_score": float(selected["score"].mean()),
                "turnover_names": turnover_names,
            }
        )

    daily_returns = pd.Series(
        {dt: float(np.mean(vals)) if vals else 0.0 for dt, vals in daily_contrib.items()}
    ).sort_index()
    return daily_returns, pd.DataFrame(sleeves)


def phase_portfolio_and_costs(features: pd.DataFrame) -> dict:
    predictions = pd.read_csv(REPRO / "walk_forward_2024_predictions.csv", parse_dates=["date"])
    summary_rows = []
    sleeve_frames = []
    return_frames = []
    for top_k in [1, 2, 3, 5]:
        for cost_bps in [0, 100, 200, 300, 400, 600]:
            rets, sleeves = simulate_overlapping_portfolio(predictions, features, top_k, cost_bps)
            metrics = portfolio_metrics(rets, sleeves)
            metrics.update({"top_k": top_k, "cost_bps": cost_bps})
            summary_rows.append(metrics)
            sleeves["strategy"] = f"top{top_k}_cost{cost_bps}"
            sleeve_frames.append(sleeves)
            return_frames.append(
                pd.DataFrame(
                    {
                        "date": [d.date().isoformat() for d in rets.index],
                        "daily_return": rets.values,
                        "top_k": top_k,
                        "cost_bps": cost_bps,
                    }
                )
            )
    summary = pd.DataFrame(summary_rows).sort_values(["top_k", "cost_bps"])
    sleeves_all = pd.concat(sleeve_frames, ignore_index=True)
    returns_all = pd.concat(return_frames, ignore_index=True)
    summary.to_csv(OUT / "portfolio_cost_sensitivity_2024.csv", index=False)
    sleeves_all.to_csv(OUT / "portfolio_sleeves_2024.csv", index=False)
    returns_all.to_csv(OUT / "portfolio_daily_returns_2024.csv", index=False)
    return {
        "best_by_sharpe": summary.sort_values("sharpe", ascending=False).head(1).to_dict(orient="records"),
        "base_top1": summary[(summary["top_k"] == 1) & (summary["cost_bps"] == 400)].to_dict(orient="records"),
    }


def feature_groups(all_cols: list[str]) -> dict[str, list[str]]:
    groups = {
        "returns_price_only": [
            c for c in all_cols
            if c in {"ret_1d", "ret_3d", "ret_5d", "ret_20d", "overnight_gap"}
            or "price" in c
            or "range" in c
        ],
        "momentum_reversal": [
            c for c in all_cols
            if "ret_" in c or "momentum" in c or "reversal" in c or c == "overnight_gap"
        ],
        "volume_liquidity": [
            c for c in all_cols
            if "volume" in c or "value_traded" in c or "liquidity" in c or "amihud" in c
            or "stale" in c or "deals" in c or "spread" in c
        ],
        "macro_all": [
            c for c in all_cols
            if "kes_usd" in c or "tbill" in c or "policy" in c or "inflation" in c
            or "market_turnover" in c or "sector_regime" in c
        ],
        "kes_usd_only": [c for c in all_cols if "kes_usd" in c],
        "nse_microstructure": [
            c for c in all_cols
            if "band" in c or "stale" in c or "zero_vol" in c or "intraday" in c
            or "session" in c or "morning" in c or "afternoon" in c or "wap" in c
            or "order_book" in c or "bid_ask" in c or "foreign" in c
        ],
        "llm_events": [c for c in all_cols if c.startswith("llm_")],
        "full_feature_set": all_cols,
    }
    return {k: v for k, v in groups.items() if v}


def phase_feature_ablations(features: pd.DataFrame) -> dict:
    slices = month_slices("2024-01-01", "2024-12-01")
    groups = feature_groups(feature_cols(features))
    summaries = []
    daily_frames = []
    pred_frames = []
    for name, cols in groups.items():
        preds, daily, summary = run_walk_forward_experiment(features, slices, cols, LABEL, name)
        summaries.append(summary)
        if not daily.empty:
            daily_frames.append(daily)
        if not preds.empty:
            pred_frames.append(preds)
    summary_df = pd.DataFrame(summaries).sort_values("mean_daily_ic", ascending=False)
    summary_df.to_csv(OUT / "feature_ablation_summary_2024.csv", index=False)
    if daily_frames:
        pd.concat(daily_frames, ignore_index=True).to_csv(OUT / "feature_ablation_daily_ic_2024.csv", index=False)
    if pred_frames:
        pd.concat(pred_frames, ignore_index=True).to_csv(OUT / "feature_ablation_predictions_2024.csv", index=False)
    return {"top_ablations": summary_df.head(5).to_dict(orient="records")}


def phase_robustness(features: pd.DataFrame) -> dict:
    all_cols = feature_cols(features)
    specs = []
    for label in ["label_1d_net_alpha", "label_3d_net_alpha", "label_5d_net_alpha"]:
        specs.append((f"{label}_expanding_seed42_lgbm", label, None, 42, "lightgbm"))
    for seed in [7, 123]:
        specs.append((f"label_3d_net_alpha_expanding_seed{seed}_lgbm", LABEL, None, seed, "lightgbm"))
    specs.append(("label_3d_net_alpha_rolling3y_seed42_lgbm", LABEL, 3, 42, "lightgbm"))
    specs.append(("label_3d_net_alpha_expanding_seed42_rf", LABEL, None, 42, "random_forest"))

    summaries = []
    daily_frames = []
    for name, label, rolling_years, seed, model_kind in specs:
        slices = month_slices("2024-01-01", "2024-12-01", rolling_years=rolling_years)
        preds, daily, summary = run_walk_forward_experiment(
            features, slices, all_cols, label, name, seed=seed, model_kind=model_kind
        )
        summary["window"] = "rolling_3y" if rolling_years else "expanding"
        summary["seed"] = seed
        summaries.append(summary)
        if not daily.empty:
            daily_frames.append(daily)
    summary_df = pd.DataFrame(summaries).sort_values("mean_daily_ic", ascending=False)
    summary_df.to_csv(OUT / "robustness_summary_2024.csv", index=False)
    if daily_frames:
        pd.concat(daily_frames, ignore_index=True).to_csv(OUT / "robustness_daily_ic_2024.csv", index=False)
    return {"robustness": summary_df.to_dict(orient="records")}


def export_features_from_db(start: date, end: date, out_path: Path) -> tuple[bool, str]:
    load_dotenv(".env")
    required = ["DB_HOST", "DB_NAME", "DB_USER"]
    if any(not os.getenv(k) for k in required):
        return False, "database environment variables are incomplete"
    query = f"""
    COPY (
      SELECT *
      FROM features_daily
      WHERE date >= DATE '{start.isoformat()}'
        AND date <= DATE '{end.isoformat()}'
      ORDER BY date, ticker
    ) TO STDOUT WITH CSV HEADER
    """
    try:
        conn = psycopg2.connect(
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
        try:
            with conn, conn.cursor() as cur, out_path.open("w", encoding="utf-8") as f:
                cur.execute("SET statement_timeout TO '180s'")
                cur.copy_expert(query, f)
        finally:
            conn.close()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    rows = sum(1 for _ in out_path.open("r", encoding="utf-8")) - 1
    return rows > 0, f"exported {rows} rows"


def phase_extended_oos(features_2024: pd.DataFrame) -> dict:
    extended_path = OUT / "features_2025_2026_oos.csv"
    if extended_path.exists() and extended_path.stat().st_size > 100:
        ok = True
        rows = sum(1 for _ in extended_path.open("r", encoding="utf-8")) - 1
        message = f"reused cached export with {rows} rows"
    else:
        ok, message = export_features_from_db(date(2025, 1, 1), date(2026, 3, 31), extended_path)
    classification = {
        "period": "2025-01-01 to 2026-03-31",
        "requested_role": "extended OOS / live-shadow candidate",
        "status": "not_evaluated",
        "classification": "blocked_by_data_availability",
        "data_message": message,
        "caveat": "Even if available, this period is not a pristine holdout unless development history proves it was not inspected during model design.",
    }
    if not ok:
        pd.DataFrame([classification]).to_csv(OUT / "extended_oos_classification_2025_2026.csv", index=False)
        return classification

    ext = read_features(extended_path)
    if ext.empty or ext[LABEL].notna().sum() == 0:
        classification["classification"] = "available_but_unlabelled_or_empty"
        classification["data_message"] = f"{message}; labelled rows={int(ext[LABEL].notna().sum()) if LABEL in ext else 0}"
        pd.DataFrame([classification]).to_csv(OUT / "extended_oos_classification_2025_2026.csv", index=False)
        return classification

    combined = pd.concat([features_2024, ext], ignore_index=True).sort_values(["date", "ticker"])
    all_cols = feature_cols(combined)
    slices = month_slices("2025-01-01", "2026-03-01")
    preds, daily, summary = run_walk_forward_experiment(
        combined, slices, all_cols, LABEL, "extended_oos_2025_2026"
    )
    classification.update(
        {
            "status": "evaluated",
            "classification": "retrospective_extended_oos_candidate_not_pristine_live_holdout",
            "rows": int(len(ext)),
            "labelled_rows": int(ext[LABEL].notna().sum()),
            "date_min": ext["date"].min().date().isoformat(),
            "date_max": ext["date"].max().date().isoformat(),
            **summary,
        }
    )
    preds.to_csv(OUT / "extended_oos_predictions_2025_2026.csv", index=False)
    daily.to_csv(OUT / "extended_oos_daily_ic_2025_2026.csv", index=False)
    pd.DataFrame([classification]).to_csv(OUT / "extended_oos_classification_2025_2026.csv", index=False)
    return classification


def phase_survivorship_audit(features: pd.DataFrame) -> dict:
    coverage = (
        features.groupby("ticker")
        .agg(
            first_date=("date", "min"),
            last_date=("date", "max"),
            n_rows=("date", "size"),
            n_labelled=(LABEL, lambda s: int(s.notna().sum())),
            positive_label_rate=(LABEL, lambda s: float((s > 0).mean())),
        )
        .reset_index()
    )
    coverage["first_date"] = coverage["first_date"].dt.date.astype(str)
    coverage["last_date"] = coverage["last_date"].dt.date.astype(str)
    coverage.to_csv(OUT / "survivorship_universe_coverage.csv", index=False)

    audit_rows = [
        {
            "audit_item": "Universe construction",
            "finding": "The reproduced snapshot contains 12 seed tickers throughout 2019-2024.",
            "risk": "If these were selected using present-day liquidity or survival, results may overstate historical investability.",
            "status": "risk_identified_not_eliminated",
        },
        {
            "audit_item": "Delisted/failed securities",
            "finding": "No delisted or failed NSE securities are present in the local reproduction snapshot.",
            "risk": "Absence of failures is a survivorship-bias risk for claims about the full NSE.",
            "status": "not_a_full_exchange_universe",
        },
        {
            "audit_item": "Per-ticker continuity",
            "finding": f"{int((coverage['n_rows'] == coverage['n_rows'].max()).sum())} of {len(coverage)} tickers have the modal row count.",
            "risk": "Continuity supports balanced-panel analysis but does not prove ex-ante investability.",
            "status": "checked",
        },
        {
            "audit_item": "Tradability flags",
            "finding": "The exported snapshot has dq_tradable false for all rows, so tradability cannot be audited from that flag.",
            "risk": "Portfolio capacity and liquidity conclusions must use separate liquidity fields or source price data.",
            "status": "data_quality_limitation",
        },
    ]
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(OUT / "survivorship_audit.csv", index=False)
    return {
        "n_tickers": int(features["ticker"].nunique()),
        "date_min": features["date"].min().date().isoformat(),
        "date_max": features["date"].max().date().isoformat(),
        "audit_status": "risk_identified_not_eliminated",
        "coverage": coverage.to_dict(orient="records"),
    }


def phase_permutation_importance(features: pd.DataFrame) -> dict:
    rng = np.random.default_rng(42)
    all_cols = feature_cols(features)
    groups = feature_groups(all_cols)
    slices = month_slices("2024-01-01", "2024-12-01")
    rows = []

    # Group permutation importance. Retrain once per month, then permute test columns.
    for slc in slices:
        train = features[
            (features["date"] >= slc.train_start) & (features["date"] <= slc.train_end)
        ].dropna(subset=[LABEL])
        test = features[
            (features["date"] >= slc.test_start) & (features["date"] <= slc.test_end)
        ].dropna(subset=[LABEL])
        if len(train) < 50 or test.empty:
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            lgb.LGBMRegressor(**{**LGB_PARAMS, "random_state": 42, "seed": 42}),
        )
        model.fit(train[all_cols], train[LABEL])
        base = test[["date", "ticker", LABEL]].copy()
        base["raw_score"] = model.predict(test[all_cols])
        base["score"] = base.groupby("date")["raw_score"].transform(
            lambda s: (s - s.min()) / (s.max() - s.min() + 1e-10)
        )
        base_daily = daily_ic(base, LABEL)
        base_ic = base_daily["ic"].mean()
        rows.append(
            {
                "slice_id": slc.slice_id,
                "feature_group": "__baseline__",
                "n_features": len(all_cols),
                "base_mean_daily_ic": base_ic,
                "permuted_mean_daily_ic": base_ic,
                "ic_drop": 0.0,
            }
        )
        for group_name, cols in groups.items():
            cols = [c for c in cols if c in all_cols]
            if not cols:
                continue
            permuted = test[all_cols].copy()
            for col in cols:
                permuted[col] = rng.permutation(permuted[col].to_numpy())
            pred = test[["date", "ticker", LABEL]].copy()
            pred["raw_score"] = model.predict(permuted)
            pred["score"] = pred.groupby("date")["raw_score"].transform(
                lambda s: (s - s.min()) / (s.max() - s.min() + 1e-10)
            )
            perm_daily = daily_ic(pred, LABEL)
            perm_ic = perm_daily["ic"].mean()
            rows.append(
                {
                    "slice_id": slc.slice_id,
                    "feature_group": group_name,
                    "n_features": len(cols),
                    "base_mean_daily_ic": base_ic,
                    "permuted_mean_daily_ic": perm_ic,
                    "ic_drop": base_ic - perm_ic,
                }
            )

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "permutation_importance_group_detail_2024.csv", index=False)
    summary = (
        detail[detail["feature_group"] != "__baseline__"]
        .groupby("feature_group")
        .agg(
            n_slices=("slice_id", "nunique"),
            mean_ic_drop=("ic_drop", "mean"),
            median_ic_drop=("ic_drop", "median"),
            mean_permuted_ic=("permuted_mean_daily_ic", "mean"),
            mean_n_features=("n_features", "mean"),
        )
        .reset_index()
        .sort_values("mean_ic_drop", ascending=False)
    )
    summary.to_csv(OUT / "permutation_importance_group_summary_2024.csv", index=False)
    return {"top_groups": summary.head(8).to_dict(orient="records")}



def main() -> None:
    t0 = time.time()
    features = read_features(REPRO / "features_daily_snapshot.csv")
    results = {
        "input_snapshot": {
            "path": str(REPRO / "features_daily_snapshot.csv"),
            "rows": int(len(features)),
            "date_min": features["date"].min().date().isoformat(),
            "date_max": features["date"].max().date().isoformat(),
            "tickers": sorted(features["ticker"].unique().tolist()),
        }
    }
    results["phase_9_10_portfolio_costs"] = phase_portfolio_and_costs(features)
    results["phase_11_feature_ablations"] = phase_feature_ablations(features)
    results["phase_12_robustness"] = phase_robustness(features)
    results["phase_13_extended_oos"] = phase_extended_oos(features)
    results["phase_14_survivorship"] = phase_survivorship_audit(features)
    results["phase_15_permutation_importance"] = phase_permutation_importance(features)
    results["runtime_seconds"] = round(time.time() - t0, 2)
    (OUT / "phases_9_15_summary.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
