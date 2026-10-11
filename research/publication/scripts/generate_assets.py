from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PUB = ROOT / "publication"
OUT = ROOT / "outputs"
PHASE = OUT / "phases_9_15"
REPRO = OUT / "reproduction"
BASE = OUT / "baselines"
DAILY = OUT / "daily_ic"

FIG = PUB / "figures"
TAB = PUB / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def fmt(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return "--"
    return f"{x:.{digits}f}"


def pct(x: float, digits: int = 1) -> str:
    if pd.isna(x):
        return "--"
    return f"{100 * x:.{digits}f}\\%"


def tex_escape(s: str) -> str:
    return str(s).replace("_", "\\_").replace("%", "\\%")


def write_table(path: Path, caption: str, label: str, cols: list[str], rows: list[list[str]]) -> None:
    spec = "l" + "r" * (len(cols) - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\small",
        f"\\begin{{tabular}}{{@{{}}{spec}@{{}}}}",
        "\\toprule",
        " & ".join(cols) + " \\\\",
        "\\midrule",
    ]
    lines += [" & ".join(row) + " \\\\" for row in rows]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def table_dataset() -> None:
    features = pd.read_csv(REPRO / "features_daily_snapshot.csv", usecols=["date", "ticker", "label_3d_net_alpha"])
    ext = pd.read_csv(PHASE / "extended_oos_classification_2025_2026.csv")
    rows = [
        ["Securities", f"{features['ticker'].nunique():d}"],
        ["2019--2024 rows", f"{len(features):,}"],
        ["2019--2024 dates", f"{features['date'].min()} to {features['date'].max()}"],
        ["2024 test security-days", "3,048"],
        ["2024 test trading days", "254"],
        ["2025--2026 extended rows", f"{int(ext.loc[0, 'rows']):,}"],
        ["2025--2026 extended trading days", f"{int(ext.loc[0, 'n_days'])}"],
        ["Primary target", "\\texttt{label\\_3d\\_net\\_alpha}"],
    ]
    write_table(
        TAB / "dataset_summary.tex",
        "Dataset and evaluation summary.",
        "tab:dataset",
        ["Quantity", "Value"],
        rows,
    )


def table_model_comparison() -> None:
    df = pd.read_csv(BASE / "baseline_comparison_2024.csv")
    order = [
        "Random Forest",
        "LightGBM",
        "1-day reversal",
        "3-day reversal",
        "Ridge regression",
        "Linear regression",
        "Random ranking",
        "5-day momentum",
        "20-day momentum",
    ]
    df["model"] = pd.Categorical(df["model"], order, ordered=True)
    df = df.sort_values("model")
    rows = []
    for _, r in df.iterrows():
        rows.append(
            [
                tex_escape(r["model"]),
                fmt(r["mean_daily_ic"]),
                fmt(r["median_daily_ic"]),
                fmt(r["icir"]),
                f"{int(r['positive_months'])}/{int(r['n_months'])}",
                pct(r["top1_hit_rate"], 1),
            ]
        )
    write_table(
        TAB / "model_comparison.tex",
        "Model comparison on the 2024 retrospective out-of-sample period.",
        "tab:model_comparison",
        ["Model", "Mean IC", "Median IC", "ICIR", "Pos. months", "Top-1 hit"],
        rows,
    )


def table_inference() -> None:
    df = pd.read_csv(DAILY / "lightgbm_hac_inference_2024.csv")
    r = df.iloc[0]
    rows = [
        ["Days", f"{int(r['n_days'])}"],
        ["Mean daily IC", fmt(r["mean_ic"], 4)],
        ["Median daily IC", fmt(r["median_ic"], 4)],
        ["IC standard deviation", fmt(r["ic_std"], 4)],
        ["ICIR", fmt(r["icir"], 4)],
        ["HAC lag", f"{int(r['hac_lag'])}"],
        ["HAC standard error", fmt(r["hac_se"], 4)],
        ["95\\% CI", f"[{fmt(r['ci95_low'], 4)}, {fmt(r['ci95_high'], 4)}]"],
        ["t-statistic", fmt(r["t_stat"], 3)],
        ["p-value", f"{r['p_value']:.2e}"],
    ]
    write_table(
        TAB / "statistical_inference.tex",
        "Dependence-aware inference for the 2024 LightGBM daily IC series.",
        "tab:inference",
        ["Statistic", "Value"],
        rows,
    )


def table_ablations() -> None:
    df = pd.read_csv(PHASE / "feature_ablation_summary_2024.csv")
    order = [
        "returns_price_only",
        "full_feature_set",
        "momentum_reversal",
        "nse_microstructure",
        "volume_liquidity",
        "macro_all",
        "kes_usd_only",
        "llm_events",
    ]
    names = {
        "returns_price_only": "Returns/price only",
        "full_feature_set": "Full feature set",
        "momentum_reversal": "Momentum/reversal",
        "nse_microstructure": "NSE microstructure",
        "volume_liquidity": "Volume/liquidity",
        "macro_all": "Macro variables",
        "kes_usd_only": "KES/USD only",
        "llm_events": "LLM/event variables",
    }
    df["model"] = pd.Categorical(df["model"], order, ordered=True)
    df = df.sort_values("model")
    rows = []
    for _, r in df.iterrows():
        rows.append(
            [
                names[str(r["model"])],
                fmt(r["mean_daily_ic"]),
                fmt(r["median_daily_ic"]),
                f"{int(r['positive_months'])}/{int(r['n_months'])}" if r["n_months"] else "--",
                pct(r["top1_hit_rate"], 1),
                f"{int(r['n_features'])}",
            ]
        )
    write_table(
        TAB / "feature_ablations.tex",
        "Feature-group ablations under the 2024 walk-forward protocol.",
        "tab:ablations",
        ["Feature set", "Mean IC", "Median IC", "Pos. months", "Top-1 hit", "Features"],
        rows,
    )


def table_extended_oos() -> None:
    r = pd.read_csv(PHASE / "extended_oos_classification_2025_2026.csv").iloc[0]
    rows = [
        ["Classification", "Retrospective extended OOS"],
        ["Period", "2025-01-02 to 2026-03-31"],
        ["Rows", f"{int(r['rows']):,}"],
        ["Trading days", f"{int(r['n_days'])}"],
        ["Mean daily IC", fmt(r["mean_daily_ic"])],
        ["Median daily IC", fmt(r["median_daily_ic"])],
        ["Positive months", f"{int(r['positive_months'])}/{int(r['n_months'])}"],
        ["Top-1 hit rate", pct(r["top1_hit_rate"], 1)],
    ]
    write_table(
        TAB / "extended_oos.tex",
        "Extended 2025--2026 retrospective out-of-sample evaluation.",
        "tab:extended_oos",
        ["Statistic", "Value"],
        rows,
    )


def table_portfolio() -> None:
    df = pd.read_csv(PHASE / "portfolio_cost_sensitivity_2024.csv")
    keep = df[((df["top_k"].isin([1, 2, 3])) & (df["cost_bps"].isin([0, 100, 400, 600])))]
    rows = []
    for _, r in keep.sort_values(["top_k", "cost_bps"]).iterrows():
        rows.append(
            [
                f"Top-{int(r['top_k'])}",
                f"{int(r['cost_bps'])}",
                pct(r["total_return"], 1),
                fmt(r["sharpe"], 2),
                pct(r["max_drawdown"], 1),
                pct(r["sleeve_hit_rate"], 1),
                pct(r["mean_sleeve_net_3d_return"], 2),
            ]
        )
    write_table(
        TAB / "portfolio_costs.tex",
        "Portfolio and transaction-cost sensitivity for overlapping three-day sleeves.",
        "tab:portfolio",
        ["Rule", "Cost bps", "Total ret.", "Sharpe", "Max DD", "Sleeve hit", "Mean sleeve ret."],
        rows,
    )


def figure_monthly_ic() -> None:
    df = pd.read_csv(DAILY / "daily_ic_all_models_2024.csv")
    models = ["Random Forest", "LightGBM", "1-day reversal", "Random ranking"]
    monthly = df[df["model"].isin(models)].groupby(["month", "model"])["ic"].mean().reset_index()
    piv = monthly.pivot(index="month", columns="model", values="ic")
    ax = piv[models].plot(figsize=(7.2, 3.6), marker="o", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Monthly mean daily IC")
    ax.set_xlabel("")
    ax.set_title("Monthly cross-sectional IC by model, 2024")
    ax.legend(ncol=2, frameon=False)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG / "fig1_monthly_ic_by_model.pdf")
    plt.close()


def figure_daily_rolling_ic() -> None:
    df = pd.read_csv(REPRO / "walk_forward_2024_daily_ic.csv")
    test = df[df["split"].eq("test")].copy()
    test["date"] = pd.to_datetime(test["date"])
    test = test.sort_values("date")
    test["rolling_21d_ic"] = test["ic"].rolling(21, min_periods=10).mean()
    ext = pd.read_csv(PHASE / "extended_oos_daily_ic_2025_2026.csv")
    ext["date"] = pd.to_datetime(ext["date"])
    ext = ext.sort_values("date")
    ext["rolling_21d_ic"] = ext["ic"].rolling(21, min_periods=10).mean()
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.plot(test["date"], test["rolling_21d_ic"], label="2024 LightGBM rolling 21d IC", linewidth=1.6)
    ax.plot(ext["date"], ext["rolling_21d_ic"], label="2025--2026 extended rolling 21d IC", linewidth=1.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Rolling mean daily IC")
    ax.set_xlabel("")
    ax.set_title("Rolling out-of-sample IC")
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(FIG / "fig2_rolling_ic.pdf")
    plt.close()


def figure_model_comparison() -> None:
    df = pd.read_csv(BASE / "baseline_comparison_2024.csv").sort_values("mean_daily_ic")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    colors = ["#4c78a8" if m in ["Random Forest", "LightGBM"] else "#9ecae9" for m in df["model"]]
    ax.barh(df["model"], df["mean_daily_ic"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean daily IC")
    ax.set_title("Model comparison, 2024")
    plt.tight_layout()
    plt.savefig(FIG / "fig3_model_comparison.pdf")
    plt.close()


def figure_ablations() -> None:
    df = pd.read_csv(PHASE / "feature_ablation_summary_2024.csv")
    df = df.sort_values("mean_daily_ic", na_position="first")
    labels = [s.replace("_", " ") for s in df["model"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    colors = ["#f58518" if s == "returns_price_only" else "#4c78a8" if s == "full_feature_set" else "#9ecae9" for s in df["model"]]
    ax.barh(labels, df["mean_daily_ic"].fillna(0), color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean daily IC")
    ax.set_title("Feature-group ablations")
    plt.tight_layout()
    plt.savefig(FIG / "fig4_feature_ablations.pdf")
    plt.close()


def figure_permutation() -> None:
    df = pd.read_csv(PHASE / "permutation_importance_group_summary_2024.csv").sort_values("mean_ic_drop")
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    ax.barh(df["feature_group"].str.replace("_", " "), df["mean_ic_drop"], color="#59a14f")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean IC drop after permutation")
    ax.set_title("Group permutation importance")
    plt.tight_layout()
    plt.savefig(FIG / "fig5_permutation_importance.pdf")
    plt.close()


def figure_cost_sensitivity() -> None:
    df = pd.read_csv(PHASE / "portfolio_cost_sensitivity_2024.csv")
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    for k in [1, 2, 3, 5]:
        sub = df[df["top_k"].eq(k)].sort_values("cost_bps")
        ax.plot(sub["cost_bps"], 100 * sub["total_return"], marker="o", linewidth=1.6, label=f"Top-{k}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Round-trip cost (bps)")
    ax.set_ylabel("2024 cumulative return (%)")
    ax.set_title("Transaction-cost sensitivity")
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(FIG / "fig6_cost_sensitivity.pdf")
    plt.close()


def figure_cumulative_returns() -> None:
    df = pd.read_csv(PHASE / "portfolio_daily_returns_2024.csv", parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    for k, cost in [(1, 0), (1, 400), (3, 400), (5, 400)]:
        sub = df[(df["top_k"].eq(k)) & (df["cost_bps"].eq(cost))].sort_values("date")
        equity = (1 + sub["daily_return"]).cumprod() - 1
        ax.plot(sub["date"], 100 * equity, linewidth=1.5, label=f"Top-{k}, {cost} bps")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Cumulative return (%)")
    ax.set_xlabel("")
    ax.set_title("Portfolio cumulative-return curves")
    ax.legend(frameon=False, ncol=2)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(FIG / "fig7_cumulative_returns.pdf")
    plt.close()


def figure_drawdowns() -> None:
    df = pd.read_csv(PHASE / "portfolio_daily_returns_2024.csv", parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    for k, cost in [(1, 400), (2, 400), (3, 400), (5, 400)]:
        sub = df[(df["top_k"].eq(k)) & (df["cost_bps"].eq(cost))].sort_values("date")
        equity = (1 + sub["daily_return"]).cumprod()
        dd = equity / equity.cummax() - 1
        ax.plot(sub["date"], 100 * dd, linewidth=1.4, label=f"Top-{k}, 400 bps")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("")
    ax.set_title("Drawdowns at 400 bps")
    ax.legend(frameon=False, ncol=2)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(FIG / "fig8_drawdowns.pdf")
    plt.close()


def main() -> None:
    table_dataset()
    table_model_comparison()
    table_inference()
    table_ablations()
    table_extended_oos()
    table_portfolio()
    figure_monthly_ic()
    figure_daily_rolling_ic()
    figure_model_comparison()
    figure_ablations()
    figure_permutation()
    figure_cost_sensitivity()
    figure_cumulative_returns()
    figure_drawdowns()
    print("Generated publication tables and figures")


if __name__ == "__main__":
    main()
