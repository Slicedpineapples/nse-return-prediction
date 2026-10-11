from __future__ import annotations

from pathlib import Path

import pandas as pd

PUB = Path("research/publication")
PUB.mkdir(parents=True, exist_ok=True)

repro = pd.read_json("research/outputs/reproduction/manifest.json")
summary = pd.read_csv("research/outputs/reproduction/walk_forward_2024_summary.csv")
baseline = pd.read_csv("research/outputs/baselines/baseline_comparison_2024.csv")
infer = pd.read_csv("research/outputs/daily_ic/lightgbm_hac_inference_2024.csv").iloc[0]
hit = pd.read_csv("research/outputs/hit_rate/hit_rate_comparison_2024.csv")

top_models = baseline[
    [
        "model",
        "mean_daily_ic",
        "median_daily_ic",
        "icir",
        "positive_months",
        "top1_hit_rate",
    ]
].copy()
top_models.columns = [
    "Model",
    "Mean daily IC",
    "Median IC",
    "ICIR",
    "Positive months",
    "Top-1 hit rate",
]

monthly = summary[["slice_id", "val_ic", "test_ic", "test_hit_rate", "n_train"]].copy()
monthly.columns = ["Month", "Validation IC", "Test IC", "Top-1 hit rate", "N train"]

def tex_table(df: pd.DataFrame, float_fmt: str = "%.3f") -> str:
    return df.to_latex(index=False, escape=True, float_format=lambda x: float_fmt % x)

baseline_table = tex_table(top_models)
monthly_table = tex_table(monthly)
hit_table = tex_table(hit.rename(columns={"metric": "Metric", "count": "Count", "n": "N", "rate": "Rate"}))

def pct(x: float) -> str:
    return f"{x:.1%}".replace("%", r"\%")

paper = rf"""
\documentclass[11pt,a4paper]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{hyperref}}
\usepackage{{amsmath}}
\usepackage{{lmodern}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{6pt}}

\title{{Cross-Sectional Return Prediction in a Small, Illiquid Equity Market:\\Machine Learning Evidence from the Nairobi Securities Exchange}}
\author{{Felix Nyabuto}}
\date{{Research Validation Draft, generated from reproducible artifacts}}

\begin{{document}}
\maketitle

\begin{{abstract}}
This validation draft re-evaluates the 2024 walk-forward ranking evidence for a
12-security Nairobi Securities Exchange universe.  Under the current code and
data snapshot, the LightGBM model produces mean monthly out-of-sample
cross-sectional Spearman IC of {summary['test_ic'].mean():.3f}, with positive
monthly IC in {(summary['test_ic'] > 0).sum()} of {len(summary)} months.  The
daily IC series has mean {infer['mean_ic']:.3f}; HAC/Newey--West inference with
lag {int(infer['hac_lag'])} gives a 95\% confidence interval of
[{infer['ci95_low']:.3f}, {infer['ci95_high']:.3f}] and p-value
{infer['p_value']:.2e}.  Baseline experiments show that Random Forest achieves
a higher mean daily IC ({baseline.iloc[0]['mean_daily_ic']:.3f}) than LightGBM
({baseline[baseline['model'] == 'LightGBM'].iloc[0]['mean_daily_ic']:.3f}).
Economic portfolio performance, transaction-cost sensitivity, feature-group
ablations, and robustness tests remain outstanding; therefore this draft
supports a limited claim of ranking predictability, not a claim of deployable
profitability.
\end{{abstract}}

\section{{Introduction}}

This report replaces the prior unreproduced manuscript value of approximately
0.091 mean 2024 out-of-sample IC with values generated from the current
machine-readable research artifacts.  The purpose is methodological
validation: to determine what the current code and data support, and to
separate measured ranking performance from untested economic claims.

\section{{Data and Reproduction}}

The reproduction uses a local snapshot exported from \texttt{{features\_daily}}
for 2019-01-01 through 2024-12-31.  The snapshot contains 18,240 feature rows.
The target is the implementation variable \texttt{{label\_3d\_net\_alpha}}.
Because benchmark availability remains incomplete, this label should be read
cautiously as net forward return where no benchmark adjustment is available.

\section{{Walk-Forward Results}}

\begin{{table}}[h]
\centering
\caption{{Monthly LightGBM walk-forward results generated from research outputs.}}
\small
{monthly_table}
\end{{table}}

The current reproduction gives mean monthly test IC of {summary['test_ic'].mean():.3f}.
This differs from the earlier manuscript value of approximately 0.091 and
should replace it unless an archived experiment explains the discrepancy.

\section{{Daily IC Inference}}

The LightGBM daily IC series contains {int(infer['n_days'])} valid days.  Mean
daily IC is {infer['mean_ic']:.3f}, median IC is {infer['median_ic']:.3f}, and
ICIR is {infer['icir']:.3f}.  HAC/Newey--West standard error with lag
{int(infer['hac_lag'])} is {infer['hac_se']:.4f}, giving t-statistic
{infer['t_stat']:.3f} and p-value {infer['p_value']:.2e}.

\section{{Baseline Comparison}}

\begin{{table}}[h]
\centering
\caption{{2024 ranking baselines.  Values are computed from saved prediction artifacts.}}
\small
{baseline_table}
\end{{table}}

The LightGBM model outperforms random ranking, simple reversal/momentum rules,
linear regression, and Ridge regression by mean daily IC.  It does not
outperform the Random Forest baseline in this run.  The manuscript should
therefore avoid claiming that LightGBM is the best model or that gradient
boosting is necessary for the observed predictability.

\section{{Hit-Rate Baseline}}

\begin{{table}}[h]
\centering
\caption{{Empirical positive net-forward-return rates.}}
\small
{hit_table}
\end{{table}}

The unconditional security-day positive rate is {pct(hit.iloc[0]['rate'])}.
LightGBM Top-1 improves on this to {pct(hit.iloc[1]['rate'])}, but equal-weight
Top-2 and Top-3 selections are lower than the unconditional rate.  The earlier
50\% hit-rate comparison is not appropriate.

\section{{Limitations}}

The current validation is incomplete.  It does not include transaction-cost
sensitivity, portfolio construction, feature-group ablations, extended
out-of-sample evaluation, robustness tests, SHAP or permutation importance, or
a full survivorship-bias audit.  Several features are entirely missing in the
2019--2024 window, including LLM and intraday fields; the 2024 predictive
result therefore cannot be attributed to those components.

\section{{Conclusion}}

The defensible current claim is narrow: the current LightGBM implementation
shows positive 2024 cross-sectional ranking information in the studied
12-security NSE universe, but it is not the strongest tested model and the
economic usefulness of the signal remains unproven.  Random Forest is the
strongest model in the completed baseline comparison.  Future manuscript
versions should be rewritten around these findings unless additional
reproducible experiments change the evidence.

\end{{document}}
"""

(PUB / "paper.tex").write_text(paper, encoding="utf-8")
print(PUB / "paper.tex")
