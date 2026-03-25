"""
Cost model for backtesting.

Three scenarios: optimistic, base, stressed.

All cost estimates are conservative for NSE:
  - Broker commission: ~1% one-way (NSE charges ~0.12% exchange fee,
    ~0.1% CDS levy, stamp duty, plus broker commission; total ~1.7-2% round-trip)
  - Slippage: varies by liquidity bucket.
  - Impact: bounded by fraction of ADV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nse.config import COSTS


SCENARIOS = {
    "optimistic": COSTS["scenario_optimistic_bps"] / 10_000,
    "base": COSTS["scenario_base_bps"] / 10_000,
    "stressed": COSTS["scenario_stressed_bps"] / 10_000,
}


def estimate_cost_bps(
    value_traded: float,
    adv_20d: float,
    scenario: str = "base",
) -> float:
    """
    Estimate round-trip cost in basis points for a single trade.

    Parameters
    ----------
    value_traded : KES value of this order
    adv_20d      : 20-day average daily value traded in KES
    scenario     : 'optimistic' | 'base' | 'stressed'

    Returns
    -------
    Cost in bps (round-trip).
    """
    # Base cost from scenario
    base_bps = SCENARIOS[scenario] * 10_000

    # Slippage: scale by how much of ADV we're using
    adv_fraction = value_traded / (adv_20d + 1e-6)
    if adv_fraction < 0.01:
        slippage_bps = COSTS["slippage_liquid_bps"]
    elif adv_fraction < 0.03:
        slippage_bps = COSTS["slippage_mid_bps"]
    else:
        slippage_bps = COSTS["slippage_thin_bps"]

    total_bps = COSTS["commission_bps"] * 2 + slippage_bps  # round-trip commission + slippage

    # Stress overlay
    if scenario == "stressed":
        total_bps *= 1.5
    elif scenario == "optimistic":
        total_bps *= 0.7

    return total_bps


def apply_costs(gross_return: float, cost_bps: float) -> float:
    """Convert gross return to net return after costs."""
    return gross_return - cost_bps / 10_000


def max_order_size_kes(adv_20d: float) -> float:
    """Maximum order size in KES (5% of 20d ADV)."""
    return adv_20d * COSTS["impact_adv_fraction"]
