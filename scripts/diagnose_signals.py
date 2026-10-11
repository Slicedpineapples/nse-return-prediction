#!/usr/bin/env python
"""
Diagnostic script to check signal backfill status and data quality.

Usage:
    python -m scripts.diagnose_signals
"""

from datetime import date, timedelta
import pandas as pd
from sqlalchemy import text

from nse.db.schema import session_scope, SignalHistory, FeaturesDaily, MarketDaily

def main():
    print("=" * 80)
    print("NSE SIGNAL DIAGNOSTICS")
    print("=" * 80)
    
    with session_scope() as s:
        # 1. Signal coverage
        print("\n1. SIGNAL COVERAGE")
        print("-" * 80)
        result = s.execute(
            text("""
            SELECT 
                COUNT(*) as total_signals,
                COUNT(CASE WHEN outcome_filled_at IS NOT NULL THEN 1 END) as filled_outcomes,
                COUNT(CASE WHEN outcome_filled_at IS NULL THEN 1 END) as unfilled_outcomes,
                MIN(signal_date) as earliest_signal,
                MAX(signal_date) as latest_signal
            FROM signal_history
            """)
        )
        for row in result:
            print(f"  Total signals:       {row[0]}")
            print(f"  Filled outcomes:     {row[1]} ({100*row[1]/row[0]:.1f}%)" if row[0] > 0 else "")
            print(f"  Unfilled outcomes:   {row[2]}")
            print(f"  Date range:          {row[3]} → {row[4]}")
        
        # 2. Trading calendar
        print("\n2. TRADING CALENDAR")
        print("-" * 80)
        result = s.execute(
            text("""
            SELECT 
                COUNT(DISTINCT date) as trading_days,
                MIN(date) as earliest_date,
                MAX(date) as latest_date
            FROM market_daily
            """)
        )
        for row in result:
            print(f"  Trading days recorded:     {row[0]}")
            print(f"  Date range:                {row[1]} → {row[2]}")
        
        # 3. Backfill eligibility
        print("\n3. BACKFILL ELIGIBILITY")
        print("-" * 80)
        cutoff = date.today() - timedelta(days=10)
        result = s.execute(
            text(f"""
            SELECT 
                sh.signal_date,
                COUNT(*) as n_signals,
                COUNT(CASE WHEN sh.outcome_filled_at IS NOT NULL THEN 1 END) as filled,
                MAX(md.date) as latest_market_date,
                MAX(md.date) - sh.signal_date as max_days_after
            FROM signal_history sh
            LEFT JOIN market_daily md ON md.date > sh.signal_date
            WHERE sh.signal_date >= '{cutoff}'
            GROUP BY sh.signal_date
            ORDER BY sh.signal_date DESC
            """)
        )
        rows_list = list(result)
        if rows_list:
            print(f"  Recent signals (last 10 days from {cutoff}):")
            for row in rows_list:
                filled_pct = 100*row[2]/row[1] if row[1] > 0 else 0
                print(f"    {row[0]}: {row[1]} signals, {row[2]} filled ({filled_pct:.0f}%), " +
                      f"max days after: {row[4]}")
        else:
            print("  No signals in recent period.")
        
        # 4. Data quality in features
        print("\n4. DATA QUALITY FLAGS (features_daily)")
        print("-" * 80)
        result = s.execute(
            text("""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(CASE WHEN dq_price_ok THEN 1 END) as dq_price_ok,
                COUNT(CASE WHEN dq_features_ok THEN 1 END) as dq_features_ok,
                COUNT(CASE WHEN dq_labels_ok THEN 1 END) as dq_labels_ok,
                COUNT(CASE WHEN dq_tradable THEN 1 END) as dq_tradable
            FROM features_daily
            WHERE date >= DATE(NOW()) - INTERVAL '30 days'
            """)
        )
        for row in result:
            if row[0] > 0:
                print(f"  Last 30 days (total {row[0]} rows):")
                print(f"    dq_price_ok:     {row[1]:>6} ({100*row[1]/row[0]:>5.1f}%)")
                print(f"    dq_features_ok:  {row[2]:>6} ({100*row[2]/row[0]:>5.1f}%)")
                print(f"    dq_labels_ok:    {row[3]:>6} ({100*row[3]/row[0]:>5.1f}%)")
                print(f"    dq_tradable:     {row[4]:>6} ({100*row[4]/row[0]:>5.1f}%)")
        
        # 5. Label distribution
        print("\n5. LABEL DISTRIBUTION")
        print("-" * 80)
        result = s.execute(
            text("""
            SELECT 
                MIN(label_3d_net_alpha) as min_label,
                MAX(label_3d_net_alpha) as max_label,
                AVG(label_3d_net_alpha) as mean_label,
                COUNT(CASE WHEN label_3d_net_alpha > 0 THEN 1 END) as rows_positive,
                COUNT(CASE WHEN label_3d_net_alpha > 0 THEN 1 END) * 100.0 / 
                    COUNT(*) as pct_positive
            FROM features_daily
            WHERE dq_labels_ok = TRUE
            """)
        )
        for row in result:
            print(f"  3d net alpha (with dq_labels_ok=True):")
            print(f"    Min:         {row[0]:>10.2f} bps" if row[0] else "    {}: N/A".format("Min"))
            print(f"    Max:         {row[1]:>10.2f} bps" if row[1] else "")
            print(f"    Mean:        {row[2]:>10.2f} bps" if row[2] else "")
            print(f"    Positive %:  {row[4]:>10.1f}%" if row[4] else "")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("""
1. If outcome fill rate is low:
   - Check if market_daily has recent data (needed for backfill_actuals)
   - Run: python -m nse.pipeline.daily --as-of <date>
   - Then: nse-monitor --start <date>

2. If dq_labels_ok is low:
   - Check if historical prices are complete
   - Labels require 3 trading days of forward data

3. If mean label is negative:
   - Market may be in drawdown
   - Or cost assumptions are too high
   
4. If dq_features_ok is low:
   - Check data ingestion pipeline
   - Run: nse-ingest --start 2025-01-01
    """)

if __name__ == "__main__":
    main()
