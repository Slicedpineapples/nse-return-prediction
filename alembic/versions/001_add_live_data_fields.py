"""Add live_stock_data fields to prices_daily and features_daily; create prices_intraday.

Revision ID: 001
Revises: (none)
Create Date: 2026-04-21
"""

from alembic import op
import sqlalchemy as sa

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- prices_daily: new columns from live_stock_data ---
    with op.batch_alter_table("prices_daily") as batch_op:
        batch_op.add_column(sa.Column("foreign_buy_kes",    sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("foreign_sell_kes",   sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("total_trades_count", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("wap",                sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("market_cap_kes",     sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("bid_price_eod",      sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("offer_price_eod",    sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("total_bid_qty",      sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("total_offer_qty",    sa.Float(), nullable=True))

    # --- features_daily: order book + intraday columns ---
    with op.batch_alter_table("features_daily") as batch_op:
        batch_op.add_column(sa.Column("bid_ask_imbalance",          sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("order_book_depth_ratio",      sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("spread_bps",                  sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("wap_vs_close_bps",            sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("foreign_net_flow_pct",        sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("foreign_participation_pct",   sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("session_return_pct",          sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("morning_return_pct",          sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("afternoon_return_pct",        sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("volume_morning_pct",          sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("intraday_high_low_range_pct", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("session_snapshot_count",      sa.Integer(), nullable=True))

    # --- prices_intraday: new table ---
    op.create_table(
        "prices_intraday",
        sa.Column("id",              sa.Integer(),                     primary_key=True, autoincrement=True),
        sa.Column("scraped_at",      sa.DateTime(timezone=True),       nullable=False),
        sa.Column("date",            sa.Date(),                        nullable=False),
        sa.Column("ticker",          sa.String(16),                    nullable=False),
        sa.Column("last_price",      sa.Float()),
        sa.Column("bid_price",       sa.Float()),
        sa.Column("bid_qty",         sa.Float()),
        sa.Column("offer_price",     sa.Float()),
        sa.Column("offer_qty",       sa.Float()),
        sa.Column("total_bid_qty",   sa.Float()),
        sa.Column("total_offer_qty", sa.Float()),
        sa.Column("total_trades",    sa.Float()),
        sa.Column("volume",          sa.Float()),
        sa.Column("turnover",        sa.Float()),
        sa.Column("foreign_buy",     sa.Float()),
        sa.Column("foreign_sell",    sa.Float()),
        sa.Column("wap",             sa.Float()),
        sa.Column("market_cap",      sa.Float()),
        sa.Column("day_open",        sa.Float()),
        sa.Column("day_high",        sa.Float()),
        sa.Column("day_low",         sa.Float()),
        sa.Column("day_close",       sa.Float()),
        sa.Column("prev_close",      sa.Float()),
        sa.Column("net_change",      sa.Float()),
        sa.UniqueConstraint("scraped_at", "ticker", name="uq_intraday_ts_ticker"),
    )
    op.create_index("ix_prices_intraday_date",   "prices_intraday", ["date"])
    op.create_index("ix_prices_intraday_ticker", "prices_intraday", ["ticker"])
    op.create_index("ix_prices_intraday_scraped_at", "prices_intraday", ["scraped_at"])


def downgrade() -> None:
    op.drop_table("prices_intraday")

    with op.batch_alter_table("features_daily") as batch_op:
        for col in [
            "bid_ask_imbalance", "order_book_depth_ratio", "spread_bps",
            "wap_vs_close_bps", "foreign_net_flow_pct", "foreign_participation_pct",
            "session_return_pct", "morning_return_pct", "afternoon_return_pct",
            "volume_morning_pct", "intraday_high_low_range_pct", "session_snapshot_count",
        ]:
            batch_op.drop_column(col)

    with op.batch_alter_table("prices_daily") as batch_op:
        for col in [
            "foreign_buy_kes", "foreign_sell_kes", "total_trades_count", "wap",
            "market_cap_kes", "bid_price_eod", "offer_price_eod",
            "total_bid_qty", "total_offer_qty",
        ]:
            batch_op.drop_column(col)
