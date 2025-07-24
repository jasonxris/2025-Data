#!/usr/bin/env python3
"""
--------------------------------------------------------------------
build_processed_trades.py
--------------------------------------------------------------------
Reads every    ./data/Portfolio Transactions - <Team>.csv      file
plus matching ./data/Holdings - <Team>.csv files, constructs a
row‑per‑completed‑trade table, and writes it to

        ./output/processed_trades.csv

Key rules implemented
---------------------
1.  Cancels   →  ignored (non‑blank “Cancel Reason” rows dropped)
2.  Each Sell / Cover ticket = ONE completed trade (partial exits
    create multiple trades).  FIFO is used to pair exit quantity
    with earlier buys/shorts, but the “entry date” recorded is the
    date on which the *majority* of those shares were opened.
3.  Open positions after the last ticket are  ⟹  closed at today’s
    price taken from the team’s   Holdings - <Team>.csv   file.
4.  Output columns generated

       team, symbol, trade_type, entry_date, exit_date,
       holding_days, qty, entry_price, exit_price,
       total_pl_usd, pct_return

Dependencies
------------
pandas ≥ 1.4, python‑dateutil (for robust timestamp parsing)

--------------------------------------------------------------------
"""

import glob
import os
import re
import datetime
from collections import deque, defaultdict
from pathlib import Path

import pandas as pd
from dateutil import parser as dtparse


# -----------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR  = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TODAY = datetime.datetime.today()          # used for forced exits

# Thresholds for risk and hold classification
RISK_THRESHOLD_PCT = 8.0                   # >= 8% of portfolio = High-Risk
HOLD_THRESHOLD_DAYS = 8                    # >= 8 days = Long-Hold


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------
def _clean_money(s: str) -> float:
    """'$1,234.56' → 1234.56  (handles negatives, commas, $)"""
    if pd.isna(s):
        return float("nan")
    return float(re.sub(r"[^0-9.\-]", "", str(s)))


def _clean_int(s: str) -> int:
    """'1,500' → 1500  (handles negatives)"""
    return int(re.sub(r"[^0-9\-]", "", str(s)))


# -----------------------------------------------------------------
# Holdings loader  (team  →  {symbol: price})
# -----------------------------------------------------------------
def load_holdings():
    mapping = {}
    for path in glob.glob(str(DATA_DIR / "Holdings - *.csv")):
        team = re.search(r"Holdings - (.*)\.csv", os.path.basename(path)).group(1)
        df   = pd.read_csv(path)
        df["Price"] = df["Price"].apply(_clean_money)
        mapping[team] = dict(zip(df["Symbol"], df["Price"]))
    return mapping


# -----------------------------------------------------------------
# Per‑team processor
# -----------------------------------------------------------------
def process_team(trade_csv: str, holdings_map: dict):
    team = re.search(r"Portfolio Transactions - (.*)\.csv",
                     os.path.basename(trade_csv)).group(1)

    df = pd.read_csv(trade_csv)

    # ----- basic cleaning ---------------------------------------------------
    df = df[df["Cancel Reason"].isna()]                               # rule 1
    df["Transaction Date"] = df["Transaction Date"].apply(
        lambda x: dtparse.parse(str(x).replace("ET", ""))
    )
    df["Amount"] = df["Amount"].apply(_clean_int)
    df["Price"]  = df["Price"].apply(_clean_money)
    df = df.sort_values("Transaction Date")

    # ----- inventory ledgers -----------------------------------------------
    long_inv  = defaultdict(deque)   # symbol → deque[(qty, price, date)]
    short_inv = defaultdict(deque)

    output_rows = []

    # ----- walk every ticket ------------------------------------------------
    for _, row in df.iterrows():
        sym, qty, price, date = row["Symbol"], row["Amount"], row["Price"], row["Transaction Date"]
        ttype = row["Type"].strip().lower()

        if ttype in {"buy", "short"}:
            # open a position
            (long_inv if ttype == "buy" else short_inv)[sym].append((qty, price, date))

        elif ttype in {"sell", "cover"}:
            is_long = ttype == "sell"
            ledger  = long_inv if is_long else short_inv
            qty_to_close = qty
            matched = []                                         # lots consumed

            while qty_to_close > 0 and ledger[sym]:
                lot_qty, lot_price, lot_date = ledger[sym][0]
                take = min(lot_qty, qty_to_close)
                matched.append((take, lot_price, lot_date))

                # shrink or pop the lot
                if take == lot_qty:
                    ledger[sym].popleft()
                else:
                    ledger[sym][0] = (lot_qty - take, lot_price, lot_date)

                qty_to_close -= take

            if not matched:
                # unmatched exit (data error) – skip gracefully
                continue

            # ---- aggregate trade metrics ----------------------------------
            total_qty = sum(q for q, _, _ in matched)
            wavg_entry_price = sum(q * p for q, p, _ in matched) / total_qty

            # majority entry date
            cum = 0
            half = total_qty / 2
            entry_date = None
            for q, _, d in matched:
                cum += q
                if cum >= half:
                    entry_date = d
                    break

            # P/L and returns
            if is_long:                                           # long trade
                profit = (price - wavg_entry_price) * total_qty
            else:                                                 # short trade
                profit = (wavg_entry_price - price) * total_qty

            pct_ret = profit / (abs(wavg_entry_price) * total_qty) * 100
            holding_days = (date.date() - entry_date.date()).days
            pct_portfolio = round((abs(wavg_entry_price * total_qty) / 1000000) * 100, 4)

            output_rows.append({
                "team"        : team,
                "symbol"      : sym,
                "trade_type"  : "Long" if is_long else "Short",
                "entry_date"  : entry_date.date(),
                "exit_date"   : date.date(),
                "holding_days": holding_days,
                "qty"         : total_qty,
                "entry_price" : round(wavg_entry_price, 4),
                "exit_price"  : price,
                "total_pl_usd": round(profit, 2),
                "pct_return"  : round(pct_ret, 2),
                "trade_result": "Positive" if profit > 0 else "Negative",
                "pct_portfolio": pct_portfolio,
                "risk_level"   : "High-Risk" if pct_portfolio >= RISK_THRESHOLD_PCT else "Low-Risk",
                "hold_type"    : "Long-Hold" if holding_days >= HOLD_THRESHOLD_DAYS else "Short-Hold",
            })

    # ----- force‑close any residual positions ------------------------------
    today = TODAY.date()
    price_map = holdings_map.get(team, {})

    def close_remaining(ledger, is_long: bool):
        for sym, deque_lots in ledger.items():
            if sym not in price_map:
                continue
            exit_price = price_map[sym]
            for qty, entry_price, entry_date in deque_lots:
                profit = ((exit_price - entry_price) if is_long
                          else (entry_price - exit_price)) * qty
                pct_ret = profit / (abs(entry_price) * qty) * 100
                holding_days = (today - entry_date.date()).days
                pct_portfolio = round((abs(entry_price * qty) / 1000000) * 100, 4)
                
                output_rows.append({
                    "team"        : team,
                    "symbol"      : sym,
                    "trade_type"  : "Long" if is_long else "Short",
                    "entry_date"  : entry_date.date(),
                    "exit_date"   : today,
                    "holding_days": holding_days,
                    "qty"         : qty,
                    "entry_price" : round(entry_price, 4),
                    "exit_price"  : exit_price,
                    "total_pl_usd": round(profit, 2),
                    "pct_return"  : round(pct_ret, 2),
                    "trade_result": "Positive" if profit > 0 else "Negative",
                    "pct_portfolio": pct_portfolio,
                    "risk_level"   : "High-Risk" if pct_portfolio >= RISK_THRESHOLD_PCT else "Low-Risk",
                    "hold_type"    : "Long-Hold" if holding_days >= HOLD_THRESHOLD_DAYS else "Short-Hold",
                })

    close_remaining(long_inv,  True)    # long leftovers → forced Sell
    close_remaining(short_inv, False)   # short leftovers → forced Cover

    return output_rows


# -----------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------
def main():
    holdings_map = load_holdings()
    trade_files  = glob.glob(str(DATA_DIR / "Portfolio Transactions - *.csv"))

    all_trades = []
    for f in trade_files:
        all_trades.extend(process_team(f, holdings_map))

    out_df = pd.DataFrame(all_trades)
    out_file = OUT_DIR / "processed_trades.csv"
    out_df.to_csv(out_file, index=False)
    print(f"Wrote {len(out_df)} completed trades → {out_file}")


if __name__ == "__main__":
    main()
