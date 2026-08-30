"""
collect.py — Step 1 of the pipeline.

Downloads daily OHLCV history for a stock ticker from Yahoo Finance
(via yfinance) and writes it to data/raw/stock_data.csv for the R
feature-engineering step to consume.

Usage:
    python data/collect.py                       # AAPL, 5 years
    python data/collect.py --ticker MSFT --years 8
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PROJECT_ROOT / "data" / "raw" / "stock_data.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download historical stock data.")
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol (default: AAPL)")
    parser.add_argument("--years", type=int, default=5, help="Years of history (default: 5)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output CSV path")
    return parser.parse_args()


def download(ticker: str, years: int) -> pd.DataFrame:
    period = f"{years}y"
    print(f"Downloading {period} of daily data for '{ticker}' from Yahoo Finance...")
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)

    if df.empty:
        raise RuntimeError(
            f"No data returned for ticker '{ticker}'. Check that the symbol is valid "
            "and that you have an internet connection."
        )

    # yfinance can return MultiIndex columns (Price, Ticker) for a single symbol
    # depending on version; flatten to plain OHLCV column names.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df = df.rename(columns={"Date": "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    keep = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = df[keep].dropna().reset_index(drop=True)
    return df


def main() -> None:
    args = parse_args()
    df = download(args.ticker, args.years)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Saved {len(df)} rows ({df['Date'].iloc[0]} -> {df['Date'].iloc[-1]}) to {args.out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # surface a clean error for the pipeline runner
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
