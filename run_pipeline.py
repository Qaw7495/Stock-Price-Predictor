"""
run_pipeline.py — runs the full stock predictor pipeline end-to-end:

    1. data/collect.py            (Python)  download raw OHLCV data
    2. data/engineer_features.R   (R)       compute technical indicators + target
    3. data/train_model.py        (Python)  train XGBoost, evaluate, save model

After this completes, launch the Shiny dashboard (dashboard/app.R) or knit
the report (report/report.Rmd) to explore the results.

Usage:
    python run_pipeline.py
    python run_pipeline.py --ticker MSFT --years 8
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full stock predictor pipeline.")
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol (default: AAPL)")
    parser.add_argument("--years", type=int, default=5, help="Years of history (default: 5)")
    return parser.parse_args()


def banner(step: int, total: int, title: str) -> None:
    print()
    print("=" * 70)
    print(f" STEP {step}/{total}: {title}")
    print("=" * 70)


def run(cmd: list[str], *, cwd: Path) -> None:
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"\nPipeline failed at: {' '.join(cmd)} (exit code {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)


def find_rscript() -> str:
    for candidate in ("Rscript", "/usr/local/bin/Rscript", "/opt/homebrew/bin/Rscript"):
        if shutil.which(candidate) or Path(candidate).exists():
            return candidate
    print(
        "ERROR: Rscript not found on PATH. Install R (e.g. `brew install r`) "
        "and ensure Rscript is available.",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    args = parse_args()
    rscript = find_rscript()
    total_steps = 3
    start = time.time()

    banner(1, total_steps, f"Download {args.years}y of data for {args.ticker} (Python)")
    run(
        [sys.executable, "data/collect.py", "--ticker", args.ticker, "--years", str(args.years)],
        cwd=PROJECT_ROOT,
    )

    banner(2, total_steps, "Engineer technical indicator features (R)")
    run([rscript, "data/engineer_features.R"], cwd=PROJECT_ROOT)

    banner(3, total_steps, "Train & evaluate XGBoost model (Python)")
    run([sys.executable, "data/train_model.py"], cwd=PROJECT_ROOT)

    elapsed = time.time() - start
    print()
    print("=" * 70)
    print(f" PIPELINE COMPLETE in {elapsed:.1f}s")
    print("=" * 70)
    print(
        "\nNext steps:\n"
        "  - Launch the dashboard: R -e \"shiny::runApp('dashboard')\"\n"
        "  - Render the report:    R -e \"rmarkdown::render('report/report.Rmd')\"\n"
    )


if __name__ == "__main__":
    main()
