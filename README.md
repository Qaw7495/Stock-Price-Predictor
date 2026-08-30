# Stock Predictor

Predicts whether a stock will close **up or down tomorrow**, using 5
years of historical daily data and a set of classic technical indicators.
The pipeline is deliberately polyglot: **Python** handles data
collection and machine learning (XGBoost), **R** handles feature
engineering (TTR), visualization (ggplot2), and serving (Shiny + R
Markdown).

> **Disclaimer:** This project is for educational purposes only. It is
> not financial advice, and predicting next-day stock direction is a
> genuinely hard, noisy problem — do not use this to make real trading
> decisions.

## How it works

```
                 ┌─────────────────────┐
                 │  data/collect.py    │  Python + yfinance
                 │  download 5y OHLCV  │
                 └──────────┬──────────┘
                            │ data/raw/stock_data.csv
                            ▼
                 ┌──────────────────────────┐
                 │ data/engineer_features.R │  R + TTR
                 │ SMA/EMA/RSI/MACD/BB/ATR  │
                 │ + next-day Up/Down target│
                 └──────────┬───────────────┘
                            │ data/features/features.csv
                            ▼
                 ┌─────────────────────┐
                 │ data/train_model.py │  Python + XGBoost
                 │ 80/20 chronological │
                 │ train/test split    │
                 └──────────┬──────────┘
                            │ models/xgb_model.json
                            │ models/eval_log.json
                 ┌──────────┴──────────┐
                 ▼                     ▼
      ┌─────────────────────┐  ┌─────────────────────┐
      │  dashboard/app.R    │  │  report/report.Rmd  │
      │Shiny live dashboard │  │  R Markdown report  │
      └─────────────────────┘  └─────────────────────┘
```

## Project structure

```
stock_predictor/
├── data/
│   ├── collect.py            # Step 1: download OHLCV data (Python)
│   ├── engineer_features.R   # Step 2: technical indicators + target (R)
│   ├── train_model.py        # Step 3: train & evaluate XGBoost (Python)
│   ├── raw/                  # generated: raw OHLCV CSV
│   └── features/             # generated: engineered feature CSV
├── models/
│   ├── xgb_model.json        # generated: trained model
│   ├── feature_columns.json  # generated: ordered feature list
│   └── eval_log.json         # generated: accuracy, confusion matrix, importances
├── dashboard/
│   └── app.R                 # Step 4: Shiny dashboard
├── report/
│   └── report.Rmd            # Step 5: R Markdown report
├── run_pipeline.py           # runs steps 1-3 end-to-end
├── install_r_packages.R      # one-time R dependency installer
├── requirements.txt          # Python dependencies
└── README.md
```

## Prerequisites

- **Python 3.9+**
- **R 4.x** with `Rscript` on your `PATH`

## Installing R

Pick your OS:

<details>
<summary><b>macOS</b></summary>

```bash
brew install r
```

If you don't have Homebrew yet: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`

On Apple Silicon, make sure you're using an **arm64** Homebrew (`/opt/homebrew/bin/brew`), not an x86_64 one under Rosetta (`/usr/local/bin/brew`) — mixing the two is a common source of "wrong architecture" errors. Check with `brew config | grep Prefix`.
</summary>
</details>

<details>
<summary><b>Ubuntu / Debian</b></summary>

```bash
sudo apt update
sudo apt install --no-install-recommends r-base
```

For a newer R version than your distro ships, use the official CRAN apt repo — see https://cran.r-project.org/bin/linux/ubuntu/.
</details>

<details>
<summary><b>Fedora / RHEL / CentOS</b></summary>

```bash
sudo dnf install R
```
</details>

<details>
<summary><b>Windows</b></summary>

Download and run the installer from https://cran.r-project.org/bin/windows/base/, or via a package manager:

```powershell
winget install --id RProject.R
# or: choco install r.project
```

After installing, add R's `bin` directory (e.g. `C:\Program Files\R\R-4.x.x\bin`) to your `PATH` so `Rscript` is available from any terminal.
</details>

Verify with:

```bash
Rscript --version
```

## Setup

```bash
# from the stock_predictor/ directory

# 1. Python dependencies
pip install -r requirements.txt

# 2. R dependencies (TTR, ggplot2, shiny, xgboost, jsonlite, dplyr, rmarkdown, knitr)
Rscript install_r_packages.R
```

### macOS note: XGBoost needs OpenMP

The Python `xgboost` wheel links against `libomp`, which isn't part of
macOS by default. If `import xgboost` fails with something like
`Library not loaded: @rpath/libomp.dylib`, install it with Homebrew —
using the **same-architecture** Homebrew as your Python interpreter
(arm64 Python needs `/opt/homebrew`'s libomp; x86_64/Rosetta Python
needs `/usr/local`'s):

```bash
brew install libomp
```

## Running the pipeline

All commands below assume your working directory is `stock_predictor/`.

### One command, end-to-end

```bash
python run_pipeline.py                       # AAPL, 5 years (defaults)
python run_pipeline.py --ticker MSFT --years 8
```

This runs, in order: `data/collect.py` → `data/engineer_features.R` →
`data/train_model.py`, printing progress and the final test accuracy /
confusion matrix.

### Or step by step

```bash
python data/collect.py --ticker AAPL --years 5
Rscript data/engineer_features.R
python data/train_model.py
```

## Exploring the results

### Shiny dashboard

```bash
R -e "shiny::runApp('dashboard')"
```

Shows the closing price chart (with SMA overlays), tomorrow's Up/Down
prediction with a confidence score, current test accuracy, and a
feature-importance chart — all read live from the pipeline's output
files.

### R Markdown report

```bash
R -e "rmarkdown::render('report/report.Rmd')"
```

Generates `report/report.html`: an approach summary, data summary,
accuracy / confusion matrix / classification report, feature
importance, and price/RSI/MACD plots.

## Notes on methodology

- **Chronological split, not random.** The 80/20 train/test split trains
  on the earliest 80% of history and evaluates on the most recent 20%,
  in date order. Shuffling would leak future information into training,
  which is invalid for time series.
- **Baseline comparison.** Both the pipeline log and the R Markdown
  report compare model accuracy against a majority-class baseline
  (always predicting the more common label), since equities drift
  upward more often than not — a model needs to beat that baseline to
  demonstrate real signal.
- **Direction, not magnitude.** The target is binary (`Close[t+1] >
  Close[t]`); the model does not estimate the size of the move.

## Changing the ticker

`run_pipeline.py --ticker <SYMBOL>` controls which stock is downloaded
in step 1; every downstream step (feature engineering, training, the
dashboard, the report) operates on whatever is currently in
`data/raw/stock_data.csv` / `data/features/features.csv`, so re-running
the pipeline with a new ticker refreshes the whole project for that
symbol.

## License

MIT License

Copyright (c) 2026 Raul Buta

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
