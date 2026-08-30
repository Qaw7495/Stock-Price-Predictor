#!/usr/bin/env Rscript
#
# engineer_features.R — Step 2 of the pipeline.
#
# Reads the raw OHLCV CSV produced by collect.py, computes a standard set
# of technical indicators with the TTR package, defines the prediction
# target (did the stock close UP the next trading day?), drops warm-up /
# trailing NA rows, and writes the result to data/features/features.csv.
#
# Usage (run from the project root):
#   Rscript data/engineer_features.R
#   Rscript data/engineer_features.R data/raw/stock_data.csv data/features/features.csv

suppressPackageStartupMessages(library(TTR))

args <- commandArgs(trailingOnly = TRUE)
in_path  <- if (length(args) >= 1) args[1] else file.path("data", "raw", "stock_data.csv")
out_path <- if (length(args) >= 2) args[2] else file.path("data", "features", "features.csv")

if (!file.exists(in_path)) {
  stop(sprintf("Raw data file not found: %s (run data/collect.py first)", in_path))
}

cat(sprintf("Reading raw data from %s\n", in_path))
raw <- read.csv(in_path, stringsAsFactors = FALSE)
raw$Date <- as.Date(raw$Date)
raw <- raw[order(raw$Date), ]

close <- raw$Close
high  <- raw$High
low   <- raw$Low
hlc   <- cbind(High = high, Low = low, Close = close)

cat("Computing technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR)...\n")

# --- Trend: simple & exponential moving averages ---
sma_20 <- SMA(close, n = 20)
sma_50 <- SMA(close, n = 50)
ema_20 <- EMA(close, n = 20)

# --- Momentum: RSI ---
rsi_14 <- RSI(close, n = 14)

# --- Momentum: MACD (12/26 EMA difference + 9-period signal line) ---
macd_raw <- MACD(close, nFast = 12, nSlow = 26, nSig = 9, maType = "EMA")
macd        <- macd_raw[, "macd"]
macd_signal <- macd_raw[, "signal"]
macd_hist   <- macd - macd_signal

# --- Volatility: Bollinger Bands (20-period, 2 std dev) ---
bb <- BBands(hlc, n = 20, sd = 2)
bb_lower <- bb[, "dn"]
bb_mid   <- bb[, "mavg"]
bb_upper <- bb[, "up"]
bb_pctb  <- bb[, "pctB"]

# --- Volatility: Average True Range ---
atr_raw <- ATR(hlc, n = 14)
atr_14 <- atr_raw[, "atr"]

features <- data.frame(
  Date = raw$Date,
  Open = raw$Open,
  High = raw$High,
  Low = raw$Low,
  Close = raw$Close,
  Volume = raw$Volume,
  SMA_20 = sma_20,
  SMA_50 = sma_50,
  EMA_20 = ema_20,
  RSI_14 = rsi_14,
  MACD = macd,
  MACD_signal = macd_signal,
  MACD_hist = macd_hist,
  BB_lower = bb_lower,
  BB_mid = bb_mid,
  BB_upper = bb_upper,
  BB_pctB = bb_pctb,
  ATR_14 = atr_14
)

# Target: 1 if TOMORROW's close is higher than TODAY's close, else 0.
n <- nrow(features)
features$Target <- c(ifelse(diff(features$Close) > 0, 1L, 0L), NA)

before <- nrow(features)
features <- na.omit(features)
after <- nrow(features)
cat(sprintf(
  "Dropped %d rows with NA (indicator warm-up + final row with no next-day target).\n",
  before - after
))

dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
write.csv(features, out_path, row.names = FALSE)

cat(sprintf(
  "Saved %d rows x %d columns to %s (%s -> %s)\n",
  nrow(features), ncol(features), out_path,
  as.character(min(features$Date)), as.character(max(features$Date))
))
cat(sprintf("Target balance: %.1f%% up days\n", 100 * mean(features$Target)))
