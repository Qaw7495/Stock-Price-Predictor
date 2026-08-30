# app.R — Step 4 of the pipeline.
#
# Shiny dashboard for the stock predictor. Loads the engineered feature
# CSV and the trained XGBoost model (both produced by the pipeline) and
# shows:
#   - the closing price chart with SMA overlays
#   - tomorrow's Up/Down prediction with a confidence score
#   - feature importance from the trained model
#
# Run from the project root with:
#   R -e "shiny::runApp('dashboard')"

suppressPackageStartupMessages({
  library(shiny)
  library(ggplot2)
  library(xgboost)
  library(jsonlite)
  library(dplyr)
})

# ---- Paths (relative to this app's directory, which Shiny sets as the
#      working directory when launched with runApp('dashboard')) --------
FEATURES_PATH <- file.path("..", "data", "features", "features.csv")
MODEL_PATH <- file.path("..", "models", "xgb_model.json")
FEATURE_COLS_PATH <- file.path("..", "models", "feature_columns.json")
EVAL_LOG_PATH <- file.path("..", "models", "eval_log.json")

missing <- Filter(function(p) !file.exists(p), c(FEATURES_PATH, MODEL_PATH, FEATURE_COLS_PATH, EVAL_LOG_PATH))
if (length(missing) > 0) {
  stop(
    "Missing pipeline output(s): ", paste(missing, collapse = ", "),
    ". Run `python run_pipeline.py` from the project root first."
  )
}

# ---- Load data / model (once at app startup) ---------------------------
features <- read.csv(FEATURES_PATH, stringsAsFactors = FALSE)
features$Date <- as.Date(features$Date)
features <- features[order(features$Date), ]

feature_cols <- fromJSON(FEATURE_COLS_PATH)
eval_log <- fromJSON(EVAL_LOG_PATH)
model <- xgb.load(MODEL_PATH)

latest <- tail(features, 1)
latest_matrix <- as.matrix(latest[, feature_cols, drop = FALSE])
storage.mode(latest_matrix) <- "double"
prob_up <- as.numeric(predict(model, latest_matrix))
prediction <- if (prob_up >= 0.5) "UP" else "DOWN"
confidence <- if (prob_up >= 0.5) prob_up else 1 - prob_up

importance_df <- xgb.importance(feature_names = feature_cols, model = model)

TICKER_LABEL <- "AAPL"  # cosmetic label; the pipeline is single-ticker per run

# ---------------------------------------------------------------------- UI
ui <- fluidPage(
  tags$head(tags$style(HTML("
    body { background-color: #f7f8fa; }
    .app-title { font-weight: 700; margin-bottom: 0; }
    .app-subtitle { color: #6b7280; margin-top: 0; }
    .metric-card {
      background: #ffffff; border-radius: 10px; padding: 18px 20px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.08); text-align: center;
    }
    .metric-label { color: #6b7280; font-size: 13px; text-transform: uppercase; letter-spacing: 0.04em; }
    .metric-value { font-size: 28px; font-weight: 700; margin-top: 4px; }
    .pred-up   { color: #15803d; }
    .pred-down { color: #b91c1c; }
    .section-title { margin-top: 28px; }
  "))),

  titlePanel(
    div(
      h2(paste0("Stock Predictor — ", TICKER_LABEL), class = "app-title"),
      p("Next-day direction prediction from technical indicators (XGBoost)", class = "app-subtitle")
    )
  ),

  fluidRow(
    column(3, div(class = "metric-card",
      div(class = "metric-label", "Tomorrow's Prediction"),
      div(class = paste("metric-value", if (prediction == "UP") "pred-up" else "pred-down"),
          paste(if (prediction == "UP") "▲ UP" else "▼ DOWN"))
    )),
    column(3, div(class = "metric-card",
      div(class = "metric-label", "Confidence"),
      div(class = "metric-value", sprintf("%.1f%%", confidence * 100))
    )),
    column(3, div(class = "metric-card",
      div(class = "metric-label", "Last Close"),
      div(class = "metric-value", sprintf("$%.2f", tail(features$Close, 1)))
    )),
    column(3, div(class = "metric-card",
      div(class = "metric-label", "Test Accuracy"),
      div(class = "metric-value", sprintf("%.1f%%", eval_log$accuracy * 100))
    ))
  ),

  fluidRow(
    column(12,
      h3("Closing Price", class = "section-title"),
      plotOutput("pricePlot", height = "380px")
    )
  ),

  fluidRow(
    column(6,
      h3("Feature Importance", class = "section-title"),
      plotOutput("importancePlot", height = "380px")
    ),
    column(6,
      h3("Model Summary", class = "section-title"),
      tableOutput("modelSummaryTable"),
      p(
        em(paste0(
          "As of ", tail(features$Date, 1), " — data last updated when the pipeline ",
          "was last run. This tool is for educational purposes only and is not financial advice."
        )),
        style = "color:#6b7280; font-size: 12px;"
      )
    )
  )
)

# ------------------------------------------------------------------ SERVER
server <- function(input, output, session) {

  output$pricePlot <- renderPlot({
    plot_df <- features %>%
      select(Date, Close, SMA_20, SMA_50) %>%
      tail(250)

    ggplot(plot_df, aes(x = Date)) +
      geom_line(aes(y = Close, color = "Close"), linewidth = 0.9) +
      geom_line(aes(y = SMA_20, color = "SMA 20"), linewidth = 0.6) +
      geom_line(aes(y = SMA_50, color = "SMA 50"), linewidth = 0.6) +
      scale_color_manual(values = c("Close" = "#1f2937", "SMA 20" = "#2563eb", "SMA 50" = "#f59e0b")) +
      labs(x = NULL, y = "Price ($)", color = NULL) +
      theme_minimal(base_size = 13) +
      theme(legend.position = "top")
  })

  output$importancePlot <- renderPlot({
    top <- head(importance_df[order(-importance_df$Gain), ], 10)
    top$Feature <- factor(top$Feature, levels = rev(top$Feature))

    ggplot(top, aes(x = Feature, y = Gain)) +
      geom_col(fill = "#2563eb") +
      coord_flip() +
      labs(x = NULL, y = "Gain (relative importance)") +
      theme_minimal(base_size = 13)
  })

  output$modelSummaryTable <- renderTable({
    data.frame(
      Metric = c("Training rows", "Test rows", "Train range", "Test range",
                 "Accuracy", "Baseline (majority class)"),
      Value = c(
        eval_log$n_train,
        eval_log$n_test,
        paste(eval_log$train_date_range, collapse = " to "),
        paste(eval_log$test_date_range, collapse = " to "),
        sprintf("%.2f%%", eval_log$accuracy * 100),
        sprintf("%.2f%%", eval_log$baseline_accuracy * 100)
      )
    )
  }, colnames = FALSE)
}

shinyApp(ui, server)
