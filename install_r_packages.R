# install_r_packages.R — one-time setup helper.
# Installs every R package the pipeline, dashboard, and report depend on.
#
# Usage:
#   Rscript install_r_packages.R

pkgs <- c("TTR", "ggplot2", "shiny", "xgboost", "jsonlite", "dplyr", "rmarkdown", "knitr")
missing <- pkgs[!pkgs %in% rownames(installed.packages())]

if (length(missing) > 0) {
  cat("Installing missing R packages:", paste(missing, collapse = ", "), "\n")
  install.packages(missing, repos = "https://cloud.r-project.org")
} else {
  cat("All required R packages are already installed.\n")
}
