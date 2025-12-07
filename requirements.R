# Required R packages for virality-prediction-framework
packages <- c(
  "tidyverse",     # Data manipulation and visualization
  "glmnet",        # Regularized regression models
  "xgboost",       # Gradient boosting models
  "randomForest",  # Random forest models
  "igraph",        # Network analysis and visualization
  "pROC",          # ROC curve analysis
  "caret",         # Classification and regression training
  "SHAPforxgboost",# SHAP values for XGBoost
  "gganimate",     # Create animated plots
  "gifski",        # Render GIFs from animations
  "gridExtra",     # Arrange multiple plots
  "broom",         # Tidy statistical outputs
  "ggcorrplot"     # Correlation matrix visualization
)

# Install missing packages
install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
  }
}

# Install all packages
sapply(packages, install_if_missing)

# Load all packages
sapply(packages, require, character.only = TRUE)

cat("✅ All required packages are installed and loaded.\n")