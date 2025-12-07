{\rtf1\ansi\ansicpg936\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Virality Prediction Framework\
\
## Paper Title\
Predicting Content Virality through Integrated Machine Learning and Network Diffusion Modeling: A Robust Framework with Baseline Comparisons and Sensitivity Analysis\
\
## Overview\
This study develops a three-stage framework for predicting viral spread of social media content. The framework integrates machine learning (content appeal prediction) with network science (diffusion simulation), providing novel explanatory insights into content propagation mechanisms.\
\
## Repository\
**GitHub**: https://github.com/Jianghaoxuan011/virality-prediction-framework\
\
## Quick Start\
\
### 1. Clone Repository\
```bash\
git clone https://github.com/Jianghaoxuan011/virality-prediction-framework.git\
cd virality-prediction-framework\
```\
\
### 2. Install Required R Packages\
```r\
# Run in R or RStudio\
install.packages(c(\
  "tidyverse",   # Data manipulation and visualization\
  "glmnet",      # Regularized regression\
  "xgboost",     # Gradient boosting models\
  "randomForest", # Random forest models\
  "igraph",      # Network analysis\
  "pROC",        # ROC curve analysis\
  "caret",       # Classification and regression training\
  "SHAPforxgboost", # SHAP analysis\
  "gganimate",   # Animated plots\
  "gifski"       # GIF rendering\
))\
```\
\
### 3. Run Complete Analysis\
```r\
# Run the entire analysis pipeline\
source("main_analysis.R")\
```\
\
## What the Script Does\
\
### Stage 1: Data Preparation (Lines 1-96)\
- Loads and cleans the dataset (48,079 short-form videos)\
- Performs feature engineering and leakage prevention\
- Creates training/test splits with temporal ordering\
\
### Stage 2: Machine Learning Models (Lines 97-250)\
- Trains XGBoost models for appeal prediction (R\'b2 = 0.832, AUC = 0.942)\
- Analyzes feature importance (saves = 77.1% gain)\
- Performs SHAP analysis for interpretability\
\
### Stage 3: Network Diffusion Simulation (Lines 251-537)\
- Constructs Watts-Strogatz small-world networks\
- Implements enhanced Independent Cascade model with appeal-dependent propagation\
- Performs baseline comparisons and sensitivity analysis with 60 replications\
- Generates animated visualizations of content spread\
\
## Key Results\
\
### Machine Learning Performance\
| Model | Test R\'b2 | Test AUC | Generalization Gap |\
|-------|---------|----------|-------------------|\
| XGBoost (Regression) | 0.832 | - | 0.002 |\
| XGBoost (Classification) | - | 0.942 | 0.003 |\
\
### Diffusion Simulation Results\
- **High Appeal Content**: Achieves 26.5% greater coverage than low appeal content at baseline propagation\
- **Statistical Significance**: Cohen's d = 0.640, p = 0.002 (60 replications)\
- **Robustness**: Results persist across parameter variations (p_base = 0.03-0.12)\
\
### Key Insights\
1. **Save Actions are Critical**: Save/bookmark actions are the strongest predictor of virality (75%+ importance gain)\
2. **Context-Dependent Effects**: Appeal advantage diminishes with increasing network activity due to saturation\
3. **Early Prediction Feasibility**: High accuracy achieved using only early metadata features\
\
## File Structure\
```\
virality-prediction-framework/\
\uc0\u9500 \u9472 \u9472  README.md              # This documentation file\
\uc0\u9500 \u9472 \u9472  main_analysis.R        # Complete R script with all analyses\
\uc0\u9500 \u9472 \u9472  LICENSE               # MIT License\
\uc0\u9492 \u9472 \u9472  results/              # Generated outputs (not tracked in Git)\
```\
\
## Dependencies\
- **R**: Version 4.3.0 or higher\
- **Data File**: `model_ready_2025-10-21_21-13-17.csv` (48,079 videos dataset)\
- **Memory**: 8 GB RAM recommended\
- **Time**: ~10 minutes for complete execution\
\
## Adjusting Parameters\
You can modify these parameters in `main_analysis.R`:\
\
```r\
# Network parameters (line ~380)\
n <- 250  # Number of nodes in the network\
\
# Simulation parameters (line ~420)\
p_base_values <- c(0.03, 0.06, 0.09, 0.12)  # Propagation probabilities\
n_replicates <- 60  # Number of replications per condition\
\
# Appeal threshold (line ~460)\
high_appeal_threshold <- 0.85  # 85th percentile\
```\
\
## Output Files\
The script generates:\
- Console output with statistical results\
- Model performance visualizations\
- Animated GIF of content spread simulation\
- Summary statistics for diffusion experiments\
\
## Troubleshooting\
\
### Common Issues\
1. **"Package not found" error**\
   ```r\
   # Install specific package\
   install.packages("package_name")\
   ```\
\
2. **Dataset file not found**\
   - Update the file path on line 15 of `main_analysis.R`\
   - Ensure the CSV file exists in the specified location\
\
3. **Memory issues**\
   - Reduce network size: Change `n <- 250` to `n <- 150`\
   - Reduce replications: Change `n_replicates <- 60` to `n_replicates <- 30`\
\
### Getting Help\
Open an issue on GitHub: https://github.com/Jianghaoxuan011/virality-prediction-framework/issues\
\
## Citation\
If you use this code in your research, please cite:\
```\
Jiang, H. (2025). Predicting Content Virality through Integrated Machine Learning \
and Network Diffusion Modeling. [Manuscript submitted for publication].\
```\
\
## License\
MIT License - see [LICENSE](LICENSE) file for details.\
\
## Contact\
- **GitHub**: [Jianghaoxuan011](https://github.com/Jianghaoxuan011)\
- **Repository**: https://github.com/Jianghaoxuan011/virality-prediction-framework\
}