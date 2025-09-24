# 💸💸💸💸 Top-24 Stocks Investment Simulation Project 📢
As a participant in the Stock Market Analytics Zoomcamp of 2025, we are required to deliver a final project that applies everything we have learned in the course to build an end-to-end machine learning pipeline. The ultimate goal is to run simulations that allow us to evaluate investment strategies before applying them in practice.


- `Course link`: https://pythoninvest.com/course
- `Github repo`: https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp/tree/main

## 🎯 Project Overview

This project builds a sophisticated machine learning system to predict whether stocks will experience significant positive growth (≥10%) over a 3-month future period. The system combines multiple data sources, extensive feature engineering, and automated model selection to create investment signals for the top 24 stocks.

The reason of top-24? Of course my love to **Kobe Bryant** 🐍

### Key Objectives

- **Predictive Modeling**: Build models to forecast positive stock price movements (≥10% growth in 3 months)
- **Feature Engineering**: Extract comprehensive features from price data, fundamentals, technical indicators, and macroeconomic factors
- **Investment Simulation**: Provide a framework for backtesting investment strategies (currently, it does not apply simulation)
- **Risk Assessment**: Incorporate volatility, momentum, and market sentiment indicators

## 🏗️ Project Architecture

The project follows a modular, production-ready architecture with clear separation of concerns:

```
├── configs/                    # Configuration files
│   ├── external_indicators.yaml    # FRED & macro data sources
│   ├── preprocessing_variables.yaml # Feature preprocessing rules
├── data/                      # Dataset storage
│   ├── full_dataset.csv          # Complete feature-engineered dataset
│   └── dataset_for_modeling.csv  # Processed data ready for ML
├── notebooks/                 # Analysis & development workflows
│   ├── dataset_creation.ipynb    # Data collection & feature engineering
│   ├── dataset_transformation.ipynb # Data preprocessing & target creation
│   └── modeling_phase.ipynb      # Model training & evaluation
├── saved_models/                    #  Model artifacts
│   ├── best_model_parameters.yaml  # Optimal model configuration
│   └── model_randomforest_*.joblib  # Trained model artifacts
├── scripts/                    #  .py files for replicating demo/full pipelines + minimal tests
│   ├── run_example.py  # Reproduce full pipeline
│   └── test_pipeline.py  # Minimal tests
└── utils/                     # Core functionality modules
    ├── data.py                # Data collection & feature engineering
    └── model.py               # ML pipeline & model evaluation
└── .pre-commit-config.yaml # Configures pre-commit hooks
└── pyproject.toml         # Ruff linter
└── main.py                # Compact file that enables the orchestration of the pipeline
└── requirements.txt       # Updated file with all the needed libraries to install
```

## 📊 Data Sources & Features

### 1. **Stock Price Data (Primary Source)**
- **Source**: Yahoo Finance API via `yfinance`
- **Coverage**: Top 24 stocks with comprehensive OHLCV data
- **Frequency**: Daily → Monthly aggregation (month-end)
- **Features**:
  - Price-based: Returns (1M, 2M, 3M), volatility measures, momentum indicators
  - Volume-based: Logarithmic volume statistics (min, median, max)
  - Technical: Moving averages (3M, 6M), relative spreads, rolling volatility

### 2. **Fundamental Data**
- **Balance Sheet Metrics**: Debt-to-equity, current ratio, working capital ratios
- **Cash Flow Indicators**: Operating CF margin, free cash flow margin
- **Valuation Metrics**: P/E ratios, P/B ratio, PEG ratio, dividend yield
- **Market Metrics**: Market cap, beta, institutional ownership

### 3. **Technical Analysis Indicators**
Powered by **TA-Lib** library with 60+ indicators:

#### Momentum Indicators (22 features)
- **Trend**: ADX, ADXR, Aroon indicators, Directional Movement Index
- **Oscillators**: RSI, Stochastic (fast/slow), Williams %R, CCI, CMO
- **MACD Family**: Standard, Extended, and Fixed versions with signals & histograms
- **Price-based**: ROC, Momentum, PPO, TRIX, Ultimate Oscillator

#### Candlestick Patterns (61 features)
- **Reversal Patterns**: Doji varieties, Hammer, Shooting Star, Engulfing
- **Continuation Patterns**: Three White Soldiers, Three Black Crows
- **Complex Patterns**: Morning/Evening Stars, Harami, Kicking patterns

### 4. **Macroeconomic Indicators**
- **FRED Economic Data**:
  - GDP (US, Germany), CPI (US, Germany)
  - Unemployment (US), Interest rates (US, EU)
- **Market Indices**: S&P 500, VIX, DAX, sector ETFs (XLV, XLK, XLF)
- **Currency/Commodities**: EUR/USD, USD Index, Gold (GLD)

### 5. **ESG & Sentiment Data**
- **ESG Scores**: Environmental, Social, Governance ratings
- **Analyst Sentiment**: Recommendation changes, strong buy ratios
- **Corporate Events**: Days to earnings, dividend stability

## 🗂️ Project Files

### Core Pipeline
- **`main.py`** - Complete end-to-end pipeline (⭐ **Recommended**)
- **`test_pipeline.py`** - Pipeline validation and testing
- **`run_example.py`** - Usage examples and demos

### Original Development Notebooks
- **`notebooks/dataset_creation.ipynb`** - Data collection & feature engineering
- **`notebooks/dataset_transformation.ipynb`** - Preprocessing & target creation
- **`notebooks/modeling_phase.ipynb`** - Model training & evaluation

### Utility Modules
- **`utils/data.py`** - Data collection & feature engineering functions
- **`utils/model.py`** - ML pipeline & model evaluation functions

### Configuration
- **`configs/external_indicators.yaml`** - External data sources configuration
- **`configs/preprocessing_variables.yaml`** - Feature preprocessing rules

### Saved models
- **`saved_models/best_model_parameters.yaml`** - Optimal model configuration (generated)
- **`saved_models/model_*.joblib`** - Trained model artifacts (generated)

### Data (Generated)
- **`data/full_dataset.csv`** - Complete feature-engineered dataset
- **`data/dataset_for_modeling.csv`** - Processed data ready for ML

## �🔧 Feature Engineering Pipeline

### 1. **Price-Based Features**
```python
# Growth rates across multiple timeframes
daily_growth = close.pct_change()
weekly_growth = close.pct_change(5)
monthly_growth = close.pct_change(21)

# Volatility measures
vol_3m = close.pct_change().rolling(3).std()
momentum_3m = return_3m / vol_3m

# Moving averages (scaled by current price)
mma_3_scaled = close.rolling(3).mean() / close
```

### 2. **Technical Indicators Integration**
```python
# TA-Lib momentum indicators
rsi = talib.RSI(close, timeperiod=14)
macd, signal, hist = talib.MACD(close)
stoch_k, stoch_d = talib.STOCH(high, low, close)

# Candlestick pattern recognition
doji = talib.CDLDOJI(open, high, low, close)
hammer = talib.CDLHAMMER(open, high, low, close)
```

### 3. **Target Variable Creation**
```python
# Binary classification: 1 if 3-month future growth ≥ 10%
target = (future_price_3m >= current_price * 1.10).astype(int)
```

## 🤖 Machine Learning Pipeline

### 1. **Data Preprocessing**
- **Missing Value Imputation**: Industry-based median imputation for financial metrics
- **Cyclical Encoding**: Sine/cosine transformation for temporal features (month)
- **Categorical Encoding**: One-hot encoding for sectors, industries, countries
- **Feature Scaling**: StandardScaler for numerical features

### 2. **Temporal Data Splitting**
```python
# Time-series aware splits to prevent data leakage
train: 70% (oldest data)
validation: 15% (middle period)
test: 15% (most recent data)
```

### 3. **Model Selection & Hyperparameter Tuning**
Automated comparison of three algorithms with TimeSeriesSplit cross-validation:

#### **Random Forest (Best Performer)**
- **Hyperparameters**: n_estimators=200, max_depth=5, min_samples_split=2
- **Performance**: 59% accuracy, 62% precision on positive class
- **Strengths**: Handles mixed data types, provides feature importance

#### **Decision Tree**
- **Hyperparameters**: max_depth, min_samples_split optimization
- **Use Case**: Interpretable baseline model

#### **K-Nearest Neighbors**
- **Hyperparameters**: n_neighbors, weights, distance metrics
- **Use Case**: Non-parametric pattern recognition

### 4. **Model Evaluation Metrics**
- **Primary Metric**: Precision on positive class (minimizes false positives)
- **Secondary Metrics**: Accuracy, F1-score, confusion matrix analysis
- **Business Focus**: High precision to reduce risk of poor investment decisions

## 📈 Results & Performance

### Current Best Model: Random Forest
```yaml
Model Performance:
  Accuracy: 59%
  Precision (Positive Class): 62%
  Recall (Positive Class): 11%
  F1-Score (Positive Class): 19%

Confusion Matrix:
  True Negatives: 118  |  False Positives: 6
  False Negatives: 82  |  True Positives: 10
```

### Key Insights
- **High Precision Strategy**: Model prioritizes avoiding false positives over capturing all positive cases
- **Threshold optimization**: Adjusting the classification threshold led to a significant improvement in model performance, yielding a gain of 9 points in precision and 1 point in accuracy. Fine-tuning this parameter proved to be crucial for optimizing prediction results.
- **Conservative Approach**: 71% precision means ~3 out of 4 predictions are correct
- **Risk Management**: Low false positive rate reduces potential losses from bad predictions

## 🛠️ Technology Stack

### Core Libraries
- **Data Manipulation**: `pandas`, `numpy`
- **Financial Data**: `yfinance`, `pandas-datareader`
- **Technical Analysis**: `TA-Lib`
- **Machine Learning**: `scikit-learn`
- **Visualization**: `matplotlib`, `seaborn`

### Development Tools
- **Environment Management**: `conda`/`pip`
- **Code Quality**: `black`, `isort`, `ruff`, `pre-commit`
- **Notebooks**: `jupyter`
- **Configuration**: `PyYAML`


## 🧪 Pipeline Testing (`test-pipeline.yml`)
- **Triggers**: Pull requests, manual dispatch
- **Purpose**: Validate pipeline functionality and code quality
- **Test Modes**:
  - `quick`: Fast test with 5 stocks
  - `full`: Complete test with up to 10 stocks
  - `dry-run`: Configuration validation only
- **Features**:
  - 🔍 Code quality checks (ruff, black)
  - 🧪 Pipeline validation tests
  - 📊 Output validation
  - 🧹 Automatic cleanup



### 📋 Workflow Requirements
- **TA-Lib Installation**: Automated in workflows
- **Python 3.12**: Specified environment
- **Dependencies**: Auto-installed from requirements.txt
- **Artifacts**: Results stored for 90 days
- **Permissions**: Requires repository write access for auto-commits

## 🚀 Usage Instructions

### Quick Start with Main Pipeline

The project includes a comprehensive `main.py` script that consolidates all functionality from the three notebooks into a single, streamlined pipeline:

```bash
# Run the complete pipeline (recommended)
python main.py

# Test the pipeline setup
python test_pipeline.py

# Run examples with different configurations being in root directory
python scripts/run_example.py demo    # Quick demo with 5 stocks
python scripts/run_example.py full    # Full pipeline with 24 stocks
python scripts/run_example.py analyze # Analyze existing data
```

### Alternative: Manual Step-by-Step Execution

If you prefer to run the analysis step-by-step using the original notebooks:

### 1. Environment Setup

Create a new Python environment with the required dependencies:

```bash
# Create conda environment
conda create -n <PREFERRED-NAME> python==3.12
conda activate <PREFERRED-NAME>

# Install dependencies
pip install -r requirements.txt

# Enable pre-commit hooks for code quality
pre-commit install
```

### 2. Project Workflow

#### **Phase 1: Data Collection & Feature Engineering**
```bash
# Run the dataset creation notebook
jupyter notebook notebooks/dataset_creation.ipynb
```
- Collects stock data for top 24 companies
- Extracts fundamental data from financial statements
- Computes 60+ technical indicators using TA-Lib
- Integrates macroeconomic data from FRED
- Outputs: `data/full_dataset.csv`

#### **Phase 2: Data Preprocessing & Target Creation**
```bash
# Run the transformation notebook
jupyter notebook notebooks/dataset_transformation.ipynb
```
- Handles missing values with domain-specific imputation
- Creates cyclical features for temporal data
- Generates binary target variable (10% growth threshold)
- Applies train/validation/test temporal splits
- Outputs: `data/dataset_for_modeling.csv`

#### **Phase 3: Model Training & Selection**
```bash
# Run the modeling notebook
jupyter notebook notebooks/modeling_phase.ipynb
```
- Trains multiple ML algorithms with hyperparameter tuning
- Performs time-series cross-validation
- Selects best model based on precision metrics
- Saves trained model and configuration files in `saved_models` folder

### 3. Key Configuration Files

#### **External Data Sources** (`configs/external_indicators.yaml`)
```yaml
fred_series:
  gdp_us: "GDPC1"      # US GDP
  cpi_us: "CPIAUCSL"   # US Consumer Price Index
  unemployment_us: "UNRATE"  # US Unemployment Rate

tickers_macro:
  sp500: "^GSPC"       # S&P 500 Index
  vix: "^VIX"          # Volatility Index
  dax: "^GDAXI"        # German DAX Index
```

#### **Preprocessing Rules** (`configs/preprocessing_variables.yaml`)
```yaml
DUMMY_VARIABLES:      # One-hot encode these categorical features
  - ticker
  - sector
  - industry

DROP_VARIABLES:       # Remove from final dataset
  - Date
  - TARGET
  - companyName
  - year

CYCLICAL_VARIABLES:   # Apply sine/cosine transformation
  - month
```

## 📊 Dataset Description

### Feature Categories Overview

| Category | Count | Examples |
|----------|-------|----------|
| **Price Features** | 22 | Returns, volatility, moving averages |
| **Fundamental Features** | 19 | P/E ratios, debt ratios, cash flow metrics |
| **Technical Indicators** | 83 | RSI, MACD, Stochastic, candlestick patterns |
| **Macroeconomic** | 12 | GDP, CPI, interest rates, market indices |
| **Categorical** | 3 | Ticker, sector, industry |
| **Temporal** | 3 | Year, month (+ cyclical encoding) |


## 🔍 Model Interpretation & Business Value

### Investment Strategy Framework
```python
# Pseudo-code for investment decision
if model.predict_proba(stock_features)[1] > threshold:
    recommendation = "BUY"  # High probability of 10%+ growth
    confidence = model.predict_proba(stock_features)[1]
else:
    recommendation = "HOLD" # Insufficient confidence for investment
```

### Risk Management Features
- **Precision-Focused**: Prioritizes avoiding bad investments over capturing all opportunities
- **Conservative Threshold**: High-confidence predictions only
- **Multi-Factor Analysis**: Combines technical, fundamental, and macro indicators
- **Temporal Validation**: Out-of-time testing prevents overfitting

### Business Applications
1. **Portfolio Construction**: Rank stocks by predicted probability
2. **Risk Screening**: Filter out high-risk investments
3. **Timing Decisions**: Identify optimal entry points
4. **Sector Rotation**: Compare opportunities across industries

## 🔬 Research & Development

### Future Enhancements
- **Alternative Data**: Sentiment analysis from news/social media
- **Deep Learning**: LSTM/Transformer models for sequential patterns
- **Multi-Asset**: Extend to bonds, commodities, crypto
- **Real-Time**: Streaming data pipeline for live predictions
- **Portfolio Optimization**: Integration with Modern Portfolio Theory

### Experimental Features
- **Ensemble Methods**: Combine multiple model predictions
- **Feature Selection**: Automated relevance determination
- **Dynamic Thresholds**: Adaptive decision boundaries
- **Market Regime Detection**: Bull/bear market awareness

## 📚 Dependencies & Requirements

### Core Dependencies
```txt
# Data & Analysis
pandas >= 2.0.0
numpy >= 1.24.0
yfinance >= 0.2.0
pandas-datareader >= 0.10.0

# Technical Analysis
TA-Lib >= 0.4.25

# Machine Learning
scikit-learn >= 1.3.0

# Visualization
matplotlib >= 3.7.0
seaborn >= 0.12.0

# Development Tools
jupyter >= 1.0.0
black >= 23.0.0
isort >= 5.12.0
ruff >= 0.0.280
pre-commit >= 3.3.0

# Configuration
PyYAML >= 6.0
requests >= 2.31.0
lxml >= 4.9.0
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Contact & Support

**Author**: Stamatis Karlos
**Course**: Stock Market Analytics Zoomcamp 2025
**Project Type**: End-to-End ML Pipeline for Investment Strategy

For questions or support, please open an issue in the GitHub repository.

---

*Built with ❤️ for the Stock Market Analytics Zoomcamp community*
