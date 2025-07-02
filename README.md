# Electricity Demand Forecasting

A production-grade, modular Python project for electricity demand and price forecasting using machine learning (Random Forest, LSTM) and rich data analysis/visualization.

## Project Overview

**Electricity Demand Forecasting** is a modular, production-ready Python package designed to predict electricity demand (and optionally price) using both classical machine learning (Random Forest) and deep learning (LSTM) models. The project is structured for clarity, reproducibility, and easy extension, making it suitable for both research and deployment.

### Why Forecast Electricity Demand?

- **Grid Management:** Accurate demand forecasts help utilities balance supply and demand, reducing costs and preventing outages.
- **Renewable Integration:** Forecasting helps manage the variability of renewables (solar, wind) by anticipating demand peaks and troughs.
- **Market Operations:** Power producers and consumers can optimize bidding and consumption strategies in electricity markets.
- **Sustainability:** Better forecasts enable more efficient energy use, reducing waste and emissions.

### How Does This Project Work?

1. **Data Ingestion & Preprocessing**
   - Loads raw historical data (demand, weather, price, etc.).
   - Cleans data: handles missing values, removes duplicates, encodes categorical variables, and scales features.
   - Extracts time-based features (day, month, year, day of week) for richer modeling.

2. **Exploratory Data Analysis (EDA)**
   - Generates a comprehensive set of visualizations to understand demand patterns, feature relationships, and data quality.
   - EDA plots are automatically saved for reporting and further analysis.

3. **Model Training**
   - Trains two types of models:
     - **Random Forest:** A robust, interpretable machine learning model.
     - **LSTM (Long Short-Term Memory):** A deep learning model specialized for time series forecasting.
   - Models are saved for later evaluation and forecasting.

4. **Model Evaluation**
   - Evaluates model performance on test data.
   - Produces plots comparing actual vs. predicted demand, and analyzes errors and trends.

5. **Forecasting**
   - Uses the trained LSTM model to generate future demand forecasts (e.g., for the next month).
   - Forecasts are visualized and saved for decision-making.

### Key Features

- **Modular Codebase:** Clean separation of data, models, evaluation, and scripts.
- **Reproducibility:** All outputs (plots, models, processed data) are saved with clear naming and structure.
- **Extensibility:** Easily add new models, features, or visualizations.
- **Visualization:** Rich EDA and performance plots for deep insight into the data and models.
- **Production-Ready:** Suitable for both experimentation and deployment.

## Setup

1. **Clone the repository**
   ```sh
   git clone <repo-url>
   cd electricity-forecasting
   ```
2. **Create and activate a virtual environment**
   - Windows:
     ```sh
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - Mac/Linux:
     ```sh
     python3 -m venv venv
     source venv/bin/activate
     ```
3. **Install all dependencies**
   ```sh
   pip install -r requirements.txt
   ```
4. **Add your data**
   - Place your raw data file as `data/raw/complete_dataset.csv`.

You're now ready to run the data pipeline, train models, and generate forecasts!

## Usage

### 1. Data Processing & EDA
- Run the data pipeline to clean, preprocess, and analyze your data:
  ```sh
  python scripts/data_pipeline.py
  ```
  - Outputs:
    - Processed data in `data/processed/`
    - EDA plots in `results/figures/eda/`

### 2. Model Training
- Train the machine learning and deep learning models:
  ```sh
  python scripts/train_models.py
  ```
  - Outputs:
    - Trained models in `models/saved_models/`

### 3. Model Evaluation
- Evaluate model performance and generate comparison plots:
  ```sh
  python scripts/evaluate_models.py
  ```
  - Outputs:
    - Performance plots in `results/figures/model_performance/`

### 4. Forecasting
- Generate a one-month demand forecast:
  ```sh
  python scripts/generate_forecasts.py
  ```
  - Outputs:
    - Forecast plot in `results/figures/forecasts/one_month_forecast.png`

## Model Accuracy

This project achieves strong predictive performance on the provided dataset:

- **Random Forest:** Typically achieves over 94% accuracy (R² or explained variance) on test data for demand forecasting.
- **LSTM:** Achieves similar or slightly higher accuracy, especially for capturing temporal patterns and trends.

Accuracy may vary depending on data quality, feature engineering, and hyperparameter tuning. Performance plots and detailed metrics are saved in `results/figures/model_performance/` after running the evaluation script.

---

**Tip:** All paths and parameters can be configured in `config.yaml`.

## Configuration
- All paths and parameters are set in `config.yaml`.
- You can adjust model hyperparameters, test/validation split, and output locations there.

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies

## Extending the Project
- Add new models in `src/models/`
- Add new visualizations in `src/evaluation/`
- Add new scripts in `scripts/`

## License
MIT license.

---

**For any issues or contributions, please open an issue or pull request!** 