# Stock Price Prediction with PyTorch (LSTM & GRU)

## Description

A time-series stock price prediction pipeline built with PyTorch. This project fetches historical stock data, performs exploratory data analysis, and trains deep learning sequence models to forecast future prices. The main goal is a **comparative study of LSTM vs. GRU architectures**, evaluating which recurrent model performs better on stock price forecasting.

## 🚀 Project Roadmap (4 Weeks)

### 📊 Week 1: Data Acquisition & Exploratory Data Analysis (EDA)
- [x] Set up the GitHub repository structure and virtual environment.
- [x] Fetch historical stock data using Yahoo Finance (yfinance) or pandas-datareader.
- [x] Perform Exploratory Data Analysis (EDA) using Pandas and Matplotlib/Seaborn.
- [x] Deliverable: Jupyter Notebook with data visualizations and clean, saved CSV datasets.

### ⚙️ Week 2: Data Preprocessing & Baseline LSTM Model
- [x] Prepare the dataset for time-series (scaling, sequence creation, train-test split).
- [x] Create PyTorch Dataset and DataLoader classes.
- [x] Build and train a baseline LSTM model in PyTorch.
- [x] Deliverable: Working PyTorch training pipeline with training loss visualization.

### 🧠 Week 3: GRU Model Implementation & Hyperparameter Tuning
- [x] Implement the GRU model architecture in PyTorch.
- [x] Train the GRU model on the same prepared dataset.
- [x] Experiment with hyperparameters (learning rate, hidden dimensions, epochs).
- [ ] Deliverable: Trained LSTM and GRU model weights saved locally.

### 📈 Week 4: Model Evaluation, Comparison & Final Polish
- [ ] Evaluate both models on test data using metrics like RMSE, MSE, and MAE.
- [ ] Plot predictions vs. actual stock prices for both models.
- [x] Document final results, limitations, and key learnings in the README.md.
- [ ] Deliverable: Fully completed GitHub repository, clean notebook, and a comparison table.

## Directory Structure

```
├── data/
│   ├── raw/                       # Raw OHLCV CSVs downloaded via fetch_data.py (not in git)
│   └── processed/                 # preprocess.py output: train/val/test.csv + scaler.pkl (not in git)
├── models/
│   └── best_model.pt              # Model weights saved at best validation loss (not in git)
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Data exploration, cleaning, technical indicators (SMA/RSI/MACD)
│   └── 02_results.ipynb            # Final metrics and prediction chart summary
├── results/
│   ├── loss_curve.png             # Train/validation loss curve
│   └── predictions_vs_actual.png  # Actual vs. predicted prices on test set
├── scripts/
│   ├── fetch_data.py              # Download OHLCV data using yfinance
│   ├── preprocess.py              # Add technical indicators, train/val/test split, scaling
│   ├── test_dataset.py            # StockDataset + DataLoader shape test
│   ├── hyperparameter_search.py   # Manual grid search for lr / hidden_size / lookback
│   ├── compare_models.py          # LSTM vs GRU: final val loss and training time comparison
│   ├── evaluate.py                # Test set evaluation (RMSE/MAE/MAPE) and prediction plot
│   └── symbols/bist30.txt         # BIST-30 ticker symbol list
├── src/
│   ├── dataset.py                 # StockDataset (sliding window PyTorch Dataset)
│   ├── models.py                  # LSTMModel & GRUModel definitions
│   └── train.py                   # train_one_epoch / validate / train_model and full training loop
├── app.py                         # Desktop GUI: BIST-30 symbol list + "Analyze" button
├── requirements.txt
└── README.md

```

## Installation

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Usage

### 1. Download Data

```bash
python scripts/fetch_data.py THYAO.IS
# Or for all symbols in the BIST-30 list:
python scripts/fetch_data.py --list-file scripts/symbols/bist30.txt
```

### 2. Preprocessing (technical indicators + chronological train/val/test split + scaling)

```bash
python scripts/preprocess.py --symbol THYAO.IS
```

`train.csv`, `val.csv`, `test.csv`, and `scaler.pkl` (fitted on train set only) are saved under `data/processed/`.

### 3. Model Training

```bash
python src/train.py
```

Trains `LSTMModel` for 25 epochs; the model is saved as `models/best_model.pt` at the best validation loss, and the loss curve is saved as `results/loss_curve.png`.

### 4. Evaluation

```bash
python scripts/evaluate.py
```
Runs `models/best_model.pt` on the test set, inverse-transforms predictions back to original price scale, prints RMSE/MAE/MAPE to the console, and generates the `results/predictions_vs_actual.png` plot.

### Other Scripts (Optional)

```bash
# Compare LSTM and GRU with the same hyperparameters (final val loss + training time)
python scripts/compare_models.py

# Manual grid search for lr / hidden_size / lookback, prints the top 3 combinations
python scripts/hyperparameter_search.py
```

## Desktop Application (GUI)

There is also a simple Tkinter interface to work from a single window instead of the command line:

```bash
python app.py
```

The home screen features the **Stock Analysis** header along with **Project Purpose** / **How to Use** info buttons (clicking them expands the explanatory text downwards). Clicking one of the dark navy BIST-30 buttons in the left sidebar displays that stock's opening price; clicking the **Analyze** button beneath it sequentially executes `fetch_data.py` → `preprocess.py` → `train_model()` (25 epochs) → `run_evaluation()` in the background (without freezing the UI) for the selected stock. A step-by-step progress log is shown during execution, and upon completion, RMSE/MAE/MAPE metrics along with loss and prediction plots are displayed in the window. Each analysis overwrites the previous run's outputs in `data/processed/`, `models/best_model.pt`, and `results/` — meaning the application always displays a single active analysis state.

## Results

`models/best_model.pt` (the LSTM model saved at the best validation loss) on the THYAO.IS test set:

| Metric | Value |
| --- | --- |
| RMSE | 9.27 TRY |
| MAE | 7.03 TRY |
| MAPE | 2.29% |

![Actual vs. predicted closing price on the test set](results/predictions_vs_actual.png)

For detailed metric calculations and plots, see `scripts/evaluate.py` and `notebooks/02_results.ipynb`.

## Limitations

- The model was trained and evaluated on a single ticker (THYAO.IS); generalization to other stocks was not tested.
- The model predicts the next day's Close price; due to high day-to-day autocorrelation in price series, the predicted curve tends to lag behind the actual price by one day (as visible in the prediction plot) — this has not yet been benchmarked against a naive baseline such as "using the previous day's price as the prediction."
- The GRU architecture was compared in a single run via `compare_models.py`; it was not saved as a separate checkpoint or evaluated under the same formal test set evaluation (RMSE/MAE/MAPE) as LSTM.
- Hyperparameter search was conducted over a small grid (3 lr × 2 hidden_size × 2 lookback) with few epochs (5); a broader search (e.g., with Optuna) might yield different and potentially better results.
- Only price/volume data and technical indicators (SMA-20, RSI-14, MACD) were used; external data such as news feeds or market sentiment was not included.
