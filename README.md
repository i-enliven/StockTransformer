# ArcLLM: Stock Forecasting Transformer

ArcLLM is a quantitative research framework for training, evaluating, and running inference with Transformer architectures on financial market data. The system forecasts S&P 500 stock components and macro asset price trends using high-capacity models optimized for Blackwell GPUs.

---

## Key Features

1. **Stock Transformer Architecture (`model.py`):**
   * Built with NVIDIA's **Transformer Engine (`te`)** utilizing causal attention masks and `bshd` format.
   * Employs input projection layers and learnable positional encodings.
   * Configured with embedding dropout ($p=0.2$) and attention/feed-forward hidden dropout ($p=0.2$) to prevent memorization of market noise.

2. **10-Model Ensemble:**
   * Trains 10 independent models with randomized seed initializations and shuffling patterns.
   * Combines predictions at inference time by averaging probabilities across all checkpoints.
   * Smooths individual model overconfidence and extracts robust consensus market signals.

3. **Multi-Modal Asset Log-Returns:**
   * Ingests S&P 500 daily close prices (`sp500_close.csv`) and volume (`sp500_volume.csv`).
   * Integrates macro features (`macro_close.csv`): Bitcoin (BTC-USD), Ethereum (ETH-USD), Gold (GC=F), Brent Crude (BZ=F), US Dollar Index (DX-Y.NYB), 10-Year Treasury Yield (^TNX), and the CBOE Volatility Index (^VIX).
   * Models raw features strictly as log-returns with clipping protection for zero/null volumes.

4. **FP8 Quantization:**
   * Utilizes NVIDIA Transformer Engine autocasting with the `DelayedScaling` recipe for standard 8-bit hardware-accelerated precision operations.

---

## Getting Started

### 1. Installation
This repository uses `uv` for dependency management. To set up the virtual environment and install dependencies:
```bash
# Using uv
uv venv
uv sync
```

### 2. Training the Ensemble
Run the training script to clean previous checkpoints, train the 10-model ensemble across 1008 trading days (approx. 4 years of history), and run the chronological backtest evaluation:
```bash
.venv/bin/python train.py
```
Checkpoints are saved separately in `checkpoints/stock_transformer_seed_{seed}.pt`.

### 3. Running Inference
Generate buy recommendations for the next trading day by averaging predictions across all ensemble checkpoints:
```bash
.venv/bin/python infer.py
```
This prints the Top-10 stock tickers to buy and saves the full ranked predictions to `predictions.csv`.

---

## Future Directions

* **Class Imbalance & Label Thresholding:** Daily market-wide directions can fluctuate due to broad macro moves, leading to high target correlations. Implementing custom label thresholding or market-neutral indexing could isolate individual stock alpha.
* **Feature Regularization:** Incorporate Z-score or MinMaxScaler transformations directly into the PyTorch pipeline to protect convergence from extreme outliers in daily return features.
* **Dynamic Learning Rate Schedulers:** Implement cosine annealing or one-cycle learning rate schedules to help the model escape suboptimal local minima.
* **Alternative Loss Functions:** Use a fused focal loss or ordinal ranking system to focus learning resources strictly on the top performers.
