# IBKR Trading Bot with Deep Learning

A fully automated trading system using Interactive Brokers API (IBAPI) and an ensemble of deep learning models (GRU, LSTM, CNN). This system supports historical data collection, model training, backtesting, Monte Carlo simulations, and live trading on IBKR.

## 🚀 Features

- **IBKR API Connection**: Place market, limit, bracket, and combo orders.
- **Historical Data Fetching**: Parallelized multi-threaded download using IBKR's rate limits.
- **Database**: Efficient storage of stock OHLCV data using SQLite.
- **Model Architectures**:
  - CNN, LSTM, GRU — with PyTorch.
  - Customizable hyperparameters via Optuna.
- **Backtesting**:
  - Custom framework with stop-loss, take-profit.
  - Handles slippage, commissions, and realistic market fills.
- **Monte Carlo Simulation**: Stress-test strategies using synthetic data generated with GBM (Geometric Brownian Motion).
- **Live Trading**:
  - Real-time 5-minute bars.
  - Predictive ensemble models per stock.
  - Auto position management, order placement, and execution.
- **Risk Management**:
  - Position sizing based on available funds.
  - Stop-loss and profit-taking brackets.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_GITHUB/ibkr-deep-learning-bot.git
   cd ibkr-deep-learning-bot
