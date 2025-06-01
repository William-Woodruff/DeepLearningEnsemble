import threading
from threading import Event, Thread, Semaphore, Lock
import pandas as pd
from datetime import datetime, timedelta
import time
from collections import deque
import warnings
from queue import Queue
import multiprocessing
import psutil
import sqlite3
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import optuna
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
import json
import inspect
from joblib import dump, load
import random
from pandas.tseries.offsets import BDay
from scipy import stats
import pytz

# Mute Warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Choices for a categorical distribution.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

# ================= Creating and Training Models =================

# - Model Classes - 
class AdvancedCNNRegressor(nn.Module):
    """ 
    CNN Architecture 
    """

    def __init__(
        self,
        input_channels,         
        conv_filters=[32, 64],  
        kernel_sizes=[3, 3],     
        pool_size=2,            
        dropout=0.3,             
        use_batchnorm=True,      
        fc_units=[128],          
        output_size=1            
    ):
        super(AdvancedCNNRegressor, self).__init__()

        self.use_batchnorm = use_batchnorm
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        prev_channels = input_channels

        
        for out_channels, kernel_size in zip(conv_filters, kernel_sizes):
            self.convs.append(
                nn.Conv2d(prev_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
            )
            if self.use_batchnorm:
                self.bns.append(nn.BatchNorm2d(out_channels))
            prev_channels = out_channels

        self.pool = nn.MaxPool2d(pool_size)
        self.dropout = nn.Dropout(dropout)

        self.fc_layers = None
        self.fc_units = fc_units
        self.output_size = output_size

    def _get_conv_output(self, x):
        for idx, conv in enumerate(self.convs):
            x = conv(x)
            if self.use_batchnorm:
                x = self.bns[idx](x)
            x = F.relu(x)
            x = self.pool(x)
        return x.view(x.size(0), -1)  

    def build_fc_layers(self, conv_out_size):
        layers = []
        prev_units = conv_out_size
        for units in self.fc_units:
            layers.append(nn.Linear(prev_units, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout.p))
            prev_units = units
        layers.append(nn.Linear(prev_units, self.output_size))  
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self._get_conv_output(x)

        if self.fc_layers is None:
          self.fc_layers = self.build_fc_layers(x.size(1)).to(x.device)

        out = self.fc_layers(x)
        return out

class AdvancedLSTMModel(nn.Module):
    """ 
    LSTM Architecture 
    """
    
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.3, use_batchnorm=False, output_size=1):
        super(AdvancedLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_batchnorm = use_batchnorm
        self.num_directions = 1  

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        if self.use_batchnorm:
            self.bn = nn.BatchNorm1d(hidden_size * self.num_directions)
        else:
            self.bn = nn.Identity()  

        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = out[:, -1, :]  
        out = self.bn(out)
        out = self.fc(out)
        return out

class AdvancedGRUModel(nn.Module):
    """ 
    GRU Architecture 
    """
    
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.3, use_batchnorm=False, output_size=1):
        super(AdvancedGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_batchnorm = use_batchnorm
        self.num_directions = 1  

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        if self.use_batchnorm:
            self.bn = nn.BatchNorm1d(hidden_size * self.num_directions)
        else:
            self.bn = nn.Identity()  

        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x):
        out, hn = self.gru(x)   
        out = out[:, -1, :]     
        out = self.bn(out)
        out = self.fc(out)
        return out

# - Trains All Models - 
def train_model(model, optimizer, criterion, epochs=100, trial=None, patience=5, CNN=False):
    """
    Trains a given model with given hyperparamters using ADAM as the optimizer, calculating training loss, 
    validation loss, and early stopping once val loss increases
    """

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    epochs_since_improvement = 0

    # Choose the correct loaders
    train_dl = train_loader_cnn if CNN else train_loader
    val_dl = val_loader_cnn if CNN else val_loader

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        train_losses.append(total_train_loss / len(train_dl))

        # === Validation ===
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dl)
        val_losses.append(avg_val_loss)
        trial_info = f"Trial {trial.number} - " if trial else ""
        print(f"{trial_info}Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.6f}, Val Loss: {avg_val_loss:.6f}")

        # === Early stopping ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    optimal_epoch = val_losses.index(best_val_loss) + 1
    return train_losses, val_losses, optimal_epoch

# - Objective Functions - 
def objective_CNN(trial):
    """
    Creates the CNN Objective function of possible hyperparamters used by Optuna to exptrapolate the combo that minimizes the loss
    """
    
    # === Suggest hyperparameters ===
    conv_filters = trial.suggest_categorical('conv_filters', [
        [32, 64],
        [64, 128],
        [32, 64, 128]
    ])

    kernel_sizes = trial.suggest_categorical('kernel_sizes', [
        [3, 3],
        [5, 3],
        [3, 5]
    ])

    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    use_batchnorm = trial.suggest_categorical('use_batchnorm', [True, False])
    fc_units = trial.suggest_categorical('fc_units', [
        [128],
        [256, 128],
        [512, 256, 128]
    ])

    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    # === Build model ===
    model = AdvancedCNNRegressor(
        input_channels=1,
        conv_filters=conv_filters,
        kernel_sizes=kernel_sizes,
        dropout=dropout,
        use_batchnorm=use_batchnorm,
        fc_units=fc_units,
        output_size=1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # === Train model ===
    train_losses, val_losses, optimal_epoch = train_model(
        model,
        optimizer,
        criterion,
        epochs=25,       
        patience=5,        
        trial=trial,
        CNN=True
    )
    CNN_losses[trial.number] = (train_losses, val_losses)

    trial.set_user_attr("best_epoch", optimal_epoch)

    return min(val_losses)

def objective_LSTM(trial):
    """
    Creates the LSTM Objective function of possible hyperparamters used by Optuna to exptrapolate the combo that minimizes the loss
    """

    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    use_batchnorm = trial.suggest_categorical('use_batchnorm', [True, False])

    model = AdvancedLSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, use_batchnorm=use_batchnorm, output_size=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses, val_losses, best_epoch = train_model(model, optimizer, criterion, epochs=20, patience=5, trial=trial, CNN = False)
    LSTM_losses[trial.number] = (train_losses, val_losses)
    trial.set_user_attr("best_epoch", best_epoch)
    return min(val_losses)

def objective_GRU(trial):
    """
    Creates the GRU Objective function of possible hyperparamters used by Optuna to exptrapolate the combo that minimizes the loss
    """

    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    use_batchnorm = trial.suggest_categorical('use_batchnorm', [True, False])

    model = AdvancedGRUModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, use_batchnorm=use_batchnorm, output_size=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses, val_losses, best_epoch = train_model(model, optimizer, criterion, epochs=20, patience=5, trial=trial, CNN = False)
    GRU_losses[trial.number] = (train_losses, val_losses)
    trial.set_user_attr("best_epoch", best_epoch)

    return min(val_losses)

# - Optuna Functions -
def CNN_optuna():
    """
    Creates an Optuna study for CNN hyperparamter optimization; 
    Then retrains the model with the lowest val loss and calculates metrics for teh testing data
    """
    
    print("\n")
    study = optuna.create_study(direction="minimize", study_name="CNN_Training")
    study.optimize(objective_CNN, n_trials=10)
    best_params = study.best_params
    print("Best Hyperparameters:", best_params)
    best_params["input_channels"] = 1
    best_params["input_size"] = input_size

    best_trial = study.best_trial
    optimal_epochs = best_trial.user_attrs["best_epoch"] + 2

    model = AdvancedCNNRegressor(
            input_channels=1,
            conv_filters=best_params['conv_filters'],
            kernel_sizes=best_params['kernel_sizes'],
            dropout=best_params['dropout'],
            use_batchnorm=best_params['use_batchnorm'],
            fc_units=best_params['fc_units'],
            output_size=1
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
    criterion = nn.MSELoss()

    train_losses, val_losses, _ = train_model(model, optimizer, criterion, epochs=optimal_epochs, patience=optimal_epochs,CNN=True)

    # === Evaluation on Test Set ===
    model.eval()
    y_preds = []
    y_true = []

    with torch.no_grad():
        for xb, yb in test_loader_cnn:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            yb = yb.cpu().numpy()
            y_preds.append(pred)
            y_true.append(yb)

    y_preds = np.concatenate(y_preds)
    y_true = np.concatenate(y_true)

    # === Regression Metrics ===
    mse = mean_squared_error(y_true, y_preds)
    mae = mean_absolute_error(y_true, y_preds)
    r2 = r2_score(y_true, y_preds)

    # === Confusion Matrix (Discretized) ===
    y_pred_disc = np.where(y_preds > 0, 1, 0)
    y_true_disc = np.where(y_true > 0, 1, 0)
    cm = confusion_matrix(y_true_disc, y_pred_disc)

    print(f"\n📊 Test Metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²:  {r2:.6f}")
    print(f"  Confusion Matrix (Gain vs Loss):\n{cm}")

    return model, best_params

def LSTM_optuna():
    """
    Creates an Optuna study for LSTM hyperparamter optimization; 
    Then retrains the model with the lowest val loss and calculates metrics for teh testing data
    """

    print("\n")
    study = optuna.create_study(direction="minimize", study_name="LSTM_Training")
    study.optimize(objective_LSTM, n_trials=10)
    best_params = study.best_params
    print("Best Hyperparameters:", best_params)
    best_params["input_size"] = input_size

    best_trial = study.best_trial
    optimal_epochs = best_trial.user_attrs["best_epoch"] + 2

    # === Retrain with Best ===
    model = AdvancedLSTMModel(
        X_train.shape[2],
        best_params['hidden_size'],
        best_params['num_layers'],
        best_params['dropout'],
        best_params['use_batchnorm'],
        1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
    criterion = nn.MSELoss()

    train_losses, val_losses, _ = train_model(model, optimizer, criterion, epochs=optimal_epochs, patience=optimal_epochs)

    # === Evaluation on Test Set ===
    model.eval()
    y_preds = []
    y_true = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            yb = yb.cpu().numpy()
            y_preds.append(pred)
            y_true.append(yb)

    y_preds = np.concatenate(y_preds)
    y_true = np.concatenate(y_true)

    # === Regression Metrics ===
    mse = mean_squared_error(y_true, y_preds)
    mae = mean_absolute_error(y_true, y_preds)
    r2 = r2_score(y_true, y_preds)

    # === Confusion Matrix (Discretized) ===
    y_pred_disc = np.where(y_preds > 0, 1, 0)
    y_true_disc = np.where(y_true > 0, 1, 0)
    cm = confusion_matrix(y_true_disc, y_pred_disc)

    print(f"\n📊 Test Metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²:  {r2:.6f}")
    print(f"  Confusion Matrix (Gain vs Loss):\n{cm}")

    return model, best_params

def GRU_optuna():
    """
    Creates an Optuna study for GRU hyperparamter optimization; 
    Then retrains the model with the lowest val loss and calculates metrics for teh testing data
    """

    print("\n")
    study = optuna.create_study(direction="minimize", study_name="GRU_Training")
    study.optimize(objective_GRU, n_trials=10)
    best_params = study.best_params
    print("Best Hyperparameters:", best_params)
    best_params["input_size"] = input_size

    best_trial = study.best_trial
    optimal_epochs = best_trial.user_attrs["best_epoch"] + 2

    # === Retrain with Best ===
    model = AdvancedGRUModel(
        X_train.shape[2],
        best_params['hidden_size'],
        best_params['num_layers'],
        best_params['dropout'],
        best_params['use_batchnorm'],
        1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
    criterion = nn.MSELoss()

    train_losses, val_losses, _ = train_model(model, optimizer, criterion, epochs=optimal_epochs, patience=optimal_epochs)

    # === Evaluation on Test Set ===
    model.eval()
    y_preds = []
    y_true = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            yb = yb.cpu().numpy()
            y_preds.append(pred)
            y_true.append(yb)

    y_preds = np.concatenate(y_preds)
    y_true = np.concatenate(y_true)

    # === Regression Metrics ===
    mse = mean_squared_error(y_true, y_preds)
    mae = mean_absolute_error(y_true, y_preds)
    r2 = r2_score(y_true, y_preds)

    # === Confusion Matrix (Discretized) ===
    y_pred_disc = np.where(y_preds > 0, 1, 0)
    y_true_disc = np.where(y_true > 0, 1, 0)
    cm = confusion_matrix(y_true_disc, y_pred_disc)

    print(f"\n📊 Test Metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²:  {r2:.6f}")
    print(f"  Confusion Matrix (Gain vs Loss):\n{cm}")

    return model, best_params

# - All Together -
def load_data_from_db(ticker):
    """
    Pulls normal market data from the internal SQLite for a [stock], TLT, and SPY
    """
    
    conn = sqlite3.connect("stocks_subset.db")

    # Load individual stock data
    stock = pd.read_sql(
        "SELECT * FROM stock_data WHERE ticker = ? ORDER BY date",
        conn,
        params=(ticker,),
        parse_dates=['date']
    )

    # Load SPY data
    market = pd.read_sql(
        "SELECT * FROM stock_data WHERE ticker = 'SPY' ORDER BY date",
        conn,
        parse_dates=['date']
    )

    # Load TLT data
    bond = pd.read_sql(
        "SELECT * FROM stock_data WHERE ticker = 'TLT' ORDER BY date",
        conn,
        parse_dates=['date']
    )

    conn.close()

    for df in [stock, market, bond]:
        df.set_index('date', inplace=True)
        df.drop(columns=['ticker'], inplace=True)
        df.sort_index(inplace=True)

    stock = stock.between_time('09:30', '16:00', inclusive='left')
    market = market.between_time('09:30', '16:00', inclusive='left')
    bond = bond.between_time('09:30', '16:00', inclusive='left')

    print(stock.shape, market.shape, bond.shape)

    return stock, market, bond

def add_all_indicators(df):
    """
    Adds 17 technical indicators to a given dataframe of stock data with (Open, High, Low, Close, Volume) aka OHLCV values
    """
    
    
    required_cols = ['close', 'high', 'low', 'volume']
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()

    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()

    rolling_mean = df['close'].rolling(window=20).mean()
    rolling_std = df['close'].rolling(window=20).std()
    df['BB_MID'] = rolling_mean
    df['BB_UPPER'] = rolling_mean + (2 * rolling_std)
    df['BB_LOWER'] = rolling_mean - (2 * rolling_std)

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = high_low.combine(high_close, max).combine(low_close, max)
    df['ATR_14'] = tr.rolling(window=14).mean()

    tp = (df['high'] + df['low'] + df['close']) / 3
    tp_ma = tp.rolling(window=20).mean()
    tp_std = tp.rolling(window=20).std()
    df['CCI_20'] = (tp - tp_ma) / (0.015 * tp_std)

    df['ROC_10'] = df['close'].pct_change(periods=10) * 100

    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['Stoch_%K'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
    df['Stoch_%D'] = df['Stoch_%K'].rolling(window=3).mean()

    df['Williams_%R'] = -100 * ((high_14 - df['close']) / (high_14 - low_14))

    return df

def clean_and_change(stock, market, bond, train = False): 
    """
    Cleans the data - no duplicates, no NA vals. Concatonates the three df into one main df
    Creates the target value and normalizes the input data
    """
    
    start = time.time()

    stock = stock.loc[:, ~stock.columns.duplicated()]
    market = market.loc[:, ~market.columns.duplicated()]
    bond = bond.loc[:, ~bond.columns.duplicated()]

    stock = stock[~stock.index.duplicated(keep='first')]
    market = market[~market.index.duplicated(keep='first')]
    bond = bond[~bond.index.duplicated(keep='first')]

    stock = add_all_indicators(stock)
    market = add_all_indicators(market)
    bond = add_all_indicators(bond)

    print(f"{time.time() - start} seconds")

    print(f"Data dimensions for Stock: {stock.shape}, Bond: {bond.shape}, Market: {market.shape}")

    df = pd.concat([
        stock.add_suffix('_stock'),
        bond.add_suffix('_bond'),
        market.add_suffix('_market')
    ], axis=1)

    df['Target'] = df['close_stock'].shift(-1) - df['close_stock']
    df.dropna(inplace=True)
    print(df.shape)
    features = df.drop(columns=['Target'])
    scaler = StandardScaler() if train == True else load(f"saved_models/{stock_ticker}/scaler.save")
    features_scaled = scaler.fit_transform(features) if train == True else scaler.transform(features)
    if train == True:
        os.makedirs(f"saved_models/{stock_ticker}", exist_ok=True)
        dump(scaler, f"saved_models/{stock_ticker}/scaler.save")
    print(features_scaled.shape)
    input_size = features_scaled.shape[1]  # number of features

    return df, features_scaled, input_size

def tensor_and_dataloader(df, features_scaled, MC = False): 
    """
    Creates a sequence such that each row has teh previous 20 rows as feautures
    Splits data into training, val, testing and creates Tensors for 
    both CNN and LSTM/GRU and loads them into dataloader for batch use
    """
    
    def create_sequences(X, y, seq_len):
        Xs, ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i:i+seq_len])
            ys.append(y[i+seq_len])
        return np.array(Xs), np.array(ys)

    SEQ_LEN = 20
    X_seq, y_seq = create_sequences(features_scaled, df['Target'].values, SEQ_LEN)

    if MC == True:

        X_test = torch.tensor(X_seq, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_seq, dtype=torch.float32).view(-1, 1).to(device)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)
        
        X_seq_cnn = X_seq.reshape(X_seq.shape[0], 1, SEQ_LEN, X_seq.shape[2])
        X_test_cnn = torch.tensor(X_seq_cnn, dtype=torch.float32).to(device)
        test_loader_cnn = DataLoader(TensorDataset(X_test_cnn, y_test), batch_size=64, shuffle=False)
        return test_loader, test_loader_cnn


    # === Split Data: 80-10-10 ===
    total = len(X_seq)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    X_train, X_val, X_test = X_seq[:train_end], X_seq[train_end:val_end], X_seq[val_end:]
    y_train, y_val, y_test = y_seq[:train_end], y_seq[train_end:val_end], y_seq[val_end:]

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False) 
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False) 

    # === CNN Input (4D): reshape X_seq ===

    X_seq_cnn = X_seq.reshape(X_seq.shape[0], 1, SEQ_LEN, X_seq.shape[2])

    X_train_cnn, X_val_cnn, X_test_cnn = X_seq_cnn[:train_end], X_seq_cnn[train_end:val_end], X_seq_cnn[val_end:]

    X_train_cnn = torch.tensor(X_train_cnn, dtype=torch.float32).to(device)
    X_val_cnn = torch.tensor(X_val_cnn, dtype=torch.float32).to(device)
    X_test_cnn = torch.tensor(X_test_cnn, dtype=torch.float32).to(device)

    train_loader_cnn = DataLoader(TensorDataset(X_train_cnn, y_train), batch_size=64, shuffle=True) 
    val_loader_cnn = DataLoader(TensorDataset(X_val_cnn, y_val), batch_size=64, shuffle=False)
    test_loader_cnn = DataLoader(TensorDataset(X_test_cnn, y_test), batch_size=64, shuffle=False) 

    return train_loader, val_loader, test_loader, train_loader_cnn, val_loader_cnn, test_loader_cnn, X_train

def Model_stats(CNN_model, LSTM_model, GRU_model):
    """
    Evaluates testing metrics for each model and each ensemble combination
    """

    GRU_model.eval()
    LSTM_model.eval()
    CNN_model.eval()

    y_GRU_preds = []
    y_LSTM_preds = []
    y_CNN_preds = []
    y_true = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)

            pred_GRU = GRU_model(xb).cpu().numpy()
            pred_LSTM = LSTM_model(xb).cpu().numpy()
            yb = yb.cpu().numpy()

            y_GRU_preds.append(pred_GRU)
            y_LSTM_preds.append(pred_LSTM)
            y_true.append(yb)

    with torch.no_grad():
        for xb, yb in test_loader_cnn:
            xb = xb.to(device)
            pred_CNN = CNN_model(xb).cpu().numpy()
            y_CNN_preds.append(pred_CNN)

    y_GRU_preds = np.concatenate(y_GRU_preds).squeeze()
    y_LSTM_preds = np.concatenate(y_LSTM_preds).squeeze()
    y_CNN_preds = np.concatenate(y_CNN_preds).squeeze()
    y_true = np.concatenate(y_true).squeeze()

    y_true_bin = (y_true > 0).astype(int)

    model_preds = {
        'GRU': y_GRU_preds,
        'LSTM': y_LSTM_preds,
        'CNN': y_CNN_preds,
        'GRU+LSTM': (y_GRU_preds + y_LSTM_preds) / 2,
        'GRU+CNN': (y_GRU_preds + y_CNN_preds) / 2,
        'LSTM+CNN': (y_LSTM_preds + y_CNN_preds) / 2,
        'GRU+LSTM+CNN': (y_GRU_preds + y_LSTM_preds + y_CNN_preds) / 3,
    }

    results = []

    for name, preds in model_preds.items():
        y_pred_bin = (preds > 0).astype(int)

        mse = mean_squared_error(y_true, preds)
        mae = mean_absolute_error(y_true, preds)
        r2 = r2_score(y_true, preds)
        cm = confusion_matrix(y_true_bin, y_pred_bin)

        results.append({
            'Model': name,
            'MSE': mse,
            'MAE': mae,
            'R²': r2,
            'Confusion Matrix': cm
        })

    results_df = pd.DataFrame(results)

    for row in results:
        print(f"\n📊 {row['Model']}:")
        print(f"  MSE: {row['MSE']:.6f}")
        print(f"  MAE: {row['MAE']:.6f}")
        print(f"  R²:  {row['R²']:.6f}")
        print(f"  Confusion Matrix:\n{row['Confusion Matrix']}")

def save_model(model, hyperparams, ticker, model_type):
    """
    Saves a models wieghts and hyperparemters
    """

    os.makedirs(f"saved_models/{ticker}", exist_ok=True)
    torch.save(model.state_dict(), f"saved_models/{ticker}/{model_type}_weights.pt")
    with open(f"saved_models/{ticker}/{model_type}_hyperparams.json", "w") as f:
        json.dump(hyperparams, f, indent=4)

def load_model(ticker, model_type, model_class):
    """
    Loads the saved models' weights and hyperparamters and creates an identical copy for use
    """

    base_path = f"saved_models/{ticker}"
    weights_path = f"{base_path}/{model_type}_weights.pt"
    hyperparams_path = f"{base_path}/{model_type}_hyperparams.json"

    with open(hyperparams_path, "r") as f:
        hyperparams = json.load(f)

    model_args = inspect.signature(model_class.__init__).parameters
    valid_keys = [k for k in model_args if k != 'self']
    filtered_params = {k: v for k, v in hyperparams.items() if k in valid_keys}

    if model_type == "CNN":
        if "input_channels" not in filtered_params:
            filtered_params["input_channels"] = 1  
        if "input_size" not in hyperparams:
            raise ValueError("Missing 'input_size' in hyperparameters for CNN.")

    model = model_class(**filtered_params).to(device)

    if model_type == "CNN":
        seq_len = 20 
        dummy_input = torch.randn(1, filtered_params["input_channels"], seq_len, hyperparams["input_size"]).to(device)
        with torch.no_grad():
            model(dummy_input) 

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

# - Back Testing -
def BackTesting_predictions(CNN_model, LSTM_model, GRU_model, df, test_loader, test_loader_cnn):
    """
    Given the models and testing data, it returns the testing df with each models' prediction and every ensemble combination
    """

    GRU_model.eval()
    LSTM_model.eval()
    CNN_model.eval()

    y_GRU_preds = []
    y_LSTM_preds = []
    y_CNN_preds = []
    y_true = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)

            pred_GRU = GRU_model(xb).cpu().numpy()
            pred_LSTM = LSTM_model(xb).cpu().numpy()
            yb = yb.cpu().numpy()

            y_GRU_preds.append(pred_GRU)
            y_LSTM_preds.append(pred_LSTM)
            y_true.append(yb)

    with torch.no_grad():
        for xb, yb in test_loader_cnn:
            xb = xb.to(device)
            pred_CNN = CNN_model(xb).cpu().numpy()
            y_CNN_preds.append(pred_CNN)

    y_GRU_preds = np.concatenate(y_GRU_preds).squeeze()
    y_LSTM_preds = np.concatenate(y_LSTM_preds).squeeze()
    y_CNN_preds = np.concatenate(y_CNN_preds).squeeze()
    y_true = np.concatenate(y_true).squeeze()

    df['predict_GRU'] = y_GRU_preds
    df['predict_LSTM'] = y_LSTM_preds
    df['predict_CNN'] = y_CNN_preds
    df['predict_GRU_LSTM'] = (y_GRU_preds + y_LSTM_preds) / 2.0
    df['predict_GRU_CNN'] = (y_GRU_preds + y_CNN_preds) / 2.0
    df['predict_LSTM_CNN'] = (y_LSTM_preds + y_CNN_preds) / 2.0
    df['predict_GRU_LSTM_CNN'] = (y_GRU_preds + y_LSTM_preds + y_CNN_preds) / 3.0
    df['True'] = y_true

    return df

def backtest(df, sl_ratio=1, threshold=0.00, pred_ler = 0):
    """
    Custom Back testing function for given data and predictions with slippage and commission costs included
    """

    trade = False
    enter_next = False
    pred_avg = df['True'].abs().mean()
    real_pred_avg = float(df['predict_GRU_LSTM_CNN'].abs().max())
    if pred_ler != 0: 
        pred_scaler = pred_ler
    else: 
        pred_scaler = pred_avg / real_pred_avg

    trades = []

    entry_price = None
    TP1 = None
    TP2 = None
    SL = None
    trade_type = None
    quantity = 0
    entry_date = None
    signal_value = None

    for idx, row in df.iterrows():
        # ===== Enter trade on next bar after signal =====
        if trade and enter_next:
            slippage = round(random.uniform(-0.02, 0.02),2)
            entry_price = round(row['open_stock'],2) + slippage
            entry_date = idx
            quantity = 2
            enter_next = False
            profit = 0

            if random.random() > 0.9:
                trade = False
            if entry_price >= TP1 or entry_price <= SL: 
                trade = False

        # ===== Check for entry condition =====
        if not trade:
            if abs(row['predict_GRU_LSTM_CNN']) > pred_avg + threshold and abs(row['predict_GRU_LSTM_CNN']) * pred_scaler > 0.05:
                TP1 = round(row['close_stock'] + row['predict_GRU_LSTM_CNN']* pred_scaler / 1,2)
                TP2 = round(row['close_stock'] + row['predict_GRU_LSTM_CNN'] * pred_scaler /0.5,2)
                TP3 = round(row['close_stock'] + row['predict_GRU_LSTM_CNN'] * 1,2)
                SL  = round(row['close_stock'] - row['predict_GRU_LSTM_CNN'] * pred_scaler / 1 * sl_ratio,2)
                signal_value = row['predict_GRU_LSTM_CNN']

                trade_type = 'Long' if signal_value > 0 else 'Short'
                trade = True
                enter_next = True
            continue

        # ===== Handle Long Trade =====
        if trade_type == 'Long':

            if row['high_stock'] >= TP1 and quantity == 2:
                profit += TP1 - entry_price
                quantity -= 1

            if row['high_stock'] >= TP2 and quantity > 0:
                profit += (TP2 - entry_price) - 0.02
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': idx,
                    'entry_price': entry_price,
                    'exit_price': TP2,
                    'TP1': TP1,
                    'TP2': TP2,
                    'SL': SL,
                    'signal': signal_value,
                    'outcome': 'TP3',
                    'profit': profit
                })
                trade = False
                quantity = 0
            
            if row['low_stock'] <= SL and quantity > 0:
                exit_price = SL
                profit += (exit_price - entry_price) * quantity - 0.02
                val = 'SL' if quantity == 2 else "SL + TP1"
                quantity = 0
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': idx,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'TP1': TP1,
                    'TP2': TP2,
                    'SL': SL,
                    'signal': signal_value,
                    'outcome': val,
                    'profit': profit
                })
                trade = False
                quantity = 0

        # ===== Handle Short Trade =====
        elif trade_type == 'Short' :

            if row['high_stock'] >= SL and quantity > 0:
                exit_price = SL
                val = 'SL' if quantity == 2 else "SL + TP1"
                profit += (entry_price - exit_price) * quantity - 0.2
                quantity = 0
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': idx,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'TP1': TP1,
                    'TP2': TP2,
                    'SL': SL,
                    'signal': signal_value,
                    'outcome': val,
                    'profit': profit
                })
                trade = False
                quantity = 0

            if row['low_stock'] <= TP1 and quantity == 2:
                profit += entry_price - TP1
                quantity -= 1

            if row['low_stock'] <= TP2 and quantity > 0:
                profit += entry_price - TP2 - 0.02
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': idx,
                    'entry_price': entry_price,
                    'exit_price': TP2,
                    'TP1': TP1,
                    'TP2': TP2,
                    'SL': SL,
                    'signal': signal_value,
                    'outcome': 'TP3',
                    'profit': profit 
                })
                trade = False
                quantity = 0

    if len(trades) < 1: 
        return [], None, None
    
    trades_df = pd.DataFrame(trades)
    total_profit = trades_df['profit'].sum()
    total_trades = len(trades_df)

    return trades_df, total_profit, total_trades

def BackTest_fully():
    """
    Backtests many different combinations of threshold and stop loss ratio values 
    for maximized profits given a minimum number of trades
    """
    
    stock, market, bond = load_data_from_db(stock_ticker)
    df, features_scaled, input_size = clean_and_change(stock, market, bond)
    df = df[:int(df.shape[0] * 0.1)]
    features_scaled = features_scaled[:int(features_scaled.shape[0] * 0.1)]
    test_loader, test_loader_cnn = tensor_and_dataloader(df, features_scaled, MC = True)

    effective_len = len(test_loader.dataset)
    df = df.iloc[-effective_len:]
    print("trades df size", df.shape)
    
    df = BackTesting_predictions(CNN_model, LSTM_model, GRU_model, df, test_loader, test_loader_cnn)

    best_profit = float('-inf')
    best_trades_df = None
    best_params = None
    best_num_trades = None
    No_trades = 0

    min_abs = float(df['predict_GRU_LSTM_CNN'].abs().min())
    max_abs = float(df['predict_GRU_LSTM_CNN'].abs().max())

    pred_avg = df['True'].abs().mean()
    real_pred_avg = float(df['predict_GRU_LSTM_CNN'].abs().max())
    pred_scaler = pred_avg / real_pred_avg

    for trial in range(300):
        sl_ratio = random.uniform(0.5, 2.0)
        threshold = random.uniform(-pred_avg + min_abs, -pred_avg + max_abs)
        
        trades_df, total_profit, total_trades = backtest(df.copy(), sl_ratio=sl_ratio, threshold=threshold)

        if len(trades_df) == 0: 
            continue

        print(f"\n Trial {trial+1}: SL Ratio = {sl_ratio:.3f}, Threshold = {threshold:.3f}, Number of Trades = {total_trades}, Profit = {total_profit:.2f}")

        if total_profit > best_profit:
            best_profit = total_profit
            best_trades_df = trades_df.copy()
            best_params = (sl_ratio, threshold)
            best_num_trades = total_trades
            No_trades = 1
   
    if best_trades_df is not None:
        print('For 1 share')
        print(f"\n Best Profit: {best_profit:.2f} with SL Ratio = {best_params[0]:.3f}, Threshold = {best_params[1]:.3f}, Number of Trades = {best_num_trades}")
        print(f"Holding profit = {round((df['close_stock'].iloc[-1] - df['open_stock'].iloc[0]) * 3,2)}")
        best_trades_df.to_csv('X_trades.csv', index=False)
    else:
        print("No successful backtest trials.")
    
    if No_trades == 0: 
        return -1, -1, -1, -1
    
    return best_params[0], best_params[1], pred_scaler, best_profit

def save_BT_params(ticker, sl_ratio, threshold, pred_scaler):
    """
    saves Back testing paramters for live trading use
    """
    
    folder = f"saved_models/{ticker}"
    os.makedirs(folder, exist_ok=True)

    trade = {
        "sl_ratio": float(sl_ratio),
        "threshold": float(threshold), 
        "pred_scaler": float(pred_scaler)
    }

    with open(f"{folder}/trade_params.json", "w") as f:
        json.dump(trade, f)

def load_best_params(ticker):
    """ 
    Loads best backtetsing paramters 
    """

    with open(f"saved_models/{ticker}/trade_params.json", "r") as f:
        return json.load(f)

# - Monte Carlos -
def generate_synthetic_ohlcv(df, n_ticks=700, interval='5min'):
    """
    Creates synthetic OHLCV data for a stock using GBM (stochastic process) and adds timestamps to each
    """
    
    df['close_open_diff'] = df['open'] - df['close']
    df['High_close_diff'] = df['high'] - df['close']
    df['Low_open_diff'] = df['open'] - df['low']

    df['log_return'] = np.log(df['close'] / df['close'].shift(1)).dropna()
    mu = df['log_return'].mean()
    sigma = df['log_return'].std()
    S0 = df['close'].iloc[-1]

    Close = np.zeros(n_ticks)
    Open = np.zeros(n_ticks)
    High = np.zeros(n_ticks)
    Low = np.zeros(n_ticks)
    Volume = np.zeros(n_ticks, dtype=int)

    Close[0] = S0
    Open[0] = S0

    for i in range(1, n_ticks):
        drift = mu - 0.5 * sigma**2
        diffusion = sigma * np.random.normal()
        Close[i] = Close[i-1] * np.exp(drift + diffusion)
        Open[i] = Close[i-1] + df['close_open_diff'].sample(1).values[0]
        High[i] = Close[i] + df['High_close_diff'].sample(1).values[0]
        Low[i] = Open[i] - df['Low_open_diff'].sample(1).values[0]
        Volume[i] = df['volume'].sample(1).values[0]

    synthetic = pd.DataFrame({
        'open': Open,
        'high': High,
        'low': Low,
        'close': Close,
        'volume': Volume
    })

    last_time = df.index[-1] if isinstance(df.index[-1], pd.Timestamp) else pd.Timestamp(df.index[-1])
    current_time = last_time + BDay(1)
    current_time = pd.Timestamp(current_time.date()) + pd.Timedelta(hours=9, minutes=30)

    current_time = pd.Timestamp.today().normalize() + pd.Timedelta(hours=9, minutes=30)

    datetime_index = []
    interval_delta = pd.to_timedelta(interval)

    for _ in range(n_ticks):
        datetime_index.append(current_time)
        current_time += interval_delta
        if current_time.time() >= pd.to_datetime('16:00').time():
            current_time += BDay(1)
            current_time = pd.Timestamp(current_time.date()) + pd.Timedelta(hours=9, minutes=30)

    synthetic['Datetime'] = datetime_index
    synthetic.set_index('Datetime', inplace=True)
    return synthetic[1:]

def MC_backtest(stock_ticker, sl_ratio=1, threshold=0.00, scaler = 0):
    """
    Runs many MC simulations by using the synthetic GBM data and using it on as testing data for the ensemble prediction models
    Calculates metrics on the many profit outcomes
    """

    print(stock_ticker, sl_ratio, threshold, scaler)
    stock, market, bond = load_data_from_db(stock_ticker)
    MC_profit = []
    MC_trades = []

    for i in range(100):
        price = float(stock['close'].iloc[-1])
        MC_stock = generate_synthetic_ohlcv(stock.copy())
        MC_market = generate_synthetic_ohlcv(market.copy())
        MC_bond = generate_synthetic_ohlcv(bond.copy())

        df, features_scaled, input_size = clean_and_change(MC_stock, MC_market, MC_bond)
        test_loader, test_loader_cnn = tensor_and_dataloader(df, features_scaled, MC = True)

        effective_len = len(test_loader.dataset)
        df = df.iloc[-effective_len:]

        df = BackTesting_predictions(CNN_model, LSTM_model, GRU_model, df, test_loader, test_loader_cnn)

        trades_df, total_profit, total_trades = backtest(df.copy(), sl_ratio=sl_ratio, threshold=threshold, pred_ler=scaler)
        
        total_profit = total_profit if len(trades_df) != 0 else 0
        total_trades = total_trades if total_trades != None else 0
        pct_profit = (((total_profit + 2 * price) / (2 * price)) - 1) * 100

        MC_profit.append(pct_profit)
        MC_trades.append(total_trades)
        days = (len(MC_stock) - 20) // 78
        print(f"MC {i+1}/100 - Profit = {total_profit}/{pct_profit}, Total Trades = {total_trades}, in {days} Days")
    
    MC_profit = [i for i in MC_profit if i != None]

    MC_profit = np.array(MC_profit)
    print(MC_trades)

    sharpe_ratio = (MC_profit.mean() - ((1 + 4.37) ** (days / 365) - 1)) / MC_profit.std()

    var_5 = np.percentile(MC_profit, 5)
    cvar_5 = MC_profit[MC_profit <= var_5].mean()

    print("💰 Monte Carlo Summary:")
    print(f"Mean Profit:        {MC_profit.mean():.2f}")
    print(f"Median Profit:      {np.median(MC_profit):.2f}")
    print(f"Std Dev:            {MC_profit.std():.2f}")
    print(f"Sharpe Ratio:       {sharpe_ratio:.2f}")
    print(f"95% CI:             [{np.percentile(MC_profit, 2.5):.2f}, {np.percentile(MC_profit, 97.5):.2f}]")
    print(f"Win Rate:           {np.mean(MC_profit > 0)*100:.1f}%")
    print(f"Skewness:           {stats.skew(MC_profit):.2f}")
    print(f"Kurtosis:           {stats.kurtosis(MC_profit):.2f}")
    print(f"5% VaR:             {var_5:.2f}")
    print(f"5% CVaR:            {cvar_5:.2f}")

# ================= Main Thread =================

if __name__ == "__main__":

    best_profit = []
    
    # - Setting Device -

    os.environ.pop("MallocStackLogging", None)
    if torch.cuda.is_available(): 
        device = "cuda"
    elif torch.backends.mps.is_available(): 
        device = "mps"
    else: 
        device = "cpu"
    print(f"Using device: {device}")

    # ================= Creating and Training Models ================= 

    needs_model = ["AMZN"]

    # - Load, Clean, tensors, dataloaders -

    for stock_ticker in needs_model: 

        stock, market, bond = load_data_from_db(stock_ticker) 
        df, features_scaled, input_size = clean_and_change(stock, market, bond, train = True)
        train_loader, val_loader, test_loader, train_loader_cnn, val_loader_cnn, test_loader_cnn, X_train = tensor_and_dataloader(df, features_scaled)

        # - Make Models -

        CNN_losses = {}
        GRU_losses = {}
        LSTM_losses = {}

        CNN_model, CNN_best_params = CNN_optuna()
        LSTM_model, LSTM_best_params = LSTM_optuna()
        GRU_model, GRU_best_params = GRU_optuna()

        # - Test Models -
        
        Model_stats(CNN_model, LSTM_model, GRU_model)
        
        # - Save Models -
        
        save_model(CNN_model, CNN_best_params, stock_ticker, "CNN")
        save_model(LSTM_model, LSTM_best_params, stock_ticker, "LSTM")
        save_model(GRU_model, GRU_best_params, stock_ticker, "GRU")

    for stock_ticker in needs_model:

        # - Load Models -
        CNN_model = load_model(stock_ticker, "CNN", AdvancedCNNRegressor)
        LSTM_model = load_model(stock_ticker, "LSTM", AdvancedLSTMModel)
        GRU_model = load_model(stock_ticker, "GRU", AdvancedGRUModel)

        # - Back Testing -
        
        sl_ratio, threshold, pred_scaler, profit = BackTest_fully()
        if sl_ratio == -1 and threshold == -1 and pred_scaler == -1 and profit == -1: 
            profit = float('-inf')
        save_BT_params(stock_ticker, sl_ratio, threshold, pred_scaler)
        best_profit.append((profit, stock_ticker))
    
    best_profit = sorted(best_profit, reverse=True)
    print(best_profit)

    # ================= Live Trading =================

    top_30_stocks = ["SPY"]
    print("Stocks that will be traded:")
    for i in best_profit: 
        if len(top_30_stocks) >= 29: 
            break
        if i[0] > 0: 
            top_30_stocks.append(i[1])
            print(f"     {i} with profit {i[0]}")
    