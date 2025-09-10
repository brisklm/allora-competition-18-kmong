import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from config import *
if optuna is not None:
    import optuna

def load_data():
    return pd.read_csv(training_price_data_path)

def compute_sentiment(texts):
    if SentimentIntensityAnalyzer is not None:
        sia = SentimentIntensityAnalyzer()
        sentiments = [sia.polarity_scores(text)['compound'] for text in texts]
        return np.array(sentiments)
    else:
        return np.zeros(len(texts))

def select_features(df, target):
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    selector = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    X = selector.fit_transform(df.drop(columns=[target]))
    features = df.drop(columns=[target]).columns[selector.get_support()]
    high_corr_features = []
    for feat in features:
        corr, _ = pearsonr(df[feat], df[target])
        if np.isnan(corr):
            continue
        if abs(corr) > CORR_THRESHOLD:
            high_corr_features.append(feat)
    return high_corr_features

def objective(trial, X, y, timesteps):
    units = trial.suggest_int('units', 50, 200)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    model = Sequential()
    model.add(LSTM(units, input_shape=(timesteps, X.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    score = model.evaluate(X, y, verbose=0)
    return -score  # Maximize negative loss for R2-like

def train_model():
    df = load_data()
    # Assume 'text' column for sentiment, replace with actual
    if 'text' in df.columns:
        df['vader_sentiment'] = compute_sentiment(df['text'])
    df = df.fillna(method='ffill').fillna(0)
    selected = select_features(df, 'log_return')
    X = df[selected].values
    y = df['log_return'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    timesteps = 10  # Example
    X_reshaped = np.array([X_scaled[i:i+timesteps] for i in range(len(X_scaled)-timesteps)])
    y = y[timesteps:]
    if optuna is not None:
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_reshaped, y, timesteps), n_trials=OPTUNA_TRIALS)
        best_params = study.best_params
        model = Sequential()
        model.add(LSTM(best_params['units'], input_shape=(timesteps, X_reshaped.shape[2])))
        model.add(Dropout(best_params['dropout']))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_reshaped, y, epochs=50, batch_size=32, verbose=0)
    else:
        model = Sequential()
        model.add(LSTM(100, input_shape=(timesteps, X_reshaped.shape[2])))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_reshaped, y, epochs=50, batch_size=32, verbose=0)
    # TODO: Add hybrid with LightGBM, ensembling, smoothing
    with open(model_file_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_file_path, 'wb') as f:
        pickle.dump(scaler, f)

if __name__ == '__main__':
    train_model()