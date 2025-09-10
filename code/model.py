import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, VarianceThreshold
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import joblib
import config
try:
    import optuna
except:
    optuna = None
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore

def compute_vader_sentiment(texts):
    if config.SentimentIntensityAnalyzer is None:
        return np.zeros(len(texts))
    sia = config.SentimentIntensityAnalyzer()
    return np.array([sia.polarity_scores(text)['compound'] for text in texts])

def preprocess_data(df):
    df = df.fillna(method='ffill').fillna(0)  # NaN handling
    selector = VarianceThreshold(threshold=0.01)  # low-variance check
    df_selected = pd.DataFrame(selector.fit_transform(df[config.FEATURES]), columns=[config.FEATURES[i] for i in selector.get_support(indices=True)])
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_selected)
    joblib.dump(scaler, config.scaler_file_path)
    return scaled, df['log_return_8h'].values

def build_model(input_shape, trial=None):
    model = Sequential()
    if trial:
        layers = trial.suggest_int('layers', 1, 3)
        units = trial.suggest_int('units', 50, 200)
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
    else:
        layers, units, dropout = 2, 100, 0.2
    model.add(LSTM(units, return_sequences=(layers > 1), input_shape=input_shape))
    model.add(Dropout(dropout))
    for _ in range(1, layers):
        model.add(LSTM(units, return_sequences=(_ < layers-1)))
        model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def optuna_objective(trial, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    model = build_model((X_train.shape[1], X_train.shape[2]), trial)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=0)
    preds = model.predict(X_val)
    return -r2_score(y_val, preds)  # maximize R2

def train_model():
    df = pd.read_csv(config.training_price_data_path)
    # Assume df has 'text' column for sentiment
    if 'text' in df.columns:
        df['vader_compound'] = compute_vader_sentiment(df['text'])
    X, y = preprocess_data(df)
    X = X.reshape((X.shape[0], 1, X.shape[1]))  # for LSTM
    if optuna:
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: optuna_objective(trial, X, y), n_trials=config.OPTUNA_PARAMS['n_trials'])
        best_trial = study.best_trial
        model = build_model((X.shape[1], X.shape[2]), best_trial)
    else:
        model = build_model((X.shape[1], X.shape[2]))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
    joblib.dump(model, config.model_file_path)
    r2 = r2_score(y_test, model.predict(X_test))
    print(f'R2: {r2}')
    # Ensembling: simple average with another model if needed
    return model

def load_model():
    return joblib.load(config.model_file_path)

def make_prediction(model, data):
    scaler = joblib.load(config.scaler_file_path)
    input_data = scaler.transform(pd.DataFrame(data))[0].reshape(1, 1, -1)
    return model.predict(input_data)[0][0]