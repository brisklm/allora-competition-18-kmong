import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import joblib
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except:
    SentimentIntensityAnalyzer = None
from config import FEATURES, HYPERPARAMS, LOW_VARIANCE_THRESHOLD, model_file_path, scaler_file_path

def add_sentiment(df):
    if SentimentIntensityAnalyzer:
        sia = SentimentIntensityAnalyzer()
        df['sentiment_score'] = df['text'].apply(lambda x: sia.polarity_scores(x)['compound'] if pd.notnull(x) else 0)
    else:
        df['sentiment_score'] = 0
    return df

def check_low_variance(df):
    variances = df[FEATURES].var()
    low_var_features = [f for f in FEATURES if variances[f] < LOW_VARIANCE_THRESHOLD]
    return [f for f in FEATURES if f not in low_var_features]

def train_model(data):
    data = add_sentiment(data)
    data = data.dropna(subset=FEATURES)  # Robust NaN handling
    selected_features = check_low_variance(data)
    X = data[selected_features]
    y = data['log_return_8h']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Placeholder for LSTM_Hybrid model, with ensembling
    models = []
    for _ in range(HYPERPARAMS['ensemble_size']):
        # Simulate training
        model = {'params': HYPERPARAMS}  # Replace with actual model
        models.append(model)
    joblib.dump(models, model_file_path)
    joblib.dump(scaler, scaler_file_path)
    # Simulate R2
    preds = np.random.rand(len(y))
    r2 = r2_score(y, preds)
    return r2

def predict(data):
    data = add_sentiment(data)
    scaler = joblib.load(scaler_file_path)
    models = joblib.load(model_file_path)
    X = data[FEATURES]
    X_scaled = scaler.transform(X)
    preds = np.mean([m.predict(X_scaled) for m in models], axis=0)  # Ensembling
    # Smoothing
    preds = pd.Series(preds).ewm(span=5).mean().values
    return preds