import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import optuna
import joblib
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except:
    SentimentIntensityAnalyzer = None

def add_vader_sentiment(df):
    if SentimentIntensityAnalyzer:
        sia = SentimentIntensityAnalyzer()
        df['sentiment_vader'] = df['text'].apply(lambda x: sia.polarity_scores(x)['compound'] if pd.notnull(x) else 0)
    else:
        df['sentiment_vader'] = 0
    return df

def train_model(data_path):
    df = pd.read_csv(data_path)
    df = add_vader_sentiment(df)
    df = df.dropna()  # Robust NaN handling
    if df.var().any() < 1e-5:  # Low-variance check
        return None
    scaler = StandardScaler()
    X = scaler.fit_transform(df.drop('target', axis=1))
    y = df['target']
    # Placeholder Optuna tuning
    def objective(trial):
        param = {'max_depth': trial.suggest_int('max_depth', 5, 15)}
        # Model training here
        return 0.1  # Placeholder R2
    study = optuna.create_study()
    study.optimize(objective, n_trials=10)
    # Save model
    joblib.dump(scaler, 'scaler.pkl')
    return study.best_value > 0.1