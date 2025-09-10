import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from config import *

# Load and preprocess data
def load_data():
    data = pd.read_csv(training_price_data_path)
    data['log_return'] = np.log(data['close']).diff()
    data['vader_sentiment'] = data['news'].apply(analyze_sentiment)
    data = data.dropna()
    return data[FEATURES]

def preprocess_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    np.save(scaler_file_path, scaler)
    return scaled_data

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = lgb.LGBMRegressor()
    model.fit(X_train, y_train)
    return model

def optimize_and_train():
    data = load_data()
    X = preprocess_data(data[FEATURES[1:]])
    y = data[FEATURES[0]]
    
    if optuna is not None:
        best_params, _ = optimize_model()
        model = lgb.LGBMRegressor(**best_params)
    else:
        model = lgb.LGBMRegressor()
    
    model.fit(X, y)
    return model

def predict(model, X):
    X = handle_nan(X)
    predictions = model.predict(X)
    return predictions

# Main execution
def main():
    data = load_data()
    X = preprocess_data(data[FEATURES[1:]])
    y = data[FEATURES[0]]
    
    model = optimize_and_train()
    predictions = predict(model, X)
    
    # Calculate R2 and correlation
    r2 = np.corrcoef(y, predictions)[0, 1] ** 2
    correlation = np.corrcoef(y, predictions)[0, 1]
    
    print(f"R2: {r2}")
    print(f"Correlation: {correlation}")

if __name__ == '__main__':
    main()