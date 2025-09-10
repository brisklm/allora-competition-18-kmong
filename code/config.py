import os
from datetime import datetime
import numpy as np
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except Exception:
    SentimentIntensityAnalyzer = None
try:
    import optuna
except Exception:
    optuna = None
data_base_path = os.path.join(os.getcwd(), 'data')
model_file_path = os.path.join(data_base_path, 'model.pkl')
scaler_file_path = os.path.join(data_base_path, 'scaler.pkl')
training_price_data_path = os.path.join(data_base_path, 'price_data.csv')
selected_features_path = os.path.join(data_base_path, 'selected_features.json')
best_model_info_path = os.path.join(data_base_path, 'best_model.json')
sol_source_path = os.path.join(data_base_path, os.getenv('SOL_SOURCE', 'raw_sol.csv'))
eth_source_path = os.path.join(data_base_path, os.getenv('ETH_SOURCE', 'raw_eth.csv'))
features_sol_path = os.path.join(data_base_path, os.getenv('FEATURES_PATH', 'features_sol.csv'))
features_eth_path = os.path.join(data_base_path, os.getenv('FEATURES_PATH_ETH', 'features_eth.csv'))
TOKEN = os.getenv('TOKEN', 'BTC')
TIMEFRAME = os.getenv('TIMEFRAME', '8h')
TRAINING_DAYS = int(os.getenv('TRAINING_DAYS', 365))
MINIMUM_DAYS = 180
REGION = os.getenv('REGION', 'com')
DATA_PROVIDER = os.getenv('DATA_PROVIDER', 'binance')
MODEL = os.getenv('MODEL', 'LSTM_Hybrid')
CG_API_KEY = os.getenv('CG_API_KEY', 'CG-xA5NyokGEVbc4bwrvJPcpZvT')
HELIUS_API_KEY = os.getenv('HELIUS_API_KEY', '70ed65ce-4750-4fd5-83bd-5aee9aa79ead')
HELIUS_RPC_URL = os.getenv('HELIUS_RPC_URL', 'https://mainnet.helius-rpc.com')
BITQUERY_API_KEY = os.getenv('BITQUERY_API_KEY', 'ory_at_LmFLzUutMY8EVb-P_PQVP9ntfwUVTV05LMal7xUqb2I.vxFLfMEoLGcu4XoVi47j-E2bspraTSrmYzCt1A4y2k')
SELECTED_FEATURES = ['sign_log_return_lag1_BTCUSDT', 'momentum_BTCUSDT', 'rsi_BTCUSDT', 'volatility_BTCUSDT', 'volatility_ETHUSDT', 'close_BTCUSDT_lag1', 'close_ETHUSDT_lag1', 'hour_of_day', 'momentum_1_BTCUSDT', 'momentum_2_BTCUSDT', 'momentum_5_BTCUSDT', 'vol_adj_return_BTCUSDT', 'direction_persistence_BTCUSDT', 'momentum_accel_BTCUSDT', 'rsi_divergence_BTCUSDT', 'resistance_distance_BTCUSDT', 'support_distance_BTCUSDT', 'vol_regime_BTCUSDT', 'trend_regime_BTCUSDT', 'mean_reversion_regime_BTCUSDT', 'range_regime_BTCUSDT', 'gap_regime_BTCUSDT', 'correlation_regime', 'near_support_BTCUSDT', 'near_resistance_BTCUSDT']
REQUIRED_FEATURES = ['sign_log_return_lag1_BTCUSDT', 'momentum_BTCUSDT', 'rsi_BTCUSDT', 'volatility_BTCUSDT', 'close_BTCUSDT_lag1', 'hour_of_day']
MODEL_PARAMS = {'n_estimators': 800, 'learning_rate': 0.006, 'num_leaves': 15, 'max_depth': 4, 'min_child_samples': 150, 'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_lambda': 8.0, 'reg_alpha': 3.0, 'min_gain_to_split': 0.05, 'feature_fraction': 0.8, 'bagging_freq': 5, 'n_jobs': 1, 'n_neighbors': 7, 'weights': 'distance', 'p': 1, 'hidden_size': 64, 'num_layers': 2, 'gamma': 0.05, 'min_child_weight': 8, 'iterations': 1200, 'depth': 5, 'l2_leaf_reg': 12.0, 'random_strength': 1.2, 'bagging_temperature': 0.7, 'border_count': 64}
OPTUNA_TRIALS = int(os.getenv('OPTUNA_TRIALS', 50))
USE_SYNTHETIC_DATA = os.getenv('USE_SYNTHETIC_DATA', 'True').lower() == 'true'
PERFORMANCE_THRESHOLDS = {'RMSE': {'excellent': 0.01, 'good': 0.015, 'acceptable': 0.025, 'current': 0.015}, 'MZTAE': {'excellent': 0.3, 'good': 0.5, 'acceptable': 0.75, 'current': 0.5}, 'directional_accuracy': {'excellent': 0.7, 'good': 0.65, 'target': 0.6, 'acceptable': 0.55, 'minimum': 0.5}, 'correlation': {'excellent': 0.5, 'good': 0.3, 'acceptable': 0.2, 'minimum': 0.1}}

def evaluate_performance(rmse, mztae, directional_acc=None, correlation=None):
    """
    Evaluate model performance against Competition 18 thresholds.
    Returns a performance rating and detailed assessment.
    """
    performance = {'overall': 'poor', 'details': {}}
    if rmse <= PERFORMANCE_THRESHOLDS['RMSE']['excellent']:
        performance['details']['rmse'] = 'excellent'
    elif rmse <= PERFORMANCE_THRESHOLDS['RMSE']['good']:
        performance['details']['rmse'] = 'good'
    elif rmse <= PERFORMANCE_THRESHOLDS['RMSE']['acceptable']:
        performance['details']['rmse'] = 'acceptable'
    else:
        performance['details']['rmse'] = 'poor'
    if mztae <= PERFORMANCE_THRESHOLDS['MZTAE']['excellent']:
        performance['details']['mztae'] = 'excellent'
    elif mztae <= PERFORMANCE_THRESHOLDS['MZTAE']['good']:
        performance['details']['mztae'] = 'good'
    elif mztae <= PERFORMANCE_THRESHOLDS['MZTAE']['acceptable']:
        performance['details']['mztae'] = 'acceptable'
    else:
        performance['details']['mztae'] = 'poor'
    if directional_acc is not None:
        if directional_acc >= PERFORMANCE_THRESHOLDS['directional_accuracy']['excellent']:
            performance['details']['directional_accuracy'] = 'excellent'
        elif directional_acc >= PERFORMANCE_THRESHOLDS['directional_accuracy']['good']:
            performance['details']['directional_accuracy'] = 'good'
        elif directional_acc >= PERFORMANCE_THRESHOLDS['directional_accuracy']['target']:
            performance['details']['directional_accuracy'] = 'target_met'
        elif directional_acc >= PERFORMANCE_THRESHOLDS['directional_accuracy']['acceptable']:
            performance['details']['directional_accuracy'] = 'acceptable'
        else:
            performance['details']['directional_accuracy'] = 'poor'
    if correlation is not None and correlation != 0.0:
        if correlation >= PERFORMANCE_THRESHOLDS['correlation']['excellent']:
            performance['details']['correlation'] = 'excellent'
        elif correlation >= PERFORMANCE_THRESHOLDS['correlation']['good']:
            performance['details']['correlation'] = 'good'
        elif correlation >= PERFORMANCE_THRESHOLDS['correlation']['acceptable']:
            performance['details']['correlation'] = 'acceptable'
        else:
            performance['details']['correlation'] = 'poor'
    else:
        performance['details']['correlation'] = 'neutral'
    core_metrics = ['rmse', 'mztae']
    if directional_acc is not None:
        core_metrics.append('directional_accuracy')
    core_ratings = [performance['details'][metric] for metric in core_metrics]
    meets_competition_thresholds = meets_current_thresholds(rmse, mztae, directional_acc)
    if meets_competition_thresholds:
        if all((r in ['excellent', 'good'] for r in core_ratings)):
            performance['overall'] = 'excellent' if 'excellent' in core_ratings else 'good'
        elif all((r in ['excellent', 'good', 'target_met'] for r in core_ratings)):
            performance['overall'] = 'good'
        else:
            performance['overall'] = 'acceptable'
    elif all((r in ['excellent', 'good', 'target_met', 'acceptable'] for r in core_ratings)):
        performance['overall'] = 'acceptable'
    else:
        performance['overall'] = 'poor'
    return performance

def meets_current_thresholds(rmse, mztae, directional_acc=None):
    """
    Check if model meets current performance thresholds.
    Returns True if RMSE, MZTAE, and directional accuracy meet current thresholds.
    """
    rmse_ok = rmse <= PERFORMANCE_THRESHOLDS['RMSE']['current']
    mztae_ok = mztae <= PERFORMANCE_THRESHOLDS['MZTAE']['current']
    da_ok = True
    if directional_acc is not None:
        da_ok = directional_acc >= PERFORMANCE_THRESHOLDS['directional_accuracy']['target']
    return rmse_ok and mztae_ok and da_ok

def get_performance_summary(rmse, mztae, directional_acc=None, correlation=None):
    """
    Generate a human-readable performance summary.
    """
    meets_threshold = meets_current_thresholds(rmse, mztae, directional_acc)
    evaluation = evaluate_performance(rmse, mztae, directional_acc, correlation)
    summary = []
    summary.append(f'Performance Summary for Competition 18:')
    summary.append(f"{'=' * 50}")
    summary.append(f"RMSE: {rmse:.6f} (Threshold: {PERFORMANCE_THRESHOLDS['RMSE']['current']}) - {evaluation['details']['rmse'].upper()}")
    summary.append(f"MZTAE: {mztae:.6f} (Threshold: {PERFORMANCE_THRESHOLDS['MZTAE']['current']}) - {evaluation['details']['mztae'].upper()}")
    if directional_acc is not None:
        da_status = evaluation['details'].get('directional_accuracy', 'N/A')
        da_display = da_status.replace('_', ' ').upper() if da_status != 'N/A' else 'N/A'
        target_threshold = PERFORMANCE_THRESHOLDS['directional_accuracy']['target']
        summary.append(f'Directional Accuracy: {directional_acc:.2%} (Target: ≥{target_threshold:.2%}) - {da_display}')
    if correlation is not None:
        summary.append(f"Correlation: {correlation:.4f} - {evaluation['details'].get('correlation', 'N/A').upper()}")
    summary.append(f"{'=' * 50}")
    summary.append(f"Overall Rating: {evaluation['overall'].upper()}")
    summary.append(f"Meets Current Thresholds: {('YES ✓' if meets_threshold else 'NO ✗')}")
    return '\n'.join(summary)