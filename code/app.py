import os
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
import numpy as np
import math

# Initialize app and env
app = Flask(__name__)
load_dotenv()

# Dynamic version tag for visibility in logs
COMPETITION = os.getenv("COMPETITION", "competition18")
TOPIC_ID = os.getenv("TOPIC_ID", "64")
TOKEN = os.getenv("TOKEN", "BTC")
TIMEFRAME = os.getenv("TIMEFRAME", "8h")
MCP_VERSION = f"{datetime.utcnow().date()}-{COMPETITION}-topic{TOPIC_ID}-app-{TOKEN.lower()}-{TIMEFRAME}"
FLASK_PORT = int(os.getenv("FLASK_PORT", 9001))

def sanitize_for_json(obj):
    """
    Recursively sanitize an object to ensure it's JSON serializable.
    Replaces NaN, inf, -inf with None or appropriate values.
    """
    if isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    elif isinstance(obj, float):
        if math.isnan(obj):
            return None
        elif math.isinf(obj):
            return 1e9 if obj > 0 else -1e9
        else:
            return obj
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj):
            return None
        elif np.isinf(obj):
            return 1e9 if obj > 0 else -1e9
        else:
            return float(obj)
    elif hasattr(obj, 'item'):  # numpy scalars
        return sanitize_for_json(obj.item())
    else:
        # For any other type, try to convert to string
        return str(obj)

# MCP Tools
TOOLS = [
    {
        "name": "optimize",
        "description": "Triggers model optimization using Optuna tuning and returns results.",
        "parameters": {}
    },
    {
        "name": "write_code",
        "description": "Writes complete source code to a specified file, overwriting existing content.",
        "parameters": {
            "file_name": {"type": "string", "description": "The name of the file to write."},
            "code": {"type": "string", "description": "The complete Python code to write."}
        }
    }
]

@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        import optuna
        # Example optimization for hyperparameters
        def objective(trial):
            max_depth = trial.suggest_int('max_depth', 3, 10)
            num_leaves = trial.suggest_int('num_leaves', 20, 50)
            # Simulate R2 score
            r2 = np.random.uniform(0.05, 0.15) + (max_depth / 100.0)
            return r2
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)
        return jsonify(sanitize_for_json({'best_params': study.best_params, 'best_r2': study.best_value}))
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    # Placeholder for prediction logic
    return jsonify({'prediction': 0.01})

if __name__ == '__main__':
    app.run(port=FLASK_PORT)