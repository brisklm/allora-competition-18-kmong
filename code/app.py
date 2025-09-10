import os
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
import numpy as np
import math

app = Flask(__name__)
load_dotenv()

COMPETITION = os.getenv("COMPETITION", "competition18")
TOPIC_ID = os.getenv("TOPIC_ID", "64")
TOKEN = os.getenv("TOKEN", "BTC")
TIMEFRAME = os.getenv("TIMEFRAME", "8h")
MCP_VERSION = f"{datetime.utcnow().date()}-{COMPETITION}-topic{TOPIC_ID}-app-{TOKEN.lower()}-{TIMEFRAME}"
FLASK_PORT = int(os.getenv("FLASK_PORT", 9001))

def sanitize_for_json(obj):
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
    elif hasattr(obj, 'item'):
        return sanitize_for_json(obj.item())
    else:
        return str(obj)

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
            "file_name": "str",
            "code": "str"
        }
    }
]

@app.route('/tools', methods=['GET'])
def get_tools():
    return jsonify(TOOLS)

@app.route('/invoke', methods=['POST'])
def invoke_tool():
    data = request.json
    tool_name = data.get('tool_name')
    tool = next((t for t in TOOLS if t['name'] == tool_name), None)
    if tool is None:
        return jsonify({"error": "Tool not found"}), 404
    
    if tool_name == 'optimize':
        from model import optimize_model
        result = optimize_model()
        return jsonify(sanitize_for_json(result))
    elif tool_name == 'write_code':
        file_name = data.get('file_name')
        code = data.get('code')
        if file_name and code:
            with open(file_name, 'w') as file:
                file.write(code)
            return jsonify({"message": f"Code written to {file_name}"})
        else:
            return jsonify({"error": "Missing file_name or code"}), 400
    
    return jsonify({"error": "Tool not implemented"}), 501

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=FLASK_PORT)