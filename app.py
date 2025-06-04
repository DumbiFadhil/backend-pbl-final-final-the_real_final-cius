from flask import Flask, request, jsonify
# from flask_cors import CORS
from services.sarimax_service import SarimaxService
# from services.xgboost_service import XGBoostService
# from services.nlp_service import NLPService

app = Flask(__name__)
# CORS(app)

sarimax_service = SarimaxService()
# xgboost_service = XGBoostService()
# nlp_service = NLPService()

@app.route('/sarimax/train', methods=['POST'])
def train_sarimax():
    # Accepts multipart/form-data with a 'file' and form fields
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400
    file = request.files['file']
    # Forward the request.form (contains user selections)
    result = sarimax_service.train(file, request.form)
    return jsonify(result)

@app.route('/sarimax/predict', methods=['POST'])
def predict_sarimax():
    # Expects JSON: { "family_val": "FAMILYNAME", "steps": 30 }
    data = request.get_json()
    family_val = data.get('family_val')
    steps = data.get('steps', 30)
    if not family_val:
        return jsonify({'error': 'family_val is required'}), 400
    result = sarimax_service.predict(family_val, steps)
    return jsonify(result)

# Example placeholders for XGBoost and NLP
# @app.route('/xgboost/train', methods=['POST'])
# def train_xgboost():
#     data = request.get_json()
#     result = xgboost_service.train(data)
#     return jsonify(result)

# @app.route('/xgboost/predict', methods=['POST'])
# def predict_xgboost():
#     data = request.get_json()
#     result = xgboost_service.predict(data)
#     return jsonify(result)

# @app.route('/nlp/predict', methods=['POST'])
# def predict_nlp():
#     data = request.get_json()
#     result = nlp_service.predict(data)
#     return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)