from flask import Flask, render_template, request, jsonify
from predict import SpamPredictor
import json

app = Flask(__name__, template_folder='../frontend/templates', static_folder='../frontend/static')

predictor = SpamPredictor('../models')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_spam():
    try:
        data = request.json
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Please enter a message'}), 400
        
        ensemble_result = predictor.get_ensemble_prediction(message)
        
        if ensemble_result is None:
            return jsonify({'error': 'Models not loaded. Please train models first.'}), 500
        
        return jsonify(ensemble_result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'models_loaded': len(predictor.models)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
