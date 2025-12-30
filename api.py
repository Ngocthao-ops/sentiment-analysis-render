from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import time  # ← THÊM ĐỂ ĐO THỜI GIAN

app = Flask(__name__)
CORS(app)  # Cho phép Dash app gọi API

# Load models và vectorizer
print("Loading models...")
with open('models/logistic_model.pkl', 'rb') as f:
    logistic_model = pickle.load(f)

with open('models/random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

print("Tải model thành công!")

def map_prediction_to_sentiment(prediction):
    sentiment_map = {
        -1: {
            'label': 'Tiêu cực',
            'label_en': 'negative',
            'rating': 1
        },
        0: {
            'label': 'Trung lập',
            'label_en': 'neutral',
            'rating': 3
        },
        1: {
            'label': 'Tích cực',
            'label_en': 'positive',
            'rating': 5
        }
    }
    
    return sentiment_map.get(prediction, sentiment_map[0])  # Default: neutral

@app.route('/')
def home():
    return jsonify({
        'status': 'API is running',
        'endpoints': {
            'predict': '/api/predict (POST)',
            'compare': '/api/compare (POST)',
            'health': '/api/health (GET)'
        }
    })

@app.route('/api/health', methods=['GET'])
def health():
    """Check API health"""
    return jsonify({'status': 'healthy', 'models_loaded': True})

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict sentiment from Vietnamese text
    Expects JSON: {"text": "review text", "model": "logistic" or "random_forest"}
    """
    try:
        start_time = time.time()  # ← BẮT ĐẦU ĐO THỜI GIAN
        
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        
        if not text or text.strip() == '':
            return jsonify({'error': 'Text is empty'}), 400
        
        model_choice = data.get('model', 'logistic')  # Default: logistic
        
        # Vectorize
        text_vector = vectorizer.transform([text])
        
        # Predict
        if model_choice == 'random_forest':
            prediction = rf_model.predict(text_vector)[0]
            proba = rf_model.predict_proba(text_vector)[0]
            model_name = "Random Forest"
        else:
            prediction = logistic_model.predict(text_vector)[0]
            proba = logistic_model.predict_proba(text_vector)[0]
            model_name = "Logistic Regression"
        
        # Map prediction to sentiment
        sentiment_info = map_prediction_to_sentiment(int(prediction))
        
        # Get confidence
        confidence = float(max(proba))
        
        # Build probabilities dict (3 classes)
        if len(proba) == 3:
            probabilities = {
                'negative': round(float(proba[0]) * 100, 2),
                'neutral': round(float(proba[1]) * 100, 2),
                'positive': round(float(proba[2]) * 100, 2)
            }
        elif len(proba) == 2:
            probabilities = {
                'negative': round(float(proba[0]) * 100, 2),
                'neutral': 0.0,
                'positive': round(float(proba[1]) * 100, 2)
            }
        else:
            probabilities = {}
        
        end_time = time.time()  # ← KẾT THÚC ĐO THỜI GIAN
        prediction_time = round((end_time - start_time) * 1000, 2)  # ms
        
        return jsonify({
            'text': text,
            'prediction': int(prediction),
            'rating': sentiment_info['rating'],
            'sentiment': sentiment_info['label'],
            'sentiment_en': sentiment_info['label_en'],
            'confidence': round(confidence * 100, 2),
            'model_used': model_name,
            'probabilities': probabilities,
            'num_classes': len(proba),
            'prediction_time_ms': prediction_time  # ← THÊM THỜI GIAN
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare', methods=['POST'])
def compare_models():
    """
    So sánh kết quả 2 models
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        
        if not text or text.strip() == '':
            return jsonify({'error': 'Text is empty'}), 400
        
        # Vectorize
        text_vector = vectorizer.transform([text])
        
        # Logistic Regression
        start_log = time.time()
        log_pred = logistic_model.predict(text_vector)[0]
        log_proba = logistic_model.predict_proba(text_vector)[0]
        log_time = round((time.time() - start_log) * 1000, 2)
        log_sentiment = map_prediction_to_sentiment(int(log_pred))
        
        # Random Forest
        start_rf = time.time()
        rf_pred = rf_model.predict(text_vector)[0]
        rf_proba = rf_model.predict_proba(text_vector)[0]
        rf_time = round((time.time() - start_rf) * 1000, 2)
        rf_sentiment = map_prediction_to_sentiment(int(rf_pred))
        
        # Build probabilities
        if len(log_proba) == 3:
            log_probs = {
                'negative': round(float(log_proba[0]) * 100, 2),
                'neutral': round(float(log_proba[1]) * 100, 2),
                'positive': round(float(log_proba[2]) * 100, 2)
            }
        else:
            log_probs = {
                'negative': round(float(log_proba[0]) * 100, 2),
                'positive': round(float(log_proba[1]) * 100, 2)
            }
        
        if len(rf_proba) == 3:
            rf_probs = {
                'negative': round(float(rf_proba[0]) * 100, 2),
                'neutral': round(float(rf_proba[1]) * 100, 2),
                'positive': round(float(rf_proba[2]) * 100, 2)
            }
        else:
            rf_probs = {
                'negative': round(float(rf_proba[0]) * 100, 2),
                'positive': round(float(rf_proba[1]) * 100, 2)
            }
        
        return jsonify({
            'text': text,
            'logistic_regression': {
                'rating': log_sentiment['rating'],
                'sentiment': log_sentiment['label'],
                'sentiment_en': log_sentiment['label_en'],
                'confidence': round(float(max(log_proba)) * 100, 2),
                'probabilities': log_probs,
                'prediction_time_ms': log_time  # ← THÊM THỜI GIAN
            },
            'random_forest': {
                'rating': rf_sentiment['rating'],
                'sentiment': rf_sentiment['label'],
                'sentiment_en': rf_sentiment['label_en'],
                'confidence': round(float(max(rf_proba)) * 100, 2),
                'probabilities': rf_probs,
                'prediction_time_ms': rf_time  # ← THÊM THỜI GIAN
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)