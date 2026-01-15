from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import time  

app = Flask(__name__)
CORS(app)  

# Load models, vectorizer
print("Loading models...")
with open('models/logistic_model.pkl', 'rb') as f:
    logistic_model = pickle.load(f)

with open('models/random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

    
try:
    print("Đang load CNN-LSTM...")
    from tensorflow import keras
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import pickle as pkl
    
    # Load CNN-LSTM
    cnn_lstm_model = keras.models.load_model('models/cnn_lstm_model.h5')
    print(f"Load model thành công! Input shape: {cnn_lstm_model.input_shape}, Output shape: {cnn_lstm_model.output_shape}")
    
    # Load tokenizer CNN-LSTM
    with open('models/tokenizer.pkl', 'rb') as f:
        tokenizer = pkl.load(f)
    print(f"Load tokenizer thành công!")
    
    cnn_lstm_available = True
    
except Exception as e:
    print(f"Xem lại model CNN-LSTM: {e}")
    import traceback
    traceback.print_exc()
    cnn_lstm_available = False

print(f"Tải model thành công! CNN-LSTM available: {cnn_lstm_available}")

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
    
    return sentiment_map.get(prediction, sentiment_map[0])  

@app.route('/')
def home():
    return jsonify({
        'status': 'API is running',
        'cnn_lstm_available': cnn_lstm_available,
        'endpoints': {
            'predict': '/api/predict (POST)',
            'compare': '/api/compare (POST)',
            'health': '/api/health (GET)'
        }
    })

@app.route('/api/health', methods=['GET'])
def health():
    """Check API health"""
    return jsonify({
        'status': 'healthy', 
        'models_loaded': True,
        'cnn_lstm_available': cnn_lstm_available
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Dự đoán: {"text": "review text", "model": "logistic" or "random_forest"}
    """
    try:
        start_time = time.time()
        
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        
        if not text or text.strip() == '':
            return jsonify({'error': 'Text is empty'}), 400
        
        model_choice = data.get('model', 'logistic')  
        
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
        
        end_time = time.time()
        prediction_time = round((end_time - start_time) * 1000, 2)
        
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
            'prediction_time_ms': prediction_time
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/predict_cnn_lstm', methods=['POST'])
def predict_cnn_lstm():
    """
     CNN-LSTM model
    """
    if not cnn_lstm_available:  
        return jsonify({'error': 'CNN-LSTM model not available'}), 503
    
    try:
        start_time = time.time()
        
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        
        if not text or text.strip() == '':
            return jsonify({'error': 'Text is empty'}), 400
        
        # Tokenize và pad sequence
        sequences = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequences, maxlen=82, padding='post', truncating='post')
        
        # Predict
        prediction = cnn_lstm_model.predict(padded, verbose=0)[0]
        
        # Chuyển đổi prediction thành class
        pred_class = np.argmax(prediction) - 1  # -1, 0, 1
        
        # Map prediction
        sentiment_info = map_prediction_to_sentiment(int(pred_class))
        
        # Get confidence
        confidence = float(max(prediction))
        
        # Build probabilities dict
        probabilities = {
            'negative': round(float(prediction[0]) * 100, 2),
            'neutral': round(float(prediction[1]) * 100, 2),
            'positive': round(float(prediction[2]) * 100, 2)
        }
        
        end_time = time.time()
        prediction_time = round((end_time - start_time) * 1000, 2)
        
        return jsonify({
            'text': text,
            'prediction': int(pred_class),
            'rating': sentiment_info['rating'],
            'sentiment': sentiment_info['label'],
            'sentiment_en': sentiment_info['label_en'],
            'confidence': round(confidence * 100, 2),
            'model_used': 'CNN-LSTM',
            'probabilities': probabilities,
            'prediction_time_ms': prediction_time
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare', methods=['POST'])
def compare_models():
    """
    So sánh 3 models
    """
    try:
        print("\n" + "="*60)
        print("Nhận request /api/compare")
        
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        print(f"Text: '{text}'")
        
        if not text or text.strip() == '':
            return jsonify({'error': 'Text is empty'}), 400
        
        # Vectorize cho traditional models
        print("\nVectorizing...")
        text_vector = vectorizer.transform([text])
        print("Vectorize xong")
        
        # Logistic Regression
        print("\nLogistic Regression...")
        start_log = time.time()
        log_pred = logistic_model.predict(text_vector)[0]
        log_proba = logistic_model.predict_proba(text_vector)[0]
        log_time = round((time.time() - start_log) * 1000, 2)
        log_sentiment = map_prediction_to_sentiment(int(log_pred))
        print(f"Result: {log_sentiment['label']} ({log_time}ms)")
        
        # Random Forest
        print("\nRandom Forest...")
        start_rf = time.time()
        rf_pred = rf_model.predict(text_vector)[0]
        rf_proba = rf_model.predict_proba(text_vector)[0]
        rf_time = round((time.time() - start_rf) * 1000, 2)
        rf_sentiment = map_prediction_to_sentiment(int(rf_pred))
        print(f"Result: {rf_sentiment['label']} ({rf_time}ms)")
        
        # CNN-LSTM
        cnn_lstm_result = None
        print(f"\nCNN-LSTM available: {cnn_lstm_available}")
        
        if cnn_lstm_available:
            try:
                print("Bắt đầu CNN-LSTM prediction...")
                start_cnn = time.time()
                
                # Tokenize
                print(f"  → Tokenizing text: '{text}'")
                sequences = tokenizer.texts_to_sequences([text])
                print(f"  → Sequences: {sequences}")
                print(f"  → Sequence length: {len(sequences[0]) if sequences and sequences[0] else 0}")
                
                # Pad
                padded = pad_sequences(sequences, maxlen=82, padding='post', truncating='post')
                print(f"  → Padded shape: {padded.shape}")
                print(f"  → Padded content (first 20): {padded[0][:20]}")
                
                # Predict
                print("  → Calling model.predict()...")
                cnn_pred_proba = cnn_lstm_model.predict(padded, verbose=0)[0]
                print(f"  → Raw output: {cnn_pred_proba}")
                print(f"  → Output type: {type(cnn_pred_proba)}")
                print(f"  → Output shape: {cnn_pred_proba.shape if hasattr(cnn_pred_proba, 'shape') else len(cnn_pred_proba)}")
                
                # Xác định class
                print("  → Determining class...")
                pred_idx = np.argmax(cnn_pred_proba)
                print(f"  → Argmax index: {pred_idx}")
                
                cnn_pred = pred_idx - 1  # Convert to -1, 0, 1
                print(f"  → Mapped class: {cnn_pred}")
                
                cnn_time = round((time.time() - start_cnn) * 1000, 2)
                cnn_sentiment = map_prediction_to_sentiment(int(cnn_pred))
                print(f"  → Sentiment: {cnn_sentiment}")
                
                cnn_lstm_result = {
                    'rating': cnn_sentiment['rating'],
                    'sentiment': cnn_sentiment['label'],
                    'sentiment_en': cnn_sentiment['label_en'],
                    'confidence': round(float(max(cnn_pred_proba)) * 100, 2),
                    'probabilities': {
                        'negative': round(float(cnn_pred_proba[0]) * 100, 2),
                        'neutral': round(float(cnn_pred_proba[1]) * 100, 2),
                        'positive': round(float(cnn_pred_proba[2]) * 100, 2)
                    },
                    'prediction_time_ms': cnn_time
                }
                print(f"CNN-LSTM complete: {cnn_sentiment['label']} ({cnn_time}ms)")
                print(f"   Result object: {cnn_lstm_result}")
                
            except Exception as e:
                print(f"\nCNN-LSTM ERROR:")
                print(f"   Type: {type(e).__name__}")
                print(f"   Message: {e}")
                import traceback
                print("   Traceback:")
                traceback.print_exc()
                cnn_lstm_result = None
        else:
            print("CNN-LSTM not available (model not loaded)")
        
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
        
        print("\n Building response...")
        result = {
            'text': text,
            'logistic_regression': {
                'rating': log_sentiment['rating'],
                'sentiment': log_sentiment['label'],
                'sentiment_en': log_sentiment['label_en'],
                'confidence': round(float(max(log_proba)) * 100, 2),
                'probabilities': log_probs,
                'prediction_time_ms': log_time
            },
            'random_forest': {
                'rating': rf_sentiment['rating'],
                'sentiment': rf_sentiment['label'],
                'sentiment_en': rf_sentiment['label_en'],
                'confidence': round(float(max(rf_proba)) * 100, 2),
                'probabilities': rf_probs,
                'prediction_time_ms': rf_time
            }
        }
        
        # Thêm CNN-LSTM
        if cnn_lstm_result is not None:
            result['cnn_lstm'] = cnn_lstm_result
            print(f"Added CNN-LSTM to response")
        else:
            print(f"CNN-LSTM result is None, NOT adding to response")
        
        print(f"\n Response ready! Keys: {list(result.keys())}")
        print("="*60 + "\n")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"\n CRITICAL ERROR in /api/compare:")
        print(f"   Type: {type(e).__name__}")
        print(f"   Message: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500        

# TÍCH HỢP DASH APP VÀO FLASK
try:
    from app import app as dash_app
    
    # Mount Dash vào Flask tại đường dẫn /dash/
    from werkzeug.middleware.dispatcher import DispatcherMiddleware
    from werkzeug.serving import run_simple
    
    # Tạo application tích hợp
    application = DispatcherMiddleware(app, {
        '/dash': dash_app.server
    })
    
    print("Dash app đã được tích hợp tại /dash/")
    DASH_INTEGRATED = True
except Exception as e:
    print(f"Không thể tích hợp Dash: {e}")
    application = app
    DASH_INTEGRATED = False

# MAIN
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    
    if DASH_INTEGRATED:
        # Chạy với Dash tích hợp
        run_simple('0.0.0.0', port, application, use_reloader=False, use_debugger=False)
    else:
        # Chạy Flask only
        app.run(host='0.0.0.0', port=port, debug=False)
