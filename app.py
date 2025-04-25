from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# Load model and tokenizer
model = load_model('lstm_fake_news_model.h5')
with open('tokenizer.pkl', "rb") as handle:
    tokenizer = pickle.load(handle)

max_len = 100

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len)

    prediction = model.predict(padded)[0][0]
    label = "Fake" if prediction > 0.5 else "Real"
    

    return jsonify({
        'text': text,
        'prediction_score': float(prediction),
        'label': label
    })

if __name__ == '__main__':
    app.run(debug=True)
