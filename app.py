from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from PIL import Image
import io
import base64
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Model yükleme
model = load_model('best_model1.keras')  # Model dosya adınızı buraya yazın

# Etiket haritası
labels_map = {0: "cardboard", 1: "glass", 2: "metal", 3: "paper", 4: "plastic"}

def preprocess_image(image):
    # Görüntüyü 224x224 boyutuna yeniden boyutlandır
    image = cv2.resize(image, (224, 224))
    # Normalize et
    image = image.astype('float32') / 255.0
    # Batch boyutunu ekle
    image = np.expand_dims(image, axis=0)
    return image

def predict_image(image):
    # Görüntüyü ön işleme
    processed_image = preprocess_image(image)
    # Tahmin yap
    prediction = model.predict(processed_image)
    # En yüksek olasılıklı sınıfı al
    predicted_class = np.argmax(prediction[0])
    # Sınıf etiketini döndür
    return labels_map[predicted_class], float(prediction[0][predicted_class])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/capture', methods=['POST'])
def capture():
    try:
        # Base64 formatındaki görüntüyü al
        image_data = request.json['image'].split(',')[1]
        # Base64'ten numpy dizisine dönüştür
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image = np.array(image)
        
        # BGR'ye dönüştür (webcam RGB formatında geliyor)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Tahmin yap
        class_name, confidence = predict_image(image)
        
        return jsonify({
            'success': True,
            'class': class_name,
            'confidence': f'{confidence:.2%}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Dosyayı oku ve numpy dizisine dönüştür
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Tahmin yap
        class_name, confidence = predict_image(image)
        
        # Görüntüyü base64'e dönüştür
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'class': class_name,
            'confidence': f'{confidence:.2%}',
            'image': f'data:image/jpeg;base64,{image_base64}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)