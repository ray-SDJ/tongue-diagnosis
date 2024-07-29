from flask import Blueprint, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

main = Blueprint('main', __name__)


model_path = r'C:\Users\Titan\OneDrive\Desktop\flask\app\models\tongue_diagnosis_model.h5'  
model = load_model(model_path)


class_names = ['blood-tongue', 'heart-tongue', 'heat-tongue', 'liver-tongue', 
               'lung-tongue', 'normal-tongue', 'pale-tongue', 'tongue-kidney', 'tongue-stomach']

def predict_tongue_condition(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    class_name = class_names[predicted_class]
    confidence = float(np.max(prediction))
    
    return class_name, confidence

@main.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
        elif 'image' in request.files: 
            file = request.files['image']
        else:
            return jsonify({'error': 'No file or image data'})

        if file.filename == '':
            return jsonify({'error': 'No selected file or image'})

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join('static/uploads', filename)
            file.save(filepath)
            img = cv2.imread(filepath)
            condition, confidence = predict_tongue_condition(img)
            return jsonify({'condition': condition, 'confidence': confidence})

    return render_template('index.html')
import tempfile
@main.route('/capture', methods=['POST'])
def capture():
    if 'image' not in request.files:
        return jsonify({'error': 'No image data'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No captured image'})
    
    if file:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                file.save(temp_file.name)
                img = cv2.imread(temp_file.name)
                condition, confidence = predict_tongue_condition(img)
            return jsonify({'condition': condition, 'confidence': confidence})
        except Exception as e:
            return jsonify({'error': f'Error processing image: {e}'})
