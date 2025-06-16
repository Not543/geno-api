from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from preprocess import preprocess_fingerprint
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Flask config
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'static/processed'
MODEL_PATH = 'final_model.keras'

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load the trained model
model = load_model(MODEL_PATH)
labels = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

def predict_blood_group(image):
    # Save uploaded image to disk
    input_path = os.path.join(UPLOAD_FOLDER, "input.jpg")
    image.save(input_path)

    # Preprocess and save
    processed_path = os.path.join(PROCESSED_FOLDER, "processed_input.jpg")
    preprocess_fingerprint(input_path, processed_path)

    # Load processed image and predict
    img = cv2.imread(processed_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = labels[np.argmax(prediction)]

    return predicted_class, processed_path

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files['image']
    predicted_class, processed_path = predict_blood_group(image)

    # Hugging Face will serve from "static" folder by default
    processed_url = f"/{processed_path}"

    return jsonify({
        "prediction": predicted_class,
        "processed_image_url": processed_url
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)

