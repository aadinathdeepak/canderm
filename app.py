import numpy as np
import tensorflow as tf
import cv2
import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load TensorFlow Lite Model (for classification)
tflite_model_path = "dog_skin_disease_densenet121_finetuned_model.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Load Keras Model (for Grad-CAM)
keras_model_path = "dog_skin_disease_densenet121_finetuned_model.h5"
keras_model = load_model(keras_model_path)

# Get input/output details for TFLite model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class labels
class_labels = {
    0: "Bacterial Dermatosis",
    1: "Fungal Infection",
    2: "Healthy",
    3: "Allergic Dermatosis"
}

# Function to preprocess image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array.astype(np.float32)

# Function to perform classification using TFLite
def predict_disease(img_path):
    img_array = preprocess_image(img_path)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = np.argmax(predictions)
    confidence_score = np.max(predictions) * 100

    return class_labels[predicted_class], confidence_score, predicted_class

# Function to generate Grad-CAM heatmap
def generate_gradcam(img_path, predicted_class):
    last_conv_layer_name = [layer.name for layer in keras_model.layers if 'conv' in layer.name][-1]
    grad_model = tf.keras.models.Model(
        [keras_model.input], [keras_model.get_layer(last_conv_layer_name).output, keras_model.output]
    )

    img_array = preprocess_image(img_path)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, predicted_class]  # Focus on predicted class

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = np.sum(pooled_grads * conv_outputs, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)  # Normalize to 0-1 range

    return heatmap

# Function to estimate severity from Grad-CAM heatmap
def estimate_severity(img_path, predicted_class):
    heatmap = generate_gradcam(img_path, predicted_class)

    # Resize and convert heatmap to grayscale
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    gray_heatmap = cv2.cvtColor(cv2.applyColorMap(heatmap, cv2.COLORMAP_JET), cv2.COLOR_BGR2GRAY)

    # Thresholding to segment lesion area
    _, lesion_mask = cv2.threshold(gray_heatmap, 100, 255, cv2.THRESH_BINARY)
    affected_area = np.sum(lesion_mask > 0) / (224 * 224) * 100  # Percentage calculation

    # Determine severity based on affected area
    if affected_area > 65:
        severity = "Severe"
    elif affected_area > 35:
        severity = "Moderate"
    elif affected_area > 15:
        severity = "Mild"
    else:
        severity = "Very Mild or Healthy"

    return affected_area, severity

# API Endpoint for Image Classification & Severity Estimation
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join("uploads", filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    disease, confidence, class_idx = predict_disease(file_path)
    affected_area, severity = estimate_severity(file_path, class_idx)

    response = {
        "Predicted Disease": disease,
        "Confidence Score": f"{confidence:.2f}%",
        "Affected Area": f"{affected_area:.2f}%",
        "Severity Level": severity
    }

    return jsonify(response)

# Home Route (to check if server is running)
@app.route("/")
def home():
    return jsonify({"message": "Flask Server is Running!"})

# Run Flask App
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
