from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# Load trained model
model = load_model("mulberry_leaf_model.h5")

# Define class labels and recommendations
class_labels = ["Disease Free Leaves", "Leaf Rust", "Leaf Spot"]
recommendations = {
    "Disease Free Leaves": "✅ Your plant is healthy! Maintain proper watering and sunlight.",
    "Leaf Rust": "⚠️ Use a fungicide spray, remove infected leaves, and avoid overhead watering.",
    "Leaf Spot": "⚠️ Trim affected leaves, ensure good air circulation, and apply copper-based fungicide."
}

# Function to preprocess image
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(150, 150))  # Resize image
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

# Route for index page
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    recommendation = None
    img_path = None

    if request.method == "POST":
        # Get uploaded file
        file = request.files["file"]
        if file:
            # Save uploaded file
            img_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(img_path)

            # Process and predict
            img_array = preprocess_image(img_path)
            pred = model.predict(img_array)
            predicted_class = np.argmax(pred, axis=1)[0]
            prediction = class_labels[predicted_class]
            recommendation = recommendations[prediction]

    return render_template("index.html", prediction=prediction, recommendation=recommendation, img_path=img_path)

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
