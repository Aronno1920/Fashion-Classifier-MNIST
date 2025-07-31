######## Import Library
import os
import io
import subprocess
import numpy as np
import base64

from PIL import Image, ImageOps
from scipy.ndimage import center_of_mass, shift

from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, redirect, url_for
# load custom model
from model_config import ModelConfig
################################################


################################################
# Start Application
app = Flask(__name__)
################################################


################################################
# Load Model Configuration
model_config_nn = ModelConfig(model_type="nn")
model_config_cnn = ModelConfig(model_type="cnn")
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Load models safely
nn_model, cnn_model = None, None
model_message = {}
try:
    print("NN Model Path -> ", model_config_nn.get_model_path())
    nn_model = load_model(model_config_nn.get_model_path())
except:
    model_message["nn"] = "⚠️ Neural Network (NN) model not found. Please train it first."

try:
    print("CNN Model Path -> ", model_config_cnn.get_model_path())
    cnn_model = load_model(model_config_cnn.get_model_path())
except:
    model_message["cnn"] = "⚠️ Convolutional Neural Network (CNN) model not found. Please train it first."
################################################


################################################
# Load Index page
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", prediction=None, image_data=None, model_message=model_message)


def preprocess_image(image):
    from scipy.ndimage import center_of_mass, shift

    # Step 1: Convert to grayscale
    image = image.convert("L") 

    # Step 2: Resize and invert
    image = ImageOps.invert(image)
    image = ImageOps.fit(image, (28, 28), Image.Resampling.LANCZOS)

    # Step 3: Normalize
    img_array = np.array(image).astype(np.float32) / 255.0

    # Step 4: Threshold noise
    img_array[img_array < 0.2] = 0.0

    # Step 5: Center foreground
    cy, cx = center_of_mass(img_array)
    if np.isnan(cx) or np.isnan(cy):
        cx, cy = 14, 14
    shift_y = np.round(14 - cy).astype(int)
    shift_x = np.round(14 - cx).astype(int)

    # Step 6: Shift to center
    img_array = shift(img_array, shift=(shift_y, shift_x), mode='constant', cval=0.0)

    # Step 7: Reshape
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return render_template("index.html", error="No image uploaded.", prediction=None, image_data=None, model_message=model_message)

    file = request.files["image"]
    if file.filename == "":
        return render_template("index.html", error="No file selected.", prediction=None, image_data=None, model_message=model_message)

    try:
        # Reset stream and read raw image bytes
        image_bytes = file.read()

        # For web preview: encode image in base64
        image_data = base64.b64encode(image_bytes).decode("utf-8")

        # Load image for preprocessing
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # Ensure it's in a compatible format
        processed_image = preprocess_image(image)

        prediction = {"nn": [], "cnn": []}

        # NN model prediction
        if nn_model:
                print("------------ nn_model --------------")

                input_shape = nn_model.input_shape  # e.g., (None, 784)
                if input_shape[-1] == 784:
                    nn_input = processed_image.reshape(1, 784) # Flatten for dense model
                else:
                    nn_input = processed_image  # In case it changes in future

                nn_probs = nn_model.predict(nn_input)[0]
                topnn3 = nn_probs.argsort()[-3:][::-1]
                for i in topnn3:
                    prediction["nn"].extend([class_names[i], round(nn_probs[i] * 100, 2)])

        # CNN model prediction
        if cnn_model:
                print("------------ cnn_model --------------")
                cnn_probs = cnn_model.predict(processed_image)[0]
                topcnn3 = cnn_probs.argsort()[-3:][::-1]
                for i in topcnn3:
                    prediction["cnn"].extend([class_names[i], round(cnn_probs[i] * 100, 2)])

        return render_template("index.html", prediction=prediction, image_data=image_data, model_message=model_message)

    except Exception as e:
        print("Prediction error:", str(e))  # Log to console for debugging
        return render_template("index.html", error=f"Prediction error: {str(e)}", prediction=None, image_data=None, model_message=model_message)


# Click for predict (GET)
# @app.route("/predict", methods=["POST"])
# def predict():
#     if "image" not in request.files:
#         return render_template("index.html", error="No image uploaded.", prediction=None, image_data=None, model_message=model_message)

#     file = request.files["image"]
#     if file.filename == "":
#         return render_template("index.html", error="No file selected.", prediction=None, image_data=None, model_message=model_message)

#     try:
#         # Encode image for preview
#         file.stream.seek(0)
#         image_data = base64.b64encode(file.read()).decode("utf-8")

#         image = Image.open(file.stream)
#         processed_image = preprocess_image(image)

#         prediction = {"nn": [], "cnn": []}

#         if nn_model:
#             print("------------ nn_model --------------")

#             nn_probs = nn_model.predict(processed_image)[0]
#             top3 = nn_probs.argsort()[-3:][::-1]

#             for i in top3:
#                 print(f"Class Name -> ", [class_names[i], round(nn_probs[i] * 100, 2)])
#                 prediction["nn"].extend([class_names[i], round(nn_probs[i] * 100, 2)])

#         if cnn_model:
#             print("------------ cnn_model --------------")
#             cnn_probs = cnn_model.predict(processed_image)[0]
#             top3 = cnn_probs.argsort()[-3:][::-1]
#             for i in top3:
#                 print(f"Class Name -> ", [class_names[i], round(cnn_probs[i] * 100, 2)])
#                 prediction["cnn"].extend([class_names[i], round(cnn_probs[i] * 100, 2)])

#         return render_template("index.html", prediction=prediction, image_data=image_data, model_message=model_message)

#     except Exception as e:
#         return render_template("index.html", error=f"Prediction error: {str(e)}", prediction=None, image_data=None, model_message=model_message)
#######################################################



################################################
# Load Training Page UI (GET)
@app.route("/train", methods=["GET"])
def train_page():
    return render_template("train.html")

# Trigger Model Training (POST)
@app.route("/train", methods=["GET", "POST"])
def train_model():
    log_output = ""

    if request.method == "POST":
        model_type = request.form.get("model_type")

        if model_type == "nn":
            cmd = ["python", "train_model_nn.py"]
        elif model_type == "cnn":
            cmd = ["python", "train_model_cnn.py"]
        else:
            log_output = "Please click any model and start training."

        if model_type in ["nn", "cnn"]:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            log_output = result.stdout

    return render_template("train.html", log_output=log_output)
################################################


################################################
# Run App
if __name__ == '__main__':
    app.run(debug=True)
################################################