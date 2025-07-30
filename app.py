######## Import Library
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import subprocess
import numpy as np
import base64

from PIL import Image
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
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat","Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

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

# Click for predict (GET)
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return render_template("index.html", error="No image uploaded.", prediction=None, image_data=None, model_message=model_message)

    file = request.files["image"]
    if file.filename == "":
        return render_template("index.html", error="No file selected.", prediction=None, image_data=None, model_message=model_message)

    try:
        img = Image.open(file).convert("L").resize((28, 28))
        img_array = np.array(img).astype("float32") / 255.0

        # Encode image for preview
        file.stream.seek(0)
        image_data = base64.b64encode(file.read()).decode("utf-8")

        nn_input = img_array.reshape(1, 28 * 28)
        cnn_input = img_array.reshape(1, 28, 28, 1)

        prediction = {"nn": [], "cnn": []}

        if nn_model:
            print("------------ nn_model --------------")

            nn_probs = nn_model.predict(nn_input, verbose=0)[0]
            top3 = np.argsort(nn_probs)[::-1][:3]

            for i in top3:
                print(f"Class Name -> ", [class_names[i], round(nn_probs[i] * 100, 2)])
                prediction["nn"].extend([class_names[i], round(nn_probs[i] * 100, 2)])

        if cnn_model:
            print("------------ cnn_model --------------")
            cnn_probs = cnn_model.predict(cnn_input, verbose=0)[0]
            top3 = np.argsort(cnn_probs)[::-1][:3]
            for i in top3:
                print(f"Class Name -> ", [class_names[i], round(cnn_probs[i] * 100, 2)])
                prediction["cnn"].extend([class_names[i], round(cnn_probs[i] * 100, 2)])

        return render_template("index.html", prediction=prediction, image_data=image_data, model_message=model_message)

    except Exception as e:
        return render_template("index.html", error=f"Prediction error: {str(e)}", prediction=None, image_data=None, model_message=model_message)
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