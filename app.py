######## Import Library
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import subprocess
from flask import Flask, render_template, request, redirect, url_for

from model_config import ModelConfig
##############


######## Start Application
app = Flask(__name__)


######## Load Model Configuration
model_config = ModelConfig()
model_path = model_config.get_model_path()


######## Load Root - Index Page
@app.route("/")
def index():
    return render_template("index.html")


######## Load Training Page UI (GET)
@app.route("/train", methods=["GET"])
def train_page():
    return render_template("train.html")


######## Trigger Model Training (POST)
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


######## Run App
if __name__ == '__main__':
    app.run(debug=True)








# ######## Import Library
# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# import tensorflow as tf

# from flask import Flask, render_template, request, jsonify
# #################
# from model_config import ModelConfig
# from train import train_models

# ######## Start Application
# app = Flask(__name__)
# #################


# ######## Load the Trained Model
# model_config = ModelConfig()
# model_path = model_config.get_model_path()
# #################


# ######## Load Root - Index page
# @app.route("/")
# def index():
#     return render_template("index.html")
# #################

# ######## Load model train page
# @app.route("/train")
# def train_model():
#     results = train_models()
#     return render_template("train.html", results=results)
# #################


# if __name__ == '__main__':
#     app.run(debug=True)