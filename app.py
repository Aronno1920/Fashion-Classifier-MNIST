######## Import Library
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

import tensorflow as tf

from flask import Flask, render_template, request, jsonify
#################
from model_config import ModelConfig

######## Start Application
app = Flask(__name__)
#################


######## Load the Trained Model
model_config = ModelConfig()
model_path = model_config.get_model_path()
#################


######## Load Root - Index page
@app.route("/")
def index():
    return render_template("index.html")
#################



if __name__ == '__main__':
    app.run(debug=True)