# 🧥 Fashion Classifier (MNIST) – TensorFlow, Flask, and FastAPI

This project is a full-stack web application that allows users to upload an image of a fashion item (e.g., shirt, sneaker, bag) and classifies it using a Neural Network (NN) and Convolutional Neural Network (CNN) trained on the [Fashion Classifier](https://github.com/Aronno1920/Fashion-Classifier-MNIST) dataset.

The app features:
- 🔍 Real-time image classification
- ⚙️ NN and CNN model build separately
- 🧠 Robust preprocessing (resize, normalize, center the object)
- 📸 Live preview of uploaded image
- 💡 Displays top 3 predictions with confidence scores
- 🎨 Beautiful and responsive HTML/CSS interface


## 🚀 Live Demo

> _Want to see it in action?_  
> 👉 [Coming Soon: Deployed on Render or Hugging Face Spaces]


## 🧠 Model Details

- **Architecture**: 3-layer CNN with BatchNorm, Dropout, and MaxPooling
- **Input shape**: 28×28 grayscale images
- **Training Dataset**: Fashion MNIST (60,000 training + 10,000 test images)
- **Accuracy**: ~91% on test data

---

## 🖼️ Supported Classes

``` ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] ```

---

## 🧪 Try It Locally

### 1️⃣ Clone the repo
```bash
git clone https://github.com/Aronno1920/Fashion-Classifier-MNIST.git
cd fashion-classifier-mnist
```
2️⃣ Install requirements
```bash
pip install -r requirements.txt
```
3️⃣ Run the app
```bash
python app.py
```
Open your browser: http://127.0.0.1:5000


📁 Project Structure

```Image-Classification-AI/
├── ImageSample/                        # Folder for example images or datasets
├── __pycache__/                        # Python cache files (auto-generated)
├── model/
│   ├── best_nn_model.keras             # Saved trained NN model weights
│   └── best_cnn_model.keras            # Saved trained CNN model weights
├── notebooks/
│   ├── fashion_mnist_selim_ahmed.py    # Saved colab file as python
│   └── Fashion_MNIST_Selim_Ahmed.ipynb # Saved colab file 
├── static/                             # logo, style and other necessary
│   ├── logo.png
│   └── style.css
├── templates/                          # Templates for web app or reports (if any)
│   ├── index.html
│   └── train.html
├── templates/                 
├── README.md                           # Project description and documentation
├── app.py                              # Main application script (e.g., for running or serving the model)
├── model_config.py                     
├── train_model_cnn.py                  # CNN model training class
└── train_model_nn.py                   # NN model training class
```

🛠 Tech Stack
```bash
    Python
    TensorFlow / Keras
    Flask
    PIL (Pillow)
    HTML5 + CSS
    JavaScript (Image preview)
    SciPy (for center-of-mass preprocessing)
```

📸 Screenshots

![Screenshot](https://github.com/Aronno1920/Fashion-Classifier-MNIST/blob/main/screenshot/Screenshot_1.png)
![Screenshot](https://github.com/Aronno1920/Fashion-Classifier-MNIST/blob/main/screenshot/Screenshot_2.png)
![Screenshot](https://github.com/Aronno1920/Fashion-Classifier-MNIST/blob/main/screenshot/Screenshot_3.png)
![Screenshot](https://github.com/Aronno1920/Fashion-Classifier-MNIST/blob/main/screenshot/Screenshot_4.png)
![Screenshot](https://github.com/Aronno1920/Fashion-Classifier-MNIST/blob/main/screenshot/Screenshot_5.png)
![Screenshot](https://github.com/Aronno1920/Fashion-Classifier-MNIST/blob/main/screenshot/Screenshot_6.png)

✍️ Author

Selim Ahmed <br/>
📫 [Connect on LinkedIn](https://www.linkedin.com/in/aronno1920/)<br/>
🌐 [GitHub Profile](https://github.com/aronno1920)

