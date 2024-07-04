from flask import Flask, request, jsonify
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import os

app = Flask(__name__)

# Define your model class
class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the model
model = SimpleNN()

# Check if model file exists
model_file = 'mnist_model.pth'
if os.path.isfile(model_file):
    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    model.eval()
else:
    print(f"Error: Model file '{model_file}' not found.")
    exit(1)

# Prediction function
def predict(image_path):
    try:
        image = Image.open(image_path).convert('L')
        image = image.resize((28, 28))
        image = np.array(image)
        image = torch.tensor(image, dtype=torch.float32)
        image = image.unsqueeze(0).unsqueeze(0)
        image = transforms.Normalize((0.5,), (0.5,))(image)

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)

        return predicted.item()

    except Exception as e:
        print(f"Prediction error: {e}")
        return None

# Flask route for root endpoint
@app.route('/')
def index():
    return "Welcome to the Digit Recognition API!"

# Flask route for prediction
@app.route('/predict', methods=['GET'])
def predict_digit():
    try:
        image_path = './two.webp'
        prediction = predict(image_path)

        if prediction is not None:
            return jsonify({'prediction': int(prediction)}), 200
        else:
            return jsonify({'error': 'Failed to predict'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)
