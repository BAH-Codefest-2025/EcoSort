from flask import Flask, request, render_template, jsonify
import torch as th
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

app = Flask(__name__) # Create a Flask app

# Set up device
device = th.device("cuda" if th.cuda.is_available() else "cpu") # Check if CUDA is available

# Load class names
class_names = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash']
num_classes = len(class_names)

# Load the trained model
model = models.resnet50(weights=None) # Load the ResNet-50 model
model.fc = nn.Linear(model.fc.in_features, num_classes) # Change the output layer to match the number of classes
model.load_state_dict(th.load("garbage_classification_model.pth", map_location=device)) # Load the trained model
model.to(device) # Move the model to
model.eval() # Set the model to evaluation

# Define the transformation to apply to the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_path):
    image_tensor = transform(image_path).unsqueeze(0).to(device) # Apply the transformation and move the image to the device
    with th.no_grad(): # Disable gradient calculation
        output = model(image_tensor) # Get the model's prediction
        _, predicted = th.max(output, 1) # Get the index of the class with the highest probability
    return class_names[predicted.item()] # Get the class name

# Define the route for the home page
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST": # If a POST request is received
        file = request.files["file"]
        if "file" not in request.files: # If no file is uploaded
            return jsonify({"error": "No file uploaded"})
        if file.filename == "":
            return jsonify({"error": "No file selected"})
        if file:
            image = Image.open(file.stream).convert("RGB")
            prediction = predict_image(image)
            return jsonify({"prediction": prediction})
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True) # Run the app in debug mode



