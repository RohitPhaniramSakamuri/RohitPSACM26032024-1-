import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2

def display_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Check if the image is loaded successfully
    if image is None:
        print("Error: Unable to read the image.")
        return

    # Display the image
    cv2.imshow('Image', image)

    # Wait for 'q' key to be pressed
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    # Close all OpenCV windows
    cv2.destroyAllWindows()

# Load the saved model
model = models.resnet18(weights = None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1000)  #original model had 1000 output units
model.load_state_dict(torch.load(r"C:\Users\Admin\PythonScripts\RohitPSACM\PepeOrNotclassifier.pth"))
model.eval()

# Create a new model with the correct final layer
new_model = models.resnet18(weights = "ResNet18_Weights.DEFAULT")
num_ftrs = new_model.fc.in_features
new_model.fc = nn.Linear(num_ftrs, 2)  # Binary Classification

# Copy the weights and biases from the loaded model to the new model
new_model.fc.weight.data = model.fc.weight.data[:2, :]  # Copy only the first 2 output units
new_model.fc.bias.data = model.fc.bias.data[:2]

# Load and preprocess the unseen image
image_path = r"C:\Users\Admin\PythonScripts\RohitPSACM\val\Pepe\isPepe545.png" # Replace with the path to your image
image = Image.open(image_path).convert('RGB')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

# Perform inference
with torch.no_grad():
    output = new_model(input_batch)

# Get the predicted class
_, predicted_class = output.max(1)

# Map the predicted class to the class name
class_names = ['Pepe', 'not_pepe']  # Make sure these class names match your training data
predicted_class_name = class_names[predicted_class.item()]

print(f'The predicted class is: {predicted_class_name}')

display_image(image_path)
