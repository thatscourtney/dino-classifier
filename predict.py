import sys
import torch
from torchvision import models, transforms, datasets
from PIL import Image

# Load class names from the training folder structure
data_dir = "data"
dataset = datasets.ImageFolder(data_dir)
class_names = dataset.classes

model_path = "model/dino_classifier.pth"

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load the image
image_path = sys.argv[1]
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0)

# Load the model architecture
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Make prediction
with torch.no_grad():
    outputs = model(image.to(device))
    _, predicted = torch.max(outputs, 1)
    predicted_class = class_names[predicted.item()]
    print(f"🦕 Prediction: {predicted_class}")
