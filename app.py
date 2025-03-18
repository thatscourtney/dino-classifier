import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from PIL import Image

# Load class names
data_dir = "data"
dataset = datasets.ImageFolder(data_dir)
class_names = dataset.classes

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("model/dino_classifier.pth", map_location=device))
model = model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# UI
st.title("🦕 Dino Image Classifier")
st.write("Upload a dinosaur image to find out what species it is.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)


    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        top_probs, top_idxs = torch.topk(probs, 3)
        top_probs = top_probs.squeeze().tolist()
        top_idxs = top_idxs.squeeze().tolist()

    predicted_class = class_names[top_idxs[0]]
    predicted_confidence = top_probs[0] * 100

    st.markdown(f"### 🦖 Prediction: **{predicted_class}** ({predicted_confidence:.2f}% confidence)")

    st.write("#### Top 3 Predictions:")
    for i in range(3):
        st.write(f"- {class_names[top_idxs[i]]}: {top_probs[i] * 100:.2f}%")
