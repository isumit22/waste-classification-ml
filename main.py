# main.py (real PyTorch classifier)
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Use relative path to static folder
STATIC_FOLDER = os.path.join(os.getcwd(), "static")

# Load class list (must match training order!)
classes = [
     'biological', 'cardboard', 'clothes',
    'glass', 'Hazardous', 'medical', 'metal',
    'paper', 'plastic', 'shoes', 'trash'
]

# Model setup
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(classes))
model.load_state_dict(torch.load("waste_classifier_final.pth", map_location="cpu"))
model.eval()

# Image transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def getPrediction(filename):
    """
    Run inference on an uploaded image and return:
      - predicted class
      - confidence score
      - filename (for frontend)
    """
    img_path = os.path.join(STATIC_FOLDER, filename)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"File not found: {img_path}")

    # Load & preprocess image
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)  # add batch dim

    with torch.no_grad():
        outputs = model(img)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        conf, pred = torch.max(probs, 0)

    label = classes[pred.item()]
    confidence = f"{conf.item() * 100:.2f}%"

    return str(label), confidence, filename

# Optional local test
if __name__ == "__main__":
    try:
        print(getPrediction("test.jpg"))
    except Exception as e:
        print("Test failed:", e)
