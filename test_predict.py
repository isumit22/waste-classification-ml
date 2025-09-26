import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ✅ Updated Classes (11 total)
classes = [
    'biological', 'cardboard', 'clothes',
    'glass', 'Hazardous', 'medical', 'metal',
    'paper', 'plastic', 'shoes', 'trash'
]

# 🔹 Load model
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(classes))

# Load trained checkpoint
model.load_state_dict(torch.load("waste_classifier_final.pth", map_location="cpu"))
model.eval()

# 🔹 Image transform (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 🔄 Change to your test image
image_path = "cardboard113.jpg"
img = Image.open(image_path).convert("RGB")
img = transform(img).unsqueeze(0)

# 🔹 Predict
with torch.no_grad():
    outputs = model(img)
    probs = torch.nn.functional.softmax(outputs, dim=1)[0]
    conf, pred = torch.max(probs, 0)

label = classes[pred.item()]

# 🔹 Waste info function
def get_waste_info(label, conf):
    recyclable_classes = ["plastic", "paper", "metal", "glass", "cardboard"]
    no_bin_items = ["Hazardous", "medical"]

    if label in recyclable_classes:
        bin_name = "Blue Bin"
    elif label in no_bin_items:
        bin_name = "No Bin"
    else:
        bin_name = "Green Bin"

    tip_dict = {
        "Hazardous": "Handle with care. Take to hazardous waste facilities.",
        "medical": "Dispose in a secure medical waste bin. Never mix with household waste."
    }

    return {
        "waste_type": label,
        "confidence": round(conf * 100, 2),
        "category": "Recyclable" if label in recyclable_classes else "Non-Recyclable",
        "recyclable": label in recyclable_classes,
        "bin": bin_name,
        "tip": tip_dict.get(label, "Dispose properly to avoid contamination")
    }

# 🔹 Get prediction info
result = get_waste_info(label, conf.item())

# 🔹 Print results
print("✅ Prediction:", result["waste_type"], f"({result['confidence']}%)")
print("Category:", result["category"])
print("Bin:", result["bin"])
print("Recyclable:", result["recyclable"])
print("Tip:", result["tip"])
