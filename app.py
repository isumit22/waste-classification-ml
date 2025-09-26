from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)
CORS(app)  # ✅ allow requests from React (localhost:5173)

# ✅ Classes (11 total)
classes = [
    'biological', 'cardboard', 'clothes',
    'glass', 'Hazardous', 'medical',
    'metal', 'paper', 'plastic',
    'shoes', 'trash'
]

# ---------------------- MODEL LOADING ---------------------- #
# Load ResNet-50 base
model = models.resnet50(weights="IMAGENET1K_V1")
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(classes))  # 11 outputs

# ✅ Load checkpoint trained for 11 classes
checkpoint = torch.load("waste_classifier_final.pth", map_location="cpu")
model.load_state_dict(checkpoint, strict=True)  # must match 11-class model
model.eval()
# ------------------------------------------------------------ #

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ♻️ Recyclables → Blue Bin
recyclables = ["plastic", "paper", "metal", "cardboard", "glass"]

# 🌱 Compostables → Green Bin
compostables = ["biological"]

# 🚫 No-bin items
no_bin_items = ["Hazardous", "medical"]

# 💡 Tips
waste_tips = {
    "biological": "Put food scraps and garden waste in the wet waste bin.",
    "cardboard": "Flatten boxes before disposal.",
    "clothes": "Donate if usable, otherwise put in dry waste.",
    "glass": "Place bottles and jars carefully. Handle broken glass with care.",
    "Hazardous": "Take to hazardous waste facilities (chemicals, electronics, etc).",
    "medical": "Dispose in a secure medical waste bin. Never mix with household waste.",
    "metal": "Put cans and tins in dry waste.",
    "paper": "Keep newspapers and paper clean and dry for recycling.",
    "plastic": "Bottles, wrappers, and containers go in dry waste.",
    "shoes": "Donate if wearable, else put in dry waste.",
    "trash": "General waste goes to landfill. Avoid mixing recyclables."
}

# 📖 Reasons (educational)
waste_reasons = {
    "biological": "Wet waste can be composted to make fertilizer.",
    "cardboard": "Recycling cardboard saves trees and reduces pollution.",
    "clothes": "Reusing clothes reduces textile waste.",
    "glass": "Glass can be recycled endlessly without loss in quality.",
    "Hazardous": "Improper disposal may release toxic chemicals.",
    "medical": "Medical waste can spread infections if not handled properly.",
    "metal": "Recycling metal saves energy and resources.",
    "paper": "Paper recycling reduces deforestation.",
    "plastic": "Recycling plastic reduces pollution.",
    "shoes": "Reusing shoes reduces landfill waste.",
    "trash": "General waste should be minimized."
}

# 🎨 Bin info
bin_info = {
    "Blue Bin": {"color": "#007BFF", "icon": "recycle"},
    "Green Bin": {"color": "#28A745", "icon": "leaf"},
    "No Bin": {"color": "#FF0000", "icon": "alert-triangle"},
    "General Bin": {"color": "#6C757D", "icon": "trash-2"}
}

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img = Image.open(file).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        conf, pred = torch.max(probs, 0)

    label = classes[pred.item()]

    # Decide bin
    if label in recyclables:
        bin_name = "Blue Bin"
        category = "Recyclable"
    elif label in compostables:
        bin_name = "Green Bin"
        category = "Compostable"
    elif label in no_bin_items:
        bin_name = "No Bin"
        category = "Special Handling Required"
    else:
        bin_name = "General Bin"
        category = "Non-Recyclable"

    result = {
        "waste_type": label,
        "confidence": round(conf.item() * 100, 2),
        "category": category,
        "bin": {
            "name": bin_name,
            "color": bin_info[bin_name]["color"],
            "icon": bin_info[bin_name]["icon"]
        },
        "recyclable": label in recyclables,
        "tip": waste_tips.get(label, "Dispose properly at home."),
        "reason": waste_reasons.get(label, "Helps keep the environment clean.")
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
