import torch

# Load the checkpoint
checkpoint = torch.load("waste_classifier.pth", map_location="cpu")

print("🔍 Keys in checkpoint:", checkpoint.keys())

# If it's a full state_dict
if "fc.weight" in checkpoint:
    print("⚡ Direct ResNet state_dict detected")
    print("fc.weight shape:", checkpoint["fc.weight"].shape)
elif isinstance(checkpoint, dict):
    # Sometimes it's wrapped in {"model_state_dict": ...}
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print("⚡ model_state_dict detected")
        print("fc.weight shape:", state_dict["fc.weight"].shape)
    else:
        print("⚠️ Unknown checkpoint format, keys:", checkpoint.keys())
else:
    print("⚠️ Not a dict checkpoint, type:", type(checkpoint))
