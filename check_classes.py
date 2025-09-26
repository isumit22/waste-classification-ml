from torchvision import datasets, transforms

# Path to your training data
data_dir = "garbage_split/train"

# Create dataset object (just for checking classes)
dataset = datasets.ImageFolder(data_dir, transform=transforms.ToTensor())

print("✅ Classes found:", dataset.classes)
print("✅ Number of classes:", len(dataset.classes))
