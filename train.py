import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# 🖼️ Custom loader to force RGB (fix PNG transparency issue)
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def main():
    # 🔧 Settings
    data_dir = "garbage_split"
    batch_size = 32
    epochs = 25   
    lr = 0.001
    patience = 5  # Early stopping patience

    # 🔄 Data transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # 📂 Datasets & Loaders
    train_dataset = datasets.ImageFolder(f"{data_dir}/train", transform=train_transform, loader=pil_loader)
    test_dataset = datasets.ImageFolder(f"{data_dir}/test", transform=test_transform, loader=pil_loader)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 📝 Class info
    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"📂 Classes found: {num_classes}")
    print(f"📝 Class names: {class_names}")

    # 🚀 Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ Using device: {device}")

    model = models.resnet50(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 📊 Tracking
    train_acc_list, test_acc_list, loss_list = [], [], []
    best_acc = 0
    early_stop_counter = 0

    # 📚 Training loop
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # 🔎 Test evaluation
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == labels).sum().item()
                test_total += labels.size(0)

        test_acc = 100 * test_correct / test_total

        # 📊 Track metrics
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        loss_list.append(avg_loss)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"- Loss: {avg_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

        # 🔹 Early stopping
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_waste_classifier.pth")
            early_stop_counter = 0
            print(f"💾 New best model saved with test acc: {best_acc:.2f}%")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("⏹️ Early stopping triggered")
                break

        scheduler.step()

        # 💾 Checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"waste_classifier_epoch{epoch+1}.pth")
            print(f"💾 Checkpoint saved at epoch {epoch+1}")

    end_time = time.time()
    print(f"⏱️ Training completed in {(end_time - start_time)/60:.2f} minutes")

    # ✅ Final save
    torch.save(model.state_dict(), "waste_classifier_final.pth")
    print("💾 Final model saved as waste_classifier_final.pth")
    print("✅ Final layer shape:", model.fc.weight.shape)

    # 📊 Plot accuracy & loss curves
    plt.figure(figsize=(10,5))
    plt.plot(train_acc_list, label="Train Accuracy")
    plt.plot(test_acc_list, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Training vs Testing Accuracy")
    plt.savefig("accuracy_curve.png")
    plt.show()

    plt.figure(figsize=(10,5))
    plt.plot(loss_list, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss Curve")
    plt.savefig("loss_curve.png")
    plt.show()

    # 📊 Confusion Matrix
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()

# ✅ Windows multiprocessing safe guard
if __name__ == "__main__":
    main()
