import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import json

# SETTINGS
DATA_DIR = "dataset"
BATCH_SIZE = 8

# TRANSFORM (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# LOAD DATA
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

# Split into train + validation (80-20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# LOAD MODEL
with open("cnn/classes.json", "r") as f:
    classes = json.load(f)

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(classes))

model.load_state_dict(torch.load("cnn/cnn_model.pth", map_location="cpu"))
model.eval()

# VALIDATION
all_preds = []
all_labels = []

correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total

print("\n✅ Validation Accuracy:", accuracy)

# Detailed report
print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))

print("\n📉 Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))