import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import json
import os

# ==============================
# SETTINGS
# ==============================

DATA_DIR = "dataset"
BATCH_SIZE = 8        # safer for CPU
EPOCHS = 10           # overnight training

# ==============================
# TRANSFORMS
# ==============================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==============================
# LOAD DATASET
# ==============================

print("Loading dataset...")

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

print(f"✅ Dataset loaded: {len(dataset)} images")
print(f"📂 Classes: {dataset.classes}")

# Save class names
os.makedirs("cnn", exist_ok=True)
with open("cnn/classes.json", "w") as f:
    json.dump(dataset.classes, f)

# DataLoader (IMPORTANT FOR WINDOWS)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

# ==============================
# DEVICE
# ==============================

device = torch.device("cpu")
print(f"🖥️ Using device: {device}")

# ==============================
# MODEL (FAST FOR CPU)
# ==============================

print("Loading model...")

model = models.mobilenet_v2(weights="IMAGENET1K_V1")

# Replace classifier
model.classifier[1] = nn.Linear(model.last_channel, len(dataset.classes))

model = model.to(device)

# ==============================
# LOSS + OPTIMIZER
# ==============================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==============================
# TRAINING
# ==============================

print("🚀 Starting training...")

for epoch in range(EPOCHS):
    total_loss = 0
    print(f"\n🔹 Epoch {epoch+1}/{EPOCHS} started")

    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Show progress every 50 batches
        if i % 50 == 0:
            print(f"   Batch {i}/{len(loader)} | Loss: {loss.item():.4f}")

    print(f"✅ Epoch {epoch+1} completed | Total Loss: {total_loss:.4f}")

# ==============================
# SAVE MODEL
# ==============================

torch.save(model.state_dict(), "cnn/cnn_model.pth")

print("\n🎉 Training complete!")
print("💾 Model saved at: cnn/cnn_model.pth")