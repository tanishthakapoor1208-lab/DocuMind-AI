import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import time
import torch.nn.functional as F

# Load classes

import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSES_PATH = os.path.join(BASE_DIR, "classes.json")

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    classes = json.load(f)


# Load model
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(classes))

MODEL_PATH = os.path.join(BASE_DIR, "cnn_model.pth")
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Prediction
def predict_image(image_path):
    start = time.time()

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probs, 1)

    end = time.time()

    return {
        "class": classes[predicted.item()],
        "confidence": round(confidence.item() * 100, 2),
        "time": round(end - start, 3)
    }
