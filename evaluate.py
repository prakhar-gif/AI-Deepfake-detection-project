import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt


# =========================
# SETTINGS
# =========================

DATASET_DIR = "dataset"
MODEL_PATH = os.path.join("model", "deepfake_model.pth")

BATCH_SIZE = 16
IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# =========================
# IMAGE TRANSFORM
# =========================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# TEST DATASET
# =========================

test_dir = os.path.join(
    DATASET_DIR,
    "test"
)

test_dataset = datasets.ImageFolder(
    test_dir,
    transform=transform
)

print("Classes:", test_dataset.class_to_idx)
print("Test images:", len(test_dataset))


# =========================
# CHECK CLASS MAPPING
# =========================

expected_mapping = {
    "fake": 0,
    "real": 1
}

if test_dataset.class_to_idx != expected_mapping:

    raise ValueError(
        f"Expected classes {expected_mapping}, "
        f"but found {test_dataset.class_to_idx}"
    )


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =========================
# LOAD MODEL
# =========================

print("Loading model...")

model = models.efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)

model.eval()

print("Model loaded successfully.")


# =========================
# PREDICTIONS
# =========================

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# =========================
# METRICS
# =========================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)


print()
print("========================================")
print("DEEPFAKE DETECTION MODEL EVALUATION")
print("========================================")

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall   : {recall * 100:.2f}%"
)

print(
    f"F1 Score : {f1 * 100:.2f}%"
)

print()
print("Classification Report:")
print()

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=["Fake", "Real"],
        zero_division=0
    )
)


# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("Confusion Matrix:")
print(cm)


# =========================
# SAVE CONFUSION MATRIX
# =========================

os.makedirs(
    "reports",
    exist_ok=True
)

plt.figure(figsize=(6, 5))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Deepfake Detection Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    [0, 1],
    ["Fake", "Real"]
)

plt.yticks(
    [0, 1],
    ["Fake", "Real"]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

output_file = os.path.join(
    "reports",
    "confusion_matrix.png"
)

plt.savefig(
    output_file,
    dpi=150
)

plt.close()

print()
print(
    "Confusion matrix saved to:",
    output_file
)

print()
print("Evaluation complete.")