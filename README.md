# AI Deepfake Detection System

Complete educational Flask + PyTorch project for screening images and videos for possible facial manipulation.

## Includes
- Flask web app
- Image and video analysis
- EfficientNet-B0 classifier
- Training script
- Evaluation script with accuracy, precision, recall, F1 and confusion matrix
- Video frame extraction utility
- SQLite scan history

## Setup
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

## Dataset
The full FaceForensics++ dataset is not bundled. Its official maintainers require an access request and its complete data is very large. Obtain it through the official project and follow its Terms of Use.

Create:
```
dataset/train/real
dataset/train/fake
dataset/val/real
dataset/val/fake
dataset/test/real
dataset/test/fake
```
Put extracted face/images in those folders. Split by source video to avoid leakage.

## Train
```powershell
python train.py --data dataset --epochs 5 --batch-size 16
```
Model: `model/deepfake_model.pth`

## Evaluate
```powershell
python evaluate.py --data dataset
```

## Extract frames
```powershell
python extract_faces.py --input path\to\videos --output dataset/train --label fake --frames 10
```

## Limitation
A detector can make false positives and false negatives, especially on unseen generators, compression levels, cameras and domains. A prediction is not forensic proof.
