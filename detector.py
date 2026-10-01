import os, cv2, numpy as np, torch
from PIL import Image
import torch.nn as nn
from torchvision import models, transforms
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); MODEL_PATH=os.path.join(ROOT,'model','deepfake_model.pth')
class DeepfakeDetector:
 def __init__(self):
  self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); self.model_available=os.path.exists(MODEL_PATH); self.model=None
  if self.model_available:
   self.model=models.efficientnet_b0(weights=None); self.model.classifier[1]=nn.Linear(self.model.classifier[1].in_features,2); self.model.load_state_dict(torch.load(MODEL_PATH,map_location=self.device)); self.model.to(self.device).eval()
  self.tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]); self.face=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
 def faces(self,img): return self.face.detectMultiScale(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),1.1,5,minSize=(60,60))
 def predict(self,pil):
  x=self.tf(pil).unsqueeze(0).to(self.device)
  with torch.no_grad(): p=torch.softmax(self.model(x),1)[0]
  fake=float(p[1]); return ('Potential Deepfake' if fake>=.5 else 'Likely Real'),round(max(fake,1-fake)*100,2)
 def analyze_image(self,path):
  img=cv2.imread(path)
  if img is None: raise ValueError('Cannot read image.')
  if not self.model_available: return {'prediction':'Model not trained','confidence':0.0,'details':'Train the model first: python train.py --data dataset --epochs 5'}
  pred,conf=self.predict(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))); return {'prediction':pred,'confidence':conf,'details':f'EfficientNet-B0. Faces detected: {len(self.faces(img))}. Device: {self.device}.'}
 def analyze_video(self,path):
  if not self.model_available: return {'prediction':'Model not trained','confidence':0.0,'details':'Train the model first: python train.py --data dataset --epochs 5'}
  cap=cv2.VideoCapture(path); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); n=min(30,total)
  if n<=0: cap.release(); raise ValueError('Cannot read video.')
  scores=[]; faces=0
  for idx in np.linspace(0,total-1,n).astype(int):
   cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx)); ok,frame=cap.read()
   if not ok: continue
   faces+=len(self.faces(frame)); pred,conf=self.predict(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))); scores.append(conf/100 if pred=='Potential Deepfake' else 1-conf/100)
  cap.release()
  if not scores: raise ValueError('No frames could be analyzed.')
  fake=float(np.mean(scores)); return {'prediction':'Potential Deepfake' if fake>=.5 else 'Likely Real','confidence':round(max(fake,1-fake)*100,2),'details':f'EfficientNet-B0. Frames analyzed: {len(scores)}/{n}. Faces detected: {faces}. Device: {self.device}.'}
