import os,argparse,torch
import torch.nn as nn
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader
p=argparse.ArgumentParser(); p.add_argument('--data',default='dataset'); p.add_argument('--epochs',type=int,default=5); p.add_argument('--batch-size',type=int,default=16); a=p.parse_args()
tf=transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]); vt=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
tr=datasets.ImageFolder(os.path.join(a.data,'train'),transform=tf); va=datasets.ImageFolder(os.path.join(a.data,'val'),transform=vt)
if tr.class_to_idx.get('fake') != 0 or tr.class_to_idx.get('real') != 1:
    raise ValueError(f"Expected fake=0,real=1; got {tr.class_to_idx}")
tl=DataLoader(tr,batch_size=a.batch_size,shuffle=True,num_workers=0); vl=DataLoader(va,batch_size=a.batch_size,shuffle=False,num_workers=0); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT); m.classifier[1]=nn.Linear(m.classifier[1].in_features,2); m.to(dev); opt=torch.optim.Adam(m.parameters(),lr=1e-4); lossf=nn.CrossEntropyLoss()
print('Device:',dev,'Train:',len(tr),'Validation:',len(va))
for ep in range(a.epochs):
 m.train(); ls=0
 for x,y in tl: x,y=x.to(dev),y.to(dev); opt.zero_grad(); out=m(x); loss=lossf(out,y); loss.backward(); opt.step(); ls+=loss.item()
 m.eval(); c=t=0
 with torch.no_grad():
  for x,y in vl: pred=m(x.to(dev)).argmax(1).cpu(); c+=(pred==y).sum().item(); t+=len(y)
 print(f'Epoch {ep+1}/{a.epochs} loss={ls/max(1,len(tl)):.4f} val_acc={c/max(1,t):.4f}')
os.makedirs('model',exist_ok=True); torch.save(m.state_dict(),'model/deepfake_model.pth'); print('Saved model/deepfake_model.pth')
