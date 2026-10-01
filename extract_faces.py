import os,cv2,argparse,numpy as np
from tqdm import tqdm
p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output',default='dataset/train'); p.add_argument('--label',required=True,choices=['real','fake']); p.add_argument('--frames',type=int,default=10); a=p.parse_args(); out=os.path.join(a.output,a.label); os.makedirs(out,exist_ok=True)
vids=[]
for dp,_,fs in os.walk(a.input):
 for f in fs:
  if f.lower().endswith(('.mp4','.mov','.avi','.mkv')): vids.append(os.path.join(dp,f))
for vp in tqdm(vids):
 cap=cv2.VideoCapture(vp); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); base=os.path.splitext(os.path.basename(vp))[0]
 if total<=0: cap.release(); continue
 for k,idx in enumerate(np.linspace(0,total-1,min(a.frames,total)).astype(int)):
  cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx)); ok,frame=cap.read()
  if ok: cv2.imwrite(os.path.join(out,f'{base}_{k:03d}.jpg'),frame)
 cap.release()
print('Done. For rigorous evaluation, split by source video before extraction.')
