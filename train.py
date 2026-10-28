"""Fine-tune on an independently annotated YOLO segmentation dataset."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--split-manifest',required=True,help='JSON mapping train/val/test to physical road IDs');p.add_argument('--weights',default='models/pothole-seg.pt');p.add_argument('--device',default='0');p.add_argument('--epochs',type=int,default=80);p.add_argument('--imgsz',type=int,default=960)
a=p.parse_args();splits=json.loads(Path(a.split_manifest).read_text())
for x in ['train','val','test']:
    if not splits.get(x):raise ValueError('Each split needs physical road IDs')
for x,y in [('train','val'),('train','test'),('val','test')]:
    if set(splits[x])&set(splits[y]):raise ValueError('Road leakage between '+x+' and '+y)
from ultralytics import YOLO
m=YOLO(a.weights)
if m.task!='segment':raise ValueError('A segmentation checkpoint is required')
m.train(data=a.data,epochs=a.epochs,imgsz=a.imgsz,device=a.device,seed=42,deterministic=True,patience=15,close_mosaic=10,project='training',name='pothole_seg',exist_ok=False)
print('Training complete. Validate the selected checkpoint on the untouched test split separately.')
