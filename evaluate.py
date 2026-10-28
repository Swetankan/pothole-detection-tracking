"""Mask precision/recall on fully annotated frames; no proxy accuracy scores.
GT JSON: {"frames":[{"frame":0,"polygons":[[[x,y],...],...]}]}.
"""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import linear_sum_assignment
p=argparse.ArgumentParser();p.add_argument('analysis');p.add_argument('ground_truth');p.add_argument('--iou',type=float,default=.5);a=p.parse_args()
d=json.loads(Path(a.analysis).read_text());gt=json.loads(Path(a.ground_truth).read_text());pred={r['frame']:r for r in d['frames']};tp=fp=fn=0;matched=[]
def mask(poly):
 m=np.zeros((d['height'],d['width']),np.uint8)
 if len(poly)>2:cv2.fillPoly(m,[np.array(poly,np.int32)],1)
 return m.astype(bool)
for r in gt['frames']:
 if r['frame'] not in pred:raise ValueError('Ground truth frame was not processed')
 g=[mask(x) for x in r['polygons']];ps=[mask(x['polygon']) for x in pred[r['frame']]['detections']]
 scores=np.zeros((len(g),len(ps)))
 for i,x in enumerate(g):
  for j,y in enumerate(ps):scores[i,j]=(x&y).sum()/max(1,(x|y).sum())
 n=0
 if scores.size:
  rows,cols=linear_sum_assignment(-(scores>=a.iou).astype(float)*1000-scores)
  for i,j in zip(rows,cols):
   if scores[i,j]>=a.iou:n+=1;matched.append(float(scores[i,j]))
 tp+=n;fp+=len(ps)-n;fn+=len(g)-n
print(json.dumps({'annotated_frames':len(gt['frames']),'TP':tp,'FP':fp,'FN':fn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'mean_matched_mask_iou':float(np.mean(matched)) if matched else None,'threshold':a.iou},indent=2))
