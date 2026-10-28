"""Pure geometry, association and review validation; no model imports."""
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

def iou(a,b):
    x=max(a[0],b[0]); y=max(a[1],b[1]); X=min(a[2],b[2]); Y=min(a[3],b[3])
    inter=max(0,X-x)*max(0,Y-y)
    return inter/max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)

def suppress(detections, threshold=.55):
    out=[]
    for d in sorted(detections,key=lambda x:x['score'],reverse=True):
        if all(iou(d['box'],o['box'])<threshold for o in out): out.append(d)
    return out

def transform_box(box,H):
    x,y,X,Y=box
    p=np.array([[x,y,1],[X,y,1],[X,Y,1],[x,Y,1]],float)@H.T
    if np.any(np.abs(p[:,2])<1e-8): return box
    p=p[:,:2]/p[:,2:]
    return [float(p[:,0].min()),float(p[:,1].min()),float(p[:,0].max()),float(p[:,1].max())]

class Tracker:
    def __init__(self,max_gap=12,min_iou=.18):
        self.tracks={}; self.next_id=1; self.max_gap=max_gap; self.min_iou=min_iou
    def update(self,detections,frame,H=None):
        if H is not None:
            for t in self.tracks.values(): t['pred']=transform_box(t['pred'],H)
        active=[k for k,t in self.tracks.items() if frame-t['last']<=self.max_gap]
        cost=np.ones((len(active),len(detections)))
        for i,k in enumerate(active):
            for j,d in enumerate(detections): cost[i,j]=1-iou(self.tracks[k]['pred'],d['box'])
        pairs=[]
        if cost.size:
            rows,cols=linear_sum_assignment(cost)
            pairs=[(active[i],j) for i,j in zip(rows,cols) if 1-cost[i,j]>=self.min_iou]
        used=set()
        for k,j in pairs:
            self.tracks[k].update(pred=detections[j]['box'],last=frame)
            detections[j]['track_id']=k; used.add(j)
        for j,d in enumerate(detections):
            if j in used: continue
            k=self.next_id; self.next_id+=1
            self.tracks[k]={'pred':d['box'],'last':frame}
            d['track_id']=k
        return detections

def resolve_review(ids,review):
    """Reject invalid IDs, cycles and negative/non-finite manual depths."""
    ids=set(map(str,ids)); entries=review.get('tracks',{})
    if set(entries)-ids: raise ValueError('Review contains unknown track IDs')
    roots={}
    for id in ids:
        seen=set(); cur=id
        while entries.get(cur,{}).get('merge_into'):
            if cur in seen: raise ValueError('Merge cycle')
            seen.add(cur); cur=str(entries[cur]['merge_into'])
            if cur not in ids: raise ValueError('Unknown merge target')
        roots[id]=cur
    for id,e in entries.items():
        depth=e.get('depth_cm')
        if depth is not None and (not isinstance(depth,(float,int)) or not math.isfinite(depth) or depth<0):
            raise ValueError('Depth must be finite and nonnegative')
        if e.get('status','candidate') not in ['candidate','confirmed','rejected','uncertain']:
            raise ValueError('Invalid status')
        if roots[id]!=id and depth is not None: raise ValueError('Enter depth on the merge target only')
    return roots

def apply_overrides(data,overrides):
    """Replace explicitly edited frames without modifying raw inference data."""
    import copy
    result=copy.deepcopy(data);frames={r['frame']:r for r in result['frames']}
    for key,ds in overrides.items():
        frame=int(key)
        if frame not in frames:raise ValueError('Correction frame was not analyzed')
        if not isinstance(ds,list):raise ValueError('Detections must be a list')
        ids=set()
        for d in ds:
            k=d.get('track_id')
            if not isinstance(k,int) or k<1 or k in ids:raise ValueError('Use distinct positive integer track IDs per frame')
            ids.add(k);box=d.get('box',[]);poly=d.get('polygon',[])
            if len(box)!=4 or not all(isinstance(v,(float,int)) and math.isfinite(v) for v in box):raise ValueError('Invalid box')
            x,y,X,Y=box
            if not (0<=x<X<=data['width'] and 0<=y<Y<=data['height']):raise ValueError('Box outside image')
            if poly and (len(poly)<3 or any(len(p)!=2 or not all(isinstance(v,(float,int)) and math.isfinite(v) for v in p) or not (0<=p[0]<=data['width'] and 0<=p[1]<=data['height']) for p in poly)):raise ValueError('Invalid polygon')
            if not isinstance(d.get('score'),(int,float)) or not 0<=d['score']<=1:raise ValueError('Score must be in [0,1]')
            d['mask_source']='manual frame correction'
        frames[frame]['detections']=copy.deepcopy(ds)
    return result
