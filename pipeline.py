"""Offline pothole segmentation and reviewable tracking. Run --help."""
import argparse,csv,hashlib,json,subprocess,time
from pathlib import Path
import cv2
import numpy as np
from core import Tracker,suppress,resolve_review,apply_overrides

def dump(path,obj): Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False))
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def motion(prev,gray):
    if prev is None:return np.eye(3),0
    pts=cv2.goodFeaturesToTrack(prev,500,.01,10)
    if pts is None or len(pts)<12:return np.eye(3),0
    nxt,ok,_=cv2.calcOpticalFlowPyrLK(prev,gray,pts,None)
    if nxt is None:return np.eye(3),0
    back,ok2,_=cv2.calcOpticalFlowPyrLK(gray,prev,nxt,None)
    if back is None:return np.eye(3),0
    good=(ok.ravel()==1)&(ok2.ravel()==1)&(np.linalg.norm(pts-back,axis=2).ravel()<1.5)
    if good.sum()<12:return np.eye(3),0
    H,inliers=cv2.findHomography(pts[good],nxt[good],cv2.RANSAC,3)
    ratio=float(inliers.mean()) if inliers is not None else 0
    if H is None or ratio<.45 or not np.isfinite(H).all():return np.eye(3),ratio
    return H,ratio

def detect(model,frame,args):
    h,w=frame.shape[:2]; crops=[(0,0,w,h)]
    if args.tiles:
        tw,th=int(w*.62),int(h*.62)
        crops += [(x,y,x+tw,y+th) for y in [0,h-th] for x in [0,w-tw]]
    det=[]
    for x,y,X,Y in crops:
        r=model.predict(frame[y:Y,x:X],conf=args.conf,imgsz=args.imgsz,device=args.device,verbose=False,retina_masks=True)[0]
        for i,b in enumerate(r.boxes):
            name=str(r.names[int(b.cls.item())]).lower()
            if 'pothole' not in name and name!='d40':continue
            box=b.xyxy[0].cpu().numpy()+np.array([x,y,x,y])
            if box[3]<args.roi_top*h:continue
            poly=[]
            if r.masks is not None:
                poly=(r.masks.xy[i]+np.array([x,y])).round(1).tolist()
            det.append({'box':box.round(1).tolist(),'polygon':poly,'score':round(float(b.conf.item()),4)})
    return suppress(det)

def analyze(args):
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(args.threads)
    out=Path(args.output)
    if (out/'analysis.json').exists():raise ValueError('This run already exists. Choose a new output directory to preserve previous reviews.')
    out.mkdir(parents=True,exist_ok=True)
    weights=Path(args.weights)
    if not weights.exists():
        if args.weights != 'models/pothole26.pt':raise FileNotFoundError('Explicit weights file does not exist: '+str(weights))
        from huggingface_hub import hf_hub_download
        weights=Path(hf_hub_download('DanielsStulpe/pothole-detection','yolo26_best.pt'))
    model=YOLO(str(weights))
    if model.task not in ('segment','detect'):raise ValueError('Supply a detection or segmentation checkpoint')
    sam=None
    if args.sam:
        from ultralytics import SAM
        sam=SAM(args.sam)
    mask_cache={}
    cap=cv2.VideoCapture(args.video)
    if not cap.isOpened():raise ValueError('Cannot open video')
    fps=cap.get(cv2.CAP_PROP_FPS);w=int(cap.get(3));h=int(cap.get(4))
    if fps<=0:raise ValueError('Invalid frame rate')
    tracker=Tracker(max_gap=args.max_gap); prev=None;rows=[];frame=0;start=time.time()
    while True:
        ok,img=cap.read()
        if not ok:break
        if args.limit and frame>=args.limit:break
        if frame%args.stride:frame+=1;continue
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        H,quality=motion(prev,gray);prev=gray
        ds=detect(model,img,args)
        ds=tracker.update(ds,frame,H)
        refresh=[]
        for d in ds:
            k=d['track_id']
            if d['polygon']:
                d['mask_source']='model'
            elif sam and (k not in mask_cache or frame-mask_cache[k][2]>=args.mask_every):
                refresh.append(d)
            elif sam and k in mask_cache:
                old_poly,old_box,last=mask_cache[k]
                x,y,X,Y=old_box; nx,ny,nX,nY=d['box']
                p=np.array(old_poly,float)
                if len(p):
                    p=(p-[x,y])*[(nX-nx)/max(1,X-x),(nY-ny)/max(1,Y-y)]+[nx,ny]
                    d['polygon']=p.round(1).tolist()
                    d['mask_source']='box-propagated; approximate'
        if refresh:
            sr=sam(img,bboxes=[d['box'] for d in refresh],device=args.device,verbose=False)[0]
            if sr.masks is not None:
                for d,p in zip(refresh,sr.masks.xy):
                    d['polygon']=p.round(1).tolist();d['mask_source']='MobileSAM box prompt'
                    mask_cache[d['track_id']]=(d['polygon'],d['box'],frame)
        rows.append({'frame':frame,'time':round(frame/fps,4),'blur':round(float(cv2.Laplacian(gray,cv2.CV_64F).var()),2),'motion_inliers':round(quality,3),'homography':H.round(7).tolist(),'detections':ds})
        if len(rows)%10==0:print(f'{frame} frames, {len(tracker.tracks)} provisional tracks, {time.time()-start:.1f}s',flush=True)
        frame+=1
    cap.release()
    data={'schema':1,'video':str(Path(args.video).resolve()),'video_sha256':sha(args.video),'fps':fps,'width':w,'height':h,'processed_frames':frame,'stride':args.stride,'model':str(weights),'model_sha256':sha(weights),'config':vars(args),'frames':rows,'runtime_seconds':round(time.time()-start,2),'depth_mode':'manual-only; no automatic metric depth inferred'}
    dump(out/'analysis.json',data)
    dump(out/'review.json',{'tracks':{str(k):{'status':'candidate','merge_into':None,'depth_cm':None,'notes':''} for k in tracker.tracks}})
    export(data,json.loads((out/'review.json').read_text()),out,args.min_hits)

def export(data,review,out,min_hits=3):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    raw_ids={str(d['track_id']) for r in data['frames'] for d in r['detections']}
    if (out/'frame_overrides.json').exists():
        data=apply_overrides(data,json.loads((out/'frame_overrides.json').read_text()))
    ids=raw_ids|{str(d['track_id']) for r in data['frames'] for d in r['detections']}
    roots=resolve_review(ids,review);entries=review.get('tracks',{})
    groups={}
    for r in data['frames']:
        for d in r['detections']:
            tid=str(d['track_id']);root=roots[tid]
            if entries.get(tid,{}).get('status')=='rejected':continue
            g=groups.setdefault(root,[]);g.append((r,d))
    records=[]; accepted={}
    for root,obs in groups.items():
        e=entries.get(root,{}); status=e.get('status','candidate')
        hits=len({r['frame'] for r,d in obs})
        if status=='rejected':continue
        eligible=hits>=min_hits or status=='confirmed'
        if eligible:accepted[root]=e
        best=max(obs,key=lambda rd:rd[1]['score'])
        records.append({'pothole_id':f'P{int(root):03}','track_id':root,'status':status,'eligible':eligible,'observations':hits,'first_s':min(r['time'] for r,d in obs),'last_s':max(r['time'] for r,d in obs),'peak_score':best[1]['score'],'depth_cm':e.get('depth_cm'),'depth_source':'manual' if e.get('depth_cm') is not None else 'unavailable','best_frame':best[0]['frame'],'notes':e.get('notes','')})
    with open(out/'potholes.csv','w',newline='') as f:
        fields=list(records[0]) if records else ['pothole_id','depth_cm','depth_source']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(records)
    dump(out/'potholes.json',records)
    frame_map={r['frame']:r for r in data['frames']}
    cap=cv2.VideoCapture(data['video'])
    if not cap.isOpened():raise ValueError('Original video is needed for export')
    w,h=data['width'],data['height'];fps=data['fps']
    temp=out/'render.avi';writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'MJPG'),fps,(w,h))
    if not writer.isOpened():raise RuntimeError('Cannot create video writer')
    crops=out/'evidence';crops.mkdir(exist_ok=True);best_frames={r['best_frame'] for r in records};seen=set();i=0
    while i<data['processed_frames']:
        ok,img=cap.read()
        if not ok:break
        original=img.copy(); row=frame_map.get(i)
        if row:
            active_roots=set()
            for d in row['detections']:
                tid=str(d['track_id']);root=roots[tid]
                if root not in accepted or entries.get(tid,{}).get('status')=='rejected':continue
                e=accepted[root];active_roots.add(root);seen.add(root)
                color=(70,210,80) if e.get('status')=='confirmed' else (30,180,255)
                p=np.array(d['polygon'],np.int32)
                if len(p)>2:
                    layer=img.copy();cv2.fillPoly(layer,[p],color);img=cv2.addWeighted(img,.72,layer,.28,0);cv2.polylines(img,[p],True,color,2)
                x,y,X,Y=map(int,d['box']);cv2.rectangle(img,(x,y),(X,Y),color,2)
                depth=e.get('depth_cm'); label=f'P{int(root):03} {d["score"]:.2f} | '+(f'{depth:g} cm MANUAL' if depth is not None else '-- cm')
                cv2.putText(img,label,(max(0,x),max(48,y-8)),cv2.FONT_HERSHEY_SIMPLEX,.46,color,1,cv2.LINE_AA)
            note=f'Active: {len(active_roots)} | Provisional unique: {len(seen)} | Review required'
        else:note='Not analyzed (stride) | no interpolated detections'
        cv2.rectangle(img,(0,0),(w,33),(24,27,34),-1);cv2.putText(img,note,(12,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(img,'-- cm = not measured | Orange: candidate | Green: reviewed | Masks may be propagated',(12,h-14),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv2.LINE_AA)
        if i in best_frames:
            cv2.imwrite(str(crops/f'frame_{i:06}.jpg'),img)
        writer.write(img);i+=1
    cap.release();writer.release()
    subprocess.run(['ffmpeg','-y','-v','error','-i',str(temp),'-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'annotated.mp4')],check=True)
    temp.unlink()
    dump(out/'summary.json',{'provisional_unique':len(accepted),'confirmed':sum(e.get('status')=='confirmed' for e in accepted.values()),'depth_values_supplied':sum(e.get('depth_cm') is not None for e in accepted.values()),'accuracy':'Not evaluated: no independent ground truth','min_observations':min_hits})
    print(json.dumps(json.loads((out/'summary.json').read_text())),flush=True)

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('analyze');a.add_argument('video');a.add_argument('--output',default='outputs/run');a.add_argument('--weights',default='models/pothole26.pt');a.add_argument('--sam',default=None,help='Optional SAM checkpoint, e.g. mobile_sam.pt');a.add_argument('--mask-every',type=int,default=5);a.add_argument('--device',default='cpu');a.add_argument('--imgsz',type=int,default=960);a.add_argument('--conf',type=float,default=.25);a.add_argument('--tiles',action='store_true');a.add_argument('--stride',type=int,default=1);a.add_argument('--limit',type=int,default=0);a.add_argument('--threads',type=int,default=4);a.add_argument('--roi-top',type=float,default=.15);a.add_argument('--max-gap',type=int,default=12);a.add_argument('--min-hits',type=int,default=3)
    e=sub.add_parser('export');e.add_argument('run');e.add_argument('--min-hits',type=int,default=None)
    args=p.parse_args()
    if args.command=='analyze':
        if args.stride<1 or args.min_hits<1 or args.threads<1 or args.mask_every<1: p.error('stride, min-hits and threads must be positive')
        analyze(args)
    else:
        run=Path(args.run)
        previous=json.loads((run/'summary.json').read_text()) if (run/'summary.json').exists() else {}
        minimum=args.min_hits if args.min_hits is not None else previous.get('min_observations',3)
        if minimum<1:p.error('min-hits must be positive')
        export(json.loads((run/'analysis.json').read_text()),json.loads((run/'review.json').read_text()),run,minimum)
if __name__=='__main__':main()
