"""Reverse-pass merge suggestions, never silently applied."""
import json,argparse
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from core import Tracker
p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();run=Path(a.run);data=json.loads((run/'analysis.json').read_text())
t=Tracker(max_gap=12);votes=defaultdict(Counter);seen=defaultdict(set);prev=None
for step,r in enumerate(reversed(data['frames'])):
 H=np.eye(3)
 if prev is not None:
  try:H=np.linalg.inv(np.array(prev.get('homography',np.eye(3))))
  except np.linalg.LinAlgError:pass
 ds=[{'box':d['box'],'forward_id':str(d['track_id'])} for d in r['detections']]
 for d in t.update(ds,step,H):
  votes[d['track_id']][d['forward_id']]+=1;seen[d['forward_id']].add(r['frame'])
 prev=r
suggestions=[]
for bid,counts in votes.items():
 ids=[k for k,v in counts.items() if v>=3]
 for i,x in enumerate(ids):
  for y in ids[i+1:]:
   if not (seen[x]&seen[y]):suggestions.append({'track_a':x,'track_b':y,'reverse_track':bid,'support_a':counts[x],'support_b':counts[y],'action':'manual inspection required'})
(run/'merge_suggestions.json').write_text(json.dumps(suggestions,indent=2));print(f'{len(suggestions)} reverse-pass merge suggestions; none applied automatically')
