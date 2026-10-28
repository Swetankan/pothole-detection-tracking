import unittest,tempfile,json,csv
from pathlib import Path
import cv2,numpy as np
from pipeline import export
class ExportTests(unittest.TestCase):
 def test_manual_depth_and_duplicate_merge(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);video=p/'input.avi';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),10,(320,180))
   for _ in range(6):writer.write(np.zeros((180,320,3),np.uint8))
   writer.release()
   data={'video':str(video),'width':320,'height':180,'fps':10,'processed_frames':6,'frames':[{'frame':i,'time':i/10,'detections':[{'track_id':1 if i<3 else 2,'box':[20,70,100,120],'polygon':[[20,70],[100,70],[100,120],[20,120]],'score':.8}]} for i in range(6)]}
   review={'tracks':{'1':{'status':'confirmed','depth_cm':4.7},'2':{'merge_into':'1'}}}
   export(data,review,p)
   records=json.loads((p/'potholes.json').read_text());self.assertEqual(len(records),1);self.assertEqual(records[0]['depth_cm'],4.7);self.assertEqual(records[0]['observations'],6)
   self.assertEqual(records[0]['depth_source'],'manual')
   cap=cv2.VideoCapture(str(p/'annotated.mp4'));self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),6);cap.release()
 def test_rejected_root_disappears(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);video=p/'input.avi';w=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),10,(320,180));w.write(np.zeros((180,320,3),np.uint8));w.release()
   data={'video':str(video),'width':320,'height':180,'fps':10,'processed_frames':1,'frames':[{'frame':0,'time':0,'detections':[{'track_id':1,'box':[20,70,100,120],'polygon':[],'score':.9}]}]}
   export(data,{'tracks':{'1':{'status':'rejected'}}},p)
   self.assertEqual(json.loads((p/'potholes.json').read_text()),[])
if __name__=='__main__':unittest.main()
