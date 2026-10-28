import unittest,tempfile,json
from pathlib import Path
import cv2,numpy as np
from streamlit.testing.v1 import AppTest
class ReviewUI(unittest.TestCase):
 def test_populated_review(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);video=p/'input.avi';w=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),10,(320,180));w.write(np.zeros((180,320,3),np.uint8));w.release()
   data={'video':str(video),'width':320,'height':180,'fps':10,'processed_frames':1,'frames':[{'frame':0,'time':0,'detections':[{'track_id':1,'box':[20,70,100,120],'polygon':[],'score':.9}]}]}
   (p/'analysis.json').write_text(json.dumps(data));(p/'review.json').write_text(json.dumps({'tracks':{'1':{'status':'candidate','merge_into':None,'depth_cm':None,'notes':''}}}));(p/'potholes.json').write_text(json.dumps([{'track_id':'1','best_frame':0}]))
   app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'review_app.py')).run(timeout=20)
   app.sidebar.text_input[0].set_value(str(p)).run(timeout=20)
   self.assertEqual(len(app.exception),0)
   app.checkbox[0].check();app.number_input[0].set_value(4.7)
   app.button[0].click().run(timeout=20)
   self.assertEqual(json.loads((p/'review.json').read_text())['tracks']['1']['depth_cm'],4.7)
if __name__=='__main__':unittest.main()
