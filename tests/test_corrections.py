import unittest
from core import apply_overrides
class Corrections(unittest.TestCase):
 def setUp(self):self.data={'width':100,'height':100,'frames':[{'frame':0,'detections':[]}]}
 def test_add_does_not_mutate_original(self):
  edit={'0':[{'track_id':1,'box':[1,1,20,20],'polygon':[[1,1],[20,1],[20,20]],'score':1.0}]}
  r=apply_overrides(self.data,edit)
  self.assertEqual(len(r['frames'][0]['detections']),1);self.assertEqual(self.data['frames'][0]['detections'],[])
 def test_reject_out_of_bounds(self):
  with self.assertRaises(ValueError):apply_overrides(self.data,{'0':[{'track_id':1,'box':[1,1,200,20],'polygon':[],'score':1.0}]})
 def test_reject_unknown_frame(self):
  with self.assertRaises(ValueError):apply_overrides(self.data,{'5':[]})
