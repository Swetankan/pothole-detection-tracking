import unittest
import numpy as np
from core import Tracker,iou,suppress,resolve_review
class Tests(unittest.TestCase):
    def test_motion_identity(self):
        t=Tracker();a=t.update([{'box':[0,0,10,10]}],0)[0]['track_id']
        H=np.array([[1,0,30],[0,1,0],[0,0,1.]])
        b=t.update([{'box':[30,0,40,10]}],1,H)[0]['track_id'];self.assertEqual(a,b)
    def test_independent_objects(self):
        t=Tracker();d=t.update([{'box':[0,0,10,10]},{'box':[50,50,60,60]}],0)
        self.assertNotEqual(d[0]['track_id'],d[1]['track_id'])
    def test_gap(self):
        t=Tracker(max_gap=2);a=t.update([{'box':[0,0,10,10]}],0)[0]['track_id'];b=t.update([{'box':[0,0,10,10]}],4)[0]['track_id'];self.assertNotEqual(a,b)
    def test_merge_cycle(self):
        with self.assertRaises(ValueError):resolve_review(['1','2'],{'tracks':{'1':{'merge_into':'2'},'2':{'merge_into':'1'}}})
    def test_bad_depth(self):
        for v in [-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):resolve_review(['1'],{'tracks':{'1':{'depth_cm':v}}})
    def test_merge_chain(self):
        self.assertEqual(resolve_review(['1','2','3'],{'tracks':{'1':{'merge_into':'2'},'2':{'merge_into':'3'}}})['1'],'3')
    def test_unknown(self):
        with self.assertRaises(ValueError):resolve_review(['1'],{'tracks':{'2':{}}})
    def test_nms(self):
        self.assertEqual(len(suppress([{'box':[0,0,10,10],'score':.9},{'box':[1,1,10,10],'score':.8}])),1)
if __name__=='__main__':unittest.main()
