import unittest
import numpy as np
from src.runtime import Detector

class TestRuntime(unittest.TestCase):
    def setUp(self):
        self.det=Detector.__new__(Detector);self.det.size=320
    def test_letterbox_channels_and_range(self):
        image=np.zeros((100,200,3),np.uint8);image[:,:,2]=255
        t,info=self.det.preprocess(image)
        self.assertEqual(t.shape,(1,3,320,320));self.assertEqual(t.dtype,np.float32)
        self.assertEqual(info,(1.6,0,80,200,100))
        self.assertEqual(t[0,0,160,160],1);self.assertEqual(t[0,2,160,160],0)
    def test_class_aware_nms_and_inverse_coordinates(self):
        # Same box/class suppresses; a different class remains a separate prediction.
        rows=np.array([[160,160,160,80,.9,.01,.01,.01],
                       [160,160,160,80,.8,.01,.01,.01],
                       [160,160,160,80,.01,.85,.01,.01]],dtype=np.float32)
        result=self.det.postprocess([rows.T[None]],(1.6,0,80,200,100),confidence=.25)
        self.assertEqual(len(result),2)
        self.assertEqual(result[0]['box'],[50.,25.,150.,75.])
        self.assertEqual({d['label'] for d in result},{'bent','color'})
    def test_wrong_output_rejected(self):
        with self.assertRaises(ValueError):
            self.det.postprocess([np.zeros((1,84,100))],(1,0,0,320,320))


    def test_nms_precedes_clipping(self):
        rows=np.array([[0,50,200,100,.9,.01,.01,.01],
                       [50,50,100,100,.8,.01,.01,.01]],dtype=np.float32)
        result=self.det.postprocess([rows.T[None]],(1,0,0,320,320),confidence=.25)
        self.assertEqual(len(result),2)
        self.assertEqual(result[0]["box"],[0.,0.,100.,100.])
