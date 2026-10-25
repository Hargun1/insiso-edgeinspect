import unittest
from src.metrics import evaluate,match,iou

def box(c=0,s=None):
    x={'box':[0,0,10,10],'class_id':c}
    if s is not None:x['score']=s
    return x

class TestMetrics(unittest.TestCase):
    def test_duplicate(self):
        p,m=match([box(s=.9),box(s=.8)],[box()])
        self.assertEqual([v for _,v in p],[True,False]);self.assertFalse(m)
    def test_wrong_class(self):
        m=evaluate([{'truth':[box()],'predictions':[box(1,.9)]}],['a','b'])
        self.assertEqual((m['tp'],m['fp'],m['fn']),(0,1,1))
    def test_good_image_fp(self):
        m=evaluate([{'truth':[box()],'predictions':[box(s=.8)]},
                    {'truth':[],'predictions':[box(s=.9)]}],['a'])
        self.assertEqual(m['precision'],.5);self.assertEqual(m['map50'],.5)
    def test_ap_independent_of_operating_threshold(self):
        m=evaluate([{'truth':[box()],'predictions':[box(s=.2)]}],['a'])
        self.assertEqual(m['recall'],0);self.assertEqual(m['map50'],1)
    def test_empty(self):
        self.assertIsNone(evaluate([],['a'])['map50'])
        self.assertEqual(iou([0,0,0,0],[0,0,0,0]),0)

if __name__=='__main__':unittest.main()

