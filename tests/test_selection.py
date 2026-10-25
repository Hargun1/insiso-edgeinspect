import unittest
from src.select import choose

def result(score,split='val'):
    return {'split':split,'status':'measured','metrics':{'map50':score}}

class TestSelection(unittest.TestCase):
    def test_tie_favors_baseline(self):
        a,b=result(.5),result(.5);self.assertIs(choose(a,b),a)
    def test_better_augmentation(self):
        a,b=result(.3),result(.5);self.assertIs(choose(a,b),b)
    def test_test_cannot_select(self):
        with self.assertRaises(ValueError):choose(result(.5,'test'),result(.4))

