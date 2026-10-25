import contextlib,io,json,tempfile,unittest
from pathlib import Path
import cv2
import numpy as np
from src.prepare import NAMES,mask_boxes,prepare

class TestPrepare(unittest.TestCase):
    def test_pixel_edges_and_disconnected_regions(self):
        m=np.zeros((10,12),np.uint8);m[1:4,2:6]=255;m[8,10]=255
        self.assertEqual(mask_boxes(m),[[2,1,6,4],[10,8,11,9]])
    def test_dedup_split_labels_and_freeze(self):
        temp_root=Path(__file__).resolve().parents[1]/'.cache'/'tests'
        temp_root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as td:
            root=Path(td)/'raw';out=Path(td)/'prepared'
            for c,name in enumerate(['good']+NAMES):
                for i in range(6):
                    folder=root/('train' if name=='good' else 'test')/name
                    folder.mkdir(parents=True,exist_ok=True)
                    image=np.full((20,30,3),c*30+i,np.uint8)
                    cv2.imwrite(str(folder/f'{i:03}.png'),image)
                    if name!='good':
                        mf=root/'ground_truth'/name;mf.mkdir(parents=True,exist_ok=True)
                        mask=np.zeros((20,30),np.uint8);mask[3:8,4:10]=255
                        cv2.imwrite(str(mf/f'{i:03}_mask.png'),mask)
            # Same good image in original train/test must not leak across custom splits.
            good_test=root/'test/good';good_test.mkdir(parents=True)
            (good_test/'000.png').write_bytes((root/'train/good/000.png').read_bytes())
            with contextlib.redirect_stdout(io.StringIO()):prepare(root,out)
            m=json.loads((out/'manifest.json').read_text())
            self.assertEqual(len(m['items']),30)
            self.assertEqual(len({i['sha256'] for i in m['items']}),30)
            for split in ['train','val','test']:
                self.assertEqual({i['category'] for i in m['items'] if i['split']==split},set(['good']+NAMES))
            for item in m['items']:
                text=(out/'labels'/item['split']/(item['id']+'.txt')).read_text()
                if item['category']=='good':self.assertEqual(text,'')
                else:
                    cls,x,y,w,h=map(float,text.split())
                    self.assertEqual(cls,NAMES.index(item['category']))
                    self.assertAlmostEqual(w,6/30);self.assertAlmostEqual(h,5/20)
            with self.assertRaises(FileExistsError):prepare(root,out)

