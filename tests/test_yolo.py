"""Kiểm tra hợp đồng dữ liệu và tích hợp YOLO khi có checkpoint."""
import unittest
import numpy as np
from src.config import ROOT
from src.preprocess import read_image, EmptyImageError


class YOLODataTests(unittest.TestCase):
    @unittest.skipUnless((ROOT/'data/yolo_digits/dataset.yaml').exists(), 'Dataset not generated')
    def test_labels_valid(self):
        for split in ('train','val','test'):
            files = list((ROOT/'data/yolo_digits/labels'/split).glob('*.txt'))
            self.assertTrue(files)
            for path in files:
                self.assertTrue((ROOT/'data/yolo_digits/images'/split/(path.stem+'.jpg')).is_file())
                for line in path.read_text().splitlines():
                    cls,x,y,w,h = map(float,line.split())
                    self.assertIn(int(cls),range(10))
                    self.assertTrue(0 < w <= 1 and 0 < h <= 1)
                    self.assertTrue(0 <= x-w/2 <= x+w/2 <= 1)
                    self.assertTrue(0 <= y-h/2 <= y+h/2 <= 1)


@unittest.skipUnless((ROOT/'models/yolo_latest.json').exists(), 'YOLO training not complete')
class YOLOIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.yolo_predict import create_predictor
        cls.predictor = create_predictor('yolo')

    def test_inference_contract(self):
        from src.detect import recognize_digits
        result, annotated, crops = recognize_digits(read_image(ROOT/'data/samples/mnist_paper_row.png'),self.predictor)
        self.assertEqual(result['backend'],'ultralytics-yolo')
        self.assertEqual(len(crops),len(result['detections']))
        self.assertTrue(np.any(np.all(annotated == [0,255,0],axis=2)))
        for d in result['detections']:
            self.assertIn(d['digit'],range(10))
            self.assertGreaterEqual(d['score'],.25)
            self.assertIsNone(d['probabilities'])

    def test_blank(self):
        with self.assertRaises(EmptyImageError):
            self.predictor.recognize(np.full((416,416,3),255,np.uint8))
