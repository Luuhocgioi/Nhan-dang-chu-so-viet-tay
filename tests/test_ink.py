import unittest
from unittest.mock import patch
import cv2
import numpy as np
from src.config import ROOT
from src.preprocess import EmptyImageError


@unittest.skipUnless((ROOT/'models/latest.json').exists(),'CNN checkpoint required')
class InkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.ink_predict import InkPredictor
        cls.predictor = InkPredictor()

    def test_grid_and_red_text_are_removed(self):
        image = np.full((280,480,3),190,np.uint8)
        for x in range(0,480,20):
            cv2.line(image,(x,0),(x,279),(110,110,110),1)
        for y in range(0,280,20):
            cv2.line(image,(0,y),(479,y),(110,110,110),1)
        cv2.line(image,(140,70),(140,220),(90,15,40),12)
        cv2.putText(image,'ABC',(260,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,220),2)
        original = image.copy()
        result,_,_ = self.predictor.recognize(image)
        self.assertEqual(result['text'],'1')
        self.assertEqual(len(result['detections']),1)
        np.testing.assert_array_equal(image,original)

    def test_red_letters_without_digits(self):
        image=np.full((280,480,3),255,np.uint8)
        cv2.putText(image,'ABC',(70,150),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,220),4)
        with self.assertRaises(EmptyImageError):
            self.predictor.recognize(image)
