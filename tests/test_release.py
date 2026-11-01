import unittest
import tkinter as tk
import numpy as np
from src.demo_photo import PhotoApp


class EmptyPagePredictor:
    def recognize(self,image):
        return {'text':'','detections':[],'message':'Không có chữ số đủ điểm nhận dạng.'},image[:,:,:3].copy(),[]


class ReleaseGUITests(unittest.TestCase):
    def test_letter_only_result_has_no_stale_prediction(self):
        root=tk.Tk();root.withdraw()
        try:
            app=PhotoApp(root,EmptyPagePredictor())
            app.image=np.full((180,280,4),255,np.uint8);app.show_image()
            app.result.config(text='123')
            app.recognize()
            self.assertEqual(app.result.cget('text'),'Không có số')
            self.assertEqual(app.last_result['detections'],[])
            self.assertIn('Không có chữ số',app.status.get())
            app.rotate(False)
            self.assertEqual(app.image.shape,(280,180,4))
        finally:root.destroy()
