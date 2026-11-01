import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import tkinter as tk
import cv2
import numpy as np
from PIL import Image
from src.config import ROOT
from src.preprocess import preprocess, read_image, EmptyImageError
from src.predict import Predictor
from src.demo_photo import PhotoApp


class PaperTests(unittest.TestCase):
    def test_shaded_paper_retains_ink(self):
        paper = np.tile(np.linspace(90, 245, 300).astype(np.uint8), (240, 1))
        cv2.line(paper, (150, 50), (150, 190), 25, 10)
        result = preprocess(paper, mode='paper')[0, ..., 0]
        # Nét mảnh được nội suy khi resize/căn tâm nên không bắt buộc đạt 1.
        self.assertGreater(result.max(), .3)
        ys, xs = np.where(result > .2)
        self.assertGreater(ys.max()-ys.min(), 15)
        self.assertLess(xs.max()-xs.min(), 6)

    def test_blank_shaded_paper(self):
        paper = np.tile(np.linspace(100, 245, 300).astype(np.uint8), (240, 1))
        with self.assertRaises(EmptyImageError):
            preprocess(paper, mode='paper')

    def test_phone_exif_rotation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'phone.jpg'
            image = Image.new('RGB', (60, 100), 'white')
            exif = image.getexif()
            exif[274] = 6
            image.save(path, exif=exif)
            self.assertEqual(read_image(path).shape, (60, 100, 4))


@unittest.skipUnless((ROOT / 'models/latest.json').exists(), 'Checkpoint required')
class PhotoAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = Predictor()

    def test_load_crop_predict_rotate_and_blank(self):
        root = tk.Tk()
        root.withdraw()
        try:
            app = PhotoApp(root, self.predictor)
            with patch('src.demo_photo.messagebox.showinfo') as info:
                app.recognize()
                info.assert_called_once()
            app.load_path(ROOT / 'data/samples/synthetic_1.png')
            ox, oy = app.offset_x, app.offset_y
            app.start_selection(SimpleNamespace(x=ox+90, y=oy+20))
            app.end_selection(SimpleNamespace(x=ox+190, y=oy+260))
            self.assertEqual(app.roi, (90, 20, 191, 261))
            app.recognize()
            root.update_idletasks()
            self.assertEqual(app.result.cget('text'), '1')
            self.assertEqual(len(app.scores.cget('text').splitlines()), 10)
            app.rotate()
            self.assertIsNone(app.roi)
            self.assertEqual(app.result.cget('text'), '—')
            app.image = np.full((100, 100, 4), 255, np.uint8)
            app.show_image()
            with patch('src.demo_photo.messagebox.showinfo') as info:
                app.recognize()
                info.assert_called_once()
            self.assertEqual(app.result.cget('text'), '—')
        finally:
            root.destroy()
