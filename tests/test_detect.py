import json
from pathlib import Path
import tempfile
import unittest
import tkinter as tk
from unittest.mock import patch
import cv2
import numpy as np
from src.config import ROOT
from src.detect import segment_digits
from src.preprocess import EmptyImageError
from src.predict import Predictor
from src.demo_photo import PhotoApp


class SegmentationTests(unittest.TestCase):
    def test_reading_order_two_rows_and_noise(self):
        paper = np.full((300, 500), 230, np.uint8)
        for x, y in [(320, 30), (40, 30), (220, 180), (60, 180)]:
            cv2.rectangle(paper, (x, y), (x+25, y+70), 20, 4)
        paper[120, 120] = 0
        _, items = segment_digits(paper)
        self.assertEqual(len(items), 4)
        self.assertEqual([item['row'] for item in items], [0, 0, 1, 1])
        self.assertLess(items[0]['bbox'][0], items[1]['bbox'][0])
        self.assertLess(items[2]['bbox'][0], items[3]['bbox'][0])

    def test_shaded_blank(self):
        paper = np.tile(np.linspace(100, 240, 500).astype(np.uint8), (300, 1))
        with self.assertRaises(EmptyImageError):
            segment_digits(paper)

    def test_large_image_coordinate_bounds(self):
        paper = np.full((2000, 2400), 255, np.uint8)
        cv2.line(paper, (1000, 600), (1000, 1300), 0, 30)
        base, items = segment_digits(paper)
        self.assertEqual(max(base.shape[:2]), 1600)
        self.assertEqual(len(items), 1)
        x, y, w, h = items[0]['bbox']
        self.assertLessEqual(x+w, base.shape[1])
        self.assertLessEqual(y+h, base.shape[0])


@unittest.skipUnless((ROOT/'models/latest.json').exists(), 'Checkpoint required')
class MultiDigitAppTests(unittest.TestCase):
    def test_automatic_row_and_save(self):
        root = tk.Tk()
        root.withdraw()
        try:
            app = PhotoApp(root, Predictor())
            app.load_path(ROOT/'data/samples/mnist_paper_row.png')
            app.recognize()
            self.assertEqual(app.result.cget('text'), '05221859')
            self.assertEqual(len(app.last_result['detections']), 8)
            self.assertTrue(np.any(np.all(app.annotated == [0,255,0], axis=2)))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'result.png'
                with patch('src.demo_photo.filedialog.asksaveasfilename', return_value=str(path)):
                    app.export_result()
                self.assertIsNotNone(cv2.imread(str(path)))
                self.assertEqual(json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))['text'], '05221859')
            app.reset_selection()
            self.assertIsNone(app.last_result)
        finally:
            root.destroy()
