"""Kiểm thử tích hợp cần checkpoint thật; webcam dùng thiết bị giả lập."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
import tkinter as tk
import numpy as np
from src.config import ROOT
from src.predict import Predictor
from src.demo_canvas import DigitApp
from src import demo_webcam


@unittest.skipUnless((ROOT / 'models/latest.json').exists(), 'Chạy train để có checkpoint trước')
class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = Predictor()

    def test_canvas_draw_predict_clear_and_blank(self):
        root = tk.Tk()
        root.withdraw()
        try:
            app = DigitApp(root, self.predictor)
            with patch('src.demo_canvas.messagebox.showinfo') as info:
                app.recognize()
                info.assert_called_once()
            app.start(SimpleNamespace(x=140, y=55))
            app.draw(SimpleNamespace(x=140, y=220))
            app.recognize()
            root.update_idletasks()
            self.assertIn(app.result.cget('text'), [str(i) for i in range(10)])
            self.assertEqual(len(app.scores.cget('text').splitlines()), 10)
            app.clear()
            self.assertFalse(np.array(app.image).any())
            self.assertEqual(app.result.cget('text'), '—')
        finally:
            root.destroy()

    def test_webcam_missing_device(self):
        camera = MagicMock()
        camera.isOpened.return_value = False
        with patch('sys.argv', ['demo_webcam']), patch.object(demo_webcam, 'Predictor', return_value=self.predictor), patch.object(demo_webcam.cv2, 'VideoCapture', return_value=camera), patch.object(demo_webcam.cv2, 'destroyAllWindows'):
            with self.assertRaises(SystemExit) as result:
                demo_webcam.main()
            self.assertEqual(result.exception.code, 2)
        camera.release.assert_called_once()

    def test_webcam_capture_predict_and_quit(self):
        camera = MagicMock()
        camera.isOpened.return_value = True
        frame = np.full((480, 640, 3), 255, np.uint8)
        frame[140:340, 310:330] = 0
        camera.read.return_value = (True, frame)
        with patch('sys.argv', ['demo_webcam']), patch.object(demo_webcam, 'Predictor', return_value=self.predictor), patch.object(demo_webcam.cv2, 'VideoCapture', return_value=camera), patch.object(demo_webcam.cv2, 'imshow') as show, patch.object(demo_webcam.cv2, 'waitKey', side_effect=[32, ord('q')]), patch.object(demo_webcam.cv2, 'getWindowProperty', return_value=1), patch.object(demo_webcam.cv2, 'destroyAllWindows'):
            demo_webcam.main()
            self.assertTrue(any(call.args[0] == 'Prediction snapshot' for call in show.call_args_list))
        camera.release.assert_called_once()
