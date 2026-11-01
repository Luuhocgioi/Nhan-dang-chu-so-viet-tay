import argparse
import json
from .config import ROOT, resolve_model
from .preprocess import preprocess, read_image
import cv2
import numpy as np


class Predictor:
    def __init__(self, model_path=None):
        import tensorflow as tf
        self.path = resolve_model(model_path)
        self.model = tf.keras.models.load_model(self.path, compile=False)

    def predict(self, image, polarity='auto', mode='standard'):
        tensor = preprocess(image, polarity, mode)
        scores = self.model(tensor, training=False).numpy()[0]
        digit = int(scores.argmax())
        return {'digit': digit, 'score': float(scores[digit]),
                'probabilities': scores.tolist()}, (tensor[0, ..., 0]*255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description='Nhận dạng một chữ số từ ảnh')
    parser.add_argument('image')
    parser.add_argument('--model')
    parser.add_argument('--polarity', choices=['auto', 'dark', 'light'], default='auto')
    parser.add_argument('--mode', choices=['paper', 'standard'], default='paper', help='paper: ảnh chụp giấy (mặc định); standard: ảnh đã tách nền')
    parser.add_argument('--output', default=str(ROOT / 'reports' / 'figures' / 'prediction_input.png'))
    args = parser.parse_args()
    try:
        image = read_image(args.image)
        result, processed = Predictor(args.model).predict(image, args.polarity, args.mode)
        from pathlib import Path
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode('.png', processed)[1].tofile(output)
        print(json.dumps(result, indent=2))
        print(f'Ảnh tiền xử lý: {output}')
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f'{exc}\n')


if __name__ == '__main__':
    main()
