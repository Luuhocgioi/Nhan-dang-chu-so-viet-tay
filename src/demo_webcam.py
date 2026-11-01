import argparse
import cv2
from .predict import Predictor
from .preprocess import EmptyImageError


def main():
    parser = argparse.ArgumentParser(description='Webcam: đặt một chữ số trong khung; SPACE dự đoán, Q thoát')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--model')
    args = parser.parse_args()
    try:
        predictor = Predictor(args.model)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f'{exc}\n')
    capture = cv2.VideoCapture(args.camera)
    try:
        if not capture.isOpened():
            parser.exit(2, 'Không mở được webcam. Kiểm tra thiết bị/quyền truy cập hoặc --camera.\n')
        while True:
            ok, frame = capture.read()
            if not ok:
                parser.exit(2, 'Không đọc được khung hình webcam.\n')
            height, width = frame.shape[:2]
            side = int(min(height, width)*.65)
            x, y = (width-side)//2, (height-side)//2
            roi = frame[y:y+side, x:x+side].copy()
            cv2.rectangle(frame, (x, y), (x+side, y+side), (0, 220, 0), 2)
            cv2.putText(frame, 'One digit in box | SPACE: predict | Q: quit', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 220, 0), 1)
            cv2.imshow('Webcam - MNIST', frame)
            key = cv2.waitKey(20) & 0xff
            if key in (ord('q'), 27) or cv2.getWindowProperty('Webcam - MNIST', cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == 32:
                try:
                    result, processed = predictor.predict(roi)
                    print(f"Digit: {result['digit']}; score: {result['score']:.1%}", flush=True)
                    preview = cv2.resize(processed, (280, 280), interpolation=cv2.INTER_NEAREST)
                    preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
                    cv2.putText(preview, f"{result['digit']} ({result['score']:.1%})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 220, 0), 2)
                    cv2.imshow('Prediction snapshot', preview)
                except EmptyImageError as exc:
                    print(exc, flush=True)
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
