"""Tách các chữ số rời nhau trên giấy và nhận dạng theo thứ tự đọc."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from .config import ROOT, save_json
from .preprocess import EmptyImageError, preprocess, read_image


def to_bgr(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        alpha = image[..., 3:4].astype(np.float32)/255
        return (image[..., :3]*alpha + 255*(1-alpha)).astype(np.uint8)
    return image.copy()


def segment_digits(image):
    """Trả ảnh BGR đã giới hạn 1600 px và các (bbox, tensor) theo từng dòng.

    Một thành phần liên thông tương ứng một chữ số. Chữ dính nhau hoặc
    nét đứt thành nhiều mảnh chưa được tách/gộp đáng tin cậy.
    """
    if image is None or image.size == 0 or image.dtype != np.uint8:
        raise ValueError('Ảnh phải là uint8 và không rỗng.')
    base = to_bgr(image)
    if max(base.shape[:2]) > 1600:
        scale = 1600/max(base.shape[:2])
        base = cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    if min(gray.shape) < 10:
        raise EmptyImageError('Ảnh quá nhỏ để tìm chữ số.')
    size = max(15, int(min(gray.shape)*.08) | 1)
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    ink = 255-cv2.divide(gray, np.maximum(background, 1), scale=255)
    threshold, _ = cv2.threshold(ink, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, mask = cv2.threshold(ink, max(20, threshold), 255, cv2.THRESH_BINARY)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    edge = set(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    candidates = []
    for idx in range(1, count):
        x, y, w, h, area = map(int, stats[idx])
        if idx in edge or area < max(12, gray.size*.00006) or h < max(10, gray.shape[0]*.015):
            continue
        if w/h > 2.5 or h > gray.shape[0]*.95:
            continue
        candidates.append((x, y, w, h, idx))
    if not candidates:
        raise EmptyImageError('Không tìm thấy chữ số. Dùng nét mực đậm trên giấy sáng, chừa viền quanh chữ.')
    # Gom dòng bằng độ chồng lấp theo chiều dọc, rồi sắp từ trái sang phải.
    rows = []
    for box in sorted(candidates, key=lambda b: b[1]+b[3]/2):
        x, y, w, h, _ = box
        for row in rows:
            top = min(b[1] for b in row)
            bottom = max(b[1]+b[3] for b in row)
            if min(y+h, bottom)-max(y, top) >= .4*min(h, bottom-top):
                row.append(box)
                break
        else:
            rows.append([box])
    items = []
    for row_id, row in enumerate(sorted(rows, key=lambda r: min(b[1] for b in r))):
        for x, y, w, h, idx in sorted(row):
            component = np.where(labels[y:y+h, x:x+w] == idx, ink[y:y+h, x:x+w], 0).astype(np.uint8)
            component = cv2.copyMakeBorder(component, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0)
            tensor = preprocess(component, polarity='light', mode='standard')
            items.append({'bbox': [x, y, w, h], 'row': row_id, 'tensor': tensor})
    return base, items


def recognize_digits(image, predictor):
    if hasattr(predictor, 'recognize'):
        return predictor.recognize(image)
    base, items = segment_digits(image)
    tensors = np.concatenate([item['tensor'] for item in items])
    scores = predictor.model(tensors, training=False).numpy()
    annotated = base.copy()
    detections = []
    for item, probabilities in zip(items, scores):
        digit = int(probabilities.argmax())
        x, y, w, h = item['bbox']
        thickness = max(2, round(max(base.shape[:2])/500))
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), thickness)
        font_scale = max(.6, min(2, h/65))
        text_height = cv2.getTextSize(str(digit), cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][1]
        baseline = y-8 if y > text_height+10 else y+text_height+4
        cv2.putText(annotated, str(digit), (x, baseline), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
        detections.append({'digit': digit, 'score': float(probabilities[digit]),
                           'probabilities': probabilities.tolist(), 'bbox': item['bbox'], 'row': item['row']})
    lines = [''.join(str(d['digit']) for d in detections if d['row'] == row)
             for row in sorted({d['row'] for d in detections})]
    result = {'text': '\n'.join(lines), 'detections': detections,
              'coordinate_space': 'output_image', 'output_size': [base.shape[1], base.shape[0]],
              'input_size': [image.shape[1], image.shape[0]]}
    thumbnails = [(item['tensor'][0, ..., 0]*255).astype(np.uint8) for item in items]
    return result, annotated, thumbnails


def save_result(path, result, annotated):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode('.png', annotated)[1].tofile(path)
    save_json(path.with_suffix('.json'), result)


def main():
    parser = argparse.ArgumentParser(description='Tự tìm nhiều chữ số trên giấy, vẽ khung và xuất chuỗi')
    parser.add_argument('image')
    parser.add_argument('--model')
    parser.add_argument('--backend', choices=['auto', 'yolo', 'cnn', 'blue-ink', 'emnist', 'yolo-emnist', 'numeric-rows'], default='auto')
    parser.add_argument('--output', default=str(ROOT/'reports/figures/detected_digits.png'))
    args = parser.parse_args()
    from .yolo_predict import create_predictor
    try:
        result, annotated, _ = recognize_digits(read_image(args.image), create_predictor(args.backend, args.model))
        save_result(args.output, result, annotated)
        print(result['text'])
        print(f'Đã lưu ảnh và JSON: {args.output}')
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f'{exc}\n')


if __name__ == '__main__':
    main()
