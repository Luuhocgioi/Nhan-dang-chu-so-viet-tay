"""Ultralytics YOLO: box và lớp chữ số đều do detector dự đoán."""
import json
import os
from pathlib import Path
import cv2
import numpy as np
from .config import ROOT
from .preprocess import EmptyImageError
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT/'data/processed/ultralytics'))


class YOLOPredictor:
    def __init__(self, model_path=None, confidence=.25):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ValueError('Thiếu Ultralytics. Cài: python -m pip install -r requirements-yolo.txt') from exc
        if model_path is None:
            pointer = ROOT/'models/yolo_latest.json'
            if not pointer.exists():
                raise FileNotFoundError('Chưa có YOLO chữ số. Chạy python -m src.train_yolo trước.')
            model_path = ROOT/json.loads(pointer.read_text(encoding='utf-8'))['model']
        self.path = Path(model_path)
        if not self.path.is_file():
            raise FileNotFoundError(f'Không tìm thấy checkpoint YOLO: {self.path}')
        self.model = YOLO(str(model_path))
        from .emnist_model import CHARACTERS
        names=[str(self.model.names[i]) for i in range(len(self.model.names))]
        if names not in ([str(i) for i in range(10)], [str(i) for i in range(10)]+['other_letter'], list(CHARACTERS)):
            raise ValueError('Cần checkpoint YOLO đã huấn luyện 10 lớp 0–9; không dùng trọng số COCO trực tiếp.')
        self.confidence = confidence

    def recognize(self, image):
        from .detect import to_bgr
        base = to_bgr(image)
        if max(base.shape[:2]) > 1600:
            scale = 1600/max(base.shape[:2])
            base = cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # Ảnh giấy có thể là dải ngang dài: 640 giữ nét nhỏ tốt hơn 416.
        prediction = self.model.predict(base, imgsz=640, conf=self.confidence, iou=.45,
                                        agnostic_nms=True, verbose=False)[0]
        detections = []
        for box in prediction.boxes:
            if int(box.cls.item()) >= 10:
                continue  # Lớp chữ cái phục vụ học phân biệt, không xuất box.
            x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
            left,top = max(0,int(x1)),max(0,int(y1))
            right,bottom = min(base.shape[1],int(np.ceil(x2))), min(base.shape[0],int(np.ceil(y2)))
            if right <= left or bottom <= top:
                continue
            detections.append({'digit':int(box.cls.item()), 'score':float(box.conf.item()),
                               'bbox':[left,top,right-left,bottom-top], 'probabilities':None})
        if not detections:
            raise EmptyImageError('Không có chữ số vượt ngưỡng nhận dạng trong ảnh. Nếu ảnh có số, hãy kiểm tra độ rõ và ánh sáng.')
        rows = []
        for detection in sorted(detections, key=lambda d:d['bbox'][1]+d['bbox'][3]/2):
            x,y,w,h = detection['bbox']
            for row in rows:
                top = min(d['bbox'][1] for d in row)
                bottom = max(d['bbox'][1]+d['bbox'][3] for d in row)
                if min(y+h,bottom)-max(y,top) >= .4*min(h,bottom-top):
                    row.append(detection)
                    break
            else:
                rows.append([detection])
        ordered, lines, thumbnails = [], [], []
        annotated = base.copy()
        for row_id,row in enumerate(rows):
            row.sort(key=lambda d:d['bbox'][0])
            lines.append(''.join(str(d['digit']) for d in row))
            for detection in row:
                detection['row'] = row_id
                ordered.append(detection)
                x,y,w,h = detection['bbox']
                thumbnails.append(cv2.resize(cv2.cvtColor(base[y:y+h,x:x+w], cv2.COLOR_BGR2GRAY),(28,28)))
                thickness = max(2,round(max(base.shape[:2])/500))
                cv2.rectangle(annotated,(x,y),(x+w,y+h),(0,255,0),thickness)
                font_scale = max(.6,min(2,h/65))
                text_height = cv2.getTextSize(str(detection['digit']),cv2.FONT_HERSHEY_SIMPLEX,font_scale,thickness)[0][1]
                baseline = y-8 if y>text_height+10 else y+text_height+4
                cv2.putText(annotated,str(detection['digit']),(x,baseline),cv2.FONT_HERSHEY_SIMPLEX,font_scale,(0,255,255),thickness,cv2.LINE_AA)
        return {'text':'\n'.join(lines),'detections':ordered,'backend':'ultralytics-yolo',
                'confidence_threshold':self.confidence,'inference_imgsz':640,'coordinate_space':'output_image',
                'output_size':[base.shape[1],base.shape[0]],'input_size':[image.shape[1],image.shape[0]]}, annotated, thumbnails


def create_predictor(backend='auto', model_path=None):
    if backend=='auto' and not model_path:
        runtime=ROOT/'models/runtime.json'
        if runtime.exists():
            selected=json.loads(runtime.read_text(encoding='utf-8')).get('backend','yolo')
            if selected not in ('numeric-rows','emnist','yolo','cnn'):
                raise ValueError('Backend trong models/runtime.json không hợp lệ.')
            backend=selected
    if backend == 'numeric-rows':
        from .emnist_predict import EMNISTPredictor
        return EMNISTPredictor(model_path,numeric_rows=True)
    if backend == 'emnist':
        from .emnist_predict import EMNISTPredictor
        return EMNISTPredictor(model_path)
    if backend == 'yolo-emnist':
        pointer=ROOT/'models/yolo_emnist_candidate.json'
        if not pointer.exists():raise FileNotFoundError('Chưa có YOLO EMNIST đã huấn luyện.')
        return YOLOPredictor(model_path or ROOT/json.loads(pointer.read_text(encoding='utf-8'))['model'])
    if backend == 'blue-ink':
        from .ink_predict import InkPredictor
        return InkPredictor()
    if backend == 'auto' and model_path:
        backend = 'yolo' if Path(model_path).suffix.lower() == '.pt' else 'cnn'
    if backend == 'yolo' or (backend == 'auto' and (ROOT/'models/yolo_latest.json').exists()):
        return YOLOPredictor(model_path)
    from .predict import Predictor
    return Predictor(model_path)
