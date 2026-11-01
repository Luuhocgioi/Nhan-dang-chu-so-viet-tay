"""Chế độ tùy chọn cho nét mực xanh/tím trên nền giấy xám/ô ly."""
import cv2
import numpy as np
from .preprocess import EmptyImageError


class InkPredictor:
    def __init__(self):
        from .predict import Predictor
        self.classifier = Predictor()

    def recognize(self, image):
        from .detect import to_bgr, recognize_digits
        base = to_bgr(image)
        if max(base.shape[:2]) > 1600:
            scale = 1600/max(base.shape[:2])
            base = cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([95,60,0]), np.array([170,255,210]))
        clean = np.full_like(base,255)
        clean[mask>0] = 0
        result, _, thumbs = recognize_digits(clean,self.classifier)
        heights = [d['bbox'][3] for d in result['detections']]
        minimum_height = .5*float(np.median(heights))
        pairs = [(d,t) for d,t in zip(result['detections'],thumbs) if d['bbox'][3]>=minimum_height]
        if not pairs:
            raise EmptyImageError('Không tìm thấy nét mực xanh/tím đủ lớn.')
        result['detections'] = [d for d,t in pairs]
        result['text'] = '\n'.join(''.join(str(d['digit']) for d in result['detections'] if d['row']==row)
            for row in sorted({d['row'] for d in result['detections']}))
        result.update(backend='blue-ink-cnn', input_size=[image.shape[1],image.shape[0]],
                      notes='Color filter; not a general letter rejection model. Small fragments below half median height omitted.')
        annotated = base.copy()
        for detection,_ in pairs:
            x,y,w,h = detection['bbox']
            cv2.rectangle(annotated,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(annotated,str(detection['digit']),(x,max(25,y-8)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_AA)
        return result,annotated,[t for d,t in pairs]
