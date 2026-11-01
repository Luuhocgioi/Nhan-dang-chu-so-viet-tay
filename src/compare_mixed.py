"""So sánh detector trên cùng ảnh có chữ và số, không ẩn lỗi nhận nhầm."""
import argparse
import json
from pathlib import Path
import numpy as np
from .config import ROOT, save_json


def measure(path, split):
    from ultralytics import YOLO
    model = YOLO(str(path))
    directory = ROOT/'data/yolo_mixed'
    tp = fp = total = letter_fp = letters = 0
    for result in model.predict(source=str(directory/'images'/split), imgsz=640,
                                conf=.25, iou=.45, agnostic_nms=True, stream=True, verbose=False):
        gt=[]
        for line in (directory/'labels'/split/(Path(result.path).stem+'.txt')).read_text().splitlines():
            c,x,y,w,h=map(float,line.split())
            width,height=result.orig_shape[1],result.orig_shape[0]
            gt.append((int(c),np.array([(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height])))
        total+=sum(c<10 for c,b in gt)
        letters+=sum(c==10 for c,b in gt)
        used=set()
        for box in result.boxes:
            cls=int(box.cls.item())
            if cls>=10:
                continue
            b=box.xyxy[0].cpu().numpy()
            overlaps=[]
            for c,g in gt:
                intersection=np.maximum(0,np.minimum(b[2:],g[2:])-np.maximum(b[:2],g[:2])).prod()
                union=np.maximum(0,b[2:]-b[:2]).prod()+np.maximum(0,g[2:]-g[:2]).prod()-intersection
                overlaps.append(float(intersection/max(union,1e-6)))
            idx=int(np.argmax(overlaps)) if overlaps else -1
            if idx>=0 and overlaps[idx]>=.5 and gt[idx][0]==cls and idx not in used:
                tp+=1;used.add(idx)
            else:
                fp+=1
                if idx>=0 and overlaps[idx]>=.5 and gt[idx][0]==10:
                    letter_fp+=1
    return {'digit_precision':tp/max(1,tp+fp),'digit_recall':tp/max(1,total),
            'correct_digits':tp,'false_digit_boxes':fp,'digit_targets':total,
            'letters_as_digits':letter_fp,'letter_targets':letters,
            'letter_false_positive_rate':letter_fp/max(1,letters)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--split',choices=['val','test'],default='val')
    args=parser.parse_args()
    baseline_path=ROOT/'models/yolo_previous.json'
    if not baseline_path.exists():
        baseline_path=ROOT/'models/yolo_latest.json'
    baseline=json.loads(baseline_path.read_text(encoding='utf-8'))
    candidate=json.loads((ROOT/'models/yolo_candidate.json').read_text(encoding='utf-8'))
    results={name:measure(ROOT/pointer['model'],args.split) for name,pointer in [('baseline',baseline),('candidate',candidate)]}
    save_json(ROOT/f'reports/yolo/mixed_comparison_{args.split}.json',results)
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
