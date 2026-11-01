import argparse
import os
from datetime import datetime
from .config import ROOT, save_json
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT/'data/processed/ultralytics'))


def main():
    from ultralytics import YOLO, settings
    import torch
    settings.update({'sync': False})
    parser = argparse.ArgumentParser(description='Huấn luyện YOLO11n cho 10 lớp chữ số')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--device', default='0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch', type=int, default=16)
    args = parser.parse_args()
    run = 'yolo_emnist_'+datetime.now().strftime('%Y%m%d_%H%M%S')
    import json
    weights = ROOT/json.loads((ROOT/'models/yolo_latest.json').read_text(encoding='utf-8'))['model']
    model = YOLO(str(weights))
    model.train(data=str(ROOT/'data/yolo_emnist/dataset.yaml'), epochs=args.epochs,
        imgsz=416, batch=args.batch, device=args.device, workers=0,
        project=str(ROOT/'models/yolo'), name=run, seed=42, deterministic=True,
        patience=6, optimizer='AdamW', lr0=.0005, fliplr=0, flipud=0,
        degrees=8, mosaic=.3, close_mosaic=5, translate=.08, scale=.2,
        hsv_h=0, hsv_s=0, hsv_v=.15, amp=False, plots=True, verbose=False)
    best = model.trainer.best
    trained = YOLO(str(best))
    metrics = trained.val(data=str(ROOT/'data/yolo_emnist/dataset.yaml'), split='test',
        imgsz=416, device=args.device, workers=0, project=str(ROOT/'reports/yolo'), name=run+'_test', plots=True)
    save_json(ROOT/'reports/yolo'/run/'summary.json', {'run':run, 'model':str(best),
        'test_metrics':metrics.results_dict, 'dataset':'EMNIST62 composites, not real photos'})
    save_json(ROOT/'models/yolo_emnist_candidate.json', {'model':best.relative_to(ROOT).as_posix(), 'run_id':run})
    print('Saved YOLO checkpoint:', best)


if __name__ == '__main__':
    main()
