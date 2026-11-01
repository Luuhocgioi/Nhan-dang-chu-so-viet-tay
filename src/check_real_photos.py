"""Bộ hồi quy phát triển: ảnh người dùng đã được xem, không phải test độc lập."""
import argparse
from pathlib import Path
import cv2
from .config import ROOT,save_json
from .preprocess import read_image
from .detect import save_result
from .yolo_predict import create_predictor

CASES=[
    {'name':'dense','path':'OIP.webp','rotate':False,'expected':''.join(str(i)*20 for i in range(10))},
    {'name':'red_webp','path':'OIP (1).webp','rotate':False,'expected':'0123456789'},
    {'name':'tiles','path':'image-15.png','rotate':False,'expected':'5041921314'},
    {'name':'grid','path':'maxresdefault (1).jpg','rotate':False,'expected':'123456789'},
    {'name':'phone','path':'1793514116442_2209075521745795385_8089266940901408211_d5688339a935afbe7d6fb5b0681671ad.jpg','rotate':True,'expected':'3512'}]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--downloads',default=str(Path.home()/'Downloads'))
    parser.add_argument('--backends',nargs='+',default=['yolo','emnist','yolo-emnist'])
    args=parser.parse_args();summary={}
    for backend in args.backends:
        predictor=create_predictor(backend)
        summary[backend]={}
        for case in CASES:
            path=Path(args.downloads)/case['path']
            if not path.exists():
                summary[backend][case['name']]={'missing_input':str(path)};continue
            image=read_image(path)
            if case['rotate']:image=cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)
            try:
                result,annotated,_=predictor.recognize(image)
                save_result(ROOT/f'reports/figures/final_{backend}_{case["name"]}.png',result,annotated)
                actual=result['text'].replace('\n','')
                summary[backend][case['name']]={'expected':case['expected'],'actual':actual,
                    'exact_match':actual==case['expected'],'boxes':len(result['detections']),
                    'rejected':len(result.get('rejected',[]))}
            except ValueError as exc:
                summary[backend][case['name']]={'expected':case['expected'],'error':str(exc),'exact_match':False}
            print(backend,case['name'],summary[backend][case['name']],flush=True)
    save_json(ROOT/'reports/real_photo_regression.json',{'scope':'2 development images, not an independent benchmark',
        'orientation':'manual expected rotation for phone case', 'results':summary})


if __name__=='__main__':main()
