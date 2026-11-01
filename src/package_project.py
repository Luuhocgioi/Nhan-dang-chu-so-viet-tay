"""Đóng gói mã, tài liệu, mô hình và minh chứng; không đưa .venv/dataset thô vào zip."""
from pathlib import Path
import json
import zipfile
from datetime import datetime
from .config import ROOT


def main():
    destination=ROOT/'dist';destination.mkdir(exist_ok=True)
    output=destination/('handwritten_digits_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.zip')
    files=set()
    if (ROOT/'.gitignore').exists():files.add(ROOT/'.gitignore')
    for pattern in ('*.md','requirements*.txt','*.cmd','src/*.py','tests/*.py','notebooks/*.ipynb',
                    'data/samples/*','reports/**/*.md','reports/**/*.json','reports/**/*.png',
                    'reports/**/split_indices.npz','models/yolo/**/args.yaml','models/yolo/**/results.csv',
                    'models/*/*.json','models/*.json'):
        files.update(p for p in ROOT.glob(pattern) if p.is_file())
    for pointer in (ROOT/'models').glob('*.json'):
        config=json.loads(pointer.read_text(encoding='utf-8-sig'))
        if 'model' in config:
            model=ROOT/config['model']
            if model.exists():files.add(model)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(files):archive.write(file,file.relative_to(ROOT))
    print(output)
    print('Files:',len(files),'Bytes:',output.stat().st_size)


if __name__=='__main__':main()
