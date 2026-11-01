import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('KERAS_HOME', str(ROOT / 'data' / 'raw' / 'keras'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'data' / 'processed' / 'matplotlib'))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def resolve_model(path=None):
    if path:
        result = Path(path)
    else:
        pointer = ROOT / 'models' / 'latest.json'
        if not pointer.exists():
            raise FileNotFoundError('Chưa có mô hình. Chạy python -m src.train trước.')
        result = ROOT / json.loads(pointer.read_text(encoding='utf-8'))['model']
    if not result.is_file():
        raise FileNotFoundError(f'Không tìm thấy mô hình: {result}')
    return result
