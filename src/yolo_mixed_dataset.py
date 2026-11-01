"""Ảnh chữ số MNIST và chữ Latin tổng hợp, có lớp loại bỏ chữ cái."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .config import ROOT, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=1000)
    parser.add_argument('--val', type=int, default=150)
    parser.add_argument('--test', type=int, default=100)
    args = parser.parse_args()
    output = ROOT/'data/yolo_mixed'
    if (output/'dataset.yaml').exists():
        parser.exit(0, 'Dataset đã tồn tại; giữ nguyên để tái lập.\n')
    # Dùng file cache, tránh import TensorFlow cùng tiến trình huấn luyện Torch.
    with np.load(ROOT/'data/raw/keras/datasets/mnist.npz') as mnist:
        images, labels = mnist['x_train'], mnist['y_train']
        test_images, test_labels = mnist['x_test'], mnist['y_test']
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(np.arange(len(labels)), test_size=6000, random_state=42, stratify=labels)
    rng = np.random.default_rng(20261101)
    sources = {'train': (images[train_idx], labels[train_idx]),
               'val': (images[val_idx], labels[val_idx]), 'test': (test_images, test_labels)}
    for split, count in vars(args).items():
        for folder in ('images', 'labels'):
            (output/folder/split).mkdir(parents=True, exist_ok=True)
        xdata, ydata = sources[split]
        for index in range(count):
            size = 416
            shade = np.linspace(rng.uniform(125, 235), rng.uniform(180, 255), size)
            paper = np.tile(shade, (size, 1)) + rng.normal(0, 1.8, (size, size))
            boxes = []
            # 4x3 ô không giao nhau; số vị trí và số dòng biến thiên.
            positions = rng.choice(12, size=int(rng.integers(1, 11)), replace=False)
            if index % 25 == 0:
                positions = []  # ảnh âm, không có chữ số
            for pos in positions:
                sample = int(rng.integers(len(xdata)))
                is_letter = rng.random() < .5 or index % 5 == 0
                if is_letter:
                    fonts = ['arial.ttf','times.ttf','calibri.ttf','comic.ttf','segoepr.ttf']
                    font_path = Path('C:/Windows/Fonts')/fonts[(index if split == 'train' else index+2)%len(fonts)]
                    if not font_path.exists():
                        font_path = Path('C:/Windows/Fonts/arial.ttf')
                    glyph = Image.new('L',(140,140),0)
                    ImageDraw.Draw(glyph).text((12,0),chr(int(rng.choice(list(range(65,91))+list(range(97,123))))),font=ImageFont.truetype(str(font_path),95),fill=255)
                    digit = np.array(glyph)
                else:
                    digit = xdata[sample]
                ys, xs = np.where(digit > 30)
                crop = digit[ys.min():ys.max()+1, xs.min():xs.max()+1]
                height = int(rng.integers(38, 86))
                width = max(8, round(height*crop.shape[1]/crop.shape[0]))
                width = min(width, 86)
                crop = cv2.resize(crop, (width, height)).astype(np.float32)/255
                x = int(pos % 4)*104 + int(rng.integers(8, max(9, 104-width-5)))
                y = int(pos // 4)*138 + int(rng.integers(12, max(13, 138-height-8)))
                paper[y:y+height, x:x+width] *= 1-crop*rng.uniform(.65, .97)
                boxes.append(f'{10 if is_letter else int(ydata[sample])} {(x+width/2)/size:.6f} {(y+height/2)/size:.6f} {width/size:.6f} {height/size:.6f}')
            image = np.clip(paper, 0, 255).astype(np.uint8)
            cv2.imwrite(str(output/'images'/split/f'{index:05d}.jpg'), image)
            (output/'labels'/split/f'{index:05d}.txt').write_text('\n'.join(boxes), encoding='utf-8')
    (output/'dataset.yaml').write_text('path: '+output.as_posix()+'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n'+''.join(f'  {i}: "{i}"\n' for i in range(10))+'  10: "other_letter"\n', encoding='utf-8')
    save_json(output/'manifest.json', {'seed': 20261101, 'counts': vars(args), 'size': 416,
        'source': 'MNIST composite on shaded paper; NOT real photographs',
        'split': 'stratified 54000 train / 6000 val MNIST source images; MNIST test reserved for test composites',
        'negative_every': 25, 'reject_class':10, 'letters':'synthetic Latin font glyphs; not real handwritten letters', 'letters_only_every':5})
    print(output/'dataset.yaml')


if __name__ == '__main__':
    main()
