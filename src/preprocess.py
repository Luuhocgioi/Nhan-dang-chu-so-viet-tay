"""Ảnh OpenCV BGR/BGRA hoặc grayscale uint8 -> tensor MNIST."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


class EmptyImageError(ValueError):
    pass


def read_image(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Không tìm thấy ảnh: {path}')
    try:
        with Image.open(path) as source:
            # Điện thoại thường lưu hướng chụp bằng EXIF thay vì xoay pixel.
            source = ImageOps.exif_transpose(source)
            rgba = np.array(source.convert('RGBA'))
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f'File không phải ảnh hợp lệ: {path}') from exc


def preprocess(image, polarity='auto', mode='standard'):
    if mode not in ('standard', 'paper'):
        raise ValueError('Chế độ phải là standard hoặc paper.')
    if polarity not in ('auto', 'dark', 'light'):
        raise ValueError('Polarity phải là auto, dark hoặc light.')
    if image is None or image.size == 0:
        raise EmptyImageError('Ảnh trống.')
    if image.dtype != np.uint8:
        raise ValueError('Ảnh đầu vào phải là uint8 (0..255).')
    if image.ndim == 3:
        if image.shape[2] == 4:
            alpha = image[..., 3:4].astype(np.float32) / 255
            image = (image[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
        if image.shape[2] != 3:
            raise ValueError('Ảnh phải có 1, 3 hoặc 4 kênh.')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image.copy()
    else:
        raise ValueError('Kích thước ảnh không hợp lệ.')
    if min(gray.shape) < 3 or int(gray.max()) - int(gray.min()) < 15:
        raise EmptyImageError('Không tìm thấy nét chữ đủ rõ.')
    if mode == 'paper':
        # Giới hạn kích thước để xử lý ảnh điện thoại mà không tốn quá nhiều RAM.
        if max(gray.shape) > 1200:
            scale = 1200 / max(gray.shape)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # Chế độ giấy giả định mực tối trên giấy sáng. Closing ước lượng nền
        # chậm biến thiên, division giảm bóng/gradient trước khi tìm nét mực.
        if polarity == 'light':
            gray = 255 - gray
        kernel_size = max(15, int(min(gray.shape) * .15) | 1)
        background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,
                                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))
        gray = 255 - cv2.divide(gray, np.maximum(background, 1), scale=255)
        if int(gray.max()) < 15:
            raise EmptyImageError('Không tìm thấy nét mực rõ trên giấy.')
        threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, mask = cv2.threshold(gray, max(15, threshold), 255, cv2.THRESH_BINARY)
    else:
        border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
        if polarity == 'dark' or (polarity == 'auto' and np.median(border) > 127):
            gray = 255 - gray
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    minimum = max(3, int(gray.size * .0005))
    keep = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= minimum]
    if mode == 'paper':
        # Mép ảnh/giấy và bóng biên không phải chữ số nằm bên trong ROI.
        # Người dùng cần chừa một viền giấy quanh nét khi khoanh vùng.
        edge_labels = set(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
        keep = [i for i in keep if i not in edge_labels]
    mask = np.isin(labels, keep)
    if not mask.any():
        raise EmptyImageError('Không tìm thấy chữ số; hãy chọn vùng có nét mực rõ hơn.')
    ys, xs = np.where(mask)
    foreground = np.where(mask, gray, 0).astype(np.uint8)
    crop = foreground[ys.min():ys.max()+1, xs.min():xs.max()+1]
    scale = 20 / max(crop.shape)
    width, height = max(1, round(crop.shape[1]*scale)), max(1, round(crop.shape[0]*scale))
    resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    padded = np.zeros((28, 28), dtype=np.uint8)
    y, x = (28-height)//2, (28-width)//2
    padded[y:y+height, x:x+width] = resized
    moments = cv2.moments(padded)
    if moments['m00']:
        dx = 13.5 - moments['m10']/moments['m00']
        dy = 13.5 - moments['m01']/moments['m00']
        padded = cv2.warpAffine(padded, np.float32([[1, 0, dx], [0, 1, dy]]), (28, 28))
    return padded.astype(np.float32)[None, ..., None] / 255
