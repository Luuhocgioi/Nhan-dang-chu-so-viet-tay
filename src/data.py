"""Tải MNIST và chia validation phân tầng; không trộn tập test."""
from .config import ROOT, save_json
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(seed=42):
    from tensorflow.keras.datasets import mnist
    (images, labels), (test_images, test_labels) = mnist.load_data()
    train_idx, val_idx = train_test_split(
        np.arange(len(labels)), test_size=6000, random_state=seed, stratify=labels)
    normalize = lambda x: x.astype('float32')[..., None] / 255.0
    return ((normalize(images[train_idx]), labels[train_idx]),
            (normalize(images[val_idx]), labels[val_idx]),
            (normalize(test_images), test_labels))


def explore():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    train, val, test = load_data()
    out = ROOT / 'reports' / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    summary = {name: {'shape': list(x.shape), 'dtype': str(x.dtype),
                     'min': float(x.min()), 'max': float(x.max()),
                     'counts': np.bincount(y, minlength=10).tolist()}
               for name, (x, y) in zip(('train', 'validation', 'test'), (train, val, test))}
    save_json(ROOT / 'reports' / 'data_summary.json', summary)
    fig, axes = plt.subplots(4, 10, figsize=(12, 5))
    for digit in range(10):
        indices = np.flatnonzero(train[1] == digit)[:4]
        for row, idx in enumerate(indices):
            axes[row, digit].imshow(train[0][idx, ..., 0], cmap='gray')
            axes[row, digit].axis('off')
            if row == 0:
                axes[row, digit].set_title(str(digit))
    fig.tight_layout()
    fig.savefig(out / 'mnist_samples.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    for offset, (name, values) in enumerate(summary.items()):
        ax.bar(np.arange(10) + (offset - 1) * .25, values['counts'], .25, label=name)
    ax.set(xticks=range(10), xlabel='Digit', ylabel='Samples')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'class_distribution.png', dpi=150)
    plt.close(fig)
    print(summary)


if __name__ == '__main__':
    explore()
