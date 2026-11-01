import argparse
import json
from .config import ROOT, resolve_model, save_json
from .data import load_data
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def main():
    parser = argparse.ArgumentParser(description='Đánh giá checkpoint đã chọn trên MNIST test')
    parser.add_argument('--model')
    args = parser.parse_args()
    import tensorflow as tf
    path = resolve_model(args.model)
    output = ROOT / 'reports' / 'experiments' / path.parent.name
    output.mkdir(parents=True, exist_ok=True)
    _, _, (x, y) = load_data()
    model = tf.keras.models.load_model(path)
    loss, accuracy = model.evaluate(x, y, verbose=0)
    probabilities = model.predict(x, batch_size=256, verbose=0)
    predicted = probabilities.argmax(axis=1)
    matrix = confusion_matrix(y, predicted, labels=range(10))
    save_json(output / 'metrics.json', {'model': str(path), 'samples': len(y),
              'test_accuracy': float(accuracy), 'test_loss': float(loss),
              'classification_report': classification_report(y, predicted, labels=range(10), output_dict=True, zero_division=0),
              'confusion_matrix': matrix.tolist()})
    np.savez_compressed(output / 'predictions.npz', labels=y, predicted=predicted, probabilities=probabilities)
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(matrix, display_labels=range(10)).plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set(xlabel='Predicted label', ylabel='True label', title=f'MNIST test accuracy: {accuracy:.2%}')
    fig.tight_layout()
    fig.savefig(output / 'confusion_matrix.png', dpi=150)
    plt.close(fig)
    history_path = output / 'history.json'
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding='utf-8'))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, metric in zip(axes, ('loss', 'accuracy')):
            epochs = range(1, len(history[metric])+1)
            ax.plot(epochs, history[metric], label='Train')
            ax.plot(epochs, history['val_'+metric], label='Validation')
            ax.set(xlabel='Epoch', ylabel=metric)
            ax.legend()
            ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(output / 'learning_curves.png', dpi=150)
        plt.close(fig)
    errors = np.flatnonzero(y != predicted)[:20]
    fig, axes = plt.subplots(4, 5, figsize=(9, 8))
    for ax in axes.flat:
        ax.axis('off')
    for ax, idx in zip(axes.flat, errors):
        ax.imshow(x[idx, ..., 0], cmap='gray')
        ax.set_title(f'True {y[idx]} / Pred {predicted[idx]}')
    fig.tight_layout()
    fig.savefig(output / 'errors.png', dpi=150)
    plt.close(fig)
    print(f'Test accuracy: {accuracy:.4%}; loss: {loss:.6f}; samples: {len(y)}\nSaved: {output}')


if __name__ == '__main__':
    main()
