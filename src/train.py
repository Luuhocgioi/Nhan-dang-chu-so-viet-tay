import argparse
from datetime import datetime
import importlib.metadata
import platform
import time
from .config import ROOT, save_json
from .data import load_data
from .model import build_model
import numpy as np
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser(description='Huấn luyện CNN trên MNIST')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error('epochs, batch-size và learning-rate phải dương.')
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    train, val, _ = load_data(args.seed)
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory = ROOT / 'models' / run_id
    directory.mkdir(parents=True, exist_ok=False)
    output = ROOT / 'reports' / 'experiments' / run_id
    output.mkdir(parents=True)
    model = build_model(args.learning_rate)
    lines = []
    model.summary(print_fn=lambda line: lines.append(line))
    (output / 'model_summary.txt').write_text('\n'.join(lines), encoding='utf-8')
    config = dict(vars(args), run_id=run_id, dropout=.3, optimizer='Adam',
                  loss='sparse_categorical_crossentropy', augmentation=False,
                  split={'train': len(train[1]), 'validation': len(val[1]), 'test': 10000},
                  python=platform.python_version(), platform=platform.platform(),
                  devices=[str(d) for d in tf.config.list_physical_devices()],
                  versions={p: importlib.metadata.version(p) for p in
                            ('tensorflow', 'keras', 'numpy', 'opencv-python', 'scikit-learn')})
    save_json(output / 'config.json', config)
    save_json(directory / 'config.json', config)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
                 tf.keras.callbacks.ModelCheckpoint(str(directory / 'best.keras'), monitor='val_loss', save_best_only=True),
                 tf.keras.callbacks.CSVLogger(str(output / 'epochs.csv'))]
    started = time.perf_counter()
    history = model.fit(*train, validation_data=val, epochs=args.epochs,
                        batch_size=args.batch_size, callbacks=callbacks, verbose=2)
    duration = time.perf_counter() - started
    save_json(output / 'history.json', history.history)
    loaded = tf.keras.models.load_model(directory / 'best.keras')
    np.testing.assert_allclose(model(val[0][:16], training=False).numpy(),
                               loaded(val[0][:16], training=False).numpy(), atol=1e-6)
    best = int(np.argmin(history.history['val_loss']))
    save_json(output / 'training_summary.json', {
        'best_epoch': best+1, 'epochs_run': len(history.history['loss']), 'seconds': duration,
        'train_accuracy': history.history['accuracy'][best],
        'validation_accuracy': history.history['val_accuracy'][best], 'reload_verified': True})
    save_json(ROOT / 'models' / 'latest.json', {'model': (directory / 'best.keras').relative_to(ROOT).as_posix(), 'run_id': run_id})
    print(f'Run: {run_id}\nModel: {directory / "best.keras"}\nThời gian: {duration:.1f}s')


if __name__ == '__main__':
    main()
