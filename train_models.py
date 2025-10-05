import os
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

DATASET_PATH  = "custom_dataset"
CHECKPOINT_DIR = "checkpoints"
EPOCHS        = 100
BATCH_SIZE    = 32
IMG_SIZE      = (128, 128)

def load_dataset():
    """Load images and labels from custom dataset"""
    images, labels = [], []
    for label_folder in sorted(os.listdir(DATASET_PATH)):
        label_path = os.path.join(DATASET_PATH, label_folder)
        if not (os.path.isdir(label_path) and label_folder.isdigit()):
            continue
        label = int(label_folder)  # 0..5
        for img_file in os.listdir(label_path):
            if img_file.lower().endswith(".png"):
                img_path = os.path.join(label_path, img_file)
                img = load_img(img_path, color_mode="grayscale", target_size=IMG_SIZE)
                arr = img_to_array(img).astype("float32") / 255.0
                images.append(arr.flatten())
                labels.append(label)
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32)

def create_model(activation, initialization, regularization):
    """Create a fully connected neural network"""
    # regularizer
    reg = regularizers.l2(0.001) if regularization == "L2" else None
    # initializer
    if initialization == "Xavier":
        init = keras.initializers.GlorotNormal()
    elif initialization == "He":
        init = keras.initializers.HeNormal()
    else:
        init = keras.initializers.GlorotNormal()

    model = keras.Sequential([
        layers.Input(shape=(128*128,)),
        layers.Dense(256, activation=activation, kernel_initializer=init, kernel_regularizer=reg),
        layers.Dropout(0.3) if regularization == "Dropout" else layers.Lambda(lambda x: x),
        layers.Dense(128, activation=activation, kernel_initializer=init, kernel_regularizer=reg),
        layers.Dropout(0.3) if regularization == "Dropout" else layers.Lambda(lambda x: x),
        layers.Dense(64, activation=activation, kernel_initializer=init,  kernel_regularizer=reg),
        layers.Dense(1, activation="linear")  # single output neuron
    ])
    return model

def train_session(cfg, X_train, y_train):
    """Train a model for one session configuration saving ONLY best & last checkpoints."""
    s = cfg["session"]
    print("\n" + "="*60)
    print(f"Training Session {s}")
    print("="*60)
    print(f"Activation: {cfg['activation']} | Init: {cfg['initialization']} | Reg: {cfg['regularization']}")
    print(f"Optimizer: {cfg['optimizer']} | LR: {cfg['lr']}")

    # Build & compile
    model = create_model(cfg["activation"], cfg["initialization"], cfg["regularization"])
    optimizer = keras.optimizers.SGD(learning_rate=cfg["lr"])
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    # Paths
    session_dir = os.path.join(CHECKPOINT_DIR, f"session_{s}")
    os.makedirs(session_dir, exist_ok=True)
    best_path = os.path.join(session_dir, "best.keras")
    last_path = os.path.join(session_dir, "last.keras")
    hist_path = os.path.join(session_dir, "history.json")

    # Callbacks
    ckpt_best = keras.callbacks.ModelCheckpoint(
        filepath=best_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    # Train (single fit; no per-2-epoch saves)
    t0 = time.time()
    h = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[ckpt_best],
        verbose=1,
        shuffle=True,
    )
    total_time = time.time() - t0

    # Save LAST checkpoint
    model.save(last_path)
    print(f"Saved last checkpoint -> {last_path}")

    # Save history (including best epoch)
    history = {k: [float(v) for v in h.history[k]] for k in h.history}
    history["epoch"] = list(range(1, len(history["loss"]) + 1))
    best_epoch = int(np.argmin(h.history["val_loss"])) + 1
    history["best_epoch_by_val_loss"] = best_epoch
    history["total_time_sec"] = float(total_time)
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved history -> {hist_path}")
    print(f"Best epoch (val_loss): {best_epoch}")

    return history

def main():
    print("Loading dataset...")
    X_train, y_train = load_dataset()
    print(f"Dataset loaded: {X_train.shape[0]} images, {X_train.shape[1]} features")

    sessions = [
        {'session': 1, 'activation': 'sigmoid', 'initialization': 'Xavier', 'regularization': 'None',    'optimizer': 'SGD', 'lr': 0.01},
        {'session': 2, 'activation': 'tanh',    'initialization': 'Xavier', 'regularization': 'None',    'optimizer': 'SGD', 'lr': 0.01},
        {'session': 3, 'activation': 'tanh',    'initialization': 'Xavier', 'regularization': 'L2',      'optimizer': 'SGD', 'lr': 0.01},
        {'session': 4, 'activation': 'tanh',    'initialization': 'Xavier', 'regularization': 'Dropout', 'optimizer': 'SGD', 'lr': 0.01},
        {'session': 5, 'activation': 'relu',    'initialization': 'Xavier', 'regularization': 'L2',      'optimizer': 'SGD', 'lr': 0.01},
        {'session': 6, 'activation': 'relu',    'initialization': 'He',     'regularization': 'L2',      'optimizer': 'SGD', 'lr': 0.01},
    ]

    for cfg in sessions:
        train_session(cfg, X_train, y_train)

    print("\n" + "="*60)
    print("All training sessions completed!")
    print("="*60)

if __name__ == "__main__":
    main()
