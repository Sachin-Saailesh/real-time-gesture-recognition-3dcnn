import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import matplotlib.pyplot as plt


# ----------------------- CONFIG -----------------------
CHECKPOINT_DIR = "checkpoints"            # e.g., checkpoints/session_1/checkpoint_100.h5

ALT_CKPT_DIR   = "checkpoints_part2"      # if Part-2 saved as checkpoints_part2/S1_sigmoid/epochNNN.keras
PART2_SUMMARY  = "part2_summary.json"     # created by Part-2 training script
EVAL_RESULTS   = "evaluation_results.json"  # created by your evaluator (epochs/accuracy per session)
TEST_IMAGES_DIR = "my_test_images"
IMG_SIZE = (128, 128)

# Map session number -> Part-2 session tag (used in part2_summary.json)
SESSION_TAG = {
    1: "S1_sigmoid",
    2: "S2_tanh",
    3: "S3_tanh_l2",
    4: "S4_tanh_do",
    5: "S5_relu",
    6: "S6_relu_l2",
}

# ----------------------- IO HELPERS -----------------------
def preprocess_image(img_path):
    """Load and preprocess a test image -> (16384,) float32 in [0,1]."""
    img = load_img(img_path, color_mode='grayscale', target_size=IMG_SIZE)
    arr = img_to_array(img).astype("float32") / 255.0
    return arr.flatten()

def _fallback_last_checkpoint(session_num):
    """Original behavior: pick last file; prefer .keras over .h5."""
    session_dir = os.path.join(CHECKPOINT_DIR, f"session_{session_num}")
    if not os.path.isdir(session_dir):
        return None
    files = sorted([f for f in os.listdir(session_dir) if f.endswith((".keras", ".h5"))])
    if not files:
        return None
    keras_files = [f for f in files if f.endswith(".keras")]
    ckpt = keras_files[-1] if keras_files else files[-1]
    return os.path.join(session_dir, ckpt)

def _best_from_part2_summary(session_num):
    """
    Use part2_summary.json -> checkpoint_eval[session_tag]['best']['checkpoint'].
    Returns absolute path to checkpoint if found, else None.
    """
    if not os.path.exists(PART2_SUMMARY):
        return None
    with open(PART2_SUMMARY, "r") as f:
        summary = json.load(f)

    tag = SESSION_TAG[session_num]
    best = summary.get("checkpoint_eval", {}).get(tag, {}).get("best")
    if not best:
        return None

    ckpt_name = best.get("checkpoint")
    if not ckpt_name:
        return None

    # Try ALT_CKPT_DIR/tag/ckpt_name (how Part-2 saved them)
    p1 = os.path.join(ALT_CKPT_DIR, tag, ckpt_name)
    if os.path.exists(p1):
        return p1

    # Try inside CHECKPOINT_DIR/session_n (in case you copied them)
    # If best checkpoint is epochNNN.keras, also look for checkpoint_NNN.h5
    session_dir = os.path.join(CHECKPOINT_DIR, f"session_{session_num}")
    p2 = os.path.join(session_dir, ckpt_name)
    if os.path.exists(p2):
        return p2

    # Heuristic: if it's "epochNNN.keras", map to "checkpoint_NNN.h5"
    import re
    m = re.search(r"epoch(\d+)\.keras$", ckpt_name or "")
    if m:
        h5_name = f"checkpoint_{int(m.group(1))}.h5"
        p3 = os.path.join(session_dir, h5_name)
        if os.path.exists(p3):
            return p3

    return None

def _best_from_eval_results(session_num):
    """
    Use evaluation_results.json -> choose epoch with highest accuracy for that session,
    then open checkpoints/session_n/checkpoint_{epoch}.h5 (or .keras if present).
    """
    if not os.path.exists(EVAL_RESULTS):
        return None
    with open(EVAL_RESULTS, "r") as f:
        res = json.load(f)

    key = str(session_num)  # saved as string keys in many dumps
    if key not in res:
        # some dumps use int keys; try direct
        if session_num not in res:
            return None
        block = res[session_num]
    else:
        block = res[key]

    epochs = block.get("epochs", [])
    accs = block.get("accuracy", [])
    if not epochs or not accs:
        return None

    best_idx = int(np.argmax(accs))
    best_epoch = int(epochs[best_idx])

    session_dir = os.path.join(CHECKPOINT_DIR, f"session_{session_num}")
    # Prefer .keras if it exists with epoch number; else .h5
    cand_keras = os.path.join(session_dir, f"epoch{best_epoch:03d}.keras")
    cand_h5    = os.path.join(session_dir, f"checkpoint_{best_epoch}.h5")
    if os.path.exists(cand_keras):
        return cand_keras
    if os.path.exists(cand_h5):
        return cand_h5
    return None

def find_best_checkpoint(session_num):
    """
    Final resolver:
      1) part2_summary.json best (lowest val MSE)
      2) evaluation_results.json best (highest accuracy)
      3) fallback: last file in checkpoints/session_n
    """
    path = _best_from_part2_summary(session_num)
    if path:
        return path
    path = _best_from_eval_results(session_num)
    if path:
        return path
    return _fallback_last_checkpoint(session_num)

def safe_labels_from_preds(preds):
    """
    Sanitize predictions:
      - replace NaN with 0.0, +inf -> 5.0, -inf -> 0.0
      - round, then clip to [0,5], return int labels
    """
    preds = np.asarray(preds, dtype=np.float32).reshape(-1)
    preds = np.nan_to_num(preds, nan=0.0, posinf=5.0, neginf=0.0)
    return np.clip(np.rint(preds), 0, 5).astype(int), preds

# ----------------------- MAIN TESTING -----------------------
def test_models():
    # Get test images
    if not os.path.exists(TEST_IMAGES_DIR):
        print(f"Test images directory '{TEST_IMAGES_DIR}' not found!")
        return
    test_images = sorted([f for f in os.listdir(TEST_IMAGES_DIR)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not test_images:
        print(f"No images found in '{TEST_IMAGES_DIR}'")
        return
    print(f"Found {len(test_images)} test images\n")

    # Preprocess ALL images once (avoids retracing)
    X = np.stack([preprocess_image(os.path.join(TEST_IMAGES_DIR, f)) for f in test_images], axis=0)

    # Load models
    session_names = {
        1: "Sigmoid + Xavier + None",
        2: "Tanh + Xavier + None",
        3: "Tanh + Xavier + L2",
        4: "Tanh + Xavier + Dropout",
        5: "ReLU + Xavier + L2",
        6: "ReLU + He + L2"
    }
    models = {}
    for session_num in range(1, 7):
        ckpt = find_best_checkpoint(session_num)
        if ckpt:
            models[session_num] = keras.models.load_model(ckpt, compile=False, safe_mode=False)
            print(f"Loaded Session {session_num}: {session_names[session_num]}  <-- {os.path.basename(ckpt)}")
        else:
            print(f"Session {session_num}: no checkpoint found")

    print("\n" + "="*80)
    print("TESTING RESULTS")
    print("="*80)

    # Predict per model (vectorized) and collect results
    all_results = []            # list of {"image": ..., "predictions": {sess: label}}
    per_session_labels = {}     # sess -> np.array of labels for all images
    per_session_raw = {}        # sess -> np.array of raw preds

    for sess, model in models.items():
        raw = model.predict(X, verbose=0).reshape(-1)
        labels, raw_clean = safe_labels_from_preds(raw)
        per_session_labels[sess] = labels
        per_session_raw[sess] = raw_clean

    # Print image-by-image summary like before
    for i, img_file in enumerate(test_images):
        print(f"\nImage: {img_file}")
        pred_map = {}
        for sess in sorted(per_session_labels.keys()):
            lbl = int(per_session_labels[sess][i])
            raw = float(per_session_raw[sess][i])
            pred_map[sess] = lbl
            print(f"  Session {sess}: {lbl} (raw: {raw:.2f})")
        all_results.append({"image": img_file, "predictions": pred_map})

    # Correctness (fill if you have GT)
    print("\n" + "="*80)
    print("CORRECTNESS ANALYSIS")
    print("="*80)
    print("\nProvide ground-truth labels to compute accuracy automatically.")
    print("You can map filenames to labels here:")
    true_labels = {
        "normalized_01.png": 4,
        "normalized_02.png": 4,
        "normalized_03.png": 1,
        "normalized_04.png": 3,
        "normalized_05.png": 2,
        "normalized_06.png": 1,
        "normalized_07.png": 2,
        "normalized_08.png": 0,
        "normalized_09.png": 5,
        "normalized_10.png": 0,
        "normalized_11.png": 4,
        "normalized_12.png": 4,
        "normalized_13.png": 1,
        "normalized_14.png": 3,
        "normalized_15.png": 2,
        "normalized_16.png": 1,
        "normalized_17.png": 2,
        "normalized_18.png": 0,
        "normalized_19.png": 5,
        "normalized_20.png": 0,
    }  # adjust to your actual files

    if true_labels:
        for sess in sorted(per_session_labels.keys()):
            ok = sum(per_session_labels[sess][i] == true_labels.get(test_images[i], -999)
                     for i in range(len(test_images)) if test_images[i] in true_labels)
            total = sum(1 for i in range(len(test_images)) if test_images[i] in true_labels)
            if total:
                print(f"Session {sess}: {ok}/{total} correct ({100.0*ok/total:.1f}%)")
    else:
        print("Ground-truth not provided; manually compare predictions with actual gestures.")

    visualize_predictions(test_images[:6], all_results[:6])

def visualize_predictions(test_images, results):
    """Show up to 6 images with per-session predicted labels."""
    if not results:
        return
    cols = min(3, len(results))
    rows = 2 if len(results) > 3 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    axes = np.array(axes).reshape(rows, cols)
    for idx, (img_file, result) in enumerate(zip(test_images, results)):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        img = Image.open(os.path.join(TEST_IMAGES_DIR, img_file)).convert('L')
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        title = img_file + "\n" + " ".join([f"S{sn}:{pl}" for sn, pl in sorted(result['predictions'].items())])
        ax.set_title(title, fontsize=9)
    plt.tight_layout()
    plt.savefig('test_predictions.png', dpi=300, bbox_inches='tight')
    print("\nPrediction visualization saved as 'test_predictions.png'")
    plt.show()

if __name__ == "__main__":
    test_models()
