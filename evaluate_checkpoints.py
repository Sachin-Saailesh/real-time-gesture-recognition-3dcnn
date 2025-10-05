import os
import numpy as np
import json
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array

DATASET_PATH = "custom_dataset"
CHECKPOINT_DIR = "checkpoints"
IMG_SIZE = (128, 128)

def load_dataset():
    """Load the same dataset used for training"""
    images = []
    labels = []
    
    for label_folder in sorted(os.listdir(DATASET_PATH)):
        label_path = os.path.join(DATASET_PATH, label_folder)
        if os.path.isdir(label_path):
            label = int(label_folder)
            
            for img_file in os.listdir(label_path):
                if img_file.endswith('.png'):
                    img_path = os.path.join(label_path, img_file)
                    img = load_img(img_path, color_mode='grayscale', target_size=IMG_SIZE)
                    img_array = img_to_array(img) / 255.0
                    images.append(img_array.flatten())
                    labels.append(label)
    
    return np.array(images), np.array(labels)

def evaluate_checkpoint(model_path, X_test, y_test):
    """Evaluate a single checkpoint (no compile on load)."""
    # Avoid compile so Keras doesn't try to resolve 'mse'/'mae' strings from the saved config
    model = keras.models.load_model(
        model_path,
        compile=False,                    # <-- key change
        custom_objects={}                 # add {'round_acc': round_acc} if you used it during save
    )

    # Forward pass
    preds = model.predict(X_test, verbose=0).reshape(-1)

    # Round to nearest integer in [0,5]
    preds_rounded = np.clip(np.rint(preds), 0, 5).astype(int)
    y_int = y_test.astype(int)

    # Metrics (manual)
    accuracy = float(np.mean(preds_rounded == y_int))
    mse      = float(np.mean((preds - y_test)**2))
    mae      = float(np.mean(np.abs(preds - y_test)))
    return accuracy, mse, mae


def evaluate_session(session_num, X_test, y_test):
    """Evaluate all checkpoints for a session"""
    session_dir = os.path.join(CHECKPOINT_DIR, f"session_{session_num}")
    
    if not os.path.exists(session_dir):
        print(f"Session {session_num} directory not found!")
        return None
    
    # Get all checkpoint files
    checkpoints = sorted([f for f in os.listdir(session_dir) if f.startswith('checkpoint_')])
    
    results = {
        'epochs': [],
        'accuracy': [],
        'mse': [],
        'mae': []
    }
    
    print(f"\nEvaluating Session {session_num}...")
    
    for checkpoint in checkpoints:
        # Extract epoch number
        epoch = int(checkpoint.split('_')[-1].replace('.h5', ''))
        model_path = os.path.join(session_dir, checkpoint)
        
        accuracy, mse, mae = evaluate_checkpoint(model_path, X_test, y_test)
        
        results['epochs'].append(epoch)
        results['accuracy'].append(accuracy)
        results['mse'].append(mse)
        results['mae'].append(mae)
        
        print(f"  Epoch {epoch}: Accuracy={accuracy:.4f}, MSE={mse:.4f}, MAE={mae:.4f}")
    
    # Find best checkpoint (highest accuracy)
    best_idx = np.argmax(results['accuracy'])
    best_epoch = results['epochs'][best_idx]
    best_acc = results['accuracy'][best_idx]
    
    print(f"  Best checkpoint: Epoch {best_epoch} with accuracy {best_acc:.4f}")
    
    return results

def plot_convergence_histories(all_results):
    """Plot convergence histories for all sessions"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Loss (MSE)
    ax1 = axes[0, 0]
    for session_num, results in all_results.items():
        ax1.plot(results['epochs'], results['mse'], marker='o', label=f'Session {session_num}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Convergence History - MSE Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: MAE
    ax2 = axes[0, 1]
    for session_num, results in all_results.items():
        ax2.plot(results['epochs'], results['mae'], marker='s', label=f'Session {session_num}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.set_title('Convergence History - MAE')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Accuracy
    ax3 = axes[1, 0]
    for session_num, results in all_results.items():
        ax3.plot(results['epochs'], results['accuracy'], marker='^', label=f'Session {session_num}')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Convergence History - Accuracy')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Summary comparison (final epoch performance)
    ax4 = axes[1, 1]
    sessions = list(all_results.keys())
    final_accuracies = [all_results[s]['accuracy'][-1] for s in sessions]
    ax4.bar(sessions, final_accuracies)
    ax4.set_xlabel('Session')
    ax4.set_ylabel('Final Accuracy')
    ax4.set_title('Final Performance Comparison')
    ax4.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('convergence_history.png', dpi=300, bbox_inches='tight')
    print("\nConvergence history plot saved as 'convergence_history.png'")
    plt.show()

def analyze_overfitting(all_results):
    """Analyze overfitting/underfitting for each session"""
    print("\n" + "="*60)
    print("OVERFITTING/UNDERFITTING ANALYSIS")
    print("="*60)
    
    for session_num, results in all_results.items():
        print(f"\nSession {session_num}:")
        
        # Check if accuracy improves then degrades (overfitting)
        accuracies = results['accuracy']
        max_acc = max(accuracies)
        max_idx = accuracies.index(max_acc)
        final_acc = accuracies[-1]
        
        # Check trend
        if max_idx < len(accuracies) - 5:  # Peak is not near the end
            if final_acc < max_acc - 0.05:
                print(f"  ⚠ OVERFITTING detected: Accuracy peaked at epoch {results['epochs'][max_idx]} "
                      f"({max_acc:.4f}) then decreased to {final_acc:.4f}")
            else:
                print(f"  ✓ Stable training: Peak accuracy {max_acc:.4f} maintained")
        else:
            if final_acc < 0.7:
                print(f"  ⚠ UNDERFITTING suspected: Final accuracy only {final_acc:.4f}")
            else:
                print(f"  ✓ Good convergence: Final accuracy {final_acc:.4f}")
        
        print(f"  Recommended checkpoint: Epoch {results['epochs'][max_idx]} "
              f"(Accuracy: {max_acc:.4f})")

def main():
    print("Loading dataset for evaluation...")
    X_test, y_test = load_dataset()
    print(f"Dataset loaded: {X_test.shape[0]} images\n")
    
    # Evaluate all sessions
    all_results = {}
    for session_num in range(1, 7):
        results = evaluate_session(session_num, X_test, y_test)
        if results:
            all_results[session_num] = results
    
    # Plot convergence histories
    if all_results:
        plot_convergence_histories(all_results)
        analyze_overfitting(all_results)
        
        # Save results to JSON
        results_file = 'evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump({k: v for k, v in all_results.items()}, f, indent=2)
        print(f"\nResults saved to '{results_file}'")

if __name__ == "__main__":
    main()
