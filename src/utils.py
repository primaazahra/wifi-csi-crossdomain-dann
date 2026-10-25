import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import numpy as np

def plot_curves(history, title, save_path):
    plt.figure(figsize=(8, 4))
    plt.plot(history['train_loss'], label='Train Loss')
    if 'val_acc' in history:
        plt.plot(history['val_acc'], label='Source Val Acc')
    if 'target_acc' in history:
        plt.plot(history['target_acc'], label='Target Acc')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_cm(y_true, y_pred, names, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=names, yticklabels=names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_tsne(z_src, z_tgt, title, save_path):
    if len(z_src) == 0 or len(z_tgt) == 0:
        return
    n_src = len(z_src)
    combined = np.vstack([z_src, z_tgt])
    labels = np.array([0] * n_src + [1] * len(z_tgt))
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(combined)-1))
    embedded = tsne.fit_transform(combined)
    
    plt.figure(figsize=(6, 5))
    plt.scatter(embedded[labels==0, 0], embedded[labels==0, 1], label='Source', alpha=0.6, s=20)
    plt.scatter(embedded[labels==1, 0], embedded[labels==1, 1], label='Target', alpha=0.6, s=20)
    plt.title(title)
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
