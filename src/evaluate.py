"""
Test-set evaluation: inference, confusion matrix, per-class F1, error analysis.

Mirrors the evaluation logic used in the project notebook's Day 4 analysis
section (notebooks/pest_classifier.ipynb), organized as reusable functions.

"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from tqdm import tqdm


@torch.no_grad()
def get_softmax_outputs(model, loader, device, save_labels=False):
    """Run model over a DataLoader and return softmax probabilities for all samples.

    Args:
        model: PyTorch model (will set to eval mode)
        loader: DataLoader yielding (images, labels) tuples
        device: torch.device
        save_labels: if True, also return labels array (used once per session)

    Returns:
        all_probs: (N, num_classes) numpy array of softmax probabilities
        all_labels: (N,) numpy array of integer class labels (or None if save_labels=False)
    """
    model.eval()
    all_probs = []
    labels_list = []
    with torch.amp.autocast('cuda'):
        for images, labels in tqdm(loader, desc="Inferring"):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = F.softmax(logits.float(), dim=1).cpu().numpy()
            all_probs.append(probs)
            if save_labels:
                labels_list.extend(labels.numpy().tolist())
    return np.concatenate(all_probs, axis=0), np.array(labels_list) if save_labels else None


def compute_per_class_f1(test_labels, predictions, num_classes=102):
    """Compute per-class F1 scores for the test set.

    Returns a (num_classes,) array of F1 scores (0 to 1 per class).
    """
    return f1_score(test_labels, predictions, labels=list(range(num_classes)),
                    average=None, zero_division=0)


def compute_confusion_matrix(test_labels, predictions, num_classes=102):
    """Full confusion matrix (rows=true class, cols=predicted class)."""
    return confusion_matrix(test_labels, predictions, labels=list(range(num_classes)))


def find_worst_classes(per_class_f1, n=10):
    """Return indices of the n worst-performing classes by F1."""
    return np.argsort(per_class_f1)[:n]


def find_best_classes(per_class_f1, n=10):
    """Return indices of the n best-performing classes by F1."""
    return np.argsort(per_class_f1)[::-1][:n]


def find_confusion_pairs_for_class(cm, true_class, top_n=3):
    """For a given true class, find the top-N most-frequent confusions.

    Returns a list of (predicted_class, count) tuples sorted by count
    descending, excluding the true class itself.
    """
    row = cm[true_class].copy()
    row[true_class] = 0  # zero out diagonal (correct predictions)
    top_indices = np.argsort(row)[::-1][:top_n]
    return [(int(idx), int(row[idx])) for idx in top_indices if row[idx] > 0]


def aggregate_to_superclass(predictions, model_idx_to_superclass):
    """Map species-level predictions to crop-superclass-level (8 categories).

    The 8 IP102 crop superclasses are: Rice, Corn, Wheat, Beet, Alfalfa,
    Vitis, Citrus, Mango.

    Args:
        predictions: (N,) array of model class indices (0-101)
        model_idx_to_superclass: (102,) array mapping each model_idx to a superclass (0-7)

    Returns:
        (N,) array of superclass indices (0-7)
    """
    return model_idx_to_superclass[predictions]


def build_superclass_mapping(folder_to_model_idx, classes_path,
                              superclass_ranges=None):
    """Build mapping from model_idx to crop superclass index.

    The 8 IP102 superclasses are defined by ranges of class IDs in
    classes.txt (1-indexed). This function maps each model_idx (0-101)
    to its superclass (0-7).

    Args:
        folder_to_model_idx: dict mapping folder name (str) to model_idx (int)
        classes_path: path to IP102 classes.txt file
        superclass_ranges: optional dict of {superclass_name: (start_id, end_id)}.
                          Defaults to standard IP102 8 crop superclasses.

    Returns:
        (numpy array of model_idx -> superclass_idx, list of superclass names)
    """
    if superclass_ranges is None:
        superclass_ranges = {
            'Rice':    (1, 14),
            'Corn':    (15, 24),
            'Wheat':   (25, 37),
            'Beet':    (38, 49),
            'Alfalfa': (50, 62),
            'Vitis':   (63, 79),
            'Citrus':  (80, 91),
            'Mango':   (92, 102),
        }

    superclass_names = list(superclass_ranges.keys())
    model_idx_to_superclass = np.zeros(102, dtype=int)
    for sc_idx, (sc_name, (start, end)) in enumerate(superclass_ranges.items()):
        for cid in range(start, end + 1):
            folder_name = str(cid - 1)  # +1 offset between folder names and classes.txt IDs
            if folder_name in folder_to_model_idx:
                m_idx = folder_to_model_idx[folder_name]
                model_idx_to_superclass[m_idx] = sc_idx

    return model_idx_to_superclass, superclass_names
