"""
Softmax-averaging ensemble across multiple architectures.

Mirrors the ensemble logic used in the project notebook
(notebooks/pest_classifier.ipynb cell 7?), organized as reusable
functions.

Combines predictions from ViT-Base/16, ResNet-50, and EfficientNet-B0
using either uniform averaging, validation-accuracy-weighted averaging,
or ViT-heavy weighting.

"""

import numpy as np


def uniform_ensemble(softmax_outputs):
    """Average softmax probabilities equally across all models.

    Args:
        softmax_outputs: list of (N, num_classes) arrays, one per model

    Returns:
        (N, num_classes) ensemble softmax probabilities
    """
    return np.mean(softmax_outputs, axis=0)


def weighted_ensemble(softmax_outputs, weights):
    """Weighted average of softmax probabilities.

    Args:
        softmax_outputs: list of (N, num_classes) arrays
        weights: list of floats (will be normalized to sum to 1)

    Returns:
        (N, num_classes) ensemble softmax probabilities
    """
    weights = np.array(weights, dtype=np.float64)
    weights = weights / weights.sum()
    stacked = np.stack(softmax_outputs, axis=0)
    return np.sum(stacked * weights[:, None, None], axis=0)


def evaluate_ensemble(ensemble_softmax, labels):
    """Compute top-1 accuracy of an ensemble's predictions.

    Args:
        ensemble_softmax: (N, num_classes) ensemble probabilities
        labels: (N,) ground truth labels

    Returns:
        float accuracy in [0, 1]
    """
    preds = ensemble_softmax.argmax(axis=1)
    return float((preds == labels).mean())


def compare_ensemble_strategies(sm_vit, sm_resnet, sm_effnet, labels,
                                  val_acc_vit=0.749, val_acc_resnet=0.643,
                                  val_acc_effnet=0.675):
    """Compare three ensemble strategies on the test set.

    The three strategies are:
        1. Uniform: equal weights for all 3 models
        2. Val-weighted: weighted by individual val accuracy
        3. ViT-heavy: 2x weight for ViT, 1x for others

    Default val accuracies match Run 3 (ViT), Run 5 (ResNet),
    and Run 6 (EfficientNet) from the project.

    Args:
        sm_vit, sm_resnet, sm_effnet: (N, num_classes) softmax arrays
        labels: (N,) ground truth labels
        val_acc_vit, val_acc_resnet, val_acc_effnet: validation accuracies for weighting

    Returns:
        dict mapping strategy name to (accuracy, ensemble_softmax) tuple
    """
    # Strategy 1: Uniform averaging
    sm_uniform = uniform_ensemble([sm_vit, sm_resnet, sm_effnet])
    acc_uniform = evaluate_ensemble(sm_uniform, labels)

    # Strategy 2: Weighted by individual val accuracy
    sm_weighted = weighted_ensemble(
        [sm_vit, sm_resnet, sm_effnet],
        [val_acc_vit, val_acc_resnet, val_acc_effnet],
    )
    acc_weighted = evaluate_ensemble(sm_weighted, labels)

    # Strategy 3: ViT-heavy
    sm_vit_heavy = weighted_ensemble(
        [sm_vit, sm_resnet, sm_effnet],
        [2.0, 1.0, 1.0],
    )
    acc_vit_heavy = evaluate_ensemble(sm_vit_heavy, labels)

    return {
        'uniform': (acc_uniform, sm_uniform),
        'weighted': (acc_weighted, sm_weighted),
        'vit_heavy': (acc_vit_heavy, sm_vit_heavy),
    }
