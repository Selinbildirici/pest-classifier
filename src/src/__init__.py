"""Pest Classifier source code modules.

Modules:
    dataset: PestDataset class and augmentation pipelines
    train: Training loop with mixed precision and checkpointing
    evaluate: Test inference, confusion matrix, per-class F1
    gradcam: Grad-CAM heatmap generation for ViT
    ensemble: Softmax-averaging ensemble across architectures
"""
