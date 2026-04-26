# Trained Models

This directory describes the trained model checkpoints. Due to GitHub's file size limit (100 MB) and the size of trained ViT-Base/16 weights (~343 MB each), the actual .pt checkpoint files are not committed to this repository.

## Available Checkpoints

| Run | Configuration | Best Val Acc | Test Acc |
|-----|---------------|--------------|----------|
| Run 1 | Frozen ViT (baseline) | 0.646 | — |
| Run 2 | Unfrozen ViT | 0.736 | — |
| Run 3 | Unfrozen + augmentation (REFERENCE) | 0.749 | 0.7452 |
| Run 4 | + class weights | 0.746 | — |
| Run 5 | ResNet-50 | 0.643 | 0.6423 |
| Run 6 | EfficientNet-B0 | 0.675 | 0.6720 |
| Run 7 | ViT LR=1e-5 (HP-tuned best) | 0.751 | 0.7476 |
| Run 8 | ViT LR=1e-4 | 0.719 | — |

## How to Access Checkpoints

Checkpoints are stored on Google Drive. To request access, contact the project author. After downloading, place .pt files in this directory.

Once placed in this directory, checkpoints can be loaded with:

    import timm, torch
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=102)
    model.load_state_dict(torch.load('models/run3_aug_vit_best.pt'))
    model.eval()

## Reference Model

For all error analysis, confusion matrix, and Grad-CAM analyses, Run 3 (run3_aug_vit_best.pt) is the reference model. Run 7 had the highest test accuracy (74.76% vs 74.52%), but the 0.2% gap is within run-to-run variance for unseeded experiments.
