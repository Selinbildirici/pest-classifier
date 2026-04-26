"""
Grad-CAM heatmap generation for Vision Transformer.

Mirrors the Grad-CAM logic used in the project notebook's
interpretability analysis section, organized as reusable functions.

Uses pytorch-grad-cam with a custom reshape transform to handle ViT's
patch-based attention. Generates heatmaps showing which image regions
the model attended to when making each prediction.

"""

import numpy as np
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ImageNet normalization stats (must match training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Image preprocessing transform for Grad-CAM input
gradcam_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


def vit_reshape_transform(tensor, height=14, width=14):
    """Convert ViT patch tokens to 2D feature map for Grad-CAM.

    ViT outputs shape (batch, num_patches+1, embed_dim). We drop the CLS
    token (first one) and reshape the remaining 196 patch tokens into a
    14x14 spatial grid that Grad-CAM can visualize as a heatmap.

    For ViT-Base/16 with 224x224 input: 14x14 = 196 patches of size 16x16.
    """
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def make_gradcam_for_vit(model):
    """Construct a Grad-CAM object targeting the last attention block of ViT-Base/16.

    The last attention block (model.blocks[-1].norm1) is the standard
    target layer for ViT Grad-CAM.
    """
    target_layers = [model.blocks[-1].norm1]
    return GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=vit_reshape_transform,
    )


def compute_heatmap_for_image(cam, image_path, target_class, device):
    """Generate a Grad-CAM heatmap overlaid on an input image.

    Args:
        cam: GradCAM object from make_gradcam_for_vit()
        image_path: path to image file
        target_class: class index whose contribution we want to visualize
        device: torch.device

    Returns:
        numpy array of shape (224, 224, 3) showing image + heatmap overlay
    """
    pil_img = Image.open(image_path).convert('RGB').resize((224, 224))
    rgb_img = np.array(pil_img).astype(np.float32) / 255.0

    img_arr = np.array(pil_img)
    transformed = gradcam_transform(image=img_arr)['image'].unsqueeze(0).to(device)

    targets = [ClassifierOutputTarget(int(target_class))]
    grayscale_cam = cam(input_tensor=transformed, targets=targets)[0]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return visualization
