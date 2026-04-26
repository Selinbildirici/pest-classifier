# Setup Instructions

This guide walks through how to set up the environment and reproduce
the experiments in this project.

## Recommended Environment

This project was developed and tested on **Google Colab Pro** with a
**T4 GPU**. The notebook can also run on a local Linux machine with
CUDA-compatible GPU and Python 3.10+.

## 1. Clone the Repository

    git clone https://github.com/Selinbildirici/pest-classifier.git
    cd pest-classifier

## 2. Install Dependencies

Install Python dependencies via pip:

    pip install -r requirements.txt

Required packages include PyTorch, torchvision, timm (for ViT/CNN
backbones), Albumentations (for data augmentation), grad-cam (for
interpretability), and kagglehub (for dataset download). See
`requirements.txt` for full versions.

## 3. Get a Kaggle API Token

The IP102 dataset is hosted on Kaggle and accessed via `kagglehub`.

1. Go to https://www.kaggle.com/settings
2. Scroll to the "API" section and click "Create New Token"
3. This downloads `kaggle.json` to your computer
4. Place it at `~/.kaggle/kaggle.json` (or `/root/.config/kaggle/kaggle.json` on Linux/Colab)
5. Restrict permissions: `chmod 600 ~/.kaggle/kaggle.json`

## 4. Download the IP102 Dataset

In a Python script or notebook:

    import kagglehub
    path = kagglehub.dataset_download("rtlmhjbn/ip02-dataset")
    print("Dataset downloaded to:", path)

The dataset is ~5–8 GB and contains 75,222 images across 102 insect
species.

## 5. (Optional) Download Pre-trained Checkpoints

The 8 trained model checkpoints (343 MB each) are not committed to
this repository due to GitHub's 100 MB file size limit. To request
access to the checkpoints on Google Drive, contact the project author.

After downloading, place `.pt` files in the `models/` directory.

## 6. Running the Notebook

The main entry point is `notebooks/pest_classifier.ipynb`. To run on
Google Colab:

1. Upload the notebook to Colab
2. Mount Google Drive (the notebook expects `/content/drive/MyDrive/cs372_project/`)
3. Run cells top to bottom

For local execution, modify the Drive paths in the notebook to point
to local directories.

## 7. Reproducing Specific Results

Different sections of the notebook correspond to different experiments:

- **Sections 5–11:** Run 1–4 (ablation study)
- **Sections 13–14:** Run 5–6 (architecture comparison: ResNet, EfficientNet)
- **Section 14:** Run 7 (HP tuning, LR=1e-5)
- **Section 16–22:** Test-set evaluation, confusion matrix, training curves
- **Sections 25–26:** Misclassified gallery, side-by-side comparison
- **Section 27–28:** Grad-CAM heatmaps
- **Section 23:** Ensemble method

(Section numbers correspond to markdown headers in the cleaned notebook.)

## Hardware Requirements

- **Recommended:** NVIDIA T4 GPU (16 GB VRAM) or better
- **Minimum:** GPU with 8 GB VRAM (will require batch size adjustments)
- **CPU-only inference is supported but training is impractically slow**

## Common Issues

- **CUDA out of memory:** Reduce `BATCH_SIZE` in training cells (default 16 for ViT, 32 for CNNs)
- **Disconnected from Colab during training:** Checkpoints save every epoch to Drive; resume from the last checkpoint
- **Drive I/O is slow:** The notebook copies the dataset to local disk for ~4× faster training
