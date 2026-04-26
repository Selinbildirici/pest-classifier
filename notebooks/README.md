# Notebooks

This directory contains the Jupyter notebook used for training, evaluation, and analysis.

## Files

- pest_classifier.ipynb — Main Colab notebook containing all 8 training runs, error analysis, Grad-CAM generation, and ensemble experiments.

## How to Run

The notebook was developed and tested on Google Colab Pro with a T4 GPU. To reproduce the results:

1. Upload pest_classifier.ipynb to Google Colab
2. Mount Google Drive (the notebook expects /content/drive/MyDrive/cs372_project/)
3. Run cells in order

For local execution, see SETUP.md for environment setup.

## Cell Organization

The notebook is organized by training day (Day 1 = setup, Day 2 = ablation study, Day 3 = architecture comparison + HP tuning, Day 4 = analysis).
