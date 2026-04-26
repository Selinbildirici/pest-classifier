# Data

This project uses the IP102 insect pest dataset (Wu et al., CVPR 2019).

## Dataset Statistics

- Total images: 75,222
- Classes: 102 insect species
- Splits: 45,095 train / 7,508 val / 22,619 test
- Crop superclasses: 8 (Rice, Corn, Wheat, Beet, Alfalfa, Vitis, Citrus, Mango)
- Size on disk: ~5–8 GB

## How to Download

The dataset is hosted on Kaggle. To download it programmatically, use kagglehub:

    import kagglehub
    path = kagglehub.dataset_download("rtlmhjbn/ip02-dataset")
    print("Dataset downloaded to:", path)

This requires a Kaggle account and kagglehub installed (see requirements.txt).

## License

IP102 is free for academic use. See the original paper for citation:

Wu et al. "IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition." CVPR 2019.

## Data is not committed to this repo

The full dataset (~5–8 GB) is not included in this repository. Use the
download command above or follow SETUP.md for full setup instructions.
