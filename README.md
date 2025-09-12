# Microglia Morphological Profiling

This repository contains the code related to the paper **"Mapping human microglial morphological diversity via handcrafted and deep learning-derived image features"**.

## Summary

In this mansucript we describe a scalable method to analyze the activation states of human microglia derived from iPSCs using high-content imaging. By combining targeted immunofluorescence and broad morphological staining (Cell Painting), we captured detailed image-based features from thousands of individual cells. These features were analyzed using both classical and deep learning approaches to identify distinct microglial phenotypes. To better reflect the continuous nature of microglial activation, we applied a probabilistic clustering method (Gaussian Mixture Models) that allows cells to be partially assigned to multiple states. This approach revealed a rich landscape of microglial morphologies and their transitions in response to inflammatory stimuli. Our framework enables sensitive detection of microglial heterogeneity and could support future efforts in drug discovery and disease modeling.

## Folder Structure

- **Data/**: Contains preprocessed datasets used in the study, organized into subdirectories for different datasets (e.g., CP_Cellprofiler, CP_deep, IF_HC).
- **Exp3-5-6_classifier/**: Includes code and configuration files for the classification stage of the study, focusing on predictive performance of different datasets for classifying microglial states.
- **Exp3-5-6_subpopulation_detection/**: Contains code and configurations for detecting different activation states of microglia cells using various datasets.
- **Figures/**: Stores data and plotting notebooks used to generate the figures presented in the paper, organized by figure number.



## Reproducing Results

To reproduce the results mentioned in the paper, you can run the following Python pipelines:

- **classification**: run `pipeline_profiling.py` with the configuration files in Exp3-5-6_classifier, then results are compared in `dataset_compare.ipynb` and `dataset_compare_leiden.ipynb`

- **clustering microglial states**: run `pipeline_clustering.py` with the configuration files in Exp3-5-6_subpopulation_detection

These scripts will use the configuration files (e.g., `conf_{dataset}_{method}.json`) to generate results. Running these pipelines will create a folder with all the results described in the paper.

**Note**: `"save_crops": false` need to be set in configuration files, since the raw images are not available in this repository.

for example:

```bash
python pipeline_clustering.py conf_CP_Cellprofiler.json
```

## Requirements

The required dependencies are listed in `requirements.txt`. To install them, use:
```
pip install -r requirements.txt
