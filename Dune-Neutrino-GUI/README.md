# Dune Experiment: Neutrino-Electron Interaction Events Classification Using CNNs

### Jordan Ellis (Supervised by Abigail Waldron)
#### Queen Mary University London

---

## Overview

This repository contains scripts designed to visualize simulated DUNE datasets, generate plots, and create datasets for training and evaluating Convolutional Neural Network (CNN) machine learning models. The datasets are sourced from the DUNE/2x2_sim Wiki - File Data Definitions (https://github.com/DUNE/2x2_sim/wiki/File-Data-Definitions), which outlines various datasets produced by the DUNE 2x2 simulation project. These datasets are crucial for simulating and analyzing particle interactions within the DUNE detector.

## Features

- **Dataset Visualization:** Explore `segments`, `md_hdr`, and `trajectory` datasets in a user-friendly dataframe format.
- **Plot Generation:** Create scatter plots, line plots, histograms, and custom visualizations for basic data analysis.
- **Dataset Creation:** Preview interaction vertices and compile datasets of all interaction vertices from selected files, organizing them into separate directories.
- **Model Creation & Evaluation:** Create tensoflow models that can be trained to classify different neutrino interactions with performance metrics.


## Requirements

Ensure you have the following Python libraries installed:

- `os`
- `argparse`
- `threading`
- `io`
- `itertools`
- `matplotlib`
- `h5py`
- `sklearn`
- `numpy`
- `seaborn`
- `pandas`
- `tqdm`
- `tkinter`
- `PIL`
- `scipy`
- `tensorflow`
- `cycler`

You can install the required libraries using `pip`:

```bash
pip install os argparse threading matplotlib h5py scikit-learn numpy seaborn pandas tqdm tk pillow scipy tensorflow cycler

```


