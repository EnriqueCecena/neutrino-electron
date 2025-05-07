# Dune Experiment: Neutrino-Electron Interaction Events Classification Using CNNs

### Jordan Ellis (Supervised by Abigail Waldron)
#### Queen Mary University London

---

## Overview

This repository contains scripts designed to visualize simulated DUNE datasets, generate plots, and create datasets for training and evaluating Convolutional Neural Network (CNN) machine learning models. The datasets are sourced from the [DUNE/2x2_sim Wiki - File Data Definitions](https://github.com/DUNE/2x2_sim/wiki/File-Data-Definitions), which outlines various datasets produced by the DUNE 2x2 simulation project. These datasets are crucial for simulating and analyzing particle interactions within the DUNE detector.

## Features

- **Dataset Visualization:** Explore `segments` , `md_hdr`, and `trajectory`  datasets in a user-friendly dataframe format.
- **Plot Generation:** Create scatter plots, line plots, histograms, and custom visualizations for basic data analysis.
- **Dataset Creation:** Preview interaction vertices and compile datasets of all interaction vertices from selected files, organizing them into separate directories.

## Requirements

Ensure you have the following Python libraries installed:

- `os`
- `argparse`
- `numpy`
- `pandas`
- `h5py`
- `scipy`
- `seaborn`
- `sklearn`
- `tensorflow`
- `tkinter`
- `PIL`
- `tqdm`
- `matplotlib`
- `threading`



Ensure you have the following scripts in the same directory as "Main_Run_File.py":

- `pdg_id_script.py`
- `Generic_Plot_script.py`
- `Pixel_Array_Script.py`
- `Generic_Plot_script.py`
- `Custom_Plot_script.py`
- `Model_Training_script.py`
- `Model_Evaluation_script.py`
- 

You can install the required libraries using `pip`:

```bash
pip install h5py argparse numpy pandas tkinter matplotlib ...

```

To execute the code, provide the path to the data directory as an argument when running the Python script. For example:

```

python Main_Run_File.py 
                         

```


