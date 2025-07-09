# common_imports.py

import os
import argparse
import threading
from io import StringIO
import itertools

from matplotlib.cm import ScalarMappable 


import h5py
import sklearn
import numpy as np
import seaborn as sn
import pandas as pd
from tqdm import tqdm
from sklearn.utils import class_weight

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from matplotlib import colormaps
import matplotlib.colors as mcolors
import matplotlib.patches as patches  

from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backends.backend_pdf import PdfPages

from PIL import Image, ImageTk
from scipy.spatial import cKDTree


import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision
from tensorflow.keras.layers import ( Input, Conv2D,  MaxPooling2D, Dropout, Flatten, GlobalAveragePooling2D, Dense, concatenate, BatchNormalization , Concatenate , InputLayer )
from tensorflow.keras.models import Model 
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.preprocessing.image import img_to_array




from cycler import cycler

__all__ = [
    "os",
    "argparse",
    "threading",
    "class_weight",
    "cKDTree",
    "h5py",
    "np",
    "pd",
    "sn",
    "sklearn",
    "StringIO",
    "itertools",
    "tk",
    "ttk",
    "messagebox",
    "filedialog",
    "matplotlib",
    "plt",
    "cm",
    "mcolors",
    "colormaps",
    "FigureCanvasTkAgg",
    "NavigationToolbar2Tk",
    "Poly3DCollection",
    "FuncAnimation",
    "PdfPages",
    "Image",
    "ImageTk",
    "tf",
    "tqdm",
    "Adam",
    "Precision",
    "Input",
    "InputLayer",
    "Conv2D",
    "Concatenate",
    "MaxPooling2D",
    "Dropout",
    "ScalarMappable",
    "Flatten",
    "GlobalAveragePooling2D",
    "Dense",
    "concatenate",
    "BatchNormalization",
    "img_to_array",
    "Model",
    "keras_load_model",
    "patches",
    "Patch",
    "cycler",
]
