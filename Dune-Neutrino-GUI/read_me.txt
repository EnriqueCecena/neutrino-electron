Dune Experiment Data Analysis App

Created By 	:	Jordan Ellis 
Supervised By	:	Abigail Waldron

(Queen Mary University London) 

 

Introduction

The Dune Experiment Data Analysis App is a graphical user interface (GUI) tool developed using Python’s Tkinter library. It is designed to facilitate the analysis of simulation data from the DUNE (Deep Underground Neutrino Experiment) project. The application allows users to view datasets, create machine learning datasets, visualize data through various plots, and manage settings related to data processing.

Features

	•	Dataset Viewing: Navigate through different datasets and inspect detailed segments and metadata.
	•	Machine Learning Dataset Creation: Generate datasets suitable for training and evaluating machine learning models.
	•	Interactive Plotting: Create scatter plots, line plots, and histograms with customizable settings.
	•	File Management: Select and manage data files efficiently through an intuitive interface.
	•	Progress Tracking: Monitor the progress of dataset creation and plotting tasks.
	•	Settings Configuration: Adjust application settings to suit different analysis requirements.

Installation

Prerequisites

	•	Python 3.7 or higher: Ensure Python is installed on your system. You can download it from python.org.


Install Dependencies

It’s recommended to use a virtual environment to manage dependencies.

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install required packages

Install the dependencies manually:

pip install h5py argparse numpy pandas matplotlib tkinter

Note: Tkinter usually comes pre-installed with Python. If not, refer to Tkinter Installation for your operating system.

Usage

Command-Line Arguments

The application accepts the following command-line arguments:
	•	-f, --Data_Directory (Required): Path to the directory containing simulation data files.


Running the Application

Navigate to the project directory and execute the script with the required arguments.

python Main_Run_File.py -f path/to/data_directory

Example:

python Main_Run_File.py -f ./data 

Application Interface

Upon launching, the application presents a multi-page interface with the following sections:
	1.	Start Page
	•	Navigate to view datasets, plot creation, dataset creation, machine learning model training, and settings.
	2.	Dataset View Page
	•	Access different views of the datasets, such as segments and metadata.
	3.	View Segments Page
	•	Select and inspect individual segments within datasets.
	4.	View mc_hdr Page
	•	(Currently Disabled) View metadata headers.
	5.	Dataset Page
	•	Options to create or load datasets.
	6.	Create Dataset Page
	•	Generate machine learning datasets with progress tracking and preview functionality.
	7.	Plot Selection Page
	•	Choose the type of plot to create: Scatter, Line, or Histogram.
	8.	Figure Creation Page
	•	Customize and generate plots based on selected parameters.
	9.	Settings Page
	•	Manage file selections and other application settings.
	10.	File Selection Page
	•	Select or deselect files from the data directory for analysis.

Navigation Tips:
	•	Use the “Back” buttons to return to previous sections.
	•	Utilize the “Select All” and “Deselect All” buttons in the File Selection Page for efficient file management.
	•	Monitor progress bars during dataset creation for real-time updates.

Dependencies

The application relies on the following Python libraries:
	•	os: Interacting with the operating system.
	•	h5py: Handling HDF5 file formats.
	•	argparse: Parsing command-line arguments.
	•	numpy: Numerical operations.
	•	pandas: Data manipulation and analysis.
	•	tkinter: Building the GUI.
	•	matplotlib: Plotting and visualization.
	•	ttk (from tkinter): Themed widgets for Tkinter.

Ensure all dependencies are installed via pip as shown in the Installation section.


Note: Ensure that pdg_id_script.py is present in the same directory as Main_Run_File.py for the application to function correctly.

Contributing

Contributions are welcome! Please follow these steps:
	1.	Fork the repository.
	2.	Create a new branch for your feature or bugfix.
	3.	Commit your changes with clear and descriptive messages.
	4.	Push your branch to your forked repository.
	5.	Open a pull request detailing your changes.

For major changes, please open an issue first to discuss your ideas.
