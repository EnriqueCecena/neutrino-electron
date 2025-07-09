Dune Experiment Data Analysis App

Created By: Jordan Ellis
Supervised By: Abigail Waldron
(Queen Mary University London)

Introduction:

The Dune Experiment Data Analysis App is a graphical user interface (GUI) tool developed using Python’s Tkinter library. It is designed to facilitate the analysis of simulation data from the DUNE (Deep Underground Neutrino Experiment) project. The application allows users to view datasets, create machine learning datasets, visualize data through various plots (including custom plots), and manage settings related to data processing.

Features:

	•	Dataset Viewing: Navigate through different datasets and inspect detailed segments, metadata, and trajectories.
	•	Machine Learning Dataset Creation: Generate datasets suitable for training and evaluating machine learning models, with progress tracking and preview functionality.
	•	Interactive Plotting: Create scatter plots (2D & 3D), line plots, histograms, and custom plots with customizable settings.
	•	Custom Plotting: Generate custom plots based on predefined templates or user-defined functions.
	•	File Management: Select and manage data files efficiently through an intuitive interface.
	•	Progress Tracking: Monitor the progress of dataset creation and plotting tasks with responsive progress bars.
	•	Settings Configuration: Adjust application settings to suit different analysis requirements, including file selections and application parameters.

Installation:

Prerequisites:

	•	Python 3.7 or higher: Ensure Python is installed on your system. Download it from python.org.

Install Dependencies:

It’s recommended to use a virtual environment to manage dependencies.

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install required packages
pip install h5py argparse numpy pandas matplotlib tkinter

Note: Tkinter usually comes pre-installed with Python. If not, refer to Tkinter installation guides for your operating system.

Usage

Command-Line Arguments


Running the Application

Navigate to the project directory and execute the script with the required arguments:

python main.py 


Application Interface:

Upon launching, the application presents a multi-page interface with the following sections:
	1.	Start Page
	•	Navigate to view datasets, plot creation, dataset creation, machine learning model training, and settings.
	2.	Dataset View Page
	•	Access different views of the datasets, such as segments, metadata (mc_hdr), and trajectories.
	3.	View Segments Page
	•	Select and inspect individual segments within datasets. Navigation buttons (Next, Back) allow for seamless exploration of events.
	4.	View mc_hdr Page
	•	View metadata headers for events.
	5.	View Trajectories Page
	•	View trajectory data associated with events.
	6.	Dataset Page
	•	Options to create or load datasets.
	7.	Create Dataset Page
	•	Generate machine learning datasets with progress tracking and preview functionality. Custom directories are created for different interaction types.
	8.	Plot Selection Page
	•	Choose the type of plot to create: Scatter (2D or 3D), Line, Histogram, or Custom Plot.
	9.	Figure Creation Page
	•	Customize and generate plots based on selected parameters, including:
	•	Axis configuration
	•	Color mapping (cmap) options
	•	3D toggle for scatter plots
	•	Grouping options for histograms
	10.	Custom Figure Page
	•	Generate custom plots based on predefined templates or user-defined functions. Includes functionality for downloading all generated plots.
	11.	Settings Page
	•	Manage file selections and other application settings.
	12.	File Selection Page
	•	Select or deselect files from the data directory for analysis. Provides options to “Select All” or “Deselect All.”

Navigation Tips:

	•	Use the “Back” buttons to return to previous sections.
	•	Utilize the “Select All” and “Deselect All” buttons in the File Selection Page for efficient file management.
	•	Monitor progress bars during dataset creation for real-time updates.
	•	In the Custom Figure Page, select custom plot functions and configure parameters for advanced visualization.

Dependencies:

The application relies on the following Python libraries:
	•	os: Interacting with the operating system.
	•	h5py: Handling HDF5 file formats.
	•	argparse: Parsing command-line arguments.
	•	numpy: Numerical operations.
	•	pandas: Data manipulation and analysis.
	•	tkinter: Building the GUI.
	•	matplotlib: Plotting and visualization.
	•	threading: Managing concurrent tasks.
	•	ttk (from tkinter): Themed widgets for Tkinter.

Ensure all dependencies are installed via pip as shown in the Installation section.

Note: Ensure that pdg_id_script.py, Generic_Plot_script.py, and Custom_Plot_script.py are present in the same directory as Main_Run_File.py for the application to function correctly.

Contributing:

Contributions are welcome! Please follow these steps:
	1.	Fork the repository.
	2.	Create a new branch for your feature or bugfix.
	3.	Commit your changes with clear and descriptive messages.
	4.	Push your branch to your forked repository.
	5.	Open a pull request detailing your changes.
