import os
import h5py
import argparse
import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg , NavigationToolbar2Tk)
from matplotlib.backends.backend_pdf import PdfPages

from PIL import Image, ImageTk # Used For displaying embedded images (png files)

import tensorflow as tf
from keras import models, layers

import matplotlib
matplotlib.use('agg')  # If I don't change the backend of matplotlib while creating plots it will clash with my progress bar backend and crash my window 😅

import threading

# Backend scripts:
import pdg_id_script
import Generic_Plot_script
import Custom_Plot_script
import Pixel_Array_Script


class App(tk.Tk):

    def _destroy_frame(self, frame_class):
        """Internal method to handle destroying a frame."""
        frame = self.frames.get(frame_class)
        if frame:
            frame.destroy()
            del self.frames[frame_class]
            # print(f"{frame_class.__name__} has been destroyed.")
        else:
            pass 


    def _reinitialize_frame(self, frame_class):
        """Internal method to handle re-initialize a frame."""
        self._destroy_frame(frame_class)
        new_frame = frame_class(self.container, self)
        self.frames[frame_class] = new_frame
        new_frame.grid(row=0, column=0, sticky="nsew")


    def __init__(self, Data_Directory=None, input_type='edep', det_complex='2x2'):
        super().__init__()
        self.title("Msci Project App")
        self.geometry("400x300")
        
        os.system('cls||clear')
        print("RUNNING")

        self.cmap = cm.plasma
        # Add destroy and reinitialize methods, because of headaches.
        self.destroy_frame      = self._destroy_frame
        self.reinitialize_frame = self._reinitialize_frame
        
        
        self.Data_Directory     = Data_Directory        # Create initial attributes for accessing the data directory 


        self.pdg_id_map         = pdg_id_script.pdg_id_map            # Apply the imported pdg map as a attribute

        self.pdg_id_map_reverse = { self.pdg_id_map[i] : i for i in list( pdg_id_script.pdg_id_map.keys() ) }
        
        self.plot_type          = 'scatter'             # Set initial plot_type state, should help with cleaning up code. 

        self.running = False
        self.model   = None

        self.max_z_for_plot = round(918.2)
        self.min_z_for_plot = round(415.8)

        self.max_y_for_plot = round(82.9)
        self.min_y_for_plot = round(-216.7)

        self.max_x_for_plot = round(350)
        self.min_x_for_plot = round(-350)

        # Retrieve and sort file names
        File_Names = os.listdir(Data_Directory)
        Temp_File_Names_Dict = {int(i.split('.')[3]): i for i in File_Names}
        sorted_keys = sorted(Temp_File_Names_Dict.keys())
        File_Names = [Temp_File_Names_Dict[i] for i in sorted_keys]
        self.File_Names = File_Names
        self.Allowed_Files = []

        self.input_type = input_type
        self.det_complex = det_complex

        # Container for stacking frames
        self.container = tk.Frame(self)
        self.container.pack(fill="both", expand=True)

        # Dictionary to keep track of frames
        self.frames = {}

        # List of all frames
        All_Frames = [
            StartPage,
            Dataset_View_Page,
            View_Segments_Page,
            View_mc_hdr_Page,
            View_traj_Page,
            Dataset_Page,
            Create_Dataset_Page,
            Load_Dataset_Page,
            Plot_Selection_Page,
            Figure_Creation_Page,
            Custom_Figure_Page,
            Training_And_Eval_Options_Page,
            Model_Architecture_Page,
            Model_Training_Page,
            Settings_Page,
            File_Selection_Page,
            Cleaning_Method_Select_Page
        ]

        # Initialize all frames
        for F in All_Frames:
            frame = F(self.container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        # Show the start page initially
        self.show_frame(StartPage)


    def set_plot_type(self, plot_type):
        self.plot_type = plot_type

        if self.plot_type == 'custom':

            self.reinitialize_frame(Custom_Figure_Page)  # Re-initialize the frame
            self.show_frame(Custom_Figure_Page)          # Show the re-initialized frame

        
        else:
            self.reinitialize_frame(Figure_Creation_Page)  # Re-initialize the frame
            self.show_frame(Figure_Creation_Page)          # Show the re-initialized frame

    def show_frame(self, page):
        """Bring a frame to the front and refresh its content if applicable."""
        frame = self.frames[page]

    
        # Dynamically resize window based on the frame
        if page == StartPage:
            self.geometry("400x300")  # Default size for other pages

        elif page == File_Selection_Page:
            self.geometry("1000x500")  # Larger size for File Selection Page
        
        
        # Larger size for View Segments Page
        elif page == View_Segments_Page or page == View_mc_hdr_Page or page == View_traj_Page:
            self.geometry("1600x500")  


        elif page == Figure_Creation_Page or page ==  Custom_Figure_Page or page == Create_Dataset_Page:
            # Get the current window size as a string
            window_size = self.geometry()
            
            # Split the geometry string into width and height
            dimensions = window_size.split('+')[0]  # Remove position offsets
            try:
                width, height = map(int, dimensions.split('x'))  # Extract and convert to integers
            except ValueError:
                # Fallback to default size if parsing fails
                width, height = 400, 300
                self.geometry("400x300")

            # Check and adjust size based on conditions
            if hasattr(self , "fig" ):
                self.geometry("900x900")

            else:
                self.geometry('900x900')

        elif page == Model_Architecture_Page:
            self.geometry('900x500')

        elif page == Model_Training_Page :
            self.geometry('1000x500')

        elif page == Load_Dataset_Page:
            self.geometry('1000x700')


        else:
            self.geometry("400x300")  # Default size for other pages

        frame.tkraise()

        # If the frame has a refresh_content method, call it
        if hasattr(frame, 'refresh_content'):
            frame.refresh_content()

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#

class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        label = tk.Label(self, text="Dune Experiment : Data Analysis🧐", font=("Helvetica", 16))
        label.pack(pady=10, padx=10, anchor='w')

        # Navigation buttons
        tk.Button(self, text="View Datasets",
                  command=lambda: controller.show_frame(Dataset_View_Page)).pack(anchor='w')
        tk.Button(self, text="Go to Plot Page",
                  command=lambda: controller.show_frame(Plot_Selection_Page)).pack(anchor='w')
        tk.Button(self, text="Create ML Dateset",
                  command=lambda: controller.show_frame(Dataset_Page)).pack(anchor='w')
        tk.Button(self, text="Train & Evaluate ML Model",
                  command=lambda: controller.show_frame(Training_And_Eval_Options_Page)).pack(anchor='w')
        tk.Button(self, text="Go to Settings",
                  command=lambda: controller.show_frame(Settings_Page)).pack(anchor='w' ,pady=(20,20))

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#



class Dataset_View_Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        tk.Label(self, text="View Datasets", font=("Helvetica", 16)).pack(pady=10, padx=10 , anchor='w' )
        tk.Button(self, text="View segments", command=lambda: controller.show_frame(View_Segments_Page)).pack( anchor='w')
        tk.Button(self, text="View mc_hdr", command=lambda: controller.show_frame(View_mc_hdr_Page) ).pack( anchor='w')
        tk.Button(self, text="View Trajectories", command=lambda: controller.show_frame(View_traj_Page)  ).pack( anchor='w' )
        tk.Button(self, text="Back to Start Page", command=lambda: controller.show_frame(StartPage)).pack( anchor='w' )




#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#


class View_Segments_Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        self.controller = controller  # Reference to the main app
        self.h5_data_name = 'segments'

        # Initialize event selection variables
        self.Event_ID_selection = 0
        self.Event_IDs = []

        # Page Title
        tk.Label(self, text="View Segments", font=("Helvetica", 16)).pack(anchor='w', pady=(10, 10))

        # Navigation Button
        tk.Button(self, text="Back to View Datasets", command=lambda: controller.show_frame(Dataset_View_Page)).pack(anchor='w', pady=(0, 10))

        # Frame for Dropdown Menu
        file_selection_frame = tk.Frame(self)
        file_selection_frame.pack(anchor='w', pady=(0, 10))

        # Dropdown Label
        tk.Label(file_selection_frame, text="Select file:").pack(side=tk.LEFT, padx=(0, 5))

        # Dropdown Menu
        self.selected_file = tk.StringVar()
        self.files_drop_down = tk.OptionMenu(file_selection_frame, self.selected_file, "")
        self.files_drop_down.pack(side=tk.LEFT)

        # Frame for Displaying DataFrame and Navigation Buttons
        self.display_frame = tk.Frame(self)
        self.display_frame.pack(anchor='w', pady=(10, 10))

        # Initialize Dropdown Options
        Frame_Manager.update_dropdown(self)

        # Navigation Buttons Frame
        navigation_buttons_frame = tk.Frame(self)
        navigation_buttons_frame.pack(anchor='w', pady=(5, 10))

        # Back Button
        self.back_button = tk.Button(
            navigation_buttons_frame, 
            text="Back", 
            command=lambda: Frame_Manager.go_back(self)
        )
        self.back_button.pack(side=tk.LEFT, padx=5)

        # Next Button
        self.next_button = tk.Button(
            navigation_buttons_frame, 
            text="Next", 
            command=lambda: Frame_Manager.go_next(self)
        )
        self.next_button.pack(side=tk.LEFT, padx=5)

        # Event Counter Label
        self.event_counter_label = tk.Label(self, text="Event 0 of 0", font=("Helvetica", 10))
        self.event_counter_label.pack(anchor='w', padx=5)

        # Bind the dropdown selection to update the DataFrame display
        self.selected_file.trace('w', lambda *args: Frame_Manager.on_file_selected(self , 'segments'))



class View_mc_hdr_Page(tk.Frame):
   def __init__(self, parent, controller):
        super().__init__(parent)

        self.controller = controller  # Reference to the main app
        self.h5_data_name = 'mc_hdr'


        # Initialize event selection variables
        self.Event_ID_selection = 0
        self.Event_IDs = []

        # Page Title
        tk.Label(self, text="View mc_hdr", font=("Helvetica", 16)).pack(anchor='w', pady=(10, 10))

        # Navigation Button
        tk.Button(self, text="Back to View Datasets", command=lambda: controller.show_frame(Dataset_View_Page)).pack(anchor='w', pady=(0, 10))

        # Frame for Dropdown Menu
        file_selection_frame = tk.Frame(self)
        file_selection_frame.pack(anchor='w', pady=(0, 10))

        # Dropdown Label
        tk.Label(file_selection_frame, text="Select file:").pack(side=tk.LEFT, padx=(0, 5))

        # Dropdown Menu
        self.selected_file = tk.StringVar()
        self.files_drop_down = tk.OptionMenu(file_selection_frame, self.selected_file, "")
        self.files_drop_down.pack(side=tk.LEFT)

        # Frame for Displaying DataFrame and Navigation Buttons
        self.display_frame = tk.Frame(self)
        self.display_frame.pack(anchor='w', pady=(10, 10))

        # Initialize Dropdown Options
        Frame_Manager.update_dropdown(self)

        # Navigation Buttons Frame
        navigation_buttons_frame = tk.Frame(self)
        navigation_buttons_frame.pack(anchor='w', pady=(5, 10))

        # Back Button
        self.back_button = tk.Button(
            navigation_buttons_frame, 
            text="Back", 
            command=lambda: Frame_Manager.go_back(self)
        )
        self.back_button.pack(side=tk.LEFT, padx=5)

        # Next Button
        self.next_button = tk.Button(
            navigation_buttons_frame, 
            text="Next", 
            command=lambda: Frame_Manager.go_next(self)
        )
        self.next_button.pack(side=tk.LEFT, padx=5)

        # Event Counter Label
        self.event_counter_label = tk.Label(self, text="Event 0 of 0", font=("Helvetica", 10))
        self.event_counter_label.pack(anchor='w', padx=5)

        # Bind the dropdown selection to update the DataFrame display
        self.selected_file.trace('w', lambda *args: Frame_Manager.on_file_selected(self , 'mc_hdr'))



class View_traj_Page(tk.Frame):
   def __init__(self, parent, controller):
        super().__init__(parent)

        self.controller = controller  # Reference to the main app
        self.h5_data_name = 'trajectories'


        # Initialize event selection variables
        self.Event_ID_selection = 0
        self.Event_IDs = []

        # Page Title
        tk.Label(self, text="View trajectories", font=("Helvetica", 16)).pack(anchor='w', pady=(10, 10))

        # Navigation Button
        tk.Button(self, text="Back to View Datasets", command=lambda: controller.show_frame(Dataset_View_Page)).pack(anchor='w', pady=(0, 10))

        # Frame for Dropdown Menu
        file_selection_frame = tk.Frame(self)
        file_selection_frame.pack(anchor='w', pady=(0, 10))

        # Dropdown Label
        tk.Label(file_selection_frame, text="Select file:").pack(side=tk.LEFT, padx=(0, 5))

        # Dropdown Menu
        self.selected_file = tk.StringVar()
        self.files_drop_down = tk.OptionMenu(file_selection_frame, self.selected_file, "")
        self.files_drop_down.pack(side=tk.LEFT)

        # Frame for Displaying DataFrame and Navigation Buttons
        self.display_frame = tk.Frame(self)
        self.display_frame.pack(anchor='w', pady=(10, 10))

        # Initialize Dropdown Options
        Frame_Manager.update_dropdown(self)

        # Navigation Buttons Frame
        navigation_buttons_frame = tk.Frame(self)
        navigation_buttons_frame.pack(anchor='w', pady=(5, 10))

        # Back Button
        self.back_button = tk.Button(
            navigation_buttons_frame, 
            text="Back", 
            command=lambda: Frame_Manager.go_back(self)
        )
        self.back_button.pack(side=tk.LEFT, padx=5)

        # Next Button
        self.next_button = tk.Button(
            navigation_buttons_frame, 
            text="Next", 
            command=lambda: Frame_Manager.go_next(self)
        )
        self.next_button.pack(side=tk.LEFT, padx=5)

        # Event Counter Label
        self.event_counter_label = tk.Label(self, text="Event 0 of 0", font=("Helvetica", 10))
        self.event_counter_label.pack(anchor='w', padx=5)

        # Bind the dropdown selection to update the DataFrame display
        self.selected_file.trace('w', lambda *args: Frame_Manager.on_file_selected(self , 'mc_hdr'))


#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#


class Dataset_Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        tk.Label(self, text="Dataset Creation", font=("Helvetica", 16)).pack(pady=10, padx=10 , anchor='w')

        tk.Button(self, text="Create",
                  command=lambda: controller.show_frame(Create_Dataset_Page)).pack( anchor='w')

        tk.Button(self, text="Load",
                  command=lambda: controller.show_frame(Load_Dataset_Page)).pack(anchor='w')

        tk.Button(self, text="Back to Start Page",
                  command=lambda: controller.show_frame(StartPage)).pack(anchor='w')
        

class Create_Dataset_Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        self.controller = controller
        self.selected_files = []
        self.file_vars = []

        # Header Frame: Back button and Header label
        self.header_frame = tk.Frame(self)
        self.header_frame.pack(anchor='w', padx=10, pady=20)

        # Back Button
        back_button = tk.Button(self.header_frame, text='Back', command=lambda: controller.show_frame(Dataset_Page))
        back_button.pack(side=tk.LEFT)

        # Header Label
        header_label = tk.Label(self.header_frame, text="Create Dataset", font=("Helvetica", 16))
        header_label.pack(side=tk.LEFT, padx=150)

        # Progress Bar and Percentage Frame
        self.progressive_frame = tk.Frame(self)
        self.progressive_frame.pack(anchor='w', padx=10, pady=(0, 20))

        self.progress = ttk.Progressbar(self.progressive_frame, orient="horizontal", length=600, mode="determinate")
        self.progress_label = tk.Label(self.progressive_frame, text='', font=("Arial", 12))
        self.progress.pack(anchor='w', side=tk.LEFT)
        self.progress_label.pack(anchor='w', side=tk.LEFT)

        # File Selection Frame (now replaced by a scrollable frame of checkboxes)
        self.file_select_frame = tk.Frame(self)
        self.file_select_frame.pack(anchor='w', padx=10, pady=(0, 20))

        tk.Label(self.file_select_frame, text="Select Files:").pack(anchor='w')

        # Scrollable frame for the file list
        scroll_frame = ScrollableFrame(self.file_select_frame)
        scroll_frame.pack(fill="both", expand=True, pady=5 )

        # Determine which files to show
        allowed_files = getattr(controller, 'Allowed_Files', [])
        if not allowed_files:
            # If no allowed files, show all files in directory
            all_files_in_dir = os.listdir(controller.Data_Directory)
            # Optionally, filter by extension
            # all_files_in_dir = [f for f in all_files_in_dir if f.endswith('.h5')]
            allowed_files = all_files_in_dir

        # Create Checkbuttons for allowed files with increased width
        for file in sorted(allowed_files):
            var = tk.IntVar()
            c = tk.Checkbutton(scroll_frame.scrollable_frame, text=file, variable=var, anchor='w', width=200)
            c.pack(fill='x', padx=5, pady=2)
            self.file_vars.append((var, file))


        # Frame for Select/Deselect/Confirm
        self.button_frame = tk.Frame(self)
        self.button_frame.pack(anchor='w', padx=10, pady=(0, 20))

        tk.Button(self.button_frame, text="Select All", command=self.select_all).pack(side=tk.LEFT, padx=5)
        tk.Button(self.button_frame, text="Deselect All", command=self.deselect_all).pack(side=tk.LEFT, padx=5)
        tk.Button(self.button_frame, text="Confirm Selection", command=self.confirm_selection).pack(side=tk.LEFT, padx=5)

        # Interact Frame for Preview and Create
        self.Interact_Frame = tk.Frame(self)
        self.Interact_Frame.pack(anchor='w', padx=10, pady=(0, 20))

        # self.Preview_Button = tk.Button(self.Interact_Frame, text='Preview', command=self.Preview_Interaction)
        self.Preview_Button = tk.Button(self.button_frame, text='Preview', command=self.Preview_Interaction)

        self.Preview_Button.pack(side=tk.LEFT, anchor='w', padx=5)

        tk.Label(self.Interact_Frame , text = "ML Dataset Name :" ).pack( anchor='w', side=tk.LEFT ,padx=10 )
        self.Text_Box_ML_Dataset_Name = tk.Text( self.Interact_Frame, bg='black',  fg='white', font=("Arial", 12), width=45, height=1, padx=10, pady=10 , state = 'normal' )
        self.Text_Box_ML_Dataset_Name.pack(anchor='w', side=tk.LEFT ,padx=(0,10), pady=5)

        self.Create_Dataset_Button = tk.Button(self.Interact_Frame, text='Create', command=lambda: Frame_Manager.setup_process(self))
        self.Create_Dataset_Button.pack(side=tk.LEFT, anchor='w')

        self.Cancel_Creation = tk.Button(self.Interact_Frame, text='Cancel', command=lambda: Frame_Manager.cancel_process(self) , state='disabled')
        self.Cancel_Creation.pack(side=tk.LEFT, anchor='w')

        self.Figure_Frame = tk.Frame(self)
        self.Figure_Frame.pack(anchor='w', side=tk.LEFT, pady=5)


    def select_all(self):
        for var, _ in self.file_vars:
            var.set(1)

    def deselect_all(self):
        for var, _ in self.file_vars:
            var.set(0)

    def confirm_selection(self):
        self.selected_files = [file for var, file in self.file_vars if var.get() == 1]
        # if not self.selected_files:
        #     # If no file selected, you can handle this scenario as needed
        #     print("No files selected!")
        # else:
        #     print("Selected files:", self.selected_files)

    def Preview_Interaction(self):
        # Ensure files are selected
        if not self.selected_files:
            print("No files selected for preview!")
            return

        # Randomly select one file from the selected list for preview
        selected_file_for_preview = np.random.choice(self.selected_files)

        self.selected_file = selected_file_for_preview 

        path = os.path.join(self.controller.Data_Directory, selected_file_for_preview)
        sim_h5 = h5py.File(path, 'r')
        temp_segments = sim_h5["segments"]
        temp_mc_hdr = sim_h5["mc_hdr"]


        unique_ids = np.unique(temp_segments['event_id']).tolist()
        random_event_id = np.random.choice(unique_ids)
        random_vertex_id = np.random.choice( temp_mc_hdr[ temp_mc_hdr['event_id'] == random_event_id ]['vertex_id'] )

        self.event_id_selected = random_event_id
        self.vertex_id_selected = random_vertex_id

        Pixel_Array_Script.Use_Pixel_Array.plot(self)

        return

    def Create_ML_Dataset_2(self):
        # Ensure files are selected
        if not self.selected_files:
            print("No files selected for dataset creation!")
            self.progress_value = 100
            self.controller.running = False
            self.Create_Dataset_Button.config(state='normal')
            return


        # Create a name for directory that will hold all dataset images
        # Test_directory = "ML_IMAGE_DATASET"
        Test_directory = str(self.Text_Box_ML_Dataset_Name.get('1.0' , tk.END) )
        print(Test_directory)

        self.Text_Box_ML_Dataset_Name.config(state = 'disabled')
        self.Cancel_Creation.config(state = 'normal')


        os.makedirs(Test_directory, exist_ok=True)

        # Create mapping for Directory Naming
        Directory_Name_Map = {
            r"$\nu$-$e^{-}$ scattering": "Neutrino_Electron_Scattering",
            r"$\nu_{e}$-CC": "Electron_Neutrino",
            r"$\nu_{e}$-NC": "Electron_Neutrino",
            "Other": "Other"
        }
        # Create sub-directories
        for Dir_Name in list(Directory_Name_Map.values()):
            _ = os.path.join(Test_directory, Dir_Name)
            os.makedirs(_, exist_ok=True)

        # Create file name counters for images generated
        Dir_File_Name_Counter = {file: 0 for file in list(Directory_Name_Map.keys())}

        # We'll process all selected files
        all_event_ids = []
        selected_fileselected_file = []
        for selected_file in self.selected_files:
            path = os.path.join(self.controller.Data_Directory, selected_file)
            sim_h5 = h5py.File(path, 'r')
            temp_segments = sim_h5["segments"]
            # temp_segments = temp_segments[(temp_segments['dE'] > 1.5)]
            # temp_segments = pd.DataFrame(temp_segments[()])
            temp_mc_hdr = sim_h5['mc_hdr']
            # unique_ids = np.unique(temp_segments['event_id']).tolist()
            unique_ids = np.unique(temp_mc_hdr['event_id']).tolist()

            all_event_ids.extend(unique_ids)

        
        # Remove duplicates if needed
        all_event_ids = list(set(all_event_ids))
        num_events = len(all_event_ids)

        self.controller.running = True
        self.progress.configure(maximum=len(all_event_ids))
        self.Create_Dataset_Button.config(state='disabled')

        # Process the event_id in each selected file where it exists

        min_z, max_z = self.controller.min_z_for_plot, self.controller.max_z_for_plot
        min_y, max_y = self.controller.min_y_for_plot, self.controller.max_y_for_plot
        min_x, max_x = self.controller.min_x_for_plot, self.controller.max_x_for_plot
        cnter = 0
        for selected_file in self.selected_files:
        # Loop for each event across all selected files

            # The condition below allows for interruption if needed
            if self.controller.running == False:
                break

            path = os.path.join(self.controller.Data_Directory, selected_file)
            sim_h5 = h5py.File(path, 'r')
            temp_segments = sim_h5['segments']
            temp_segments = temp_segments[ temp_segments['dE'] > 1.5 ]
            temp_mc_hdr   = sim_h5['mc_hdr']

            seg_unique_ids = np.unique( temp_segments['event_id'] )

            for event_id in seg_unique_ids:
                cnter += 1
                self.progress_value = (cnter / num_events) * 100

                # Check if processing should continue
                if not self.controller.running:
                    break

                # Find indices where event_id matches
                indices = np.where(temp_segments['event_id'] == event_id)[0]
                if len(indices) == 0:
                    continue

                # Extract segments for the current event_id
                temp_segments_event = temp_segments[temp_segments['event_id'] == event_id]
                temp_segments_event = pd.DataFrame(temp_segments_event)

                # Extract mc_hdr entries for the current event_id
                temp_mc_hdr_event = temp_mc_hdr[temp_mc_hdr['event_id'] == event_id]

                # Get unique vertex_ids from mc_hdr_event
                mc_hdr_vertex_ids = np.unique(temp_mc_hdr_event['vertex_id']).tolist()

                # Optimize noise_indexes calculation using set operations
                vertex_ids_set = set(mc_hdr_vertex_ids)
                segments_vertex_ids = set(temp_segments_event['vertex_id'])
                noise_indexes = list(segments_vertex_ids - vertex_ids_set)

                for true_vertex in mc_hdr_vertex_ids:
                    # Extract mc_hdr_event_vertex and segments_event_vertex
                    mask_vertex = temp_mc_hdr_event['vertex_id'] == true_vertex
                    temp_mc_hdr_event_vertex = temp_mc_hdr_event[mask_vertex]
                    temp_segments_event_vertex = temp_segments_event[temp_segments_event['vertex_id'] == true_vertex]

                    # Ensure there is exactly one vertex per event
                    if temp_mc_hdr_event_vertex.shape[0] != 1:
                        continue  # Skip if multiple or no vertices found

                    # Extract coordinates to local variables
                    z = temp_mc_hdr_event_vertex['z_vert'][0]
                    y = temp_mc_hdr_event_vertex['y_vert'][0]
                    x = temp_mc_hdr_event_vertex['x_vert'][0]

                    # Optimized Spatial Filtering
                    if (z <= min_z or z >= max_z or
                        y <= min_y or y >= max_y or
                        x <= min_x or x >= max_x):
                        continue

                    # Determine interaction label based on reaction and neutrino properties
                    reaction = temp_mc_hdr_event_vertex['reaction'][0]
                    nu_pdg = temp_mc_hdr_event_vertex['nu_pdg'][0]
                    isCC = temp_mc_hdr_event_vertex['isCC'][0]

                    if reaction == 7:
                        interaction_label = r"$\nu$-$e^{-}$ scattering"
                    elif nu_pdg == 12 and isCC:
                        interaction_label = r"$\nu_{e}$-CC"
                    elif nu_pdg == 12 and not isCC:
                        interaction_label = r"$\nu_{e}$-NC"
                    else:
                        interaction_label = 'Other'

                    # Extract noise segments
                    if noise_indexes:
                        noise_df = temp_segments_event[temp_segments_event['vertex_id'].isin(noise_indexes)]
                    else:
                        noise_df = pd.DataFrame(columns=temp_segments_event.columns)  # Empty DataFrame

                    # Construct the file path for saving
                    loop_path = os.path.join(Test_directory, Directory_Name_Map.get(interaction_label, 'Other'))
                    loop_filename = f"IMG_{Dir_File_Name_Counter.get(interaction_label, 0)}_{event_id}_{true_vertex}.png"
                    loop_path = os.path.join(loop_path, loop_filename)

                    # Concatenate only non-empty DataFrames
                    dfs_to_concat = [df for df in [temp_segments_event_vertex, noise_df] if not df.empty]
                    if dfs_to_concat:
                        DF = pd.concat(dfs_to_concat, ignore_index=True)
                    else:
                        DF = pd.DataFrame(columns=temp_segments_event_vertex.columns)  # or handle as needed

                    # Save the DataFrame for Machine Learning
                    Pixel_Array_Script.Use_Pixel_Array.Save_For_ML(self, DF, loop_path)

                    # Update the directory file name counter
                    Dir_File_Name_Counter[interaction_label] = Dir_File_Name_Counter.get(interaction_label, 0) + 1

        self.progress_value = 100
        self.controller.running = False
        self.Create_Dataset_Button.config(state='normal')
        self.Text_Box_ML_Dataset_Name.config(state = 'normal')
        self.Cancel_Creation.config(state = 'disabled')

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#


class Load_Dataset_Page( tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller


        page_title_frame = tk.Frame(self)
        page_title_frame.pack( anchor= 'w' , pady=10)

        # Title and Navigation
        tk.Button(page_title_frame, text="Back", command=lambda: controller.show_frame(Training_And_Eval_Options_Page)).pack(anchor='w', padx=10 , side= tk.LEFT)
        tk.Label(page_title_frame, text="Load Image ", font=("Helvetica", 16)).pack( padx=50, anchor='w' , side = tk.LEFT)

        dir_select_frame = tk.Frame(self)
        dir_select_frame.pack( anchor='w' , pady = (10 , 50) )

        self.selected_dir = tk.StringVar()

        self.entry_box = tk.Entry(dir_select_frame, textvariable=self.selected_dir ,  bg='black',  fg='white' , width=80 , state='readonly')
        self.entry_box.pack(anchor='w', side=tk.LEFT, padx=10, pady=10)

        self.selected_dir.trace_add('write', lambda *args: self.Class_Select_Dropdown_Func() )

        select_dir_buttnon = tk.Button( dir_select_frame , text= "Dataset Directory" , command= lambda: Frame_Manager.select_directory_window(self , self.entry_box))
        select_dir_buttnon.pack( anchor='w' , side= tk.LEFT )

        self.View_Dataset_Control_Frame = tk.Frame(self)
        self.View_Dataset_Control_Frame.pack( anchor= 'w' , pady=10)

        tk.Label(self.View_Dataset_Control_Frame, text="Class :" ).pack(anchor='w', side=tk.LEFT) 


        self.Class_selected = tk.StringVar()
        self.Class_dropdown = ttk.Combobox( self.View_Dataset_Control_Frame , textvariable= self.Class_selected , state= 'readonly'  )
        self.Class_dropdown.pack(anchor='w', side=tk.LEFT) 

        self.Class_selected.trace_add('write', lambda *args: self.Image_Select_Dropdown_Func() )

        tk.Label(self.View_Dataset_Control_Frame, text="Image :" ).pack(anchor='w', side=tk.LEFT) 

        self.Image_Selected = tk.StringVar()
        self.Image_dropdown = ttk.Combobox( self.View_Dataset_Control_Frame , textvariable= self.Image_Selected , state= 'readonly'  )
        self.Image_dropdown.pack(anchor='w', side=tk.LEFT) 

        self.Image_Selected.trace_add( 'write' , lambda *args: self.Load_Image()  )

        self.Image_Frame = tk.Frame(self)
        self.Image_Frame.pack(anchor= 'w' , pady=10)


    def Class_Select_Dropdown_Func(self):

        Class_Dir_Names  = os.listdir(self.entry_box.get())

        self.Class_dropdown.config( values = sorted(Class_Dir_Names) )


    def Image_Select_Dropdown_Func(self):
        
        path = os.path.join( str(self.entry_box.get()) , str(self.Class_selected.get() ) )
        Image_File_Names  = os.listdir( path )

        sorted_Image_File_Names = sorted(Image_File_Names, key=lambda x: int(x.split('_')[1].split('.')[0]))

        self.Image_dropdown.config( values = sorted_Image_File_Names )
        self.Image_dropdown.set('')

    def Load_Image(self):

        try:
            photo_path = os.path.join(  str(self.selected_dir.get()) , str(self.Class_selected.get())   ) 
            photo_path = os.path.join(  photo_path , str(self.Image_Selected.get())  ) 


            img = Image.open(photo_path)
            original_width, original_height = img.size
            new_size = (original_width // 4, original_height // 4)
            img_resized = img.resize(new_size, Image.Resampling.LANCZOS) 
            Photo = ImageTk.PhotoImage(img_resized)

            # Clear previous images in the frame
            for widget in self.Image_Frame.winfo_children():
                widget.destroy()


            label = tk.Label(self.Image_Frame, image=Photo)
            label.image = Photo  # Keep a reference so the image is not garbage collected
            label.pack(anchor='w', pady=10)

        except:
            pass

        return

        

        
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#

class Training_And_Eval_Options_Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        page_title_frame = tk.Frame(self)
        page_title_frame.pack( anchor= 'w' , pady=10)

        tk.Button(page_title_frame , text = "back", command= lambda: controller.show_frame(StartPage) ).pack( padx=10 , anchor='w', side = tk.LEFT)
        tk.Label(page_title_frame, text="Select_Option", font=("Helvetica", 16)).pack( padx=50 , anchor='w' , side = tk.LEFT)

        tk.Button(self, text="Architecture Configuration",
                  command=lambda: controller.show_frame(Model_Architecture_Page)).pack( padx = 10 ,pady=(20,0) ,anchor='w')

        tk.Button(self, text="Train Model",
                  command=lambda: controller.show_frame(Model_Training_Page)).pack(  padx=10  ,anchor='w')

        tk.Button(self, text="Evaluate Model",
                  command=lambda: controller.show_frame(StartPage)).pack( padx=10   ,anchor='w')

class Model_Architecture_Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller


        # Title and Navigation
        page_title_frame = tk.Frame(self)
        page_title_frame.pack( anchor= 'w' , pady=10)

        tk.Button(page_title_frame , text = "back", command= lambda: controller.show_frame(Training_And_Eval_Options_Page) ).pack( padx=10 , anchor='w', side = tk.LEFT)
        tk.Label(page_title_frame, text="Model Architecture Configuration", font=("Helvetica", 16)).pack( padx=50 , anchor='w' , side = tk.LEFT)
        

        # Create a canvas and scrollbar for the entire page
        self.canvas = tk.Canvas(self)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill='y')
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Create a frame inside the canvas to hold all configuration
        self.inner_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), height = 1000 , width = 1000 , window=self.inner_frame, anchor='nw')
        # self.canvas.create_window((0, 0), height = 10 , width = 10 , window=self.inner_frame, anchor='nw')


        # Bind the configuration event to update scroll region
        self.inner_frame.bind("<Configure>", lambda event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # Initialize variables and create the layout
        self.model = None
        self.conv_layers = []
        self.dense_layers = []
        self.flatten_added = False

        # Create the layout
        self.create_layout()

    def create_layout(self):
        """Create the entire layout for the model configuration in inner_frame."""
        for widget in self.inner_frame.winfo_children():
            widget.destroy()

        #Input Shape 
        input_shape_frame = tk.Frame(self.inner_frame)
        input_shape_frame.pack(anchor='w', padx=10, pady=5)
        tk.Label(input_shape_frame, text="Input Shape (H,W,Channels):").pack(side=tk.LEFT, padx=(0,5))

        self.input_height = tk.StringVar(value="256")
        self.input_width = tk.StringVar(value="256")
        self.input_channels = tk.StringVar(value="3")

        tk.Entry(input_shape_frame, textvariable=self.input_height, width=5).pack(side=tk.LEFT)
        tk.Entry(input_shape_frame, textvariable=self.input_width, width=5).pack(side=tk.LEFT, padx=5)
        tk.Entry(input_shape_frame, textvariable=self.input_channels, width=5 , state='disabled').pack(side=tk.LEFT)

        # Convolution Layers 
        conv_frame = tk.Frame(self.inner_frame)
        conv_frame.pack(anchor='w', padx=10, pady=5, fill='x')
        tk.Label(conv_frame, text="Convolution Layers").pack(anchor='w')
        tk.Button(conv_frame, text="Add Convolution Layer", command=self.add_conv_layer).pack(anchor='w', pady=5)
        self.conv_layer_container = tk.Frame(conv_frame)
        self.conv_layer_container.pack(anchor='w', padx=10, pady=5)

        # Flatten Layer 
        flatten_frame = tk.Frame(self.inner_frame)
        flatten_frame.pack(anchor='w', padx=10, pady=5)
        tk.Button(flatten_frame, text="Add Flatten Layer", command=self.add_flatten_layer).pack(anchor='w')

        # Dense Layers 
        dense_frame = tk.Frame(self.inner_frame)
        dense_frame.pack(anchor='w', padx=10, pady=5)
        tk.Label(dense_frame, text="Dense (Fully Connected) Layers").pack(anchor='w')
        tk.Button(dense_frame, text="Add Dense Layer", command=self.add_dense_layer).pack(anchor='w', pady=5)
        self.dense_layer_container = tk.Frame(dense_frame)
        self.dense_layer_container.pack(anchor='w', padx=10, pady=5)

        #  Output Classes 
        output_shape_frame = tk.Frame(self.inner_frame)
        output_shape_frame.pack(anchor='w', padx=10, pady=5)
        tk.Label(output_shape_frame, text="Output Classes:").pack(side=tk.LEFT, padx=(0,5))
        self.output_classes = tk.StringVar(value="3")
        tk.Entry(output_shape_frame, textvariable=self.output_classes, width=5).pack(side=tk.LEFT)

        #  Build & Reset Buttons 
        build_frame = tk.Frame(self.inner_frame)
        build_frame.pack(anchor='w', padx=10, pady=10)
        tk.Button(build_frame, text="Build & Compile Model", command=self.build_and_compile_model).pack(side=tk.LEFT, padx=5)
        tk.Button(build_frame, text="Reset Architecture", command=self.reset_architecture).pack(side=tk.LEFT, padx=5)

        #  Model Summary 
        summary_frame = tk.Frame(self.inner_frame )
        # summary_frame.pack(anchor='w', padx=10, pady=10, fill='both', expand=True)
        summary_frame.pack(anchor='w', padx=40, pady=10, fill='both', expand=True)

        # Increase height here as requested
        # self.model_summary_text = tk.Text(summary_frame, width=80, height=30, wrap='none', font=("Courier", 8))
        self.model_summary_text = tk.Text(summary_frame, width=3, height=30, wrap='none', font=("Courier", 8))
        self.model_summary_text.pack(side=tk.LEFT, fill='both', expand=True)

        summary_scrollbar_vertical = ttk.Scrollbar(summary_frame, orient='vertical', command=self.model_summary_text.yview)
        summary_scrollbar_vertical.pack(side=tk.RIGHT, fill='y')
        self.model_summary_text.config(yscrollcommand=summary_scrollbar_vertical.set)

        summary_scrollbar_horizontal = ttk.Scrollbar(summary_frame, orient='horizontal', command=self.model_summary_text.xview)
        summary_scrollbar_horizontal.pack(side=tk.BOTTOM, fill='x')
        self.model_summary_text.config(xscrollcommand=summary_scrollbar_horizontal.set)

    def add_conv_layer(self):
        layer_frame = tk.Frame(self.conv_layer_container)
        layer_frame.pack(anchor='w', pady=5)

        tk.Label(layer_frame, text="Filters:").pack(side=tk.LEFT)
        filters_var = tk.StringVar(value="32")
        tk.Entry(layer_frame, textvariable=filters_var, width=5).pack(side=tk.LEFT, padx=5)

        tk.Label(layer_frame, text="Kernel Size:").pack(side=tk.LEFT)
        kernel_var = tk.StringVar(value="3")
        tk.Entry(layer_frame, textvariable=kernel_var, width=5).pack(side=tk.LEFT, padx=5)

        tk.Label(layer_frame, text="Activation:").pack(side=tk.LEFT)
        activation_var = tk.StringVar(value="leaky_relu")
        activation_combobox = ttk.Combobox(layer_frame, textvariable=activation_var, 
                                           values=["leaky_relu","relu", "sigmoid", "tanh", "linear"], state='readonly', width=8)
        activation_combobox.pack(side=tk.LEFT, padx=5)

        self.conv_layers.append((filters_var, kernel_var, activation_var))
        self.inner_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def add_flatten_layer(self):
        if self.flatten_added:
            return
        self.flatten_added = True
        flatten_label_frame = tk.Frame(self.inner_frame)
        flatten_label_frame.pack(anchor='w', padx=20, pady=5)
        tk.Label(flatten_label_frame, text="Flatten Layer Added", fg='blue').pack(anchor='w')

        self.inner_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def add_dense_layer(self):
        layer_frame = tk.Frame(self.dense_layer_container)
        layer_frame.pack(anchor='w', pady=5)

        tk.Label(layer_frame, text="Units:").pack(side=tk.LEFT)
        units_var = tk.StringVar(value="64")
        tk.Entry(layer_frame, textvariable=units_var, width=5).pack(side=tk.LEFT, padx=5)

        tk.Label(layer_frame, text="Activation:").pack(side=tk.LEFT)
        dense_activation_var = tk.StringVar(value="leaky_relu")
        dense_activation_combobox = ttk.Combobox(layer_frame, textvariable=dense_activation_var, 
                                                 values=["leaky_relu","relu", "sigmoid", "tanh", "linear"], 
                                                 state='readonly', width=8)
        dense_activation_combobox.pack(side=tk.LEFT, padx=5)

        self.dense_layers.append((units_var, dense_activation_var))
        self.inner_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def build_and_compile_model(self):

        self.model_summary_text.delete(1.0, tk.END)

        model = models.Sequential()

        input_h = int(self.input_height.get())
        input_w = int(self.input_width.get())
        input_c = int(self.input_channels.get())

        if self.conv_layers:
            first_filters, first_kernel, first_activation = self.conv_layers[0]
            model.add(layers.Conv2D(filters=int(first_filters.get()),
                                    kernel_size=(int(first_kernel.get()), int(first_kernel.get())),
                                    activation=first_activation.get(),
                                    input_shape=(input_h, input_w, input_c)))
            for (f_var, k_var, a_var) in self.conv_layers[1:]:
                model.add(layers.Conv2D(filters=int(f_var.get()),
                                        kernel_size=(int(k_var.get()), int(k_var.get())),
                                        activation=a_var.get()))
                model.add(layers.MaxPooling2D((2, 2)))
        else:
            model.add(layers.InputLayer(input_shape=(input_h, input_w, input_c)))

        if self.flatten_added:
            model.add(layers.Flatten())

        for (u_var, a_var) in self.dense_layers:
            model.add(layers.Dense(int(u_var.get()), activation=a_var.get()))

        num_classes = int(self.output_classes.get())
        model.add(layers.Dense(num_classes, activation='softmax'))

        model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['accuracy'])

        self.controller.model = model

        stringlist = []
        self.controller.model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        self.model_summary_text.insert(tk.END, short_model_summary)
        self.model_summary_text.insert(tk.END, "\n\nModel compiled successfully!")

    def reset_architecture(self):
        self.controller.model = None
        self.conv_layers.clear()
        self.dense_layers.clear()
        self.flatten_added = False

        self.create_layout()
        self.inner_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))



class Model_Training_Page(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller


        page_title_frame = tk.Frame(self)
        page_title_frame.pack( anchor= 'w' , pady=10)

        # Title and Navigation
        tk.Button(page_title_frame, text="Back", command=lambda: controller.show_frame(Training_And_Eval_Options_Page)).pack(anchor='w', padx=10 , side= tk.LEFT)
        tk.Label(page_title_frame, text="Model Training ", font=("Helvetica", 16)).pack( padx=50, anchor='w' , side = tk.LEFT)

        dir_select_frame = tk.Frame(self)
        dir_select_frame.pack( anchor='w' , pady = (10 , 50) )

        self.selected_dir = tk.StringVar()
        text_box = tk.Entry( dir_select_frame, textvariable=self.selected_dir , bg='black',  fg='white', font=("Arial", 12), width=90, state = 'disabled' )
        text_box.pack(anchor='w', side=tk.LEFT ,padx=10, pady=10)

        select_dir_buttnon = tk.Button( dir_select_frame , text= "Select Directory" , command= lambda: Frame_Manager.select_directory_window(self , text_box))
        select_dir_buttnon.pack( anchor='w' , side= tk.LEFT )

        train_split_params_frame = tk.Frame(self)
        train_split_params_frame.pack( anchor='w' , pady = (10 , 20) )

        self.train_size_text_selected = tk.IntVar()
        tk.Label(train_split_params_frame , text = "train_size :" ).pack(anchor='w' , side = tk.LEFT )
        train_size_text_box = tk.Entry( train_split_params_frame , textvariable= self.train_size_text_selected  , bg = 'black', fg='white', font=("Arial", 12), state = 'normal' )
        train_size_text_box.pack( anchor='w', side = tk.LEFT  )
        train_size_text_box.delete(0, tk.END)
        train_size_text_box.insert(0,70)
        

        self.val_size_text_selected = tk.IntVar()
        tk.Label(train_split_params_frame , text = "val_size :" ).pack(anchor='w' , side = tk.LEFT )
        val_size_text_box = tk.Entry( train_split_params_frame , textvariable= self.val_size_text_selected, bg='black',  fg='white', font=("Arial", 12), state = 'normal' )
        val_size_text_box.pack( anchor='w' ,side = tk.LEFT  )
        val_size_text_box.delete(0, tk.END)
        val_size_text_box.insert(0,20)


        self.test_size_text_selected = tk.IntVar()
        tk.Label(train_split_params_frame , text = "test_size :" ).pack(anchor='w' , side = tk.LEFT )
        test_size_text_box = tk.Entry( train_split_params_frame, textvariable= self.test_size_text_selected,  bg='black',  fg='white', font=("Arial", 12), state = 'normal')
        test_size_text_box.pack( anchor='w' ,side = tk.LEFT  )
        test_size_text_box.delete(0, tk.END)
        test_size_text_box.insert(0,10)

        Model_Training_Params_Frame = tk.Frame(self)
        Model_Training_Params_Frame.pack( anchor='w' , pady = (10 , 20) )

        self.Epoch_size_text_selected = tk.IntVar()
        tk.Label(Model_Training_Params_Frame , text = "Num Epoches :" ).pack(anchor='w' , side = tk.LEFT )
        Epoch_size_text_box = tk.Entry( Model_Training_Params_Frame , textvariable= self.Epoch_size_text_selected  , bg = 'black', fg='white', font=("Arial", 12), state = 'normal' )
        Epoch_size_text_box.pack( anchor='w', side = tk.LEFT  )

        self.Batch_size_text_selected = tk.IntVar()
        tk.Label(Model_Training_Params_Frame , text = "batch_size :" ).pack(anchor='w' , side = tk.LEFT )
        Batch_size_text_box = tk.Entry( Model_Training_Params_Frame , textvariable= self.Batch_size_text_selected  , bg = 'black', fg='white', font=("Arial", 12), state = 'normal' )
        Batch_size_text_box.pack( anchor='w', side = tk.LEFT  )


        tk.Button(self, text='Test_Train', command=lambda: self.Train_Model()).pack(anchor='w')

        self.Training_Plots_Output_Frame = tk.Frame(self)
        self.Training_Plots_Output_Frame.pack( anchor='w' , side=tk.LEFT )



    def Train_Model(self):

        selected_dir = self.selected_dir.get()

        # Function to find the first image file in the directory or its subdirectories
        def find_first_image(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        return os.path.join(root, file)
            return None

        first_image_path = find_first_image(selected_dir)

        if not first_image_path:
            print("No image files found in the selected directory.")
            return

        try:
            with Image.open(first_image_path) as img:
                original_size = img.size  # (width, height)
        except Exception as e:
            print(f"Error opening image {first_image_path}: {e}")
            return

        original_width, original_height = original_size

        # Load dataset with original image size
        Raw_Data = tf.keras.utils.image_dataset_from_directory(
            selected_dir,
            # image_size=(original_height, original_width),
            # batch_size=32,  # Adjust as needed
            shuffle=True,
            seed=123,
            label_mode='int'
        )

        # Calculate total size
        total_size = Raw_Data.cardinality().numpy()

        # Ensure sizes add up to total dataset size
        train_percent = self.train_size_text_selected.get()
        val_percent = self.val_size_text_selected.get()
        test_percent = self.test_size_text_selected.get()

        if (train_percent + val_percent + test_percent) != 100:
            print("Train, validation, and test sizes do not add up to 100%.")
            return

        train_size = int(total_size * (train_percent / 100))
        val_size = int(total_size * (val_percent / 100))
        test_size = int(total_size * (test_percent / 100))

        # Split the dataset
        train = Raw_Data.take(train_size)
        val = Raw_Data.skip(train_size).take(val_size)
        test = Raw_Data.skip(train_size + val_size).take(test_size)

        # Compile and train the model
        self.controller.model.compile(
            optimizer='adam',
            loss='SparseCategoricalCrossentropy',
            metrics=['accuracy']
        )
        print(self.controller.model.summary())

        logdir = 'logs'
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        hist = self.controller.model.fit(
            train,
            epochs=int(self.Epoch_size_text_selected.get()),
            validation_data=val,
            callbacks=[tensorboard_callback]
        )

        # Plot training history
        self.train_fig, self.train_ax = plt.subplots(nrows=1, ncols=2)

        ax_loss, ax_acc = self.train_ax

        canvas = FigureCanvasTkAgg(self.train_fig, master=self.Training_Plots_Output_Frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        ax_loss.plot(hist.history['loss'], color='teal', label='loss')
        ax_loss.plot(hist.history['val_loss'], color='orange', label='val_loss')
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Loss", fontsize=17)

        ax_acc.plot(hist.history['accuracy'], color='teal', label='accuracy')
        ax_acc.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy %")
        ax_acc.set_title("Accuracy", fontsize=17)

        self.train_fig.suptitle('Training Metrics', fontsize=20)
        ax_loss.legend(loc="upper left")
        ax_acc.legend(loc="upper left")
        self.train_fig.tight_layout()
        canvas.draw()

        # Add a navigation toolbar to the Figure_Frame
        toolbar = NavigationToolbar2Tk(canvas, self.Training_Plots_Output_Frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.LEFT, fill=tk.X)


    

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#



class Plot_Selection_Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        tk.Label(self, text="Plot Figures", font=("Helvetica", 16)).pack(pady=10, padx=10, anchor='w')
        
        tk.Button(self, text="Create Scatter Plot",
                    command=lambda: controller.set_plot_type('scatter')).pack(anchor='w', pady=2)

        tk.Button(self, text="Create Line Plot",
                    command=lambda: controller.set_plot_type('line')  ).pack(anchor='w', pady=2)

        tk.Button(self, text="Create Histogram",
                    command=lambda: controller.set_plot_type('hist')).pack(anchor='w', pady=2)

        tk.Button(self, text="Custom Plot",
                    command=lambda: controller.set_plot_type('custom') ).pack(anchor='w', pady=2)    

        tk.Button(self, text='Plot Settings',
                    command=lambda: controller.show_frame(Dataset_View_Page) , state= 'disabled').pack(anchor='w', pady=2)

        tk.Button(self, text="Back to Start Page",
                    command=lambda: controller.show_frame(StartPage)).pack(anchor='w', pady=2)



    def create_line_plot(self):
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        plt.figure()
        plt.plot(x, y)
        plt.title("Line Plot")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.show()


class Figure_Creation_Page(tk.Frame):
    def __init__(self, parent, controller):

        super().__init__(parent)
        self.controller = controller

        # Locate all the column names within the "segments" dataset and make them CheckButtons that can be pressed
        try:
            path = os.path.join( controller.Data_Directory , controller.File_Names[0] ) 
            with h5py.File(path , 'r') as sim_h5:
                column_names  = pd.DataFrame( sim_h5['segments'][()][:1] ).columns.to_list() 


        except:
            pass
        
        self.back_frame = tk.Frame(self)
        self.back_frame.pack(anchor='w', pady = 5)
        self.Back_Button = tk.Button(
                                self.back_frame, text='Back to Figure Selection' , command= lambda: controller.show_frame(Plot_Selection_Page)
                                        )
        self.Back_Button.pack(anchor='w', side=tk.LEFT, padx = 10)

        # Page Title
        Page_Title_str = tk.StringVar()
        Page_Title = tk.Label(self.back_frame, text="Figure Creation", font=("Helvetica", 16) , textvariable = Page_Title_str ).pack(anchor='w', pady=(0, 10) , padx=50)



        # Progress Bar and Percentage Frame
        self.progressive_frame = tk.Frame(self)
        self.progressive_frame.pack(anchor='w', padx=10, pady=(0, 20))  

        self.progress = ttk.Progressbar( self.progressive_frame,  orient="horizontal", length=600,  mode="determinate" ) 
        self.progress_label = tk.Label(self.progressive_frame , text = '' , font=("Arial", 12))
        self.progress.pack(anchor='w' , side=tk.LEFT)
        self.progress_label.pack(anchor='w' , side=tk.LEFT )

        # Create file selection Frame 
        self.file_select_frame = tk.Frame(self)
        self.file_select_frame.pack(anchor='w', pady=20)

        self.file_selected = tk.StringVar()

        tk.Label(self.file_select_frame , text="file : ").pack(side=tk.LEFT)
        self.file_combobox = ttk.Combobox(
            self.file_select_frame , textvariable = self.file_selected , values = controller.Allowed_Files , state='readonly' , width=60
        )
        self.file_combobox.pack(anchor='w' , side = tk.LEFT)

        # Bind the dropdown selection to update the "event_id" selection dropdown
        self.file_selected.trace('w', self.on_file_selected)

        # Add a button for toggling on and off 3d Plots

        self.Dropdown_3D_frame = tk.Frame(self)
        self.Dropdown_3D_frame.pack(anchor='w' , pady=5)


        # Add a Frame to organize dropdowns in one row
        axis_select_frame = tk.Frame(self)
        axis_select_frame.pack(anchor='w', pady=20)
        

        # Variables to hold selected values for dropdown menus
        self.x_selected = tk.StringVar()
        self.y_selected = tk.StringVar()
        self.z_selected = tk.StringVar()

        tk.Label(axis_select_frame, text='x axis: ').pack(side=tk.LEFT)
        self.x_combobox = ttk.Combobox(
            axis_select_frame, textvariable=self.x_selected, values=column_names, state='readonly', width=10
        )
        self.x_combobox.pack(anchor='w',side=tk.LEFT , padx=6 )

    
        tk.Label(axis_select_frame, text='y axis: ').pack(side=tk.LEFT)

        self.y_combobox = ttk.Combobox(
            axis_select_frame, textvariable=self.y_selected, values=column_names, state='readonly', width=10
        )
        self.y_combobox.pack(anchor='w',side=tk.LEFT)

        tk.Label(axis_select_frame, text='z axis: ').pack(side=tk.LEFT)

        self.z_combobox = ttk.Combobox(
            axis_select_frame, textvariable=self.z_selected, values=column_names, state='disabled', width=10
        )
        self.z_combobox.pack(anchor='w',side=tk.LEFT)

        # Variable to hold the cmap selection
        self.cmap_yes_no = tk.StringVar()
        self.cmap_option_select = tk.StringVar()

        self.colour_map_frame = tk.Frame(self)
        self.colour_map_frame.pack(anchor='w', pady=5)

        tk.Label(self.colour_map_frame, text="cmap:  ").pack(side=tk.LEFT)

        self.cmap_combobox = ttk.Combobox(
            self.colour_map_frame, textvariable=self.cmap_yes_no, values=['No', 'Yes'], 
            state='readonly', width=10,
        )
        self.cmap_combobox.set('No')
        self.cmap_combobox.pack(anchor='w', side=tk.LEFT)
        
        self.cmap_selection_combobox = ttk.Combobox(
            self.colour_map_frame, textvariable=self.cmap_option_select, 
            values=['viridis', 'plasma', 'inferno', 'magma', 'cividis'], 
            width=10, state='disabled'
        )
        self.cmap_selection_combobox.pack(anchor='w', side=tk.LEFT)
        self.cmap_yes_no.trace('w', lambda *args: self.Lock_Unlock_Cmap(self.cmap_yes_no, self.cmap_selection_combobox))


        if self.controller.plot_type == 'scatter':

            tk.Label( self.Dropdown_3D_frame , text= "3D:").pack( side=tk.LEFT , padx = 10 )
            self.dropdown_3d_select = tk.StringVar()
            self.dropdown_3d = ttk.Combobox( self.Dropdown_3D_frame , textvariable= self.dropdown_3d_select , values= ['No' , 'Yes'] , width=10  )
            self.dropdown_3d.pack( side= tk.LEFT)
            self.dropdown_3d_select.trace('w', lambda *args: self.Lock_Unlock_Cmap(self.dropdown_3d_select, self.z_combobox))

            tk.Label( self.Dropdown_3D_frame , text = "pixel array: " ).pack(side = tk.LEFT , padx= 10)
            self.pixel_array_select = tk.StringVar()
            self.dropdown_pixel = ttk.Combobox( self.Dropdown_3D_frame , textvariable= self.pixel_array_select , values= ['No' , 'Yes'] , width=10 )
            self.dropdown_pixel.pack( side = tk.LEFT)

            # self.pixel_array_select.trace('w', lambda *args: self.Lock_Unlock_Cmap(self.pixel_array_select, self.z_combobox))


            Page_Title_str.set("Figure Creation : Scatter Plot")
        
        if self.controller.plot_type == 'line': 
            self.particle_select = tk.StringVar()

            self.particle_select_frame = tk.Frame(self)
            self.particle_select_frame.pack(anchor='w', pady=5)

            tk.Label(self.particle_select_frame, text="particle: ").pack(side=tk.LEFT)

            self.particle_select_combobox = ttk.Combobox(
                self.particle_select_frame, textvariable=self.particle_select, 
                values= list( self.controller.pdg_id_map.values() ) , width=10)

            self.particle_select_combobox.pack(anchor='w', side=tk.LEFT)

            Page_Title_str.set("Figure Creation : Line Plot")


        if self.controller.plot_type == 'hist':
            self.group_yes_no = tk.StringVar()
            self.hist_option_select = tk.StringVar()

            self.hist_group_frame = tk.Frame(self)
            self.hist_group_frame.pack(anchor='w', pady=5)

            tk.Label(self.hist_group_frame, text="group: ").pack(side=tk.LEFT)

            self.hist_group_combobox = ttk.Combobox(
                self.hist_group_frame, textvariable=self.group_yes_no, 
                values=['No', 'Yes'], state='readonly', width=10,
            )
            self.hist_group_combobox.set('No')
            self.hist_group_combobox.pack(anchor='w', side=tk.LEFT)

            self.hist_selection_combobox = ttk.Combobox(
                self.hist_group_frame, textvariable=self.hist_option_select, 
                values=column_names, width=10, state='disabled'
            )
            self.hist_selection_combobox.pack(anchor='w', side=tk.LEFT)
            
            self.y_combobox.set('')
            self.y_combobox.state(["disabled"])
            self.group_yes_no.trace('w', lambda *args: self.Lock_Unlock_Cmap(self.group_yes_no, self.hist_selection_combobox))


            self.cmap_combobox.set('')
            self.cmap_combobox.state(['disabled'])

            Page_Title_str.set("Figure Creation : Hist Plot")
      


        # Add "Create Button" button below which can create a plt fig 
        self.Create_Fig_Button = tk.Button(self, text='Create' , command= lambda : self.Plot_Type_Map(self) )
        self.Create_Fig_Button.pack(anchor='w', pady=10)

        self.Figure_Frame = tk.Frame(self)
        self.Figure_Frame.pack(anchor='w', side= tk.LEFT ,pady=5)


    def Lock_Unlock_Cmap(self, yes_no, selection_combobox, *args):
        # Switch the state of the selection_combobox dropdown

        if yes_no.get() == 'Yes':
            selection_combobox.config(state='readonly')
        elif yes_no.get() == 'No':
            selection_combobox.set('')
            selection_combobox.config(state='disabled')
        



    def on_file_selected(self, *args):
        """Callback triggered when a new file is selected from the dropdown."""

        path = os.path.join( self.controller.Data_Directory , self.file_selected.get() )
        self.event_id_selected = tk.StringVar()
        with h5py.File(path , 'r') as sim_h5:
                unique_event_ids = list(np.unique( sim_h5["segments"]['event_id'] ))

        if hasattr(self, 'event_combobox') and self.event_combobox:
            self.event_combobox['values'] = unique_event_ids  
            self.event_combobox.set('')  

        else:
            tk.Label(self.file_select_frame , text= "event id :").pack(padx=(10,10) , side=tk.LEFT ) 
            self.event_combobox = ttk.Combobox(
                self.file_select_frame,
                textvariable=self.event_id_selected,
                values=unique_event_ids,
                state='readonly'
            )
            self.event_combobox.pack(anchor='w', side=tk.LEFT)


    def Plot_Type_Map(self, *args):
        if self.controller.plot_type == 'scatter':
            
            if self.pixel_array_select.get() != 'Yes' : 
                Generic_Plot_script.Generic_Plot.Create_Scatter_Fig(self)

            else:
                Frame_Manager.setup_process(self)

            pass

        elif self.controller.plot_type == 'line':

            Generic_Plot_script.Generic_Plot.Create_Line_PLot(self)

            pass

        elif self.controller.plot_type == 'hist':

            Generic_Plot_script.Generic_Plot.Create_Hist_Fig(self)

            pass

        return
    

        
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#


class Custom_Figure_Page(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent)

        # Page Title
        tk.Label(self, text="Custom Figure", font=("Helvetica", 16)).pack(anchor='w', pady=(0, 10))

        self.controller = controller
        # Locate all the column names within the "segments" dataset and make them CheckButtons that can be pressed
        try:
            path = os.path.join( controller.Data_Directory , controller.File_Names[0] ) 
            with h5py.File(path , 'r') as sim_h5:
                column_names  = pd.DataFrame( sim_h5['segments'][()][:1] ).columns.to_list() 


        except:
            pass
        
        self.back_frame = tk.Frame(self)
        self.back_frame.pack(anchor='w', pady = 5)
        self.Back_Button = tk.Button(
                                self.back_frame, text='Back to Figure Selection' , command= lambda: controller.show_frame(Plot_Selection_Page)
                                        )
        self.Back_Button.pack(anchor='w', side=tk.LEFT)

        # Select custom plot 

        self.custom_fig_select_frame = tk.Frame(self)
        self.custom_fig_select_frame.pack(anchor='w', pady=20)

        self.custom_fig_seleceted = tk.StringVar()

        tk.Label(self.custom_fig_select_frame , text="Custom Figure: ").pack(side=tk.LEFT)
        
    
        Custom_Plot_Names = [attr for attr in dir(Custom_Plot_script.Custom_Plot) if callable(getattr(Custom_Plot_script.Custom_Plot, attr)) and not attr.startswith("__")]

        self.custom_combobox = ttk.Combobox(
            self.custom_fig_select_frame , textvariable = self.custom_fig_seleceted , values = Custom_Plot_Names , state='readonly' , width=20
        )

        self.custom_fig_seleceted.trace('w', self.on_custom_selected)

        self.custom_combobox.pack(anchor='w' , side = tk.LEFT)

        # Create file selection Frame 

        self.file_select_frame = tk.Frame(self)
        self.file_select_frame.pack(anchor='w', pady=20)

        self.file_selected = tk.StringVar()

        tk.Label(self.file_select_frame , text="file : ").pack(side=tk.LEFT)
        self.file_combobox = ttk.Combobox(
            self.file_select_frame , textvariable = self.file_selected , values = controller.Allowed_Files , state='readonly' , width=55
        )
        self.file_combobox.pack(anchor='w' , side = tk.LEFT)



        self.particle_select = tk.StringVar()

        self.particle_frame = tk.Frame(self)
        self.particle_frame.pack(anchor='w' )
        tk.Label( self.particle_frame , text = "particle :" ).pack(anchor='w', side=tk.LEFT)

        self.particle_combobox = ttk.Combobox(
                                        self.particle_frame, textvariable = self.particle_select ,  values = list( self.controller.pdg_id_map.values() )
                                                )
        self.particle_combobox.pack(anchor='w', side=tk.LEFT)



        self.plot_button_frame = tk.Frame(self)
        self.plot_button_frame.pack(anchor='w', pady = 5)
        self.Plot_Button = tk.Button(
                                self.plot_button_frame, text='Plot' , command= lambda : self.Custom_Selection()
                                        )

        self.Plot_Button.pack(anchor='w', side=tk.LEFT)

        self.Custom_Figure_Frame = tk.Frame(self)
        self.Custom_Figure_Frame.pack(anchor='w', side= tk.LEFT ,pady=5)



    def Custom_Selection(self):

        if hasattr(self, 'fig'):
            plt.close(self.custom_fig)  # Close the old figure
        for widget in self.Custom_Figure_Frame.winfo_children():
            widget.destroy()


        # if not hasattr(self, 'particle_frame'):


        if str(self.custom_fig_seleceted.get()) == 'Track_dE_Analysis':
            self.custom_fig = plt.figure( figsize=(6, 6) )
            canvas = FigureCanvasTkAgg( self.custom_fig, master= self.Custom_Figure_Frame)  
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            if not hasattr(self, "Download_dir"):
                self.Download_dir = tk.Button( self.plot_button_frame, text='Download All File Plots', command= lambda:  Custom_Plot_script.Custom_Plot.Track_dE_Analysis(self , {'Plot_Mode' : 'Download_Dir' ,  'canvas' : canvas  ,'fig' : self.custom_fig   } ) )  
                self.Download_dir.pack(anchor='w', side=tk.LEFT )

 

            # Custom_Plot_script.Custom_Plot.Track_dE_Analysis(self , {   'Plot_Mode'     : 'Single_Plot' , 
            #                                                             'canvas'        : canvas        ,
            #                                                             'fig'           : self.custom_fig,   } ) 


            getattr(Custom_Plot_script.Custom_Plot, str(self.custom_fig_seleceted.get()) )( self , {   'Plot_Mode'     : 'Single_Plot' , 
                                                                                                        'canvas'        : canvas        ,
                                                                                                        'fig'           : self.custom_fig,   } )
            


        getattr(Custom_Plot_script.Custom_Plot, str(self.custom_fig_seleceted.get()) )( self , {   'Plot_Mode'     : 'Single_Plot' , 
                                                                                            'canvas'        : 1        ,
                                                                                            'fig'           : 1,   } )
                





    def on_custom_selected(self, *args):
        """Callback triggered when a new custom figure is selected from the dropdown."""


        if self.custom_fig_seleceted.get() ==  'Track_dE_Analysis':
            
            self.file_selected.trace('w', self.on_file_selected)


    def on_file_selected(self, *args):
        """Callback triggered when a new file is selected from the dropdown."""

        path = os.path.join( self.controller.Data_Directory , self.file_selected.get() )
        self.event_id_selected = tk.StringVar()
        with h5py.File(path , 'r') as sim_h5:
                unique_event_ids = list(np.unique( sim_h5["segments"]['event_id'] ))

        if hasattr(self, 'event_combobox') and self.event_combobox:
            self.event_combobox['values'] = unique_event_ids  
            self.event_combobox.set('')  

        else:
            tk.Label(self.file_select_frame , text= "event id :").pack(padx=(10,10) , side=tk.LEFT ) 
            self.event_combobox = ttk.Combobox(
                self.file_select_frame,
                textvariable=self.event_id_selected,
                values=unique_event_ids,
                width=5,
                state='readonly'
            )
            self.event_combobox.pack(anchor='w', side=tk.LEFT)

            self.event_id_selected.trace('w', self.on_event_selected)



    def on_event_selected(self, *args):
        """Callback triggered when a new event is selected from the dropdown."""

        path = os.path.join( self.controller.Data_Directory , self.file_selected.get() )
        self.vertex_id_selected = tk.StringVar()
        with h5py.File(path , 'r') as sim_h5:
                sim_h5 = sim_h5["segments"]
                event_segment = sim_h5[ (sim_h5['event_id'] == int(self.event_id_selected.get()) ) ]
                unique_vertex_ids = list(np.unique( event_segment['vertex_id'] ))


        if hasattr(self, 'vertex_combobox') and self.vertex_combobox:
            self.vertex_combobox['values'] = unique_vertex_ids  
            self.vertex_combobox.set('')  

        else:
            tk.Label(self.file_select_frame , text= "vertex id :").pack(padx=(10,10) , side=tk.LEFT ) 
            self.vertex_combobox = ttk.Combobox(
                self.file_select_frame,
                textvariable=self.vertex_id_selected,
                values=unique_vertex_ids,
                width=10,
                state='readonly'
            )
            self.vertex_combobox.pack(anchor='w', side=tk.LEFT)



        return

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#



class Settings_Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        page_title_frame = tk.Frame(self)
        page_title_frame.pack( anchor= 'w' , pady=(10, 30))

        tk.Button(page_title_frame , text = "back", command= lambda: controller.show_frame(StartPage) ).pack( padx=10 , anchor='w', side = tk.LEFT)
        tk.Label(page_title_frame, text="Settings", font=("Helvetica", 16)).pack( padx=50 , anchor='w' , side = tk.LEFT)

        tk.Button(self, text='Select Files',
                  command= lambda: controller.show_frame(File_Selection_Page)).pack( anchor='w' , padx=10  )
        

        tk.Button(self, text='Cleaning Method',
                command= lambda: controller.show_frame(Cleaning_Method_Select_Page) , state='disabled' ).pack( anchor='w' , padx=10 )
        

        



#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#


class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        # Create a canvas
        canvas = tk.Canvas(self, borderwidth=0 , width=600 , height=100)
        canvas.pack(side="left", fill="both", expand=True)

        # Create a scrollbar
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")

        # Frame that will hold the actual widgets
        self.scrollable_frame = tk.Frame(canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")   # Update scrollregion to encompass all widgets
            )
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)


#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#


class File_Selection_Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.file_vars = []  # List to hold IntVar for each file Checkbutton

        # Title and Back Navigation
        page_title_frame = tk.Frame(self)
        page_title_frame.pack( anchor= 'w' , pady=(10, 30))

        tk.Button(page_title_frame , text = "back", command= lambda: controller.show_frame(Settings_Page) ).pack( padx=10 , anchor='w', side = tk.LEFT)
        tk.Label(page_title_frame, text="File Selection", font=("Helvetica", 16)).pack( padx=50 , anchor='w' , side = tk.LEFT)

        # Frame for Select/Deselect buttons
        button_frame = tk.Frame(self)
        button_frame.pack(pady=5, padx=10, anchor='w')

        # Select All Button
        select_all_button = tk.Button(button_frame, text="Select All", command=self.select_all)
        select_all_button.pack(side=tk.LEFT, padx=(0, 5))

        # Deselect All Button
        deselect_all_button = tk.Button(button_frame, text="Deselect All", command=self.deselect_all)
        deselect_all_button.pack(side=tk.LEFT)

        # Access Data_Directory via controller
        if hasattr(controller, 'Data_Directory') and controller.Data_Directory:
            try:
                # Assuming controller.File_Names is a list of filenames
                File_Names = controller.File_Names


                if File_Names:
                    tk.Label(self, text="Files:", font=("Helvetica", 12)).pack(pady=5, anchor='w', padx=10)
                    
                    # Create a scrollable frame
                    scrollable_frame = ScrollableFrame(self)
                    scrollable_frame.pack(pady=5, padx=10, fill="both", expand=True)

                    # Add file names to the scrollable frame
                    for file in File_Names:
                        var = tk.IntVar()
                        file_check = tk.Checkbutton(
                            scrollable_frame.scrollable_frame,
                            text=file,
                            anchor="w",
                            variable=var
                        )
                        file_check.pack(fill="x", padx=10, pady=2)
                        self.file_vars.append(var)
                else:
                    tk.Label(self, text="No files found in the directory.").pack(pady=5, padx=10, anchor='w')
            except FileNotFoundError:
                tk.Label(self, text="Data Directory not found.").pack(pady=5, padx=10, anchor='w')
            except Exception as e:
                tk.Label(self, text=f"Error: {e}").pack(pady=5, padx=10, anchor='w')
        else:
            tk.Label(self, text="No Data Directory provided.", fg="red").pack(pady=5, padx=10, anchor='w')

        # "Confirm Selection" button to update the Allowed_Files list
        confirm_button = tk.Button(self, text="Confirm Selection", command=self.confirm_selection)
        confirm_button.pack(pady=10, padx=10, anchor='w')



    def select_all(self):
        """Select all file checkboxes."""
        for var in self.file_vars:
            var.set(1)

    def deselect_all(self):
        """Deselect all file checkboxes."""
        for var in self.file_vars:
            var.set(0)

    def get_selected_files(self):
        """Retrieve the list of selected files."""
        selected_files = [file for file, var in zip(self.controller.File_Names, self.file_vars) if var.get()]
        return selected_files

    def confirm_selection(self):
        """Update the Allowed_Files in the controller based on user selection."""

        selected = self.get_selected_files()
        if not selected:
            # If no specific files selected, show all files
            self.controller.Allowed_Files = sorted(self.controller.File_Names)
        else:
            self.controller.Allowed_Files = sorted(selected) # Update the Allowed_Files list


        # Reinitialize frames that rely on Allowed_Files
        self.controller.reinitialize_frame(View_Segments_Page)
        self.controller.reinitialize_frame(View_mc_hdr_Page)
        self.controller.reinitialize_frame(View_traj_Page)
        self.controller.reinitialize_frame(Figure_Creation_Page)
        self.controller.reinitialize_frame(Custom_Figure_Page)
        self.controller.reinitialize_frame(Create_Dataset_Page)

        self.controller.show_frame(File_Selection_Page)

        print("Allowed Files Updated:", self.controller.Allowed_Files)


#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#

class Cleaning_Method_Select_Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.file_vars = []  # List to hold IntVar for each file Checkbutton

        # Header Frame: Back button and Header label
        self.header_frame = tk.Frame(self)
        self.header_frame.pack( side = tk.LEFT , anchor='nw', padx=10, pady=20)  

        # Back Button
        back_button = tk.Button( self.header_frame,  text='Back',   command=lambda: controller.show_frame(Settings_Page)  )
        back_button.pack(side=tk.LEFT , anchor='w')

        # Header Label
        header_label = tk.Label( self.header_frame,  text="Cleanning Methood Selection",  font=("Helvetica", 16) )
        header_label.pack(side=tk.LEFT, anchor = 'w',  padx=50)

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#

class Frame_Manager():

    def __init__(self, frame):
        self.frame = frame
        self.controller = frame.controller  # Access the controller from the frame

    def update_dropdown(self):
        """Refresh the dropdown menu with the latest Allowed_Files."""
        menu = self.files_drop_down["menu"]
        menu.delete(0, "end")  # Clear existing options

    def refresh_content(self):
        """Refresh dropdown and DataFrame display when the frame is shown."""
        self.update_dropdown()
        self.Display_DF_in_Frame(self.selected_file.get())


    def refresh_frame(self):
        """Clear all widgets in the display_frame."""
        for widget in self.display_frame.winfo_children():
            widget.destroy()

    def Display_DF_in_Frame(self, dropdown_file_name , h5_data_name = str):
        """Display the DataFrame based on the selected file."""
        Frame_Manager.refresh_frame(self)  # Clear previous content


        # Display the selected file name
        tk.Label(self.display_frame, text=f"Selected File: {dropdown_file_name}", font=("Helvetica", 12)).pack(anchor='w')

        # Construct the file path
        File_path = os.path.join(self.controller.Data_Directory, dropdown_file_name)

        try:

            if h5_data_name == 'mc_hdr':
                h5_DataFrame = pd.DataFrame.from_records( h5py.File(File_path)[ str(h5_data_name) ]  , columns = np.dtype( h5py.File(File_path)[ str(h5_data_name) ] ).names ) 

            else:
                h5_DataFrame = pd.DataFrame( h5py.File(File_path)[ str(h5_data_name) ][()] )


            self.Event_IDs = np.unique(h5_DataFrame['event_id'])


            if not self.Event_IDs.size:
                raise ValueError("No Event IDs found in the selected file.")

            # Ensure Event_ID_selection is within bounds
            self.Event_ID_selection = max(0, min(self.Event_ID_selection, len(self.Event_IDs) - 1))

            # Get the current Event ID
            current_event_id = self.Event_IDs[self.Event_ID_selection]
            h5_DataFrame_event = h5_DataFrame[h5_DataFrame['event_id'] == current_event_id]
            os.system('cls||clear')
            h5_DataFrame_event = pd.DataFrame(h5_DataFrame_event)

            # Update Event Counter Label
            self.event_counter_label.config(text=f"Event {self.Event_ID_selection + 1} of {len(self.Event_IDs)}")

            # Update navigation buttons' state
            Frame_Manager.update_navigation_buttons(self)

            # Create a Treeview widget to display the DataFrame
            self.tree = ttk.Treeview(self.display_frame, columns=list(h5_DataFrame_event.columns), show="headings")

            # Configure Treeview Style
            style = ttk.Style()
            style.configure("Treeview", font=("Helvetica", 7))  # Row font size
            style.configure("Treeview.Heading", font=("Helvetica", 8, "bold"))  # Header font size

            # Define columns and headings
            for col in h5_DataFrame_event.columns:

                self.tree.heading(col, text=col)
                self.tree.column(col, width=52, anchor="center")  # Adjust width as needed

            # Insert DataFrame rows into the Treeview
            for _, row in h5_DataFrame_event.iterrows():
                self.tree.insert("", "end", values=list(row))

            # Add a vertical scrollbar to the Treeview
            self.scrollbar = ttk.Scrollbar(self.display_frame, orient="vertical", command=self.tree.yview)
            self.tree.configure(yscrollcommand=self.scrollbar.set)
            self.scrollbar.pack(side="right", fill="y")

            # Pack the Treeview
            self.tree.pack(fill="both", expand=True)

        except Exception as e:
            # Display error message if file loading fails
            tk.Label(self.display_frame, text=f"Error loading file: {e}", fg="red").pack(anchor='w')
    

    def go_back(self):
        """Navigate to the previous event."""
        if self.Event_ID_selection > 0:
            self.Event_ID_selection -= 1
            self.Display_DF_in_Frame( self.selected_file.get() )

    def go_next(self):
        """Navigate to the next event."""
        if self.Event_ID_selection < len(self.Event_IDs) - 1:
            self.Event_ID_selection += 1
            self.Display_DF_in_Frame( self.selected_file.get() )

    def update_navigation_buttons(self):
        """Enable or disable navigation buttons based on the current event selection."""
        if self.Event_ID_selection <= 0:
            self.back_button.config(state=tk.DISABLED)
        else:
            self.back_button.config(state=tk.NORMAL)

        if self.Event_ID_selection >= len(self.Event_IDs) - 1:
            self.next_button.config(state=tk.DISABLED)
        else:
            self.next_button.config(state=tk.NORMAL)

    def go_back(self):
        """Navigate to the previous event."""
        if self.Event_ID_selection > 0:
            self.Event_ID_selection -= 1
            Frame_Manager.Display_DF_in_Frame(self , self.selected_file.get() , self.h5_data_name )

    def go_next(self):
        """Navigate to the next event."""
        if self.Event_ID_selection < len(self.Event_IDs) - 1:
            self.Event_ID_selection += 1
            Frame_Manager.Display_DF_in_Frame(self , self.selected_file.get() , self.h5_data_name)

    def on_file_selected(self , h5_data_name ):
        """Callback triggered when a new file is selected from the dropdown."""
        selected_file = self.selected_file.get()
        self.Event_ID_selection = 0  # Reset to first event when a new file is selected
        Frame_Manager.Display_DF_in_Frame(self , selected_file , h5_data_name)


    def update_dropdown(self):
        """Refresh the dropdown menu with the latest Allowed_Files."""
        menu = self.files_drop_down["menu"]
        menu.delete(0, "end")  # Clear existing options

        # Retrieve the latest Allowed_Files from the controller
        if hasattr(self.controller, 'Allowed_Files') and self.controller.Allowed_Files:
            files = self.controller.Allowed_Files
        elif hasattr(self.controller, 'File_Names') and self.controller.File_Names:
            files = self.controller.File_Names
        else:
            files = ["No Files Available"]

        # Populate the dropdown menu
        for file in files:
            menu.add_command(label=file, command=lambda value=file: self.selected_file.set(value))

        # Set the default selection
        if files and files[0] != "No Files Available":
            self.selected_file.set(files[0])
        else:
            self.selected_file.set("No Files Available")

    def select_directory_window(self , Text_Box ):
        
        directory_path = tk.filedialog.askdirectory( initialdir= str(os.getcwd()) , title="Select a Directory")

        # Text_Box.config( state = 'normal')
        # Text_Box.set( str(directory_path) )
        if directory_path:
            Text_Box.config( state = 'normal' )
            Text_Box.delete( 0 , tk.END)
            Text_Box.insert( 0, directory_path)
            Text_Box.config(state = 'disabled')

        else:
            Text_Box.config(state='normal')
            Text_Box.delete("1.0", tk.END)
            Text_Box.insert("1.0", directory_path)
            Text_Box.config(state='disabled')
        return

    def setup_process(self):
        if not self.controller.running:
            self.controller.running = True
            # self.frame.Create_Dataset_Button.config(state='disabled')
            
            self.progress_value = 0
            self.progress['value'] = 0
            self.progress.config(maximum=100) 

            self.controller.running = True


            if str(self.__class__.__name__) == 'Create_Dataset_Page':
                self.Create_Dataset_Button.config(state='disabled')
                Frame_Manager.check_progress(self)

                threading.Thread(target=self.Create_ML_Dataset_2).start()


            else:
                print(self.__class__.__name__ )
                # self.Create_Fig_Button.config(state='disabled')
                Frame_Manager.check_progress(self)

                threading.Thread(target=Pixel_Array_Script.Use_Pixel_Array.plot, args=(self,)).start()

    def cancel_process(self):
        if self.controller.running:
            self.progress['value'] = 100
            self.controller.running = False
            print('Cancelled' , self.controller.running )


    def check_progress(self):
        self.progress['value'] = self.progress_value
        self.progress_label.config(text=f"{self.progress_value:.2f}%")
        # print("running")
        if self.controller.running:
            # self.after(100, self.check_progress)  # Check every 100 ms
            self.after(100, lambda : Frame_Manager.check_progress(self))  # Check every 100 ms

        else:
            if str(self.__class__.__name__) == 'Create_Dataset_Page':
                # self.frame.Create_Dataset_Button.config(state='normal')
                pass 


#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--Data_Directory', required=True, type=str, help='Path to simulation directory')
    args = parser.parse_args()
    app = App( Data_Directory=args.Data_Directory )
    app.mainloop()
