import os
import h5py
import argparse
import numpy as np
import pandas as pd
import sklearn

import tkinter as tk
from tkinter import ttk

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg , NavigationToolbar2Tk)
from matplotlib.backends.backend_pdf import PdfPages

from PIL import Image, ImageTk # Used For displaying embedded images (png files)

import sklearn.metrics
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision

from tensorflow.keras.losses import CategoricalFocalCrossentropy
from focal_loss import SparseCategoricalFocalLoss

from keras import models, layers

import matplotlib
matplotlib.use('agg')  # If I don't change the backend of matplotlib while creating plots it will clash with my progress bar backend and crash my window 😅

import threading

# Backend scripts:
import pdg_id_script
import Generic_Plot_script
import Custom_Plot_script
import Pixel_Array_Script
import Model_Training_script
import Model_Evaluation_script


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Colour-Blind Stuff

from cycler import cycler

# Retrieve the colors from the 'tab20' colormap
colors = plt.get_cmap('tab20').colors

# Set the color cycle for all plots
plt.rcParams['axes.prop_cycle'] = cycler('color', colors)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class App(tk.Tk):

    def toggle_fullscreen_event(self, event=None):
        self.toggle_fullscreen()

    def toggle_fullscreen(self):
        # Toggle full-screen mode.
        self.is_fullscreen = not self.is_fullscreen
        self.attributes("-fullscreen", self.is_fullscreen)

    def enter_fullscreen_event(self, event=None):
        #  Exit and Entering Full screen func. 
        self.is_fullscreen = True
        self.attributes("-fullscreen", True)

    def exit_fullscreen_event(self, event=None):
        #  Exit and Entering Full screen func. 
        self.is_fullscreen = False
        self.attributes("-fullscreen", False)

    def _destroy_frame(self, frame_class):
        # handle destroying a frame
        frame = self.frames.get(frame_class)
        if frame:
            frame.destroy()
            del self.frames[frame_class]
            # print(f"{frame_class.__name__} has been destroyed.")
        else:
            pass 

    def _reinitialize_frame(self, frame_class):
        # method to handle re-initialize a frame.
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.selected_directory = ''  # This may have to be removed, testing for stability

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.running = False

        self.model   = None
        self.model_learning_rate  = 0.0001
        # self.model.history_dict = None
        self.test_images = None
        self.is_fullscreen = False  # Track the full-screen state
        

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
            Advance_Class_Selection_Page,
            Monitor_Training_Page,
            Show_Confusion_Page,
            Show_Model_Filters_Page,
            Evaluate_Model_Page,
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

        if page == StartPage:
            self.geometry("400x300")  # Default size for other pages

        elif page == File_Selection_Page:
            self.geometry("1000x800")  # Larger size for File Selection Page
        
        elif page == Advance_Class_Selection_Page:
            self.geometry("1600x900") 
            self.update_idletasks() 


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
            self.geometry('1100x750')

        elif page == Model_Training_Page or page == Evaluate_Model_Page :
            self.geometry('1000x500')

        elif page == Load_Dataset_Page:
            self.geometry('1000x700')

        elif page == Show_Confusion_Page or page == Show_Model_Filters_Page or page == Monitor_Training_Page:
            pass

        else:
            self.geometry("400x300")  

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


        self.Event_ID_selection = 0
        self.Event_IDs = []


        tk.Label(self, text="View Segments", font=("Helvetica", 16)).pack(anchor='w', pady=(10, 10))


        tk.Button(self, text="Back to View Datasets", command=lambda: controller.show_frame(Dataset_View_Page)).pack(anchor='w', pady=(0, 10))


        file_selection_frame = tk.Frame(self)
        file_selection_frame.pack(anchor='w', pady=(0, 10))


        tk.Label(file_selection_frame, text="Select file:").pack(side=tk.LEFT, padx=(0, 5))


        self.selected_file = tk.StringVar()
        self.files_drop_down = tk.OptionMenu(file_selection_frame, self.selected_file, "")
        self.files_drop_down.pack(side=tk.LEFT)


        self.display_frame = tk.Frame(self)
        self.display_frame.pack(anchor='w', pady=(10, 10))


        Frame_Manager.update_dropdown(self)


        navigation_buttons_frame = tk.Frame(self)
        navigation_buttons_frame.pack(anchor='w', pady=(5, 10))


        self.back_button = tk.Button( navigation_buttons_frame, text="Back",  command=lambda: Frame_Manager.go_back(self) )
        self.back_button.pack(side=tk.LEFT, padx=5)


        self.next_button = tk.Button( navigation_buttons_frame, text="Next",  command=lambda: Frame_Manager.go_next(self)  )
        self.next_button.pack(side=tk.LEFT, padx=5)


        self.event_counter_label = tk.Label(self, text="Event 0 of 0", font=("Helvetica", 10))
        self.event_counter_label.pack(anchor='w', padx=5)

        self.selected_file.trace('w', lambda *args: Frame_Manager.on_file_selected(self , 'segments'))



class View_mc_hdr_Page(tk.Frame):
   def __init__(self, parent, controller):
        super().__init__(parent)

        self.controller = controller  # Reference to the main app
        self.h5_data_name = 'mc_hdr'


        self.Event_ID_selection = 0
        self.Event_IDs = []


        tk.Label(self, text="View mc_hdr", font=("Helvetica", 16)).pack(anchor='w', pady=(10, 10))

        tk.Button(self, text="Back to View Datasets", command=lambda: controller.show_frame(Dataset_View_Page)).pack(anchor='w', pady=(0, 10))


        file_selection_frame = tk.Frame(self)
        file_selection_frame.pack(anchor='w', pady=(0, 10))


        tk.Label(file_selection_frame, text="Select file:").pack(side=tk.LEFT, padx=(0, 5))


        self.selected_file = tk.StringVar()
        self.files_drop_down = tk.OptionMenu(file_selection_frame, self.selected_file, "")
        self.files_drop_down.pack(side=tk.LEFT)

        self.display_frame = tk.Frame(self)
        self.display_frame.pack(anchor='w', pady=(10, 10))


        Frame_Manager.update_dropdown(self)


        navigation_buttons_frame = tk.Frame(self)
        navigation_buttons_frame.pack(anchor='w', pady=(5, 10))


        self.back_button = tk.Button( navigation_buttons_frame, text="Back", command=lambda: Frame_Manager.go_back(self) )
        self.back_button.pack(side=tk.LEFT, padx=5)


        self.next_button = tk.Button( navigation_buttons_frame, text="Next", command=lambda: Frame_Manager.go_next(self) )
        self.next_button.pack(side=tk.LEFT, padx=5)


        self.event_counter_label = tk.Label(self, text="Event 0 of 0", font=("Helvetica", 10))
        self.event_counter_label.pack(anchor='w', padx=5)


        self.selected_file.trace('w', lambda *args: Frame_Manager.on_file_selected(self , 'mc_hdr'))



class View_traj_Page(tk.Frame):
   def __init__(self, parent, controller):
        super().__init__(parent)

        self.controller = controller  # Reference to the main app
        self.h5_data_name = 'trajectories'

        self.Event_ID_selection = 0
        self.Event_IDs = []


        tk.Label(self, text="View trajectories", font=("Helvetica", 16)).pack(anchor='w', pady=(10, 10))


        tk.Button(self, text="Back to View Datasets", command=lambda: controller.show_frame(Dataset_View_Page)).pack(anchor='w', pady=(0, 10))


        file_selection_frame = tk.Frame(self)
        file_selection_frame.pack(anchor='w', pady=(0, 10))


        tk.Label(file_selection_frame, text="Select file:").pack(side=tk.LEFT, padx=(0, 5))


        self.selected_file = tk.StringVar()
        self.files_drop_down = tk.OptionMenu(file_selection_frame, self.selected_file, "")
        self.files_drop_down.pack(side=tk.LEFT)


        self.display_frame = tk.Frame(self)
        self.display_frame.pack(anchor='w', pady=(10, 10))


        Frame_Manager.update_dropdown(self)


        navigation_buttons_frame = tk.Frame(self)
        navigation_buttons_frame.pack(anchor='w', pady=(5, 10))


        self.back_button = tk.Button( navigation_buttons_frame,  text="Back",  command=lambda: Frame_Manager.go_back(self) )
        self.back_button.pack(side=tk.LEFT, padx=5)


        self.next_button = tk.Button( navigation_buttons_frame,  text="Next",  command=lambda: Frame_Manager.go_next(self) )
        self.next_button.pack(side=tk.LEFT, padx=5)


        self.event_counter_label = tk.Label(self, text="Event 0 of 0", font=("Helvetica", 10))
        self.event_counter_label.pack(anchor='w', padx=5)


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


        self.header_frame = tk.Frame(self)
        self.header_frame.pack(anchor='w', padx=10, pady=20)


        back_button = tk.Button(self.header_frame, text='Back', command=lambda: controller.show_frame(Dataset_Page))
        back_button.pack(side=tk.LEFT)


        header_label = tk.Label(self.header_frame, text="Create Dataset", font=("Helvetica", 16))
        header_label.pack(side=tk.LEFT, padx=150)


        self.progressive_frame = tk.Frame(self)
        self.progressive_frame.pack(anchor='w', padx=10, pady=(0, 20))

        self.progress = ttk.Progressbar(self.progressive_frame, orient="horizontal", length=600, mode="determinate")
        self.progress_label = tk.Label(self.progressive_frame, text='', font=("Arial", 12))
        self.progress.pack(anchor='w', side=tk.LEFT)
        self.progress_label.pack(anchor='w', side=tk.LEFT)


        self.file_select_frame = tk.Frame(self)
        self.file_select_frame.pack(anchor='w', padx=10, pady=(0, 20))

        tk.Label(self.file_select_frame, text="Select Files:").pack(anchor='w')

        # Scrollable frame for the file list
        scroll_frame = ScrollableFrame(self.file_select_frame)
        scroll_frame.pack(fill="both", expand=True, pady=5 )

        # Determine which files to show
        allowed_files = getattr(controller, 'Allowed_Files', [])
        if not allowed_files:

            all_files_in_dir = os.listdir(controller.Data_Directory)

            allowed_files = all_files_in_dir


        for file in sorted(allowed_files):
            var = tk.IntVar()
            c = tk.Checkbutton(scroll_frame.scrollable_frame, text=file, variable=var, anchor='w', width=200)
            c.pack(fill='x', padx=5, pady=2)
            self.file_vars.append((var, file))



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
        self.Text_Box_ML_Dataset_Name = tk.Text( self.Interact_Frame, bg='black',  fg='white', font=("Arial", 12), width=35, height=1, padx=10, pady=10 , state = 'normal' )
        self.Text_Box_ML_Dataset_Name.pack(anchor='w', side=tk.LEFT ,padx=(0,10), pady=5)


        tk.Label(self.Interact_Frame , text = "File Tag :" ).pack( anchor='w', side=tk.LEFT ,padx=10 )
        self.Text_Box_File_Tag = tk.Text( self.Interact_Frame, bg='black',  fg='white', font=("Arial", 12), width=10, height=1, padx=10, pady=10 , state = 'normal' )
        self.Text_Box_File_Tag.pack(anchor='w', side=tk.LEFT ,padx=(0,10), pady=5)


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

        if not self.selected_files:
            print("No files selected for dataset creation!")
            self.progress_value = 100
            self.controller.running = False
            self.Create_Dataset_Button.config(state='normal')
            return


        Test_directory = self.Text_Box_ML_Dataset_Name.get('1.0', tk.END).strip()
        File_tag       = self.Text_Box_File_Tag.get('1.0', tk.END).strip()
        print(Test_directory)

        self.Text_Box_ML_Dataset_Name.config(state='disabled')
        self.Cancel_Creation.config(state='normal')
        os.makedirs(Test_directory, exist_ok=True)

        # Map LaTeX labels → subfolders
        Directory_Name_Map = {
            r"$\nu$-$e^{-}$ scattering": "Neutrino_Electron_Scattering",
            r"$\nu_{e}$-CC":            "Electron_Neutrino_CC",
            'QES - CC':                 "QES_CC",
            'QES - NC':                 "QES_NC",
            'MEC - CC':                 "MEC_CC",
            'MEC - NC':                 "MEC_NC",
            'DIS - CC':                 "DIS_CC",
            'DIS - NC':                 "DIS_NC",
            'COH - CC':                 "COH_CC",
            'COH - NC':                 "COH_NC",
            r"$\nu$-CC":                "Neutrino_CC_Other",
            r"$\nu$-NC":                "Neutrino_NC_Other",
        }
        for dirname in Directory_Name_Map.values():
            os.makedirs(os.path.join(Test_directory, dirname), exist_ok=True)

        Dir_File_Name_Counter = {label: 0 for label in Directory_Name_Map.keys()}


        all_event_ids = []
        for fname in self.selected_files:
            path = os.path.join(self.controller.Data_Directory, fname)
            with h5py.File(path, 'r') as sim_h5:
                all_event_ids.extend(
                    np.unique(sim_h5['mc_hdr']['event_id']).tolist()
                )
        all_event_ids = list(set(all_event_ids))
        num_events = len(all_event_ids)

        self.controller.running = True
        self.Create_Dataset_Button.config(state='disabled')


        min_z, max_z = self.controller.min_z_for_plot, self.controller.max_z_for_plot
        min_y, max_y = self.controller.min_y_for_plot, self.controller.max_y_for_plot
        min_x, max_x = self.controller.min_x_for_plot, self.controller.max_x_for_plot

        cnter = 0
        for fname in self.selected_files:
            if not self.controller.running:
                break

            path = os.path.join(self.controller.Data_Directory, fname)
            with h5py.File(path, 'r') as sim_h5:
                seg_ds = sim_h5['segments']

                seg_mask      = seg_ds['dE'] > 1.5
                temp_segments = seg_ds[seg_mask]
                hdr_ds        = sim_h5['mc_hdr']

                for event_id in np.unique(temp_segments['event_id']):
                    cnter += 1
                    self.progress_value = (cnter / num_events) * 100
                    if not self.controller.running:
                        break

                    # build a DataFrame of all surviving segments for this event
                    seg_data = temp_segments[temp_segments['event_id'] == event_id]
                    seg_df   = pd.DataFrame(seg_data)

                    # grab the mc_hdr rows for this event as a numpy structured array
                    hdr_array = hdr_ds[hdr_ds['event_id'] == event_id]
                    mc_hdr_vertex_ids = np.unique(hdr_array['vertex_id']).tolist()

                    # compute which vertex_ids are purely noise
                    noise_ids = list(set(seg_df['vertex_id']) - set(mc_hdr_vertex_ids))

                    for true_v in mc_hdr_vertex_ids:

                        hdr_rows = hdr_array[hdr_array['vertex_id'] == true_v]
                        if hdr_rows.shape[0] != 1:
                            continue


                        z = hdr_rows['z_vert'][0]
                        y = hdr_rows['y_vert'][0]
                        x = hdr_rows['x_vert'][0]


                        if (z <= min_z or z >= max_z or
                            y <= min_y or y >= max_y or
                            x <= min_x or x >= max_x):
                            continue


                        reaction = hdr_rows['reaction'][0]
                        nu_pdg   = hdr_rows['nu_pdg'][0]
                        isCC     = hdr_rows['isCC'][0]
                        isQES    = hdr_rows['isQES'][0]
                        isMEC    = hdr_rows['isMEC'][0]
                        isDIS    = hdr_rows['isDIS'][0]
                        isCOH    = hdr_rows['isCOH'][0]

                        if reaction == 7:
                            interaction_label = r"$\nu$-$e^{-}$ scattering"
                        elif nu_pdg == 12 and isCC:
                            interaction_label = r"$\nu_{e}$-CC"
                        elif isQES and isCC:
                            interaction_label = 'QES - CC'
                        elif isQES and not isCC:
                            interaction_label = 'QES - NC'
                        elif isMEC and isCC:
                            interaction_label = 'MEC - CC'
                        elif isMEC and not isCC:
                            interaction_label = 'MEC - NC'
                        elif isDIS and isCC:
                            interaction_label = 'DIS - CC'
                        elif isDIS and not isCC:
                            interaction_label = 'DIS - NC'
                        elif isCOH and isCC:
                            interaction_label = 'COH - CC'
                        elif isCOH and not isCC:
                            interaction_label = 'COH - NC'
                        elif not any([isCC, isQES, isMEC, isDIS, isCOH]):
                            interaction_label = r"$\nu$-NC"
                        elif isCC and not any([isQES, isMEC, isDIS, isCOH]):
                            interaction_label = r"$\nu$-CC"
                        else:
                            interaction_label = 'Other'

                        # Skip any vertex with no real segments dE>1.5
                        real_df = seg_df[seg_df['vertex_id'] == true_v]
                        if real_df.empty:
                            continue

                        # collect noise if present
                        if noise_ids:
                            noise_df = seg_df[seg_df['vertex_id'].isin(noise_ids)]
                        else:
                            noise_df = pd.DataFrame(columns=seg_df.columns)

                        # final DataFrame + save
                        DF = pd.concat([real_df, noise_df], ignore_index=True)
                        out_dir = os.path.join(Test_directory, Directory_Name_Map.get(interaction_label, 'Other'))
                        out_fname = (
                            f"{File_tag}_IMG_"
                            f"{Dir_File_Name_Counter[interaction_label]}_"
                            f"{event_id}_{true_v}.png"
                        )
                        out_path = os.path.join(out_dir, out_fname)

                        Pixel_Array_Script.Use_Pixel_Array.Save_For_ML(self, DF, out_path)
                        Dir_File_Name_Counter[interaction_label] += 1

            print(f"{fname} (finished)")


        self.progress_value = 100
        self.controller.running = False
        self.Create_Dataset_Button.config(state='normal')
        self.Text_Box_ML_Dataset_Name.config(state='normal')
        self.Cancel_Creation.config(state='disabled')


#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#


class Load_Dataset_Page( tk.Frame  ):

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller


        page_title_frame = tk.Frame(self)
        page_title_frame.pack( anchor= 'w' , pady=10)

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

        sorted_Image_File_Names = sorted(Image_File_Names, key=lambda x: int(x.split('_')[2].split('.')[0]))

        # sorted_Image_File_Names = sorted( Image_File_Names )

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
                  command=lambda: controller.show_frame(Evaluate_Model_Page)).pack( padx=10   ,anchor='w')




class Model_Architecture_Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # Title and Navigation
        page_title_frame = tk.Frame(self)
        page_title_frame.pack(anchor='w', pady=10)
        tk.Button(page_title_frame, text="Back", command=lambda: controller.show_frame(Training_And_Eval_Options_Page)).pack(padx=10, anchor='w', side=tk.LEFT)
        tk.Label(page_title_frame, text="Model Architecture Configuration", font=("Helvetica", 16)).pack(padx=50, anchor='w', side=tk.LEFT)
        tk.Button(page_title_frame, text="Load Model", command=self.load_tf_model).pack(padx=50, anchor='w', side=tk.LEFT)

        # Canvas + scrollbar for scrolling
        self.canvas = tk.Canvas(self, height=670)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill='y')
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.inner_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.inner_frame, anchor='nw')
        self.inner_frame.bind("<Configure>",lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))


        self.layers = []
        self.editing_index = None
        self.update_mode = False   # False = adding, True = editing
        self.output_activation = tk.StringVar(value="softmax")


        self.create_layout()

    def create_layout(self):

        for w in self.inner_frame.winfo_children():
            w.destroy()


        input_shape_frame = tk.Frame(self.inner_frame); input_shape_frame.pack(anchor='w', padx=10, pady=5)
        tk.Label(input_shape_frame, text="Input Shape (H, W, C):").pack(side=tk.LEFT)
        self.input_height = tk.StringVar(value="334")
        self.input_width  = tk.StringVar(value="200")
        self.input_channels = tk.StringVar(value="3")
        tk.Entry(input_shape_frame, textvariable=self.input_height, width=5).pack(side=tk.LEFT)
        tk.Entry(input_shape_frame, textvariable=self.input_width, width=5).pack(side=tk.LEFT, padx=5)
        tk.Entry(input_shape_frame, textvariable=self.input_channels, width=5, state='disabled').pack(side=tk.LEFT)


        lr_frame = tk.Frame(self.inner_frame); lr_frame.pack(anchor='w', padx=10, pady=5)
        tk.Label(lr_frame, text="Learning Rate:").pack(side=tk.LEFT, padx=(0,50))
        self.learning_rate_float = tk.DoubleVar(value=0.0001)
        tk.Entry(lr_frame, textvariable=self.learning_rate_float, width=12).pack(side=tk.LEFT)

        # Layer addition options
        layer_options_frame = tk.Frame(self.inner_frame)
        layer_options_frame.pack(anchor='w', padx=10, pady=5, fill='x')
        tk.Label(layer_options_frame, text="Layer Type:").pack(side=tk.LEFT)
        self.layer_type = tk.StringVar()
        layer_types = ["Conv2D","MaxPooling2D","GlobalAveragePooling2D","Dropout","Flatten","Dense"]
        self.layer_type_combobox = ttk.Combobox( layer_options_frame, textvariable=self.layer_type, values=layer_types, state='readonly', width=15 )
        self.layer_type_combobox.pack(side=tk.LEFT, padx=5)
        self.layer_type_combobox.bind("<<ComboboxSelected>>", self.show_layer_options)

        # Add / Update button
        self.action_button = tk.Button( layer_options_frame, text="Add Layer", command=self.add_or_update_layer )
        self.action_button.pack(side=tk.LEFT, padx=5)
        self.cancel_edit_button = tk.Button( layer_options_frame, text="Cancel Edit", command=self.cancel_edit )
        # Initially hidden:
        self.cancel_edit_button.pack_forget()

        # Container for layer-specific options
        self.layer_options_container = tk.Frame(self.inner_frame)
        self.layer_options_container.pack(anchor='w', padx=20, pady=5, fill='x')

        # Layers list
        layers_list_frame = tk.Frame(self.inner_frame)
        layers_list_frame.pack(anchor='w', padx=10, pady=5, fill='x')
        tk.Label(layers_list_frame, text="Layers:").pack(anchor='w')
        self.layers_container = tk.Frame(layers_list_frame)
        self.layers_container.pack(anchor='w', padx=10, pady=5, fill='x')

        # Output classes + activation
        output_frame = tk.Frame(self.inner_frame); output_frame.pack(anchor='w', padx=10, pady=5)
        tk.Label(output_frame, text="Output Classes:").pack(side=tk.LEFT)
        self.output_classes = tk.StringVar(value="3")
        tk.Entry(output_frame, textvariable=self.output_classes, width=5).pack(side=tk.LEFT, padx=(0,15))
        tk.Label(output_frame, text="Activation:").pack(side=tk.LEFT)
        ttk.Combobox(output_frame, textvariable=self.output_activation,
                     values=["softmax","sigmoid","linear","relu","tanh"],
                     state='readonly', width=10).pack(side=tk.LEFT)

        # Build & Reset
        build_frame = tk.Frame(self.inner_frame); build_frame.pack(anchor='w', padx=10, pady=10)
        tk.Button(build_frame, text="Build & Compile Model", command=self.build_and_compile_model)\
            .pack(side=tk.LEFT, padx=5)
        tk.Button(build_frame, text="Reset Architecture", command=self.reset_architecture)\
            .pack(side=tk.LEFT, padx=5)

        # Summary text
        summary_frame = tk.Frame(self.inner_frame)
        summary_frame.pack(anchor='w', padx=40, pady=10, fill='both', expand=True)
        self.model_summary_text = tk.Text(summary_frame, width=100, height=30, wrap='none', font=("Courier",8))
        self.model_summary_text.pack(side=tk.LEFT, fill='both', expand=True)
        sb_v = ttk.Scrollbar(summary_frame, orient='vertical', command=self.model_summary_text.yview)
        sb_v.pack(side=tk.RIGHT, fill='y')
        self.model_summary_text.config(yscrollcommand=sb_v.set)
        sb_h = ttk.Scrollbar(summary_frame, orient='horizontal', command=self.model_summary_text.xview)
        sb_h.pack(side=tk.BOTTOM, fill='x')
        self.model_summary_text.config(xscrollcommand=sb_h.set)


        self.refresh_layers_list()

    def show_layer_options(self, event=None):
        # Clear previous
        for w in self.layer_options_container.winfo_children():
            w.destroy()
        t = self.layer_type.get()
        if t == "Conv2D":
            # Filters
            tk.Label(self.layer_options_container, text="Filters:").grid(row=0, column=0, padx=5, pady=2)
            self.conv_filters = tk.StringVar(value="32")
            tk.Entry(self.layer_options_container, textvariable=self.conv_filters, width=8).grid(row=0, column=1, padx=5, pady=2)
            # Kernel
            tk.Label(self.layer_options_container, text="Kernel:").grid(row=0, column=2, padx=5, pady=2)
            self.conv_kernel = tk.StringVar(value="3")
            tk.Entry(self.layer_options_container, textvariable=self.conv_kernel, width=8).grid(row=0, column=3, padx=5, pady=2)
            # Activation
            tk.Label(self.layer_options_container, text="Activation:").grid(row=0, column=4, padx=5, pady=2)
            self.conv_activation = tk.StringVar(value="relu")
            ttk.Combobox( self.layer_options_container, textvariable=self.conv_activation, values=["relu","sigmoid","tanh","leaky_relu","linear"], state='readonly', width=10).grid(row=0, column=5, padx=5, pady=2)

        elif t == "MaxPooling2D":
            tk.Label(self.layer_options_container, text="Pool Size:").grid(row=0, column=0, padx=5, pady=2)
            self.pool_size = tk.StringVar(value="2")
            tk.Entry(self.layer_options_container, textvariable=self.pool_size, width=8).grid(row=0, column=1, padx=5, pady=2)

        elif t == "GlobalAveragePooling2D":
            tk.Label(self.layer_options_container, text="GlobalAveragePooling2D").grid(row=0, column=0, padx=5, pady=2)

        elif t == "Dropout":
            tk.Label(self.layer_options_container, text="Rate:").grid(row=0, column=0, padx=5, pady=2)
            self.dropout_rate = tk.StringVar(value="0.1")
            tk.Entry(self.layer_options_container, textvariable=self.dropout_rate, width=8).grid(row=0, column=1, padx=5, pady=2)

        elif t == "Dense":
            tk.Label(self.layer_options_container, text="Units:").grid(row=0, column=0, padx=5, pady=2)
            self.dense_units = tk.StringVar(value="64")
            tk.Entry(self.layer_options_container, textvariable=self.dense_units, width=8).grid(row=0, column=1, padx=5, pady=2)
            tk.Label(self.layer_options_container, text="Activation:").grid(row=0, column=2, padx=5, pady=2)
            self.dense_activation = tk.StringVar(value="relu")
            ttk.Combobox(self.layer_options_container, textvariable=self.dense_activation, values=["relu","sigmoid","tanh","leaky_relu","linear"], state='readonly', width=10 ).grid(row=0, column=3, padx=5, pady=2)

        elif t == "Flatten":
            tk.Label(self.layer_options_container, text="Flatten Layer").grid(row=0, column=0, padx=5, pady=2)

    def add_or_update_layer(self):
        if self.update_mode:
            self.update_layer()
        else:
            self.add_layer()

    def add_layer(self):
        params = {}
        t = self.layer_type.get()
        try:
            if t == "Conv2D":
                params = { 'filters': int(self.conv_filters.get()), 'kernel_size': int(self.conv_kernel.get()), 'activation': self.conv_activation.get() }
            elif t == "MaxPooling2D":
                params = {'pool_size': int(self.pool_size.get())}

            elif t == "GlobalAveragePooling2D":
                params = {}

            elif t == "Dropout":
                rate = float(self.dropout_rate.get())
                if not 0 < rate < 1: 
                    raise ValueError
                params = {'rate': rate}
                
            elif t == "Dense":
                params = { 'units': int(self.dense_units.get()),  'activation': self.dense_activation.get() }
            elif t == "Flatten":
                params = {}
        except Exception:
            tk.messagebox.showerror("Invalid Input", f"Please enter valid parameters for {t}.")
            return

        self.layers.append({'type': t, 'params': params})
        self.refresh_layers_list()
        self.clear_layer_options()

    def edit_layer(self, idx):

        layer = self.layers[idx]
        self.editing_index = idx
        self.update_mode = True
        # Change button text
        self.action_button.config(text="Update Layer")
        self.cancel_edit_button.pack(side=tk.LEFT, padx=5)

        # Set layer type
        self.layer_type.set(layer['type'])
        # Show appropriate fields
        self.show_layer_options()

        p = layer['params']
        t = layer['type']
        if t == "Conv2D":
            self.conv_filters.set(str(p['filters']))
            self.conv_kernel.set(str(p['kernel_size']))
            self.conv_activation.set(p['activation'])
        elif t == "MaxPooling2D":
            self.pool_size.set(str(p['pool_size']))
        elif t == "Dropout":
            self.dropout_rate.set(str(p['rate']))
        elif t == "Dense":
            self.dense_units.set(str(p['units']))
            self.dense_activation.set(p['activation'])


        # Scroll to options
        self.canvas.yview_moveto(0)

    def update_layer(self):
        """Save edited values back into the layer list."""
        idx = self.editing_index
        if idx is None:
            return
        t = self.layer_type.get()
        params = {}
        try:
            if t == "Conv2D":
                params = { 'filters': int(self.conv_filters.get()), 'kernel_size': int(self.conv_kernel.get()), 'activation': self.conv_activation.get() }
            elif t == "MaxPooling2D":
                params = {'pool_size': int(self.pool_size.get())}
            elif t == "GlobalAveragePooling2D":
                params = {}
            elif t == "Dropout":
                rate = float(self.dropout_rate.get())
                if not 0 < rate < 1: raise ValueError
                params = {'rate': rate}
            elif t == "Dense":
                params = { 'units': int(self.dense_units.get()), 'activation': self.dense_activation.get() }
            elif t == "Flatten":
                params = {}
        except Exception:
            tk.messagebox.showerror("Invalid Input", f"Please enter valid parameters for {t}.")
            return


        self.layers[idx] = {'type': t, 'params': params}
        self.refresh_layers_list()
        self.cancel_edit()

    def cancel_edit(self):
        self.editing_index = None
        self.update_mode = False
        self.action_button.config(text="Add Layer")
        self.cancel_edit_button.pack_forget()
        self.clear_layer_options()
        self.layer_type.set('')

    def clear_layer_options(self):
        for w in self.layer_options_container.winfo_children():
            w.destroy()

    def refresh_layers_list(self):
        for w in self.layers_container.winfo_children():
            w.destroy()
        for i, layer in enumerate(self.layers):
            desc = self.get_layer_description(layer)
            fr = tk.Frame(self.layers_container)
            fr.pack(anchor='w', pady=2, fill='x')
            tk.Label(fr, text=f"{i+1}. {desc}", anchor='w')\
                .pack(side=tk.LEFT, fill='x', expand=True)
            tk.Button(fr, text="Edit", command=lambda idx=i: self.edit_layer(idx))\
                .pack(side=tk.RIGHT, padx=2)
            tk.Button(fr, text="Remove", command=lambda idx=i: self.remove_layer(idx))\
                .pack(side=tk.RIGHT)

    def get_layer_description(self, layer):
        t, p = layer['type'], layer['params']
        if t == "Conv2D":
            return f"Conv2D(filters={p['filters']}, kernel={p['kernel_size']}, activation={p['activation']})"
        if t == "MaxPooling2D":
            return f"MaxPooling2D(pool_size={p['pool_size']})"
        if t == "GlobalAveragePooling2D":
            return "GlobalAveragePooling2D"
        if t == "Dropout":
            return f"Dropout(rate={p['rate']})"
        if t == "Flatten":
            return "Flatten"
        if t == "Dense":
            return f"Dense(units={p['units']}, activation={p['activation']})"
        return t

    def remove_layer(self, idx):
        del self.layers[idx]
        self.refresh_layers_list()

    def build_and_compile_model(self):
        self.model_summary_text.delete(1.0, tk.END)
        model = models.Sequential()
        lr = self.learning_rate_float.get()
        self.controller.model_learning_rate = lr
        ih, iw, ic = map(int, (self.input_height.get(), self.input_width.get(), self.input_channels.get()))
        first = True
        for layer in self.layers:
            t, p = layer['type'], layer['params']
            if t == "Conv2D":
                if first:
                    model.add(layers.Conv2D(p['filters'], (p['kernel_size'],)*2,  activation=p['activation'], input_shape=(ih, iw, ic)))
                    first = False
                else:
                    model.add(layers.Conv2D(p['filters'], (p['kernel_size'],)*2, activation=p['activation']))
            elif t == "MaxPooling2D":
                model.add(layers.MaxPooling2D(pool_size=(p['pool_size'],)*2))
            elif t == "GlobalAveragePooling2D":
                model.add(layers.GlobalAveragePooling2D())
            elif t == "Dropout":
                model.add(layers.Dropout(rate=p['rate']))
            elif t == "Flatten":
                model.add(layers.Flatten())
            elif t == "Dense":
                model.add(layers.Dense(units=p['units'], activation=p['activation']))

        if not self.layers:
            model.add(layers.InputLayer(input_shape=(ih, iw, ic)))

        # Output layer
        num_classes = int(self.output_classes.get())
        act = self.output_activation.get()
        model.add(layers.Dense(num_classes, activation=act))

        model.compile(optimizer=Adam(learning_rate=lr),
                      loss='SparseCategoricalCrossentropy',
                      metrics=['accuracy'])
        self.controller.model = model

        summary_lines = []
        model.summary(print_fn=lambda x: summary_lines.append(x))
        self.model_summary_text.insert(tk.END, "\n".join(summary_lines) +
                                       "\n\nModel compiled successfully!")

    def reset_architecture(self):
        self.controller.model = None
        self.layers.clear()
        self.editing_index = None
        self.update_mode = False
        self.create_layout()
        self.inner_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def load_tf_model(self):
        path = tk.filedialog.askopenfilename(
            title="Select a Model Directory", initialdir=os.getcwd())
        if not path:
            return
        try:
            loaded_model = tf.keras.models.load_model(path)
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to load model:\n{e}")
            return


        self.controller.model = loaded_model
        try:
            lr = float(loaded_model.optimizer.lr.numpy())
            self.learning_rate_float.set(lr)
            self.controller.model_learning_rate = lr
        except Exception:
            pass

        if loaded_model.input_shape and len(loaded_model.input_shape) == 4:
            _, h, w, c = loaded_model.input_shape
            self.input_height.set(str(h))
            self.input_width.set(str(w))
            self.input_channels.set(str(c))

        model_layers = loaded_model.layers[:]
        if model_layers and isinstance(model_layers[0], tf.keras.layers.InputLayer):
            model_layers.pop(0)
        # Extract output
        if model_layers and isinstance(model_layers[-1], tf.keras.layers.Dense):
            cfg = model_layers[-1].get_config()
            self.output_classes.set(str(cfg["units"]))
            self.output_activation.set(cfg["activation"])
            model_layers.pop()

        # Populate self.layers
        self.layers.clear()
        for layer in model_layers:
            lt = layer.__class__.__name__
            cfg = layer.get_config()
            if lt == "Conv2D":
                ks = cfg["kernel_size"][0] if isinstance(cfg["kernel_size"], (list,tuple)) else cfg["kernel_size"]
                self.layers.append({"type":"Conv2D","params":{
                    "filters":cfg["filters"],"kernel_size":ks,
                    "activation":cfg["activation"]}})
            elif lt == "MaxPooling2D":
                ps = cfg["pool_size"][0] if isinstance(cfg["pool_size"], (list,tuple)) else cfg["pool_size"]
                self.layers.append({"type":"MaxPooling2D","params":{"pool_size":ps}})
            elif lt == "GlobalAveragePooling2D":
                self.layers.append({"type":"GlobalAveragePooling2D","params":{}})
            elif lt == "Dropout":
                self.layers.append({"type":"Dropout","params":{"rate":cfg["rate"]}})
            elif lt == "Flatten":
                self.layers.append({"type":"Flatten","params":{}})
            elif lt == "Dense":
                self.layers.append({"type":"Dense","params":{
                    "units":cfg["units"],"activation":cfg["activation"]}})

        self.refresh_layers_list()
        self.model_summary_text.delete(1.0, tk.END)
        lines = []
        loaded_model.summary(print_fn=lambda x: lines.append(x))
        self.model_summary_text.insert(tk.END, "\n".join(lines) + "\n\nModel loaded successfully!")



class Model_Training_Page(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        # self.loss_fn = SparseCategoricalFocalLoss( gamma=2.0 )

        # self.optimizer = tf.keras.optimizers.Adam( learning_rate  = 0.00001 )
        self.controller = controller


        page_title_frame = tk.Frame(self)
        page_title_frame.pack( anchor= 'w' , pady=10)


        tk.Button(page_title_frame, text="Back", command=lambda: controller.show_frame(Training_And_Eval_Options_Page)).pack(anchor='w', padx=10 , side= tk.LEFT)
        tk.Label(page_title_frame, text="Model Training ", font=("Helvetica", 16)).pack( padx=50, anchor='w' , side = tk.LEFT)


        dir_select_frame = tk.Frame(self)
        dir_select_frame.pack(anchor='w', pady=(10, 50))

        self.selected_dir = tk.StringVar()

        text_box = tk.Entry(  dir_select_frame  , textvariable=self.selected_dir, bg='black', fg='white', font=("Arial", 12),  width=90, state='disabled')
        text_box.pack(anchor='w', side=tk.LEFT, padx=10, pady=10)

        select_dir_button = tk.Button(  dir_select_frame,  text="Select Directory", command=lambda: self.select_directory_window(text_box))
        select_dir_button.pack(anchor='w', side=tk.LEFT)


        # self.class_setting_button = tk.Button( dir_select_frame  , text='Class Settings' , command=lambda: controller.show_frame(Advance_Class_Selection_Page), state='disabled' )
        self.class_setting_button = tk.Button( dir_select_frame  , text='Class Settings' , command=lambda: self.Show_Advanced_Class_Page() , state='disabled' )

        self.class_setting_button.pack(anchor='w', side=tk.LEFT)

        train_split_params_frame = tk.Frame(self)
        train_split_params_frame.pack( anchor='w' , pady = (10 , 20) )

        self.train_size_text_selected = tk.IntVar()
        tk.Label(train_split_params_frame , text = "train_size :" ).pack(anchor='w' , side = tk.LEFT )
        train_size_text_box = tk.Entry( train_split_params_frame , textvariable= self.train_size_text_selected  , bg = 'black', fg='white', font=("Arial", 12), state = 'normal' )
        train_size_text_box.pack( anchor='w', side = tk.LEFT  )
        train_size_text_box.delete(0, tk.END)
        train_size_text_box.insert(0,40)
        

        self.val_size_text_selected = tk.IntVar()
        tk.Label(train_split_params_frame , text = "val_size :" ).pack(anchor='w' , side = tk.LEFT )
        val_size_text_box = tk.Entry( train_split_params_frame , textvariable= self.val_size_text_selected, bg='black',  fg='white', font=("Arial", 12), state = 'normal' )
        val_size_text_box.pack( anchor='w' ,side = tk.LEFT  )
        val_size_text_box.delete(0, tk.END)
        val_size_text_box.insert(0,40)


        self.test_size_text_selected = tk.IntVar()
        tk.Label(train_split_params_frame , text = "test_size :" ).pack(anchor='w' , side = tk.LEFT )
        test_size_text_box = tk.Entry( train_split_params_frame, textvariable= self.test_size_text_selected,  bg='black',  fg='white', font=("Arial", 12), state = 'normal')
        test_size_text_box.pack( anchor='w' ,side = tk.LEFT  )
        test_size_text_box.delete(0, tk.END)
        test_size_text_box.insert(0,20)

        Model_Training_Params_Frame = tk.Frame(self)
        Model_Training_Params_Frame.pack( anchor='w' , pady = (10 , 20) )

        self.Epoch_size_text_selected = tk.IntVar()
        tk.Label(Model_Training_Params_Frame , text = "Num Epoches :" ).pack(anchor='w' , side = tk.LEFT )
        Epoch_size_text_box = tk.Entry( Model_Training_Params_Frame , textvariable= self.Epoch_size_text_selected  , bg = 'black', fg='white', font=("Arial", 12), state = 'normal' )
        Epoch_size_text_box.pack( anchor='w', side = tk.LEFT  )
        Epoch_size_text_box.delete(0, tk.END)
        Epoch_size_text_box.insert(0,100)


        self.Batch_size_text_selected = tk.IntVar()
        tk.Label(Model_Training_Params_Frame , text = "batch_size :" ).pack(anchor='w' , side = tk.LEFT )
        Batch_size_text_box = tk.Entry( Model_Training_Params_Frame , textvariable= self.Batch_size_text_selected  , bg = 'black', fg='white', font=("Arial", 12), state = 'normal' )
        Batch_size_text_box.pack( anchor='w', side = tk.LEFT  )
        Batch_size_text_box.delete(0, tk.END)
        Batch_size_text_box.insert(0,25)




        Control_Training_Frame = tk.Frame(self)
        Control_Training_Frame.pack( anchor='w' , pady = (10 , 20) )
        # tk.Button(Control_Training_Frame, text='Train', command=lambda: self.Train_Model()).pack(anchor='w' , side= tk.LEFT)
        tk.Button(Control_Training_Frame, text='Train', command=lambda: Frame_Manager.setup_process(self) ).pack(anchor='w' , side= tk.LEFT)
        tk.Button(Control_Training_Frame, text='Stop', command=lambda: Frame_Manager.cancel_process(self) ).pack(anchor='w' , side= tk.LEFT)

        # tk.Button(Control_Training_Frame, text='Train', command=lambda: Model_Training_script.Model_Training.Train_Model(self) ).pack(anchor='w' , side= tk.LEFT)


        Model_Training_script.Model_Training.Train_Model


        # tk.Button(Control_Training_Frame, text='Monitor', command= lambda : controller.show_frame(Monitor_Training_Page) ).pack(anchor='w' ,  side= tk.LEFT)
        tk.Button(Control_Training_Frame, text='Monitor', command= lambda :  self.Monitor_Launch_func() ).pack(anchor='w' ,  side= tk.LEFT)
        # tk.Button(Control_Training_Frame, text='Monitor', command = self.Monitor_Launch_func ).pack(anchor='w' ,  side= tk.LEFT)

        tk.Button(Control_Training_Frame, text='Save Model', command= lambda :  self.Save_Trained_Model() ).pack(anchor='w' ,  side= tk.LEFT)
        tk.Button(Control_Training_Frame, text='Test Batch Sizes', command=lambda: self.Test_Batch_Sizes() ).pack(anchor='w' , side= tk.LEFT)



    def select_directory_window(self, text_box):
        directory_path = tk.filedialog.askdirectory( initialdir=os.getcwd(), title="Select a Directory" )
        if directory_path:
            text_box.config(state='normal')
            text_box.delete(0, tk.END)
            text_box.insert(0, directory_path)
            text_box.config(state='disabled')

            self.class_setting_button.config(state='normal')

            self.controller.selected_directory = directory_path

            training_class_page = self.controller.frames[Advance_Class_Selection_Page]
            training_class_page.Update_Page_With_Class(directory_path)



    def Show_Advanced_Class_Page(self):
        self.controller.show_frame(Advance_Class_Selection_Page)
        Monitor_Training_Page.toggle_fullscreen(self)

    def Monitor_Launch_func(self):

        self.controller.enter_fullscreen_event()
        self.controller.show_frame(Monitor_Training_Page)
        return
    
    def Save_Trained_Model(self):


        model_path_input = tk.filedialog.asksaveasfilename( title="Save file as...", defaultextension=".keras",  filetypes=[("Text Files", "*.keras"), ("All Files", "*.*")] , initialdir=os.getcwd()  )

        if model_path_input:
            print("File will be saved to:", model_path_input)
        else:
            print("Save cancelled.")

        # try:
        self.controller.model.save(model_path_input)

        # except:
        #     print("ERROR SAVING MODEL")

        dict_path = os.path.dirname(model_path_input)
        Model_hist_file = open(f"{dict_path}/Test.txt", "w")
        try:
            Model_hist_file.write( str(self.controller.model.history_dict) )
        except:
            Model_hist_file.write("Error But we go on")

        Model_hist_file.close()


    def Test_Batch_Sizes(self):


        self.output_dir = tk.filedialog.askdirectory( title="Select Output Directory for Batch Size Test Results" , initialdir=os.getcwd() )
        if not self.output_dir:
            print("No output directory selected. Aborting Test_Batch_Sizes.")
            return


        self.batch_sizes_to_test = [10, 25, 50, 100, 250]
        self.current_batch_index = 0


        self.original_model = self.controller.model
        self.original_batch = self.Batch_size_text_selected.get()


        self.run_next_batch()


    def run_next_batch(self):


        if self.current_batch_index >= len(self.batch_sizes_to_test):
            print("All batch size tests completed.")
            self.Batch_size_text_selected.set(self.original_batch)
            self.controller.model = self.original_model
            return


        bs = self.batch_sizes_to_test[self.current_batch_index]
        print(f"Starting training for batch size {bs}...")


        self.Batch_size_text_selected.set(bs)


        new_model = tf.keras.models.clone_model(self.original_model)

        new_model.compile(
            optimizer=Adam(learning_rate=self.controller.model_learning_rate),
            loss='SparseCategoricalCrossentropy',
            # loss='SparseCategoricalFocalLoss',
            metrics=['accuracy']
        )

        # model.compile(
        #     optimizer=Adam(learning_rate=self.controller.model_learning_rate),
        #     loss=SparseCategoricalCrossentropy(from_logits=True),              # or use the string 'sparse_categorical_crossentropy'
        #     metrics=[Precision(name='precision')]
        # )

        self.controller.model = new_model

        # Start training using your existing process.
        Frame_Manager.setup_process(self)

        # Periodically check whether training has finished.
        self.check_training_finished()


    def check_training_finished(self):
        # Use after() so that the UI remains responsive.
        if self.controller.running:
            self.after(1000, self.check_training_finished)
        else:
            # When training is finished, first save the training history and figures.
            bs = self.batch_sizes_to_test[self.current_batch_index]
            self.save_results_for_batch(bs)
            # Move on to the next batch size after a brief pause.
            self.current_batch_index += 1
            self.after(1000, self.run_next_batch)


    def save_results_for_batch(self, batch_size):

        # Create a nested directory for this batch size.
        batch_dir = os.path.join(self.output_dir, f"Batch_size_{batch_size}")
        os.makedirs(batch_dir, exist_ok=True)

        # Save the training history.
        history = self.controller.model.history_dict
        history_file = os.path.join(batch_dir, "history.txt")
        try:
            with open(history_file, "w") as f:
                f.write(str(history))
            print(f"History for batch size {batch_size} saved to {history_file}")
        except Exception as e:
            print(f"Error saving history for batch size {batch_size}: {e}")

        # Save metric plots from the Monitor Training page.
        monitor_page = self.controller.frames.get(Monitor_Training_Page)
        if monitor_page is not None:
            # Example: Save the class accuracy plot.
            if hasattr(monitor_page, "class_accuracy_ax"):
                try:
                    monitor_page.class_accuracy_ax.figure.savefig(
                        os.path.join(batch_dir, "class_accuracy.png")
                    )
                    print("Class accuracy plot saved.")
                except Exception as e:
                    print("Error saving class accuracy plot:", e)
            # Save class precision plot.
            if hasattr(monitor_page, "class_precision_ax"):
                try:
                    monitor_page.class_precision_ax.figure.savefig(
                        os.path.join(batch_dir, "class_precision.png")
                    )
                    print("Class precision plot saved.")
                except Exception as e:
                    print("Error saving class precision plot:", e)
            # Save class recall plot.
            if hasattr(monitor_page, "class_recall_ax"):
                try:
                    monitor_page.class_recall_ax.figure.savefig(
                        os.path.join(batch_dir, "class_recall.png")
                    )
                    print("Class recall plot saved.")
                except Exception as e:
                    print("Error saving class recall plot:", e)
            # Save loss plot if available.
            if hasattr(monitor_page, "loss_fig_ax"):
                try:
                    monitor_page.loss_fig_ax.figure.savefig(
                        os.path.join(batch_dir, "loss.png")
                    )
                    print("Loss plot saved.")
                except Exception as e:
                    print("Error saving loss plot:", e)
        else:
            print("Monitor_Training_Page not found.")

        # Save the confusion matrix from Show_Confusion_Page.
        confusion_page = self.controller.frames.get(Show_Confusion_Page)
        if confusion_page is not None:
            if hasattr(confusion_page, "confusion_fig_ax"):
                try:
                    confusion_page.confusion_fig_ax.figure.savefig(
                        os.path.join(batch_dir, "confusion_matrix.png")
                    )
                    print("Confusion matrix image saved.")
                except Exception as e:
                    print("Error saving confusion matrix image:", e)
        else:
            print("Show_Confusion_Page not found.")

        

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#



#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#


class SliderSection:
    def __init__(self, parent, section_name, allocated_var, update_flag, default_max=100):

        self.parent = parent
        self.section_name = section_name
        self.allocated_var = allocated_var
        self.update_flag = update_flag
        self.default_max = default_max

        # Create a labeled frame for all sliders in this section.
        self.frame = ttk.LabelFrame(parent, text=f"{self.section_name} Sliders")
        self.frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Entry to set all slider values in this section
        self.all_slider_entry = ttk.Entry(self.frame, width=6)
        self.all_slider_entry.pack(side=tk.RIGHT, padx=5)
        self.all_slider_entry.insert(0, "")
        self.all_slider_entry.bind("<Return>", self.set_all_sliders)
        self.all_slider_entry.bind("<FocusOut>", self.set_all_sliders)

        # Scrollable frame for slider rows.
        self.scrollable_frame = ScrollableFrame_2(self.frame)
        self.scrollable_frame.pack(fill="both", expand=True)

        # Containers for slider widgets and values.
        self.slider_vars = []
        self.slider_widgets = []
        self.entry_vars = []
        self.percent_labels = []
        self.slider_max_values = []
        self.slider_original_counts = []

        # Avoid recursive updates.
        self.allocated_var.trace_add('write', self.on_allocation_change)

    def add_slider(self, slider_label=None, max_value=None, initial_value=None, original_count=None):
        if self.update_flag.get(): return
        self.update_flag.set(True)
        try:
            max_value = max_value if max_value is not None else self.default_max
            initial_value = initial_value if initial_value is not None else max_value

            slider_var = tk.DoubleVar(value=initial_value)
            idx = len(self.slider_vars)

            row = ttk.Frame(self.scrollable_frame.scrollable_frame)
            row.pack(fill='x', padx=10, pady=2)

            ttk.Label(row, text=slider_label or f"{self.section_name} {idx+1}", width=30).grid(row=0, column=0, sticky='w')

            slider = ttk.Scale(
                row, from_=0, to=max_value, orient='horizontal',
                variable=slider_var, command=lambda v, var=slider_var, i=idx: self._update_percentage(var, i)
            )
            slider.grid(row=0, column=1, sticky='ew')
            row.columnconfigure(1, weight=1)

            entry_var = tk.StringVar(value=str(int(initial_value)))
            entry = ttk.Entry(row, textvariable=entry_var, width=6)
            entry.grid(row=0, column=2, padx=5)
            entry.bind('<Return>', lambda e, i=idx: self._entry_changed(i))
            entry.bind('<FocusOut>', lambda e, i=idx: self._entry_changed(i))

            percent = (initial_value / max_value * 100) if max_value else 0
            pct_lbl = ttk.Label(row, text=f"{percent:.1f}%")
            pct_lbl.grid(row=0, column=3, padx=5)

            self.slider_vars.append(slider_var)
            self.slider_widgets.append(slider)
            self.entry_vars.append(entry_var)
            self.percent_labels.append(pct_lbl)
            self.slider_max_values.append(max_value)
            self.slider_original_counts.append(original_count if original_count is not None else max_value)

            slider_var.trace_add('write', lambda *a, var=slider_var, i=idx: self._update_percentage(var, i))
        finally:
            self.update_flag.set(False)

    def _update_percentage(self, var, idx):
        val = var.get()
        max_v = self.slider_max_values[idx]
        self.entry_vars[idx].set(str(int(val)))
        pct = (val/max_v*100) if max_v else 0
        self.percent_labels[idx].config(text=f"{pct:.1f}%")

    def _entry_changed(self, idx):
        if self.update_flag.get(): return
        self.update_flag.set(True)
        try:
            try: val = float(self.entry_vars[idx].get())
            except ValueError: return
            max_v = self.slider_widgets[idx].cget('to')
            val = min(val, float(max_v))
            self.slider_vars[idx].set(val)
            self._update_percentage(self.slider_vars[idx], idx)
        finally:
            self.update_flag.set(False)

    def set_all_sliders(self, event=None):
        try: new = float(self.all_slider_entry.get())
        except ValueError: return
        for i, var in enumerate(self.slider_vars):
            var.set(min(new, self.slider_max_values[i]))

    def on_allocation_change(self, *args):
        pass

class ScrollableFrame_2(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self, width=600, height=250, borderwidth=0)
        canvas.pack(side='left', fill='both', expand=True)
        sb = ttk.Scrollbar(self, orient='vertical', command=canvas.yview)
        sb.pack(side='right', fill='y')
        self.scrollable_frame = tk.Frame(canvas)
        self.scrollable_frame.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0,0), window=self.scrollable_frame, anchor='nw')
        canvas.configure(yscrollcommand=sb.set)

class Advance_Class_Selection_Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.style = ttk.Style()  # for dynamic styling
        self.test_set_dir = None
        self.different_classes = []
        self.slider_sections = {}
        self.alloc_vars = {}
        self.update_flag = tk.BooleanVar(value=False)

    def clear_page(self):
        for w in self.winfo_children(): w.destroy()

    def select_test_directory(self):
        test_dir = tk.filedialog.askdirectory(title="Select Test Set Directory", initialdir=os.getcwd())
        if not test_dir: return
        missing = [c for c in self.different_classes if not os.path.isdir(os.path.join(test_dir,c))]
        if missing:
            messagebox.showerror("Error", f"Missing classes in test set: {missing}")
            return
        self.test_set_dir = test_dir

        self.style.configure("TestSet.TLabelframe.Label", foreground="red")
        self.slider_sections["Test"].frame.configure(style="TestSet.TLabelframe")

        for i, cls in enumerate(self.different_classes):
            cnt = len(os.listdir(os.path.join(test_dir, cls)))
            sec = self.slider_sections["Test"]
            sec.slider_max_values[i] = cnt
            sec.slider_widgets[i].config(to=cnt)
            sec.slider_vars[i].set(cnt)

    def Update_Page_With_Class(self, dir_path=None):
        dir_path = dir_path or self.controller.selected_directory
        self.clear_page()

        # Header row
        hdr = tk.Frame(self)
        hdr.pack(fill='x', padx=10, pady=10)
        tk.Button(hdr, text='Back', command=lambda: [Monitor_Training_Page.toggle_fullscreen(self), self.controller.show_frame(Model_Training_Page)]).pack(side='left')
        tk.Label(hdr, text="Advanced Training Configuration", font=("Helvetica",16)).pack(side='left', padx=50)
        tk.Button(hdr, text='Full', command=lambda: Monitor_Training_Page.toggle_fullscreen(self)).pack(side='left')
        tk.Button(hdr, text='Select Test Set', command=self.select_test_directory).pack(side='left')

        if not dir_path or not os.path.isdir(dir_path):
            tk.Label(self, text="Invalid Directory").pack()
            return

        # Discover classes and counts
        self.different_classes = sorted([d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path,d))])
        counts = {c: len(os.listdir(os.path.join(dir_path,c))) for c in self.different_classes}
        total = sum(counts.values())
        tk.Label(self, text=f"Found {total} images across {len(counts)} classes.").pack(anchor='w', padx=10)

        # Allocation entries
        self.alloc_vars = {s: tk.DoubleVar(value=40 if s!='Test' else 20) for s in ["Train","Validate","Test"]}
        self.error_label = ttk.Label(self, text="", foreground="red")
        self.error_label.pack()
        for v in self.alloc_vars.values(): v.trace_add('write', lambda *a: self.check_entries_sum())

        af = ttk.Frame(self)
        af.pack(fill='x', pady=10, padx=10)
        for i, sec in enumerate(["Train","Validate","Test"]):
            ttk.Label(af, text=f"{sec}:").grid(row=0, column=2*i, sticky='e', padx=5)
            entry = ttk.Entry(af, textvariable=self.alloc_vars[sec], width=6,
                              validate='key', validatecommand=(self.register(self.validate_digit),'%P'))
            entry.grid(row=0, column=2*i+1, padx=5)

        # Enable & epochs
        self.Enable_Value = tk.BooleanVar()
        ttk.Label(af, text='Enable:').grid(row=0, column=10, sticky='e', padx=(5,1))
        ttk.Checkbutton(af, variable=self.Enable_Value).grid(row=0, column=11, padx=2)
        self.Epochs_Before_Refresh = tk.IntVar()
        ttk.Label(af, text='Epochs Before Refresh:').grid(row=0, column=12, sticky='e', padx=(5,1))
        ttk.Entry(af, textvariable=self.Epochs_Before_Refresh, width=6).grid(row=0, column=13, padx=2)

        # Create slider sections
        for sec in ["Train","Validate","Test"]:
            self.slider_sections[sec] = SliderSection(self, sec, tk.DoubleVar(value=100), self.update_flag)

        # Populate sliders
        for cls in self.different_classes:
            cnt = counts[cls]
            pcts = {s: self.alloc_vars[s].get()/100 for s in self.alloc_vars}
            self.slider_sections["Train"].add_slider(cls, cnt*pcts["Train"], cnt*pcts["Train"], cnt)
            self.slider_sections["Validate"].add_slider(cls, cnt*pcts["Validate"], cnt*pcts["Validate"], cnt)
            self.slider_sections["Test"].add_slider(cls, cnt*pcts["Test"], cnt*pcts["Test"], cnt)

        self.check_entries_sum()
        for idx in range(len(self.different_classes)):
            self.slider_sections["Train"].slider_vars[idx].trace_add('write', lambda *a, i=idx: self.train_slider_changed(i))

    def train_slider_changed(self, idx, *args):
        if self.slider_sections["Train"].slider_vars[idx].get()==0:
            self.slider_sections["Validate"].slider_vars[idx].set(0)
            self.slider_sections["Test"].slider_vars[idx].set(0)

    def check_entries_sum(self):
        if self.update_flag.get(): return
        total = sum(v.get() for v in self.alloc_vars.values())
        if abs(total-100)>1e-6:
            self.error_label.config(text="Train + Validate + Test must sum to 100!")
            for sec in self.slider_sections.values():
                sec.allocated_var.set(0)
                for row in sec.sliders if hasattr(sec,'sliders') else []:
                    for ch in row.winfo_children(): ch.configure(state='disabled')
        else:
            self.error_label.config(text="")
            for name,sec in self.slider_sections.items():
                sec.allocated_var.set(100)
                for row in sec.sliders if hasattr(sec,'sliders') else []:
                    for ch in row.winfo_children(): ch.configure(state='normal')
                pct = self.alloc_vars[name].get()/100
                for i,orig in enumerate(sec.slider_original_counts):
                    if name=='Test' and self.test_set_dir:
                        new_max = sec.slider_max_values[i]
                    else:
                        new_max = orig*pct
                    sec.slider_max_values[i]=new_max
                    if sec.slider_vars[i].get()>new_max: sec.slider_vars[i].set(new_max)
                    sec.slider_widgets[i].config(to=new_max)
                    sec._update_percentage(sec.slider_vars[i], i)

    def validate_digit(self, nv):
        if nv=="": return True
        try: float(nv); return True
        except ValueError: return False

        
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#

    

class Monitor_Training_Page(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # Initialize legend toggle flag
        self.legend_visible = True

        # ─── Title & Controls ────────────────────────────────────────
        page_title_frame = tk.Frame(self)
        page_title_frame.pack(anchor='w', pady=10, fill='x')

        # Back
        tk.Button(page_title_frame, text="Back", command=self.exit_button_event).pack(anchor='w', padx=10, side=tk.LEFT)

        # Model Filters
        tk.Button(page_title_frame, text='Model Filters',
                  command=lambda: controller.show_frame(Show_Model_Filters_Page))\
          .pack(anchor='w', side=tk.LEFT)

        # Title Label
        tk.Label(page_title_frame, text="AI Command Center", font=("Helvetica", 16))\
          .pack(padx=50, anchor='w', side=tk.LEFT)

        # Confusion Matrix
        tk.Button(page_title_frame, text='Confusion Matrix',
                  command=lambda: controller.show_frame(Show_Confusion_Page))\
          .pack(anchor='w', side=tk.LEFT)

        # Toggle Legend
        self.Toggle_legend_button = tk.Button(
            page_title_frame, text='Toggle Legend', command=self.toggle_legend)
        self.Toggle_legend_button.pack(anchor='w', side=tk.LEFT)

        # Fullscreen
        self.full_button = tk.Button(
            page_title_frame, text='Full', command=self.toggle_fullscreen)
        self.full_button.pack(anchor='w', side=tk.LEFT)

        # ─── Classes ▾ Dropdown with Checkboxes ───────────────────────
        self.class_filter_mb = tk.Menubutton(
            page_title_frame, text='Classes ▾', relief=tk.RAISED)
        self.class_filter_menu = tk.Menu(self.class_filter_mb, tearoff=False)
        self.class_filter_mb.config(menu=self.class_filter_menu)
        self.class_filter_mb.pack(anchor='w', side=tk.LEFT, padx=(10,0))

        # Holds label → BooleanVar
        self.class_filter_vars = {}

        # ─── Plot Frames ──────────────────────────────────────────────
        self.First_Canvas_Row = tk.Frame(self); self.First_Canvas_Row.pack(anchor='w', side=tk.TOP, pady=10)
        self.First_Canvas_Row_Accuracy  = tk.Frame(self.First_Canvas_Row); self.First_Canvas_Row_Accuracy.pack(side=tk.LEFT, padx=10)
        self.First_Canvas_Row_Precision = tk.Frame(self.First_Canvas_Row); self.First_Canvas_Row_Precision.pack(side=tk.LEFT, padx=10)
        self.First_Canvas_Row_Recall    = tk.Frame(self.First_Canvas_Row); self.First_Canvas_Row_Recall.pack(side=tk.LEFT, padx=10)
        self.First_Canvas_Row_Loss      = tk.Frame(self.First_Canvas_Row); self.First_Canvas_Row_Loss.pack(side=tk.LEFT, padx=10)

        self.Second_Canvas_Row = tk.Frame(self); self.Second_Canvas_Row.pack(anchor='w', side=tk.TOP, pady=10)
        self.Second_Canvas_Row_Class_Accuracy  = tk.Frame(self.Second_Canvas_Row); self.Second_Canvas_Row_Class_Accuracy.pack(side=tk.LEFT, padx=10)
        self.Second_Canvas_Row_Class_Precision = tk.Frame(self.Second_Canvas_Row); self.Second_Canvas_Row_Class_Precision.pack(side=tk.LEFT, padx=10)
        self.Second_Canvas_Row_Class_Recall    = tk.Frame(self.Second_Canvas_Row); self.Second_Canvas_Row_Class_Recall.pack(side=tk.LEFT, padx=10)
        self.Second_Canvas_delta_Loss          = tk.Frame(self.Second_Canvas_Row); self.Second_Canvas_delta_Loss.pack(side=tk.LEFT, padx=10)

        # ─── Placeholders for Axes & Canvases ────────────────────────
        self.class_accuracy_ax      = None
        self.class_accuracy_canvas  = None
        self.class_precision_ax     = None
        self.class_precision_canvas = None
        self.class_recall_ax        = None
        self.class_recall_canvas    = None
        self.loss_fig_ax            = None
        self.loss_fig_canvas        = None


    def setup_class_filter(self, *args):

        # Extract the labels list from the last argument
        if not args:
            return
        translated_labels = args[-1]

        # Clear old menu
        self.class_filter_menu.delete(0, 'end')
        self.class_filter_vars.clear()

        for lbl in translated_labels:
            var = tk.BooleanVar(value=True)
            self.class_filter_vars[lbl] = var
            self.class_filter_menu.add_checkbutton(
                label=lbl,
                variable=var,
                command=self.update_class_visibility
            )


    def update_class_visibility(self):

        for lbl, var in self.class_filter_vars.items():
            vis = var.get()
            for ax in (self.class_accuracy_ax,
                       self.class_precision_ax,
                       self.class_recall_ax):
                if ax:
                    for line in ax.get_lines():
                        if line.get_label() == lbl:
                            line.set_visible(vis)


        for ax, canvas in (
            (self.class_accuracy_ax,   self.class_accuracy_canvas),
            (self.class_precision_ax,  self.class_precision_canvas),
            (self.class_recall_ax,     self.class_recall_canvas),
        ):
            if ax and canvas:

                vis_lines = [l for l in ax.get_lines() if l.get_visible()]
                if vis_lines:
                    handles = vis_lines
                    labels  = [l.get_label() for l in vis_lines]
                    ax.legend(handles, labels, loc="upper left", prop={'size':7})
                else:

                    if ax.legend_:
                        ax.legend_.remove()
                canvas.draw_idle()


    def toggle_legend(self):

        self.legend_visible = not self.legend_visible
        for ax, canvas in (
            (self.class_accuracy_ax,   self.class_accuracy_canvas),
            (self.class_precision_ax,  self.class_precision_canvas),
            (self.class_recall_ax,     self.class_recall_canvas),
        ):
            if ax and canvas and ax.get_legend():
                ax.get_legend().set_visible(self.legend_visible)
                canvas.draw_idle()


    def toggle_fullscreen(self):

        self.controller.toggle_fullscreen()
        if hasattr(self, 'full_button'):
            txt = 'Exit Full Screen' if self.controller.is_fullscreen else 'Full'
            self.full_button.config(text=txt)


    def exit_button_event(self):
        """Exit fullscreen and return to Model_Training_Page."""
        self.controller.exit_fullscreen_event()
        self.controller.show_frame(Model_Training_Page)


class Show_Confusion_Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller


        page_title_frame = tk.Frame(self)
        page_title_frame.pack( anchor= 'w' , pady=10 ,fill='x')

        # tk.Button(page_title_frame, text="Back", command = lambda:  self.exit_button_event() ).pack(anchor='w', padx=10 , side= tk.LEFT)
        tk.Button(page_title_frame, text="Back", command = lambda:  self.controller.show_frame(Monitor_Training_Page) ).pack(anchor='w', padx=10 , side= tk.LEFT)

        tk.Label(page_title_frame, text="Confusion Matrix ", font=("Helvetica", 16)).pack( padx=50, anchor='w' , side = tk.LEFT)

        # 'Full' button to toggle full-screen
        self.full_button = tk.Button( page_title_frame, text='Full', command= lambda: self.toggle_fullscreen()  )
        self.full_button.pack(anchor='w' , side=tk.LEFT)


        self.Confusion_Canvas_Frame = tk.Frame( self )
        self.Confusion_Canvas_Frame.pack( anchor='w' , side= tk.TOP , pady= 10 )

        self.Confusion_Matrix = tk.Frame( self.Confusion_Canvas_Frame )
        self.Confusion_Matrix.pack( anchor='w' , side= tk.TOP , pady= 10 )


        self.confusion_fig_ax = None
        self.confusion_fig_canvas = None





    def toggle_fullscreen(self):
        """Toggle full-screen mode and update button text."""
        self.controller.toggle_fullscreen()
        if self.controller.is_fullscreen:
            self.full_button.config(text='Exit Full Screen')
        else:
            self.full_button.config(text='Full Screen')


class Show_Model_Filters_Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        page_title_frame = tk.Frame(self)
        page_title_frame.pack( anchor= 'w' , pady=10 ,fill='x')

        # tk.Button(page_title_frame, text="Back", command = lambda:  self.exit_button_event() ).pack(anchor='w', padx=10 , side= tk.LEFT)
        tk.Button(page_title_frame, text="Back", command = lambda:  self.controller.show_frame(Monitor_Training_Page) ).pack(anchor='w', padx=10 , side= tk.LEFT)


        tk.Label(page_title_frame, text="Model_Filters ", font=("Helvetica", 16)).pack( padx=50, anchor='w' , side = tk.LEFT)

        # 'Full' button to toggle full-screen
        self.full_button = tk.Button( page_title_frame, text='Full', command= lambda: self.toggle_fullscreen()  )
        self.full_button.pack(anchor='w' , side=tk.LEFT)

        self.test_print = tk.Button( page_title_frame , text= 'Print Layers' , command= lambda : self.Update_Model_display() )
        self.test_print.pack( anchor = 'w' , side = tk.LEFT )

        self.Filter_Canvas_Frame = tk.Frame( self )
        self.Filter_Canvas_Frame.pack( anchor='w' , side= tk.TOP , pady= 10 )

        self.Filter_Plot = tk.Frame( self.Filter_Canvas_Frame )
        self.Filter_Plot.pack( anchor='w' , side= tk.TOP , pady= 10 )


    def toggle_fullscreen(self):
        """Toggle full-screen mode and update button text."""
        self.controller.toggle_fullscreen()
        if self.controller.is_fullscreen:
            self.full_button.config(text='Exit Full Screen')
        else:
            self.full_button.config(text='Full Screen')

    def Update_Model_display(self):

        for layer in self.controller.model.layers :

            try:
                print(layer , layer.activation)
            except:
                print(layer , '< No ativation >')

    def Model_Filters_Ready(self):

        # Build the list of layers from the current model.
        # For each layer a label is shown. In the case of Conv2D layers, a “Show” button is added that will call Plot_Model_Filters for that layer.
        # Any previously added widgets (labels/buttons) are cleared first.

        # Clear all children in Filter_Canvas_Frame except the Filter_Plot (which is used for plotting)
        Filter_Page = next( (frame for cls, frame in self.controller.frames.items() if cls.__name__ == "Show_Model_Filters_Page"), None  )
        for widget in Filter_Page.Filter_Canvas_Frame.winfo_children():
            # Do not remove the plot area frame
            if widget != Filter_Page.Filter_Plot:
                widget.destroy()
        # Also clear any plot already in the Filter_Plot area.
        for widget in Filter_Page.Filter_Plot.winfo_children():
            widget.destroy()

        model = self.controller.model
        if model is None:
            tk.Label(Filter_Page.Filter_Canvas_Frame, text="No model loaded.").pack(anchor='w')
            return

        # Add a title label for the architecture
        tk.Label(Filter_Page.Filter_Canvas_Frame, text="Model Architecture:",   font=("Helvetica", 14, "bold") ).pack(anchor='w', pady=5)

        # Loop through each layer in the model and create a small frame
        # that contains a label with the layer’s description.
        # If the layer is a Conv2D layer, add a button that calls Plot_Model_Filters.
        for layer in model.layers:
            layer_frame = tk.Frame(Filter_Page.Filter_Canvas_Frame)
            layer_frame.pack(anchor='w', fill='x', pady=2)

            # Create a label that shows the layer name and its class type.
            layer_desc = f"{layer.name}: {layer.__class__.__name__}"
            tk.Label(layer_frame, text=layer_desc).pack(side='left')

            # For Conv2D layers, add a "Show" button.
            if isinstance(layer, tf.keras.layers.Conv2D):
                tk.Button(layer_frame, text="Show", command=lambda l=layer: Show_Model_Filters_Page.Plot_Model_Filters( self = self , conv_layer = l) ).pack(side='left', padx=5)

    def Plot_Model_Filters(self, conv_layer):

        # Clear the current plot area and display a new plot of the filters from the given conv_layer. If the button is pressed again (or another conv layer’s button is clicked) the previous plot is removed.

        # Clear the plot area first.
        Filter_Page = next( (frame for cls, frame in self.controller.frames.items() if cls.__name__ == "Show_Model_Filters_Page"), None  )
        for widget in Filter_Page.Filter_Plot.winfo_children():
            widget.destroy()

        # Attempt to get the weights for the layer.
        try:
            filters, biases = conv_layer.get_weights()
        except Exception as e:
            tk.messagebox.showerror("Error", f"Could not retrieve filters: {e}")
            return

        n_filters = filters.shape[-1]
        n_channels = filters.shape[-2]

        # Determine a grid size so that all filters can be displayed
        grid_r = int(np.ceil(np.sqrt(n_filters)))
        grid_c = int(np.ceil(n_filters / grid_r))

        # Create a new matplotlib Figure object.
        fig = plt.Figure( dpi = 100)
        # fig = plt.Figure(figsize=(grid_c * 2, grid_r * 2) , dpi = 100)
        axes = fig.subplots(grid_r, grid_c)
        # Ensure axes is a flat list.
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]


        for i in range(n_filters):

            f = filters[:, :, :, i]

            f_min, f_max = f.min(), f.max()
            if f_max - f_min != 0:
                f = (f - f_min) / (f_max - f_min)
            else:
                f = f - f_min


            if n_channels == 3:
                axes[i].imshow(f)
            else:
                axes[i].imshow(f[:, :, 0], cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f"Filter {i+1}", fontsize=8)


        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        fig.suptitle(f"Filters from layer: {conv_layer.name}", fontsize=12)
        fig.tight_layout()


        canvas = FigureCanvasTkAgg(fig, master=Filter_Page.Filter_Plot)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        toolbar_filter= NavigationToolbar2Tk(canvas, Filter_Page.Filter_Plot , pack_toolbar=False)
        toolbar_filter.update()
        toolbar_filter.pack(side=tk.LEFT, fill=tk.X)
            



#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#

class Evaluate_Model_Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # Title and Navigation Frame
        title_frame = tk.Frame(self)
        title_frame.pack(fill='x', pady=(10, 5), padx=10)

        back_button = tk.Button( title_frame ,  text="Back" , command=lambda: controller.show_frame(Training_And_Eval_Options_Page) )
        back_button.pack(side='left')

        title_label = tk.Label(  title_frame  ,  text="Evaluate Model"  ,  font=("Helvetica", 16) )
        title_label.pack(side='left', padx=(20, 0))

        # Directory Selection Frame
        dir_frame = tk.Frame(self)
        dir_frame.pack(fill='x', pady=(5, 10), padx=10)

        self.selected_dir = tk.StringVar()
        dir_entry = tk.Entry( dir_frame  ,  textvariable=self.selected_dir  ,   bg='black' ,  fg='white',  font=("Arial", 12) ,  width=90,  state='disabled' )
        dir_entry.pack(side='left', fill='x', expand=True)
        self.selected_dir.trace_add('write', lambda *args: self.Add_Dir_Classes() )

        select_button = tk.Button( dir_frame  ,  text="Select Directory" ,  command=lambda: Frame_Manager.select_directory_window(self , dir_entry) )
        select_button.pack(side='left', padx=(5, 0))

        dropdown_frame = tk.Frame( self )
        dropdown_frame.pack(fill='x', pady=(10, 0), padx=10)

        self.Class_dropdown_var = tk.StringVar()
        self.Image_dropdown_var = tk.StringVar()

        tk.Label(dropdown_frame , text= 'Class : ' ).pack(side = tk.LEFT , anchor= 'w')
        self.Class_dropdown = ttk.Combobox( dropdown_frame  , textvariable= self.Class_dropdown_var , state='disabled', width= 30 )
        self.Class_dropdown.pack(side=tk.LEFT , anchor='w')
        self.Class_dropdown_var.trace_add('write', lambda *args: self.Add_Image_Options() )



        tk.Label(dropdown_frame , text= 'Image : ' ).pack(side = tk.LEFT , anchor= 'w')
        self.Image_dropdown = ttk.Combobox( dropdown_frame, textvariable = self.Image_dropdown_var , state='disabled' , width= 15 )
        self.Image_dropdown.pack(side=tk.LEFT , anchor='w')


        # Evaluation Options Frame
        eval_options_frame = tk.Frame(self)
        eval_options_frame.pack(fill='x', pady=(10, 0), padx=10)

        heatmap_button = tk.Button( eval_options_frame ,  text='Heatmap' , command= lambda : self.Heatmap_Pressed( path = os.path.join( self.selected_dir.get() , self.Class_dropdown_var.get() , self.Image_dropdown_var.get() )) )
        heatmap_button.pack(side='left', padx=(0, 5))

        heatmap_button = tk.Button( eval_options_frame ,  text='Heatmap (Random 100)' , command= lambda : self.Heatmap_Pressed( path = os.path.join( self.selected_dir.get() , self.Class_dropdown_var.get() ) , random_100 = True) )
        heatmap_button.pack(side='left', padx=(0, 5))

        correlation_button = tk.Button( eval_options_frame  ,  text='Correlation' , command= lambda : self.Correlation_Pressed( path = os.path.join( self.selected_dir.get() , self.Class_dropdown_var.get() , self.Image_dropdown_var.get() ))  )
        correlation_button.pack(side='left')


        correlation_button = tk.Button( eval_options_frame  ,  text='Metrics' , command= lambda : self.Mertrics_Pressed( path = os.path.join( self.selected_dir.get() , self.Class_dropdown_var.get() , self.Image_dropdown_var.get() ))  )
        correlation_button.pack(side='left')

        Single_Pred_Button = tk.Button( eval_options_frame  ,  text='Single Prediction' , command= lambda : self.Prob_image_func( path = os.path.join( self.selected_dir.get() , self.Class_dropdown_var.get() , self.Image_dropdown_var.get() ))  )
        Single_Pred_Button.pack(side='left')

        Big_Pred_Button = tk.Button( eval_options_frame  ,  text='Class Probs' , command= lambda : self.Whole_Class_prediction( path = os.path.join( self.selected_dir.get() , self.Class_dropdown_var.get() ))  )
        Big_Pred_Button.pack(side='left')

        Deposite_button = tk.Button( eval_options_frame  ,  text='Image_to_Array' , command= lambda : self.Important_Regions( path = os.path.join( self.selected_dir.get()  ))  )
        Deposite_button.pack(side='left')
        
        Energy_Hist_button = tk.Button( eval_options_frame  ,  text='Energy_Hist' , command= lambda : self.Energy_Prediction_Hist( path = os.path.join( self.selected_dir.get()  ))  )
        Energy_Hist_button.pack(side='left')

        Pixel_Hist_button = tk.Button( eval_options_frame  ,  text='Pixel_Count_Hist' , command= lambda : self.Pixel_Count_Prediction_Hist( path = os.path.join( self.selected_dir.get()  ))  )
        Pixel_Hist_button.pack(side='left')

        Pixel_PDF = tk.Button( eval_options_frame  ,  text='Download_Pixel_PDF' , command= lambda : self.Pixel_Predictions_PDF( path = os.path.join( self.selected_dir.get()  ))  )
        Pixel_PDF.pack(side='left')

        FP_PDF = tk.Button( eval_options_frame  ,  text='False_Positve_NES' , command= lambda : self.Scattering_Softmax_PDF( path = os.path.join( self.selected_dir.get()  ))  )
        FP_PDF.pack(side='left')

        FP_2_Class_PDF = tk.Button( eval_options_frame  ,  text='Two_Class_Threshold' , command= lambda : self.Two_Class_Binary_Threshold_PDF( path = os.path.join( self.selected_dir.get()  ))  )
        FP_2_Class_PDF.pack(side='left')

        Lepton_Angle_PDF = tk.Button( eval_options_frame  ,  text='Lepton_Angle' , command= lambda : self.Two_Class_Lepton_Angle( path = os.path.join( self.selected_dir.get()  ))  )
        Lepton_Angle_PDF.pack(side='left')

        self.Figure_Canvas_Frame = tk.Frame( self )
        self.Figure_Canvas_Frame.pack( anchor='w' , side= tk.TOP , pady= 10 )


    def Add_Dir_Classes(self):
        possible_classes = sorted ( os.listdir( self.selected_dir.get() ) ) 
        self.Class_dropdown_var.set('')
        self.Class_dropdown.config(state='readonly' , values= possible_classes )

        return

    def Add_Image_Options(self):
        path = os.path.join( self.selected_dir.get()  , self.Class_dropdown_var.get() )
        self.Image_dropdown_var.set('')
        possible_images = sorted ( os.listdir( path) ) 

        # sorted_possible_images = sorted(possible_images, key=lambda x: int(x.split('_')[2].split('.')[0]))
        self.Image_dropdown.config(state='readonly' , values= possible_images)

    def Heatmap_Pressed(self , path  , random_100 = False):
        
        if random_100 == False:
            Model_Evaluation_script.Evaluating_Model.Heatmap_func( self , path = path)
        else:
            Model_Evaluation_script.Evaluating_Model.Heatmap_func( self , path = path , random_100 = True)

        return  

    def Prob_image_func(self, path ):
        
        print(path)

        Model_Evaluation_script.Evaluating_Model.Plot_Prob_Single_Image_func( self , path = path)



    def Correlation_Pressed(self , path ):

        Model_Evaluation_script.Evaluating_Model.Correlation_func( self)

        return  
    

    def Mertrics_Pressed(self , path ):

        Model_Evaluation_script.Evaluating_Model.Plot_metrics_func( self)

        return  

    def Whole_Class_prediction(self , path):

        Model_Evaluation_script.Evaluating_Model.Whole_Class_Predictinos( self , path = path)

        return

    def Important_Regions(self, path):

        print( path )
        Model_Evaluation_script.Evaluating_Model.Detector_Important_Regions( self , path = path)

        return


    def Energy_Prediction_Hist(self, path):

        Model_Evaluation_script.Evaluating_Model.energy_prediction_distribution( self , path = path)

        return


    def Pixel_Count_Prediction_Hist(self, path):

        Model_Evaluation_script.Evaluating_Model.Active_Pixel_Count( self , path = path)

        return


    def Pixel_Predictions_PDF(self, path):

        Model_Evaluation_script.Evaluating_Model.Download_Pixel_Eval_Plots( self , path = path)

        return


    def Scattering_Softmax_PDF(self, path):

        Model_Evaluation_script.Evaluating_Model.Scattering_False_Positve_Analysis( self , path = path)

        return


    def Two_Class_Binary_Threshold_PDF(self, path):

        Model_Evaluation_script.Evaluating_Model.Two_Class_Scattering_False_Positve_Analysis( self , path = path)

        return

    def Two_Class_Lepton_Angle(self, path):

        Model_Evaluation_script.Evaluating_Model.two_class_lepton_angel_PDF( self , path = path)

        return

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
        self.Back_Button = tk.Button( self.back_frame, text='Back to Figure Selection' , command= lambda: controller.show_frame(Plot_Selection_Page)  )
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
        self.file_combobox = ttk.Combobox( self.file_select_frame , textvariable = self.file_selected , values = controller.Allowed_Files , state='readonly' , width=60 )
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
        self.x_combobox = ttk.Combobox( axis_select_frame, textvariable=self.x_selected, values=column_names, state='readonly', width=10  )
        self.x_combobox.pack(anchor='w',side=tk.LEFT , padx=6 )

    
        tk.Label(axis_select_frame, text='y axis: ').pack(side=tk.LEFT)

        self.y_combobox = ttk.Combobox( axis_select_frame, textvariable=self.y_selected, values=column_names, state='readonly', width=10 )
        self.y_combobox.pack(anchor='w',side=tk.LEFT)

        tk.Label(axis_select_frame, text='z axis: ').pack(side=tk.LEFT)

        self.z_combobox = ttk.Combobox(  axis_select_frame, textvariable=self.z_selected, values=column_names, state='disabled', width=10 )
        self.z_combobox.pack(anchor='w',side=tk.LEFT)

        # Variable to hold the cmap selection
        self.cmap_yes_no = tk.StringVar()
        self.cmap_option_select = tk.StringVar()

        self.colour_map_frame = tk.Frame(self)
        self.colour_map_frame.pack(anchor='w', pady=5)

        tk.Label(self.colour_map_frame, text="cmap:  ").pack(side=tk.LEFT)

        self.cmap_combobox = ttk.Combobox( self.colour_map_frame, textvariable=self.cmap_yes_no, values=['No', 'Yes'],  state='readonly', width=10 )
        self.cmap_combobox.set('No')
        self.cmap_combobox.pack(anchor='w', side=tk.LEFT)
        
        self.cmap_selection_combobox = ttk.Combobox(  self.colour_map_frame, textvariable=self.cmap_option_select,   values=['viridis', 'plasma', 'inferno', 'magma', 'cividis'],  width=10, state='disabled'  )
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

        self.custom_combobox = ttk.Combobox( self.custom_fig_select_frame , textvariable = self.custom_fig_seleceted , values = Custom_Plot_Names , state='readonly' , width=20 )

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
        # Close the old figure (if any) and clear the frame
        if hasattr(self, 'custom_fig'):
            plt.close(self.custom_fig)
        for widget in self.Custom_Figure_Frame.winfo_children():
            widget.destroy()

        # Create the custom figure if the selected type is one of the Track_dE_Analysis types
        if self.custom_fig_seleceted.get() in ['Track_dE_Analysis', 'Track_dE_Analysis_Thesis']:
            self.custom_fig = plt.figure(figsize=(9, 6))
            canvas = FigureCanvasTkAgg(self.custom_fig, master=self.Custom_Figure_Frame)
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            if not hasattr(self, "Download_dir"):
                self.Download_dir = tk.Button(
                    self.plot_button_frame,
                    text='Download All File Plots',
                    command=lambda: Custom_Plot_script.Custom_Plot.Track_dE_Analysis(
                        self,
                        {'Plot_Mode': 'Download_Dir', 'canvas': canvas, 'fig': self.custom_fig}
                    )
                )
                self.Download_dir.pack(anchor='w', side=tk.LEFT)

            # Call the plotting function based on the selection
            getattr(Custom_Plot_script.Custom_Plot, self.custom_fig_seleceted.get())(
                self,
                {'Plot_Mode': 'Single_Plot', 'canvas': canvas, 'fig': self.custom_fig}
            )

        elif self.custom_fig_seleceted.get() == 'Specific_Vertex':

            self.custom_fig = plt.figure(figsize=(9, 6))
            canvas = FigureCanvasTkAgg(self.custom_fig, master=self.Custom_Figure_Frame)
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            getattr(Custom_Plot_script.Custom_Plot, self.custom_fig_seleceted.get())(
                self,
                {'Plot_Mode': 'Single_Plot', 'canvas': canvas, 'fig': self.custom_fig}
            )


    def on_custom_selected(self, *args):
        """Callback triggered when a new custom figure is selected from the dropdown."""
        if self.custom_fig_seleceted.get() in ['Track_dE_Analysis', 'Track_dE_Analysis_Thesis' , 'Specific_Vertex']:
            self.file_selected.trace('w', self.on_file_selected)


    def on_file_selected(self, *args):
        path = os.path.join(self.controller.Data_Directory, self.file_selected.get())
        with h5py.File(path, 'r') as sim_h5:
            unique_event_ids = list(np.unique(sim_h5["segments"]['event_id']))

        # Create a new StringVar and add the trace
        self.event_id_selected = tk.StringVar()
        self.event_id_selected.trace_add('write', self.on_event_selected)

        if hasattr(self, 'event_combobox') and self.event_combobox:
            self.event_combobox['values'] = unique_event_ids
            self.event_combobox.config(textvariable=self.event_id_selected)
            if hasattr(self, 'vertex_combobox'):
                self.vertex_combobox['values'] = []
                if hasattr(self, 'vertex_id_selected'):
                    self.vertex_id_selected.set('')
        else:
            tk.Label(self.file_select_frame, text="event id :").pack(padx=(10, 10), side=tk.LEFT)
            self.event_combobox = ttk.Combobox(
                self.file_select_frame,
                textvariable=self.event_id_selected,
                values=unique_event_ids,
                width=5,
                state='readonly'
            )
            self.event_combobox.pack(anchor='w', side=tk.LEFT)


    def on_event_selected(self, *args):
        """Callback triggered when a new event is selected from the dropdown."""
        print('Im getting called')
        path = os.path.join(self.controller.Data_Directory, self.file_selected.get())
        with h5py.File(path, 'r') as sim_h5:
            segments = sim_h5["segments"]
            event_segment = segments[segments['event_id'] == int(self.event_id_selected.get())]
            unique_vertex_ids = list(np.unique(event_segment['vertex_id']))

        # If the vertex combobox exists, update its values and reuse the same StringVar.
        if hasattr(self, 'vertex_combobox') and self.vertex_combobox:
            if not hasattr(self, 'vertex_id_selected'):
                self.vertex_id_selected = tk.StringVar()
                self.vertex_combobox.config(textvariable=self.vertex_id_selected)
            self.vertex_combobox['values'] = unique_vertex_ids
            self.vertex_id_selected.set('')
        else:
            # Create the vertex combobox along with its StringVar.
            self.vertex_id_selected = tk.StringVar()
            tk.Label(self.file_select_frame, text="vertex id :").pack(padx=(10, 10), side=tk.LEFT)
            self.vertex_combobox = ttk.Combobox(
                self.file_select_frame,
                textvariable=self.vertex_id_selected,
                values=unique_vertex_ids,
                width=10,
                state='readonly'
            )
            self.vertex_combobox.pack(anchor='w', side=tk.LEFT)

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

        # print("Allowed Files Updated:", self.controller.Allowed_Files)
        print("Allowed Files Updated:")


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

    def select_directory_window(self , Text_Box , select_class_setting_button = None):
        
        directory_path = tk.filedialog.askdirectory( initialdir= str(os.getcwd()) , title="Select a Directory")

        # Text_Box.config( state = 'normal')
        # Text_Box.set( str(directory_path) )
        if directory_path:
            # Update the Entry
            Text_Box.config(state='normal')
            Text_Box.delete(0, tk.END)
            Text_Box.insert(0, directory_path)
            Text_Box.config(state='disabled')

            if select_class_setting_button != None:
                # Enable next button
                select_class_setting_button.config(state='normal')

            # Save it in the controller’s global attribute
            self.controller.selected_directory = directory_path

            if str(self.__class__.__name__) == 'Model_Training_Page':
                # Now call the method on Model_Training_Class_Page
                training_class_page = self.controller.frames[Advance_Class_Selection_Page]
                training_class_page.Update_Page_With_Class()

        else:
            Text_Box.config(state='normal')
            Text_Box.delete("1.0", tk.END)
            Text_Box.insert("1.0", directory_path)
            Text_Box.config(state='disabled')
        return

    def setup_process(self):
        if not self.controller.running:
            # self.controller.running = True
            # self.frame.Create_Dataset_Button.config(state='disabled')
            
            try:
                self.progress_value = 0
                self.progress['value'] = 0
                self.progress.config(maximum=100) 

                self.controller.running = True
            except:
                pass

            if str(self.__class__.__name__) == 'Create_Dataset_Page':
                self.Create_Dataset_Button.config(state='disabled')
                Frame_Manager.check_progress(self)

                threading.Thread(target=self.Create_ML_Dataset_2).start()

            elif str(self.__class__.__name__) == 'Model_Training_Page':
                # Frame_Manager.check_progress(self)
                # threading.Thread(target=self.Train_Model).start()
                self.controller.running = True
                threading.Thread( target= lambda : Model_Training_script.Model_Training.Train_Model(self) ).start()


            else:
                print(self.__class__.__name__ )
                # self.Create_Fig_Button.config(state='disabled')
                Frame_Manager.check_progress(self)

                threading.Thread(target=Pixel_Array_Script.Use_Pixel_Array.plot, args=(self,)).start()

    def cancel_process(self):
        if self.controller.running:
            try:
                self.progress['value'] = 100
            except:
                pass
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


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-f', '--Data_Directory', required=True, type=str, help='Path to simulation directory')
#     args = parser.parse_args()
#     app = App( Data_Directory=args.Data_Directory )
#     app.mainloop()

if __name__ == '__main__':
    # root = tk.Tk()
    # root.withdraw()  # Hide the extra root
    directory_path = tk.filedialog.askdirectory(title="Select a Data Directory" , initialdir = os.getcwd() )
    app = App(Data_Directory=directory_path)
    app.mainloop()

