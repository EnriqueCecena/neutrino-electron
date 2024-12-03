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


import matplotlib
matplotlib.use('agg')  # If I don't change the backend of matplotlib while creating plots it will clash with my progress bar backend and crash my window 😅

import threading

# Backend scripts:
import Custom_Plot_script
import pdg_id_script


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
            # print(f"{frame_class.__name__} not found in frames.")

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




        # Import and Shuffle pdg_id_map for cmapping
        # temp_items = list( pdg_id_map.items() )
        # np.random.shuffle(temp_items)
        # pdg_id_map = dict( temp_items )
        self.cmap = cm.plasma
        # Add destroy and reinitialize methods, because of headaches.
        self.destroy_frame      = self._destroy_frame
        self.reinitialize_frame = self._reinitialize_frame
        
        
        self.Data_Directory     = Data_Directory        # Create initial attributes for accessing the data directory 


        self.pdg_id_map         = pdg_id_script.pdg_id_map            # Apply the imported pdg map as a attribute

        self.pdg_id_map_reverse = { self.pdg_id_map[i] : i for i in list( pdg_id_script.pdg_id_map.keys() ) }
        
        self.plot_type          = 'scatter'             # Set initial plot_type state, should help with cleaning up code. 

        self.running = False

        self.max_z_for_plot = round(950)
        self.max_y_for_plot = round(100)
        self.max_x_for_plot = round(350)

        self.min_z_for_plot = round(400)
        self.min_y_for_plot = round(-250)
        self.min_x_for_plot = round(-350)

        # Retrieve and sort file names
        File_Names = os.listdir(Data_Directory)
        Temp_File_Names_Dict = {int(i.split('.')[3]): i for i in File_Names}
        sorted_keys = sorted(Temp_File_Names_Dict.keys())
        File_Names = [Temp_File_Names_Dict[i] for i in sorted_keys]
        self.File_Names = File_Names
        self.Allowed_Files = self.File_Names.copy()

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
            Plot_Selection_Page,
            Figure_Creation_Page,
            Custom_Figure_Page,
            Settings_Page,
            File_Selection_Page
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

       
        # self.current_frame = 
        # Dynamically resize window based on the frame
        if page == File_Selection_Page:
            self.geometry("800x500")  # Larger size for File Selection Page
        
        
        # Larger size for View Segments Page
        elif page == View_Segments_Page or page == View_mc_hdr_Page or page == View_traj_Page:
            self.geometry("1600x500")  


        elif page == Figure_Creation_Page or Custom_Figure_Page or page == Create_Dataset_Page:
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
                  command=lambda: controller.show_frame(Dataset_Page)).pack(anchor='w')
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
                  command=lambda: controller.show_frame(StartPage)).pack(anchor='w')

        tk.Button(self, text="Back to Start Page",
                  command=lambda: controller.show_frame(StartPage)).pack(anchor='w')
        

class Create_Dataset_Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        self.controller = controller

        # Header Frame: Back button and Header label
        self.header_frame = tk.Frame(self)
        self.header_frame.pack(anchor='w', padx=10, pady=20)  

        # Back Button
        back_button = tk.Button( self.header_frame,  text='Back',   command=lambda: controller.show_frame(Dataset_Page)  )
        back_button.pack(side=tk.LEFT)

        # Header Label
        header_label = tk.Label( self.header_frame,  text="Create Dataset",  font=("Helvetica", 16) )
        header_label.pack(side=tk.LEFT, padx=150)

        # Progress Bar and Percentage Frame
        self.progressive_frame = tk.Frame(self)
        self.progressive_frame.pack(anchor='w', padx=10, pady=(0, 20))  

        # self.progress = ttk.Progressbar( self.progressive_frame,  orient="horizontal", length=600,  mode="determinate" ) 
        self.progress = ttk.Progressbar( self.progressive_frame,  orient="horizontal", length=600,  mode="determinate" ) 
        self.progress_label = tk.Label(self.progressive_frame , text = '' , font=("Arial", 12))
        self.progress.pack(anchor='w' , side=tk.LEFT)
        self.progress_label.pack(anchor='w' , side=tk.LEFT )

        # File Selection Frame
        self.file_select_frame = tk.Frame(self)
        self.file_select_frame.pack(anchor='w', padx=10, pady=(0, 20))  
        # File Selection Label
        file_label = tk.Label(self.file_select_frame, text="File: ")
        file_label.pack(side=tk.LEFT)

        # File Dropdown (Combobox)
        self.file_selected = tk.StringVar()
        self.file_combobox = ttk.Combobox( self.file_select_frame,  textvariable=self.file_selected,  values=controller.Allowed_Files,  state='readonly',  width=60 )
        self.file_combobox.pack(side=tk.LEFT, padx=(5, 0))  

        # Interact Frame ... running out of names

        self.Interact_Frame = tk.Frame(self)
        self.Interact_Frame.pack(anchor='w', padx=10, pady=(0, 20))  

        self.Preview_Button = tk.Button(self.Interact_Frame , text = 'Preview' , command= lambda : self.Preview_Interaction() )
        self.Preview_Button.pack( side = tk.LEFT , anchor='w')
        # self.Create_Dataset_Button = tk.Button(self.Interact_Frame , text = 'Create' , command= lambda: self.Create_ML_Dataset() )
        self.Create_Dataset_Button = tk.Button(self.Interact_Frame , text = 'Create' , command = self.setup_process )

        self.Create_Dataset_Button.pack( side = tk.LEFT , anchor='w')

        self.Preview_Fig_Frame =  tk.Frame(self)
        self.Preview_Fig_Frame.pack(anchor='w', side= tk.LEFT ,pady=5)

    def Preview_Interaction(self):

        path = os.path.join(  self.controller.Data_Directory , self.file_selected.get() )

        # print(path)
        sim_h5 = h5py.File(path , 'r')

        temp_segments =  pd.DataFrame(sim_h5["segments"][()])
        temp_mc_hdr =  sim_h5['mc_hdr'] 

        temp_segments = temp_segments[ ( temp_segments['dE'] > 1.5 )   ]
        unique_ids = np.unique(temp_segments['event_id']).tolist()

        random_event_id  = np.random.choice(  unique_ids )

        temp_segments = temp_segments[ (temp_segments['event_id'] == random_event_id)]
        temp_mc_hdr   = temp_mc_hdr[ (temp_mc_hdr['event_id'] == random_event_id)]

        cmap = cm.plasma
        norm = plt.Normalize(vmin=0, vmax=max(temp_segments['dE']))

        # Create Canvas
        # Destroy old widgets in the Preview_Fig_Frame
        if hasattr(self, 'fig'):
            plt.close(self.fig)  # Close the old figure
        for widget in self.Preview_Fig_Frame.winfo_children():
            widget.destroy()

        # Create subplot
        self.preview_fig, self.preview_ax = plt.subplots( )

        canvas = FigureCanvasTkAgg( self.preview_fig, master= self.Preview_Fig_Frame)  
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        mc_hdr_unique_vertex_ids = np.unique( temp_mc_hdr['vertex_id'] ).tolist()

        noise_indexes = [                           index 
                         for index , vertex_id in zip( temp_segments.index.to_list() , temp_segments['vertex_id'].to_list() ) 
                            if vertex_id not in  mc_hdr_unique_vertex_ids]
        
        test_group_df = temp_segments[ (temp_segments['vertex_id'] == mc_hdr_unique_vertex_ids[0] )]
        

        # Plot the clean vertex
        c = test_group_df['dE']

        self.preview_ax.scatter( test_group_df['z'] , test_group_df['y'] , c = c ,cmap = cmap , norm= norm , s = 7 )

        # Add the noise back in 
        noise_df = temp_segments[ temp_segments.index.isin( noise_indexes )]

        c = noise_df['dE']
        self.preview_ax.scatter( noise_df['z'] , noise_df['y'] , c = c ,cmap = cmap , norm= norm , s = 6)

        self.preview_ax.set_xlim( self.controller.min_z_for_plot , self.controller.max_z_for_plot  )
        self.preview_ax.set_ylim( self.controller.min_y_for_plot , self.controller.max_y_for_plot  )
        self.preview_ax.set_xlabel( 'Z' )
        self.preview_ax.set_ylabel( 'Y' )
        canvas.draw()

        toolbar = NavigationToolbar2Tk(canvas, self.Preview_Fig_Frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.LEFT, fill=tk.X)
        self.controller.geometry("900x900")


        return




    def setup_process(self):
        if not self.controller.running:
            self.controller.running = True
            self.Create_Dataset_Button.config(state='disabled')

            # Initialize progress tracking variables
            self.progress_value = 0
            self.progress['value'] = 0
            # self.progress['maximum'] = 100  # Set to 100%, we'll compute actual percentages
            self.progress.config(maximum=100)   # Set to 100%, we'll compute actual percentages

            threading.Thread(target=self.Create_ML_Dataset).start()
            self.check_progress()  # Start checking progress in the main thread


    def check_progress(self):
        # Update the progress bar in the main thread
        self.progress['value'] = self.progress_value
        self.progress_label.config(text=f"{self.progress_value:.0f}%")
        if self.controller.running:
            # Schedule the next check
            self.after(1, self.check_progress)
        else:
            # Re-enable the button when done
            self.Create_Dataset_Button.config(state='normal')


    def Create_ML_Dataset(self):

        #Create a name for directory that will hold all of the directory datasets
        Test_directory = "Test_ML_File"
        os.makedirs(Test_directory, exist_ok=True)

        # Create mapping for Directory Naming
        Directory_Name_Map = {r"$\nu$-$e^{-}$ scattering"   : "Neutrino_Electron_Scattering" , 
                                r"$\nu_{e}$-CC"               : "Neutrino_Electron_CC" , 
                                r"$\nu_{e}$-NC"               : "Neutrino_Electron_NC" , 
                                "Other"                       : "Other"}
        # Create sub-directories
        for Dir_Name in list(Directory_Name_Map.values()) : 
            _  = os.path.join( Test_directory , Dir_Name )
            os.makedirs( _ , exist_ok = True)

        # Create file name counters for images generated
        Dir_File_Name_Counter = { file : 0 for file in list(Directory_Name_Map.keys() )}

        # Load the selected path & Get the segment data set.
        path = os.path.join(  self.controller.Data_Directory , self.file_selected.get() )
        sim_h5 = h5py.File(path , 'r')
        temp_segments = sim_h5["segments"]
        temp_segments = temp_segments[ ( temp_segments['dE'] > 1.5 )   ]
        temp_segments =  pd.DataFrame(temp_segments[()])
        temp_mc_hdr =  sim_h5['mc_hdr'] 

        unique_ids = np.unique(temp_segments['event_id']).tolist()

        cmap = cm.plasma

        self.controller.running = True
        self.progress.configure(maximum= len(unique_ids))
        self.Create_Dataset_Button.config(state='disabled')
        # Loop for each event & and each vertex and create the desired adding it to the correct sub-directory
        for i , event_id  in enumerate(unique_ids , start = 1) :


            temp_segments_event = temp_segments[ (temp_segments['event_id'] == event_id)]
            temp_mc_hdr_event   = temp_mc_hdr[ (temp_mc_hdr['event_id'] == event_id)]

            mc_hdr_vertex_ids = np.unique( temp_mc_hdr_event['vertex_id'] ).tolist()
            noise_indexes = [ i for i in  temp_segments_event['vertex_id'] if i not in mc_hdr_vertex_ids]

            for true_vertex in mc_hdr_vertex_ids:

                # try:
                temp_mc_hdr_event_vetex = temp_mc_hdr_event[ (temp_mc_hdr_event['vertex_id'] == true_vertex  ) ]
                temp_segments_event_vertex = temp_segments_event[( temp_segments_event['vertex_id'] == true_vertex )]

                if self.controller.min_z_for_plot < temp_mc_hdr_event_vetex['z_vert'] < self.controller.max_z_for_plot  and  self.controller.min_y_for_plot < temp_mc_hdr_event_vetex['y_vert'] < self.controller.max_y_for_plot and self.controller.min_x_for_plot < temp_mc_hdr_event_vetex['x_vert'] < self.controller.max_x_for_plot:
                    pass
                else:
                    continue

                if temp_mc_hdr_event_vetex['reaction'] == 7:
                    interaction_label   = r"$\nu$-$e^{-}$ scattering"

                elif temp_mc_hdr_event_vetex['nu_pdg'] == 12 and temp_mc_hdr_event_vetex['isCC'] == True:
                    interaction_label   = r"$\nu_{e}$-CC"

                elif temp_mc_hdr_event_vetex['nu_pdg'] == 12 and temp_mc_hdr_event_vetex['isCC'] == False:
                    interaction_label   = r"$\nu_{e}$-NC"
                else:
                    interaction_label = 'Other'


                norm = plt.Normalize(vmin=0, vmax=75 )
                
                noise_df = temp_segments_event[ temp_segments_event.index.isin(noise_indexes) ]

                plt.scatter( temp_segments_event_vertex['z'] , temp_segments_event_vertex['y'] , c = temp_segments_event_vertex['dE'] , cmap = cmap , norm = norm , s = 6 )
                plt.scatter( noise_df['z'] , noise_df['y'] , c = noise_df['dE'] , cmap = cmap , norm = norm , s = 6)

                plt.xlim( self.controller.min_z_for_plot , self.controller.max_z_for_plot  )
                plt.ylim( self.controller.min_y_for_plot , self.controller.max_y_for_plot  )
                plt.axis('off')

                Dir_File_Name_Counter[ interaction_label ] += 1 
                loop_path = os.path.join( Test_directory ,Directory_Name_Map[ interaction_label ] )
                loop_path = os.path.join( loop_path , f"IMG_{Dir_File_Name_Counter[interaction_label]}.png" )

                plt.savefig( loop_path )
                plt.close()
                #     pass
                    
                # except:
                #     continue

            self.progress_value = (i / len(unique_ids)) * 100  # Calculate percentage



        self.controller.running = False
        self.progress_value = 100  # Ensure progress bar reaches 100%
        self.check_progress()      # Update the GUI one last time
        self.Create_Dataset_Button.config(state='normal')





    pass
        

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

        # tk.Button(self, text='Compare "Cleaning" Methods',
        #             command=lambda: controller.show_frame(Dataset_View_Page)).pack(anchor='w', pady=2)

        tk.Button(self, text='Plot Settings',
                    command=lambda: controller.show_frame(Dataset_View_Page) , state= 'disabled').pack(anchor='w', pady=2)

        tk.Button(self, text="Back to Start Page",
                    command=lambda: controller.show_frame(StartPage)).pack(anchor='w', pady=2)
        # tk.Button(self, text="Create Line Plot", command=self.create_line_plot).pack( anchor='w' )


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

        Page_Title_str = tk.StringVar()
        # Page Title
        Page_Title = tk.Label(self, text="Figure Creation", font=("Helvetica", 16) , textvariable = Page_Title_str ).pack(anchor='w', pady=(0, 10))
        # tk.Label(self, text="Figure Creation", font=("Helvetica", 16)).pack(anchor='w' )

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
        self.x_combobox.pack(anchor='w',side=tk.LEFT)

    
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
        tk.Button(self, text='Create' , command= lambda : self.Plot_Type_Map() ).pack(anchor='w', pady=10)

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
            self.Create_Scatter_Fig()
            pass

        elif self.controller.plot_type == 'line':
            self.Create_Line_PLot()
            pass

        elif self.controller.plot_type == 'hist':
            self.Create_Hist_Fig()
            pass

        return
    
    #All of the Scatter Plots should be handled here ... in a ideal world. 
    def Create_Scatter_Fig(self, *args):


        path = os.path.join(  self.controller.Data_Directory , self.file_selected.get() )
        with h5py.File(path , 'r') as sim_h5:
            temp_df =  pd.DataFrame(sim_h5["segments"][()])
            temp_df = temp_df[ ( temp_df['dE'] > 2 ) & ( temp_df['event_id'] == int(self.event_combobox.get()) ) ]

        # Create Canvas
        # Destroy old widgets in the Figure_Frame
        if hasattr(self, 'fig'):
            plt.close(self.fig)  # Close the old figure
        for widget in self.Figure_Frame.winfo_children():
            widget.destroy()


        if self.dropdown_3d_select.get() == 'Yes':
            self.fig, self.ax = plt.subplots( subplot_kw=dict(projection='3d') )
    
        else:
            self.fig, self.ax = plt.subplots( )

        canvas = FigureCanvasTkAgg( self.fig, master= self.Figure_Frame)  
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        if str(self.cmap_yes_no.get()) == 'Yes' and str(self.cmap_selection_combobox.get()) != '' :
            cmap = colormaps[ self.cmap_selection_combobox.get() ]
            # cmap = cm['plasma']

            norm = plt.Normalize(vmin=0, vmax= 75)
            c = temp_df['dE']

        else:
            cmap = None
            norm = None
            c    = None

        if self.dropdown_3d_select.get() == 'Yes':

            scatter = self.ax.scatter( temp_df[ self.x_combobox.get() ] , temp_df[ self.y_combobox.get() ] , temp_df[ self.z_combobox.get() ] , c = c ,cmap = cmap , norm= norm)
            self.ax.set_zlabel( self.z_combobox.get() )


        else:
            self.ax.scatter( temp_df[ self.x_combobox.get() ] , temp_df[ self.y_combobox.get() ] , c = c ,cmap = cmap , norm= norm)


        self.ax.set_xlabel( self.x_combobox.get() )
        self.ax.set_ylabel( self.y_combobox.get() )
        canvas.draw()

        self.fig.colorbar( scatter , ax=self.ax ,  shrink=0.5, aspect=10)

        toolbar = NavigationToolbar2Tk(canvas, self.Figure_Frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.LEFT, fill=tk.X)

        self.controller.geometry("900x900")

    # All generic line plots should be creeated here... line plots are not usefull
    def Create_Line_PLot(self , *args):

        path = os.path.join(  self.controller.Data_Directory , self.file_selected.get() )
        with h5py.File(path , 'r') as sim_h5:
            temp_df =  pd.DataFrame(sim_h5["segments"][()])
            temp_df = temp_df[ ( temp_df['dE'] > 2 ) & ( temp_df['event_id'] == int(self.event_combobox.get()) ) ]

        # Create Canvas
        # Destroy old widgets in the Figure_Frame
        if hasattr(self, 'fig'):
            plt.close(self.fig)  # Close the old figure
        for widget in self.Figure_Frame.winfo_children():
            widget.destroy()


        self.fig, self.ax = plt.subplots()

        canvas = FigureCanvasTkAgg( self.fig, master= self.Figure_Frame)  
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


        self.ax.plot( temp_df[ self.x_combobox.get() ] , temp_df[ self.y_combobox.get() ] )
        self.ax.set_xlabel( self.x_combobox.get() )
        self.ax.set_ylabel( self.y_combobox.get() )
        canvas.draw()

        toolbar = NavigationToolbar2Tk(canvas, self.Figure_Frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.LEFT, fill=tk.X)

        self.controller.geometry("900x900")


    # All of the Hist Plots should be handled here ... in a ideal world. 
    def Create_Hist_Fig(self, *args):

        path = os.path.join(  self.controller.Data_Directory , self.file_selected.get() )
        with h5py.File(path , 'r') as sim_h5:
            temp_df =  pd.DataFrame(sim_h5["segments"][()])
            temp_df = temp_df[ ( temp_df['dE'] > 2 ) & ( temp_df['event_id'] == int(self.event_combobox.get()) ) ]

        # Create Canvas
        # Destroy old widgets in the Figure_Frame
        if hasattr(self, 'fig'):
            plt.close(self.fig)  # Close the old figure
        for widget in self.Figure_Frame.winfo_children():
            widget.destroy()

        self.fig, self.ax = plt.subplots()
        canvas = FigureCanvasTkAgg( self.fig, master= self.Figure_Frame)  
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        if str(self.group_yes_no.get()) == 'Yes' and str(self.hist_option_select.get() )!= '':
            hist_list = []
            name_list = []
            for group_name, group_df in temp_df.groupby( self.hist_option_select.get() ):
                hist_list.append( group_df[ self.x_combobox.get() ].to_list() )

                if self.hist_option_select.get() == 'pdg_id':
                    name_list.append( self.controller.pdg_id_map[ str(group_name) ]  )
                else:
                    name_list.append(group_name)

            self.ax.hist( hist_list , bins = 100 , stacked = True , label = name_list )
            self.ax.set_title(f"Grouped By : { self.hist_option_select.get() }")
            plt.legend(fontsize = 7 , loc="upper right")


        else:
            self.ax.hist( temp_df[ self.x_combobox.get() ] , bins = 100)

        self.ax.set_xlabel( self.x_combobox.get() )
        self.ax.set_ylabel( 'Frequency' )

        toolbar = NavigationToolbar2Tk(canvas, self.Figure_Frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.LEFT, fill=tk.X)
        self.controller.geometry("900x900")
        return


    def Progression(self , possision , termination , note =''):
        progress = 100 * (possision / float(termination))
        bar = str('~'*(100-(100-int(progress)))+'→' +' ' * (100-int(progress)))
        print( "\r¦{}¦ {:.0f}% {:}".format( bar ,progress , note ),end='')





        
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
        if str(self.custom_fig_seleceted.get()) == 'Track_dE_Analysis':

            if hasattr(self, 'fig'):
                plt.close(self.custom_fig)  # Close the old figure
            for widget in self.Custom_Figure_Frame.winfo_children():
                widget.destroy()


            # if not hasattr(self, 'particle_frame'):


            self.custom_fig = plt.figure( figsize=(6, 6) )
            canvas = FigureCanvasTkAgg( self.custom_fig, master= self.Custom_Figure_Frame)  
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            if not hasattr(self, "Download_dir"):
                self.Download_dir = tk.Button( self.plot_button_frame, text='Download All File Plots', command= lambda:  Custom_Plot_script.Custom_Plot.Track_dE_Analysis(self , {'Plot_Mode' : 'Download_Dir' ,  'canvas' : canvas  ,'fig' : self.custom_fig   } ) )  
                self.Download_dir.pack(anchor='w', side=tk.LEFT )

 

            Custom_Plot_script.Custom_Plot.Track_dE_Analysis(self , {   'Plot_Mode'     : 'Single_Plot' , 
                                                                        'canvas'        : canvas        ,
                                                                        'fig'           : self.custom_fig,   } ) 
                                                            # 'ax'            : self.custom_ax } )
            






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

                # print(event_segment)

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

            # self.event_id_selected.trace('w', self.on_event_selected)




        return

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#



class Settings_Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        tk.Label(self, text="Settings", font=("Helvetica", 16)).pack(pady=10, padx=10 , anchor='w' )

        tk.Button(self, text='Select Files',
                  command= lambda: controller.show_frame(File_Selection_Page)).pack( anchor='w' )
        
        tk.Button(self, text="Back to Start Page",
                  command=lambda: controller.show_frame(StartPage)).pack( anchor='w' )
        



#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#


class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        canvas = tk.Canvas(self, borderwidth=0 , width = 600)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas)  # Use tk.Frame here for background

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")


#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#


class File_Selection_Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.file_vars = []  # List to hold IntVar for each file Checkbutton

        # Header Label
        tk.Label(self, text="File Selection", font=("Helvetica", 16)).pack(pady=10, padx=10, anchor='w')

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

        # Back Button
        tk.Button(self, text="Back to Settings Page", command=lambda: controller.show_frame(Settings_Page)).pack(pady=10, padx=10, anchor='w')   

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
        self.controller.Allowed_Files = selected  # Update the Allowed_Files list
        print("Allowed Files Updated:", self.controller.Allowed_Files)


        


#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#

class Frame_Manager():


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


#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--Data_Directory', required=True, type=str, help='Path to simulation directory')
    args = parser.parse_args()
    app = App( Data_Directory=args.Data_Directory )
    app.mainloop()
