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
        # print(f"{frame_class.__name__} has been re-initialized.")

    def __init__(self, Data_Directory=None, input_type='edep', det_complex='2x2'):
        super().__init__()
        self.title("Msci Project App")
        self.geometry("400x300")

        # Import and Shuffle pdg_id_map for cmapping
        from pdg_id_script import pdg_id_map
        # temp_items = list( pdg_id_map.items() )
        # np.random.shuffle(temp_items)
        # pdg_id_map = dict( temp_items )

        # Add destroy and reinitialize methods, because of headaches.
        self.destroy_frame      = self._destroy_frame
        self.reinitialize_frame = self._reinitialize_frame
        
        
        self.Data_Directory     = Data_Directory        # Create initial attributes for accessing the data directory 


        self.pdg_id_map         = pdg_id_map            # Apply the imported pdg map as a attribute

        self.pdg_id_map_reverse = { self.pdg_id_map[i] : i for i in list( pdg_id_map.keys() ) }
        
        self.plot_type          = 'scatter'             # Set initial plot_type state, should help with cleaning up code. 


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

        # Dynamically resize window based on the frame
        if page == File_Selection_Page:
            self.geometry("800x500")  # Larger size for File Selection Page
        elif page == View_Segments_Page:
            self.geometry("1600x500")  # Larger size for View Segments Page


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


class Dataset_View_Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        tk.Label(self, text="View Datasets", font=("Helvetica", 16)).pack(pady=10, padx=10 , anchor='w' )
        tk.Button(self, text="View segments", command=lambda: controller.show_frame(View_Segments_Page)).pack( anchor='w')
        tk.Button(self, text="View mc_hdr", command=lambda: controller.show_frame(View_mc_hdr_Page) , state='disabled').pack( anchor='w')
        tk.Button(self, text="View Trajectories", command=lambda: controller.show_frame(StartPage) , state= 'disabled').pack( anchor='w' )
        tk.Button(self, text="Back to Start Page", command=lambda: controller.show_frame(StartPage)).pack( anchor='w' )


#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#


class View_Segments_Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        self.controller = controller  # Reference to the main app

        # Initialize event selection variables
        self.Event_ID_selection = 0
        self.Event_IDs = []

        # Page Title
        tk.Label(self, text="View Segments", font=("Helvetica", 16)).pack(anchor='w', pady=(10, 10))

        # Navigation Button
        tk.Button(self, text="Go to Start Page", command=lambda: controller.show_frame(StartPage)).pack(anchor='w', pady=(0, 10))

        # Frame for Dropdown Menu
        file_selection_frame = tk.Frame(self)
        file_selection_frame.pack(anchor='w', pady=(0, 10))

        # Dropdown Label
        tk.Label(file_selection_frame, text="Select file:").pack(side=tk.LEFT, padx=(0, 5))

        # Dropdown Menu
        self.selected_file = tk.StringVar()
        self.files_drop_down = tk.OptionMenu(file_selection_frame, self.selected_file, "")
        self.files_drop_down.pack(side=tk.LEFT)

        # Bind the dropdown selection to update the DataFrame display
        self.selected_file.trace('w', self.on_file_selected)

        # Frame for Displaying DataFrame and Navigation Buttons
        self.display_frame = tk.Frame(self)
        self.display_frame.pack(anchor='w', pady=(10, 10))

        # Initialize Dropdown Options
        self.update_dropdown()

        # Navigation Buttons Frame
        navigation_buttons_frame = tk.Frame(self)
        navigation_buttons_frame.pack(anchor='w', pady=(5, 10))

        # Back Button
        self.back_button = tk.Button(navigation_buttons_frame, text="Back", command=self.go_back)
        self.back_button.pack(side=tk.LEFT, padx=5)

        # Next Button
        self.next_button = tk.Button(navigation_buttons_frame, text="Next", command=self.go_next)
        self.next_button.pack(side=tk.LEFT, padx=5)

        # Event Counter Label
        self.event_counter_label = tk.Label(self, text="Event 0 of 0", font=("Helvetica", 10))
        self.event_counter_label.pack(anchor='w', padx=5)

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

    def on_file_selected(self, *args):
        """Callback triggered when a new file is selected from the dropdown."""
        selected_file = self.selected_file.get()
        self.Event_ID_selection = 0  # Reset to first event when a new file is selected
        self.Display_DF_in_Frame(selected_file)

    def refresh_content(self):
        """Refresh dropdown and DataFrame display when the frame is shown."""
        self.update_dropdown()
        self.Display_DF_in_Frame(self.selected_file.get())


    def refresh_frame(self):
        """Clear all widgets in the display_frame."""
        for widget in self.display_frame.winfo_children():
            widget.destroy()

    def Display_DF_in_Frame(self, dropdown_file_name):
        """Display the DataFrame based on the selected file."""
        self.refresh_frame()  # Clear previous content

        # Display the selected file name
        tk.Label(self.display_frame, text=f"Selected File: {dropdown_file_name}", font=("Helvetica", 12)).pack(anchor='w')

        # Construct the file path
        File_path = os.path.join(self.controller.Data_Directory, dropdown_file_name)

        try:
            # Open the HDF5 file and extract data
            with h5py.File(File_path, 'r') as sim_h5:
                segments = sim_h5['segments'][()]
                # segments = pd.DataFrame(segments)
                self.Event_IDs = np.unique(segments['event_id'])

            if not self.Event_IDs.size:
                raise ValueError("No Event IDs found in the selected file.")

            # Ensure Event_ID_selection is within bounds
            self.Event_ID_selection = max(0, min(self.Event_ID_selection, len(self.Event_IDs) - 1))

            # Get the current Event ID
            current_event_id = self.Event_IDs[self.Event_ID_selection]
            segments_event = segments[segments['event_id'] == current_event_id]
            segments_event = pd.DataFrame(segments_event)

            # Update Event Counter Label
            self.event_counter_label.config(text=f"Event {self.Event_ID_selection + 1} of {len(self.Event_IDs)}")

            # Update navigation buttons' state
            self.update_navigation_buttons()

            # Create a Treeview widget to display the DataFrame
            self.tree = ttk.Treeview(self.display_frame, columns=list(segments_event.columns), show="headings")
            # self.tree = ttk.Treeview(self.display_frame, columns=list(column_names), show="headings")

            # Configure Treeview Style
            style = ttk.Style()
            style.configure("Treeview", font=("Helvetica", 7))  # Row font size
            style.configure("Treeview.Heading", font=("Helvetica", 8, "bold"))  # Header font size

            # Define columns and headings
            for col in segments_event.columns:

                self.tree.heading(col, text=col)
                self.tree.column(col, width=50, anchor="center")  # Adjust width as needed

            # Insert DataFrame rows into the Treeview
            for _, row in segments_event.iterrows():
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
            self.Display_DF_in_Frame(self.selected_file.get())

    def go_next(self):
        """Navigate to the next event."""
        if self.Event_ID_selection < len(self.Event_IDs) - 1:
            self.Event_ID_selection += 1
            self.Display_DF_in_Frame(self.selected_file.get())

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


#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#




class View_mc_hdr_Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        tk.Label(self, text="View mc_hdr", font=("Helvetica", 16)).pack(pady=10, padx=10, anchor='w')
        tk.Button(self, text="Back to Start Page", command=lambda: controller.show_frame(StartPage)).pack( anchor='w' )






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

        self.progress = ttk.Progressbar( self.progressive_frame,  orient="horizontal", length=600,  mode="determinate" ) 
        self.progress_label = tk.Label(self.progressive_frame , text = '0%' , font=("Arial", 12))
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
        self.Create_Dataset = tk.Button(self.Interact_Frame , text = 'Create' , command= lambda: self.Create_ML_Dataset() )
        self.Create_Dataset.pack( side = tk.LEFT , anchor='w')

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

        temp_segments = temp_segments[ (temp_segments['event_id'] == unique_ids[0])]
        temp_mc_hdr   = temp_mc_hdr[ (temp_mc_hdr['event_id'] == unique_ids[0])]

        cmap = cm.plasma
        norm = plt.Normalize(vmin=0, vmax=max(temp_segments['dE']))
        # Create Canvas
        # Destroy old widgets in the Preview_Fig_Frame
        if hasattr(self, 'fig'):
            plt.close(self.fig)  # Close the old figure
        for widget in self.Preview_Fig_Frame.winfo_children():
            widget.destroy()

        # dpi and 4mm granularity
        dpi         = 400 
        hit_radius  = 0.4 # cm -> 44mm

        # Create subplot
        self.preview_fig, self.preview_ax = plt.subplots( )

        canvas = FigureCanvasTkAgg( self.preview_fig, master= self.Preview_Fig_Frame)  
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        mc_hdr_unique_vertex_ids = np.unique( temp_mc_hdr['vertex_id'] ).tolist()

        noise_indexes = [                           index 
                         for index , vertex_id in zip( temp_segments.index.to_list() , temp_segments['vertex_id'].to_list() ) 
                            if vertex_id not in  mc_hdr_unique_vertex_ids]
        
        test_group_df = temp_segments[ (temp_segments['vertex_id'] == mc_hdr_unique_vertex_ids[0] )]
        

        radius_in_points = (hit_radius * dpi / 2.54) ** 2  # Convert to points^2

        # Plot the clean vertex
        c = test_group_df['dE']
        # self.preview_ax.scatter( test_group_df['z'] , test_group_df['y'] , c = c ,cmap = cmap , norm= norm , s = radius_in_points )
        self.preview_ax.scatter( test_group_df['z'] , test_group_df['y'] , c = c ,cmap = cmap , norm= norm , s = 7 )

        # Add the noise back in 
        noise_df = temp_segments[ temp_segments.index.isin( noise_indexes )]

        # print(len(mc_hdr_unique_vertex_ids))
        # print(len(np.unique(temp_segments['vertex_id'])) )

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


    def Progression(self , possision , termination , note =''):
        progress = 100 * (possision / float(termination))
        bar = str('~'*(100-(100-int(progress)))+'→' +' ' * (100-int(progress)))
        print( "\r¦{}¦ {:.0f}% {:}".format( bar ,progress , note ),end='')

    def Create_ML_Dataset(self):

        Test_directory = "Test_ML_File"
        os.makedirs(Test_directory, exist_ok=True)

        Directory_Name_Map = {r"$\nu$-$e^{-}$ scattering"  : "Neutrino_Electron_Scattering" , r"$\nu_{e}$-CC" : "Neutrino_Electron_CC" , r"$\nu_{e}$-NC" : "Neutrino_Electron_NC" , "Other" : "Other"}
        
        for Dir_Name in list(Directory_Name_Map.values()) : 
            _  = os.path.join( Test_directory , Dir_Name )
            os.makedirs( _ , exist_ok = True)

        Dir_File_Name_Counter = { file : 0 for file in list(Directory_Name_Map.keys() )}

        path = os.path.join(  self.controller.Data_Directory , self.file_selected.get() )
        sim_h5 = h5py.File(path , 'r')
        temp_segments = sim_h5["segments"]
        temp_segments = temp_segments[ ( temp_segments['dE'] > 1.5 )   ]
        temp_segments =  pd.DataFrame(temp_segments[()])
        temp_mc_hdr =  sim_h5['mc_hdr'] 

        unique_ids = np.unique(temp_segments['event_id']).tolist()

        cmap = cm.plasma

        for i , event_id  in enumerate(unique_ids , start = 1) :
            self.Progression( possision = i , termination= len(unique_ids) , note='test' )

            temp_segments_event = temp_segments[ (temp_segments['event_id'] == event_id)]
            temp_mc_hdr_event   = temp_mc_hdr[ (temp_mc_hdr['event_id'] == event_id)]

            mc_hdr_vertex_ids = np.unique( temp_mc_hdr_event['vertex_id'] ).tolist()
            noise_indexes = [ i for i in  temp_segments_event['vertex_id'] if i not in mc_hdr_vertex_ids]

            for true_vertex in mc_hdr_vertex_ids:

                self.test_fig, self.text_ax = plt.subplots( )

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

                try:

                    # norm = plt.Normalize(vmin=0, vmax=max(temp_segments_event_vertex['dE']))
                    norm = plt.Normalize(vmin=0, vmax=75 )
                    
                    noise_df = temp_segments_event[ temp_segments_event.index.isin(noise_indexes) ]

                    self.text_ax.scatter( temp_segments_event_vertex['z'] , temp_segments_event_vertex['y'] , c = temp_segments_event_vertex['dE'] , cmap = cmap , norm = norm , s = 6 )
                    self.text_ax.scatter( noise_df['z'] , noise_df['y'] , c = noise_df['dE'] , cmap = cmap , norm = norm , s = 6)

                    self.text_ax.set_xlim( self.controller.min_z_for_plot , self.controller.max_z_for_plot  )
                    self.text_ax.set_ylim( self.controller.min_y_for_plot , self.controller.max_y_for_plot  )
                    self.text_ax.axis('off')

                    Dir_File_Name_Counter[ interaction_label ] += 1 
                    loop_path = os.path.join( Test_directory ,Directory_Name_Map[ interaction_label ] )
                    loop_path = os.path.join( loop_path , f"IMG_{Dir_File_Name_Counter[interaction_label]}.png" )

                    plt.savefig( loop_path )
                    plt.close()
                    
                except:
                    continue

                




            pass



        return


    pass
        

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#


class Plot_Selection_Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        tk.Label(self, text="Plot Figures", font=("Helvetica", 16)).pack(pady=10, padx=10, anchor='w')
        
        tk.Button(self, text="Create Scatter Plot",
                    command=lambda: controller.set_plot_type('scatter')).pack(anchor='w', pady=2)

        tk.Button(self, text="Create Line Plot",
                    command=lambda: controller.set_plot_type('line') , state='disabled' ).pack(anchor='w', pady=2)

        tk.Button(self, text="Create Histogram",
                    command=lambda: controller.set_plot_type('hist')).pack(anchor='w', pady=2)

        tk.Button(self, text="Custom Plot",
                    command=lambda: controller.set_plot_type('custom') , state='disabled' ).pack(anchor='w', pady=2)    

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

        # Page Title
        tk.Label(self, text="Figure Creation", font=("Helvetica", 16)).pack(anchor='w', pady=(0, 10))
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


        # ttk.Label( self.file_select_frame , text = 'hello').pack( anchor = 'w' , side = tk.LEFT)

        # Add a Frame to organize dropdowns in one row
        axis_select_frame = tk.Frame(self)
        axis_select_frame.pack(anchor='w', pady=5)
        

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

        if self.controller.plot_type == 'line': 
            self.particle_select = tk.StringVar()

            self.particle_select_frame = tk.Frame(self)
            self.particle_select_frame.pack(anchor='w', pady=5)

            tk.Label(self.particle_select_frame, text="particle: ").pack(side=tk.LEFT)

            self.particle_select_combobox = ttk.Combobox(
                self.particle_select_frame, textvariable=self.particle_select, 
                values= list( self.controller.pdg_id_map.values() ) , width=10)

            self.particle_select_combobox.pack(anchor='w', side=tk.LEFT)
             


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
                # print(self.unique_event_ids)

        if hasattr(self, 'event_combobox') and self.event_combobox:
            # Update the Combobox with new event IDs
            # Set the new values & Reset the dropdown to a blank state
            self.event_combobox['values'] = unique_event_ids  
            self.event_combobox.set('')  

        else:
            # Create a new Combobox widget if one doesn't exist 
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

        # else:
        self.fig, self.ax = plt.subplots()

        canvas = FigureCanvasTkAgg( self.fig, master= self.Figure_Frame)  
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        if str(self.cmap_yes_no.get()) == 'Yes' and str(self.cmap_selection_combobox.get()) != '' :
            cmap = colormaps[ self.cmap_selection_combobox.get() ]
            # cmap = cm['plasma']

            norm = plt.Normalize(vmin=0, vmax=max(temp_df['dE']))
            c = temp_df['dE']
        else:
            cmap = None
            norm = None
            c    = None

        self.ax.scatter( temp_df[ self.x_combobox.get() ] , temp_df[ self.y_combobox.get() ] , c = c ,cmap = cmap , norm= norm)
        self.ax.set_xlabel( self.x_combobox.get() )
        self.ax.set_ylabel( self.y_combobox.get() )
        canvas.draw()

        toolbar = NavigationToolbar2Tk(canvas, self.Figure_Frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.LEFT, fill=tk.X)

        self.controller.geometry("900x900")

        return self

    #All of the Hist Plots should be handled here ... in a ideal world. 
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

            # plt.hist( hist_list, bins = int(100), stacked=True, color = [ p_cmap( Particle_cmap[ str(pdg_id_map_inverse[str(id)]) ] ) for id in  np.unique( list(particle_distance_dict[Interaction].keys()) )  ] , label = hist_list_particles  )
            
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

    def Create_Line_PLot(self , *args):
        print( self.particle_select.get()  )
        path = os.path.join(  self.controller.Data_Directory , self.file_selected.get() )

        with h5py.File(path , 'r') as sim_h5:
            temp_df =  pd.DataFrame(sim_h5["segments"][()])
            temp_df_particle = temp_df[ ( temp_df['dE'] > 2 ) & ( temp_df['pdg_id'] == int(self.controller.pdg_id_map_reverse[ self.particle_select.get() ] )) ]
            temp_df          = temp_df[ ( temp_df['dE'] > 2 ) ]  

        unique_event_ids = np.unique( temp_df['event_id'] ).tolist()
        # Create Canvas
        # Destroy old widgets in the Figure_Frame
        if hasattr(self, 'fig'):
            plt.close(self.fig)  # Close the old figure
        for widget in self.Figure_Frame.winfo_children():
            widget.destroy()

        # else:

        # particle_index_holder  = []
        pdf_name = f'Test_Track_Analysis_{ self.particle_select.get() }.pdf'
        norm = plt.Normalize(vmin=0, vmax=75 )
        cmap = cm.plasma


        with PdfPages(pdf_name) as output:

            for location , event in enumerate(unique_event_ids , start=1):
                self.Progression( possision= location , termination=len(unique_event_ids))
                temp_df_event = temp_df[ temp_df['event_id'] == event ]
                temp_df_particle_event = temp_df_particle[temp_df_particle['event_id'] == event ]

                for vetex_name , vertex_df in temp_df_particle_event.groupby('vertex_id'):

                    for track_name, vertex_df_track in vertex_df.groupby('traj_id'):
                        
                        if vertex_df_track.shape[0] < 3 :
                            continue

                        # Create figure with custom gridspec
                        fig = plt.figure(figsize=(10, 10))  # Adjusted for better spacing
                        gs = fig.add_gridspec(3, 2)  # Define a grid with 3 rows and 2 columns


                        # Left plot (dE vs index)
                        ax_left = fig.add_subplot(gs[0, 0])
                        ax_left.plot(np.arange(vertex_df_track['dE'].shape[0]), vertex_df_track['dE'], marker='.', alpha=0.2)
                        scatter_left = ax_left.scatter( np.arange(vertex_df_track['dE'].shape[0]),  vertex_df_track['dE'],  c=vertex_df_track['dE'],  cmap=cmap , norm = norm )
                        fig.colorbar(scatter_left, ax=ax_left, shrink=0.5, aspect=10)
                        ax_left.set_ylim(bottom=0)
                        ax_left.set_xlabel('Index')
                        ax_left.set_ylabel('dE [MeV]')

                        # Right plot (dEdx vs index)
                        ax_right = fig.add_subplot(gs[0, 1])
                        ax_right.plot(np.arange(vertex_df_track['dEdx'].shape[0]), vertex_df_track['dEdx'], marker='.')
                        ax_right.set_xlabel('Index')
                        ax_right.set_ylabel('dEdx  [MeV/cm]')

                        # 3D scatter plot (x, y, z)
                        ax_bottom_left = fig.add_subplot(gs[1 , 0], projection='3d')  # Span rows 2 and 3
                        ax_bottom_left.scatter( vertex_df_track['z'],  vertex_df_track['y'],  vertex_df_track['x'] ,c = vertex_df_track['dE'] , cmap=cmap , norm = norm , s = 7)

                        ax_bottom_right = fig.add_subplot(gs[1 , 1], projection='3d')  # Span rows 2 and 3
                        ax_bottom_right.scatter( temp_df_event['z'],  temp_df_event['y'], temp_df_event['x'], c = temp_df_event['dE'] , cmap=cmap , norm = norm , s = 7 )

                        for ax_i in [ax_bottom_left , ax_bottom_right ]:

                            ax_i.set_xlabel('Z')
                            ax_i.set_ylabel('Y')
                            ax_i.set_zlabel('X')

                            ax_i.set_xlim( self.controller.min_z_for_plot , self.controller.max_z_for_plot )
                            ax_i.set_ylim( self.controller.min_y_for_plot , self.controller.max_y_for_plot )
                            ax_i.set_zlim( self.controller.min_x_for_plot , self.controller.max_x_for_plot )

                        fig.suptitle(f" file  : { self.file_selected.get() } -- event_id : {event} -- vertex_id : {vetex_name}-- traj_id : {track_name} -- particle : { self.particle_select.get() }" , fontsize = 6)
                        output.savefig(  )


                    plt.close()

                if location == 5:
                    break
                
        print('\n\n Complete')

        return 


        

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#

# class Figure_Creation_Page(tk.Frame):
class Custom_Figure_Page(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent)

        # Page Title
        tk.Label(self, text="Custom Figure", font=("Helvetica", 16)).pack(anchor='w', pady=(0, 10))
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

        # Select custom plot 

        self.custom_fig_select_frame = tk.Frame(self)
        self.custom_fig_select_frame.pack(anchor='w', pady=20)

        self.custom_fig_seleceted = tk.StringVar()

        tk.Label(self.custom_fig_select_frame , text="Custom Figure: ").pack(side=tk.LEFT)

        self.custom_combobox = ttk.Combobox(
            self.custom_fig_select_frame , textvariable = self.custom_fig_seleceted , values = ['Track_dE_Analysis'] , state='readonly' , width=20
        )
        self.custom_combobox.pack(anchor='w' , side = tk.LEFT)


        # Create file selection Frame 

        self.file_select_frame = tk.Frame(self)
        self.file_select_frame.pack(anchor='w', pady=20)

        self.file_selected = tk.StringVar()

        tk.Label(self.file_select_frame , text="file : ").pack(side=tk.LEFT)
        self.file_combobox = ttk.Combobox(
            self.file_select_frame , textvariable = self.file_selected , values = controller.Allowed_Files , state='readonly' , width=60
        )
        self.file_combobox.pack(anchor='w' , side = tk.LEFT)


        self.plot_button_frame = tk.Frame(self)
        self.plot_button_frame.pack(anchor='w', pady = 5)
        self.Plot_Button = tk.Button(
                                self.plot_button_frame, text='Plot' , command= lambda : self.Custom_Selection()
                                        )
        self.Plot_Button.pack(anchor='w', side=tk.LEFT)


    def Custom_Selection(self):
        if str(self.custom_fig_seleceted.get()) == 'Track_dE_Analysis':

            from Custom_Plot_script import Track_dE_Analysis 

            Track_dE_Analysis(self)
            

            print("yess")
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
        
        # var = tk.BooleanVar()
        # tk.Checkbutton(self, text="I agree", variable=var).pack( anchor='w' )


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
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--Data_Directory', required=True, type=str, help='Path to simulation file')
    args = parser.parse_args()
    app = App( Data_Directory=args.Data_Directory )
    app.mainloop()
