# import os
# import h5py
# import numpy as np
# import pandas as pd
# import tkinter as tk
# from tkinter import ttk
# from Helpers.Frame_Manager_Script import Frame_Manager
# from Helpers.Scrollable_Frame_Script import ScrollableFrame
# from Backend.Generic_Plot_script import Generic_Plot
from Imports.common_imports import *


class Scatter_Creation_Page(tk.Frame):
    def __init__(self, parent, controller):

        super().__init__(parent)
        self.controller = controller

        self.manager = self.controller.Frame_Manager(self)

        # Locate all the column names within the "segments" dataset and make them CheckButtons that can be pressed
        try:
            path = os.path.join(controller.Data_Directory, controller.File_Names[0])
            with h5py.File(path, 'r') as sim_h5:
                column_names = pd.DataFrame(sim_h5['segments'][()][:1]).columns.to_list()
        except Exception:
            column_names = []

        self.back_frame = tk.Frame(self)
        self.back_frame.pack(anchor='w', pady=5)
        self.Back_Button = tk.Button( self.back_frame, text='Back to Figure Selection', command=lambda: controller.show_frame("Plot_Selection_Page") )
        self.Back_Button.pack(anchor='w', side=tk.LEFT, padx=10)

        # Page Title
        Page_Title_str = tk.StringVar()
        Page_Title = tk.Label( self.back_frame, text="Figure Creation", font=("Helvetica", 16), textvariable=Page_Title_str ).pack(anchor='w', pady=(0, 10), padx=50)

        # Progress Bar and Percentage Frame
        self.progressive_frame = tk.Frame(self)
        self.progressive_frame.pack(anchor='w', padx=10, pady=(0, 20))

        self.progress = ttk.Progressbar( self.progressive_frame, orient="horizontal", length=600, mode="determinate" )
        self.progress_label = tk.Label(  self.progressive_frame, text='', font=("Arial", 12) )
        self.progress.pack(anchor='w', side=tk.LEFT)
        self.progress_label.pack(anchor='w', side=tk.LEFT)

        # Create file selection Frame 
        self.file_select_frame = tk.Frame(self)
        self.file_select_frame.pack(anchor='w', pady=20)

        self.file_selected = tk.StringVar()

        tk.Label(self.file_select_frame, text="file : ").pack(side=tk.LEFT)
        self.file_combobox = ttk.Combobox( self.file_select_frame, textvariable=self.file_selected, values=controller.Allowed_Files, state='readonly', width=60 )
        self.file_combobox.pack(anchor='w', side=tk.LEFT)

        # Bind the dropdown selection to update the "event_id" selection dropdown
        self.file_selected.trace_add('write', self.on_file_selected)

        # Add a button for toggling on and off 3d Plots
        self.Dropdown_3D_frame = tk.Frame(self)
        self.Dropdown_3D_frame.pack(anchor='w', pady=5)

        # Add a Frame to organize dropdowns in one row
        axis_select_frame = tk.Frame(self)
        axis_select_frame.pack(anchor='w', pady=20)

        # Variables to hold selected values for dropdown menus
        self.x_selected = tk.StringVar()
        self.y_selected = tk.StringVar()
        self.z_selected = tk.StringVar()

        tk.Label(axis_select_frame, text='x axis: ').pack(side=tk.LEFT)
        self.x_combobox = ttk.Combobox( axis_select_frame, textvariable=self.x_selected, values=column_names, state='readonly', width=10 )
        self.x_combobox.pack(anchor='w', side=tk.LEFT, padx=6)

        tk.Label(axis_select_frame, text='y axis: ').pack(side=tk.LEFT)
        self.y_combobox = ttk.Combobox(   axis_select_frame, textvariable=self.y_selected, values=column_names, state='readonly',  width=10 )
        self.y_combobox.pack(anchor='w', side=tk.LEFT)

        tk.Label(axis_select_frame, text='z axis: ').pack(side=tk.LEFT)
        self.z_combobox = ttk.Combobox( axis_select_frame, textvariable=self.z_selected,  values=column_names, state='disabled', width=10 )
        self.z_combobox.pack(anchor='w', side=tk.LEFT)

        # Variable to hold the cmap selection
        self.cmap_yes_no = tk.StringVar()
        self.cmap_option_select = tk.StringVar()

        self.colour_map_frame = tk.Frame(self)
        self.colour_map_frame.pack(anchor='w', pady=5)

        tk.Label(self.colour_map_frame, text="cmap:  ").pack(side=tk.LEFT)

        self.cmap_combobox = ttk.Combobox( self.colour_map_frame,  textvariable=self.cmap_yes_no, values=['No', 'Yes'], state='readonly', width=10 )
        self.cmap_combobox.set('No')
        self.cmap_combobox.pack(anchor='w', side=tk.LEFT)

        self.cmap_selection_combobox = ttk.Combobox(  self.colour_map_frame, textvariable=self.cmap_option_select, values=['viridis', 'plasma', 'inferno', 'magma', 'cividis'], width=10, state='disabled' )
        self.cmap_selection_combobox.pack(anchor='w', side=tk.LEFT)
        self.cmap_yes_no.trace_add( 'write', lambda *args: self.Lock_Unlock_Cmap(self.cmap_yes_no, self.cmap_selection_combobox) )


        tk.Label(self.Dropdown_3D_frame, text="3D:").pack(side=tk.LEFT, padx=10)
        self.dropdown_3d_select = tk.StringVar()
        self.dropdown_3d = ttk.Combobox( self.Dropdown_3D_frame, textvariable=self.dropdown_3d_select, values=['No', 'Yes'], width=10 )
        self.dropdown_3d.pack(side=tk.LEFT)
        self.dropdown_3d_select.trace_add( 'write', lambda *args: self.Lock_Unlock_Cmap(self.dropdown_3d_select, self.z_combobox) )

        tk.Label(self.Dropdown_3D_frame, text="pixel array: ").pack(side=tk.LEFT, padx=10)
        self.pixel_array_select = tk.StringVar()
        self.dropdown_pixel = ttk.Combobox( self.Dropdown_3D_frame, textvariable=self.pixel_array_select,values=['No', 'Yes'],  width=10 )
        self.dropdown_pixel.pack(side=tk.LEFT)

        Page_Title_str.set("Figure Creation : Scatter Plot")



        # Add "Create Button" button below which can create a plt fig 
        self.Create_Fig_Button = tk.Button(  self, text='Create', command=lambda: self.Plot_Type_Map() )
        self.Create_Fig_Button.pack(anchor='w', pady=10)

        self.Figure_Frame = tk.Frame(self)
        self.Figure_Frame.pack(anchor='w', side=tk.LEFT, pady=5)

    def Lock_Unlock_Cmap(self, yes_no, selection_combobox, *args):
        # Switch the state of the selection_combobox dropdown
        if yes_no.get() == 'Yes':
            selection_combobox.config(state='readonly')
        elif yes_no.get() == 'No':
            selection_combobox.set('')
            selection_combobox.config(state='disabled')

    def on_file_selected(self, *args):
        """Callback triggered when a new file is selected from the dropdown."""
        path = os.path.join(self.controller.Data_Directory, self.file_selected.get())
        self.event_id_selected = tk.StringVar()
        with h5py.File(path, 'r') as sim_h5:
            unique_event_ids = list(np.unique(sim_h5["segments"]['event_id']))

        if hasattr(self, 'event_combobox') and self.event_combobox:
            self.event_combobox['values'] = unique_event_ids
            self.event_combobox.set('')
        else:
            tk.Label(self.file_select_frame, text="event id :").pack(
                padx=(10, 10), side=tk.LEFT)
            self.event_combobox = ttk.Combobox(
                self.file_select_frame,
                textvariable=self.event_id_selected,
                values=unique_event_ids,
                state='readonly'
            )
            self.event_combobox.pack(anchor='w', side=tk.LEFT)

    def Plot_Type_Map(self, *args):

        if self.pixel_array_select.get() != 'Yes':
            # Generic_Plot.Create_Scatter_Fig(self)

            # print( dir( self.controller ) )
            # self.controller.Create_Scatter_Fig(self)
            self.controller.Generic_Plot.Create_Scatter_Fig(self)
        else:
            # Frame_Manager.setup_process(self)
            self.manager.setup_process()
            # print( dir(self.controller.Frame_Manager.setup_process) )
            # self.controller.Frame_Manager.setup_process( self )


        return

        
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#