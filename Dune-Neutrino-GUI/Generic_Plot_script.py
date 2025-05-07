import os
import tkinter as tk
import h5py
import argparse
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_tkagg import ( FigureCanvasTkAgg,  NavigationToolbar2Tk )

# Import everything from Main_Run_File File ( might delete this import later)
from Main_Run_File import *

class Generic_Plot:


    def Create_Scatter_Fig(self, *args):

        path = os.path.join(self.controller.Data_Directory, self.file_selected.get())
        
        if self.pixel_array_select.get() != 'Yes':
            with h5py.File(path, 'r') as sim_h5:
                temp_df = pd.DataFrame(sim_h5["segments"][()])
                # Filter the DataFrame for dE > 1.5 and the selected event_id
                temp_df = temp_df[
                    (temp_df['dE'] > 1.5) & 
                    (temp_df['event_id'] == int(self.event_combobox.get()))
                ]

            # Close the previous figure if it exists
            if hasattr(self, 'fig'):
                plt.close(self.fig)
            
            # Remove all existing widgets from the Figure_Frame
            for widget in self.Figure_Frame.winfo_children():
                widget.destroy()

            # Determine whether to create a 3D scatter plot based on user selection
            if self.dropdown_3d_select.get() == 'Yes':
                self.fig, self.ax = plt.subplots(subplot_kw=dict(projection='3d'))
            else:
                self.fig, self.ax = plt.subplots()


            canvas = FigureCanvasTkAgg(self.fig, master=self.Figure_Frame)
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            # Determine if a colormap should be applied
            if (str(self.cmap_yes_no.get()) == 'Yes' and  str(self.cmap_selection_combobox.get()) != ''):
                cmap = colormaps[self.cmap_selection_combobox.get()]
                norm = plt.Normalize(vmin=0, vmax=75)
                c = temp_df['dE']
                
            else:
                cmap = None
                norm = None
                c = None

            # Create the scatter plot (3D or 2D)
            if self.dropdown_3d_select.get() == 'Yes':
                scatter = self.ax.scatter( temp_df[self.x_combobox.get()], temp_df[self.y_combobox.get()], temp_df[self.z_combobox.get()],  c=c, cmap=cmap,  norm=norm  )
                self.ax.set_zlabel(self.z_combobox.get())

                self.ax.set_xlim( (self.controller.min_z_for_plot ,self.controller.max_z_for_plot) )
                self.ax.set_ylim( (self.controller.min_y_for_plot ,self.controller.max_y_for_plot) )
                self.ax.set_zlim( (self.controller.min_x_for_plot ,self.controller.max_x_for_plot) )

            else:
                scatter = self.ax.scatter( temp_df[self.x_combobox.get()],  temp_df[self.y_combobox.get()],  c=c, cmap=cmap,  norm=norm  )


            self.ax.set_xlabel(self.x_combobox.get())
            self.ax.set_ylabel(self.y_combobox.get())

            # Render the canvas
            canvas.draw()


            if cmap is not None:
                cbar = self.fig.colorbar(scatter, ax=self.ax, shrink=0.5, aspect=10)
                cbar.set_label('dE [MeV]')


            toolbar = NavigationToolbar2Tk(canvas, self.Figure_Frame, pack_toolbar=False)
            toolbar.update()
            toolbar.pack(side=tk.LEFT, fill=tk.X)


    def Create_Line_PLot(self, *args):

        path = os.path.join(self.controller.Data_Directory, self.file_selected.get())
        

        with h5py.File(path, 'r') as sim_h5:
            temp_df = pd.DataFrame(sim_h5["segments"][()])

            temp_df = temp_df[ (temp_df['dE'] > 1.5) &   (temp_df['event_id'] == int(self.event_combobox.get())) ]


        if hasattr(self, 'fig'):
            plt.close(self.fig)
        

        for widget in self.Figure_Frame.winfo_children():
            widget.destroy()

        # Create a new matplotlib figure and axes
        self.fig, self.ax = plt.subplots()

        # Create a matplotlib canvas and add it to the Figure_Frame
        canvas = FigureCanvasTkAgg(self.fig, master=self.Figure_Frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Plot the selected data as a line plot
        self.ax.plot( temp_df[self.x_combobox.get()],  temp_df[self.y_combobox.get()] )

        # Set axis labels based on user selections
        self.ax.set_xlabel(self.x_combobox.get())
        self.ax.set_ylabel(self.y_combobox.get())


        canvas.draw()

        toolbar = NavigationToolbar2Tk(canvas, self.Figure_Frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.LEFT, fill=tk.X)


    def Create_Hist_Fig(self, *args):

        # Construct the full path to the selected HDF5 file
        path = os.path.join(self.controller.Data_Directory, self.file_selected.get())
        
        # Open the HDF5 file and load the "segments" dataset into a pandas DataFrame
        with h5py.File(path, 'r') as sim_h5:
            temp_df = pd.DataFrame(sim_h5["segments"][()])
            # Filter the DataFrame for dE > 1.5 and the selected event_id
            temp_df = temp_df[
                (temp_df['dE'] > 1.5) & 
                (temp_df['event_id'] == int(self.event_combobox.get()))
            ]

        temp_df['pdg_group'] = temp_df['pdg_id'].apply(lambda x: 'Nuclei' if x > 10000 else x)


        if hasattr(self, 'fig'):
            plt.close(self.fig)
        

        for widget in self.Figure_Frame.winfo_children():
            widget.destroy()


        self.fig, self.ax = plt.subplots()
        

        canvas = FigureCanvasTkAgg(self.fig, master=self.Figure_Frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


        if (str(self.group_yes_no.get()) == 'Yes' and 
            str(self.hist_option_select.get()) != ''):
            hist_list = []
            name_list = []


            if self.hist_option_select.get() != 'pdg_id':
                for group_name, group_df in temp_df.groupby(self.hist_option_select.get()):
                    hist_list.append(group_df[self.x_combobox.get()].to_list())


                    if self.hist_option_select.get() == 'pdg_id':
                        name_list.append(self.controller.pdg_id_map[str(group_name)])
                    else:
                        name_list.append(group_name)

            else:

                for group_name, group_df in temp_df.groupby('pdg_group'):
                    hist_list.append(group_df[self.x_combobox.get()].to_list())

                    # # Map group names if grouping by 'pdg_id', else use the group name directly
                    # if self.hist_option_select.get() == 'pdg_id':
                    #     name_list.append(self.controller.pdg_id_map[str(group_name)])
                    # else:
                    #     name_list.append(group_name)

                    # If the group is 'Nuclei', use that label; otherwise look up the pdg id in the dictionary
                    if group_name == 'Nuclei':
                        label = 'Nuclei'
                    else:
                        label = self.controller.pdg_id_map.get(str(group_name), 'Other')

                    name_list.append(label)


            # Plot the histogram with grouped data
            self.ax.hist( hist_list, bins=100, stacked=True, label=name_list )
            # Set the title to indicate the grouping criterion
            self.ax.set_title(f"Grouped By: {self.hist_option_select.get()}")
            # Add a legend with appropriate font size and location
            plt.legend(fontsize=7, loc="upper right")
        else:
            # Plot a simple histogram without grouping
            self.ax.hist(temp_df[self.x_combobox.get()], bins=100)

        # Set axis labels based on user selections
        if self.x_combobox.get() == 'dE':
            self.ax.set_xlabel('dE [MeV]')

        else:
            self.ax.set_xlabel(self.x_combobox.get())


        self.ax.set_ylabel('Frequency')

        # Add a navigation toolbar to the Figure_Frame
        toolbar = NavigationToolbar2Tk(canvas, self.Figure_Frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.LEFT, fill=tk.X)

        return


    # This is a old function that might be deleted but keeping for now
    def Progression(self, position, termination, note=''):

        # Calculate the progress percentage
        progress = 100 * (position / float(termination))
        # Create a visual representation of the progress bar
        bar = '~' * (100 - (100 - int(progress))) + '→' + ' ' * (100 - int(progress))
        # Print the progress bar with carriage return to update in place
        print(f"\r¦{bar}¦ {progress:.0f}% {note}", end='')
