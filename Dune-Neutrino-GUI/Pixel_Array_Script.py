# Import necessary libraries and modules
import os
import tkinter as tk 
import h5py
import argparse
import numpy as np  
import pandas as pd
import matplotlib.cm as cm 
import matplotlib.pyplot as plt  
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)  
import matplotlib.patches as patches  
from matplotlib.cm import ScalarMappable 
from matplotlib import colormaps 


import Main_Run_File 

class Use_Pixel_Array:
    """
    A class to handle pixel-based data visualization and processing.
    """

    def plot(self):
        """
        Plots pixel-based data from an HDF5 file and displays it in a Tkinter GUI.
        """
        # Close the previous figure if it exists to prevent memory leaks
        if hasattr(self, 'fig'):
            plt.close(self.fig)
        
        # Remove all existing widgets from the Figure_Frame to prepare for new plots
        for widget in self.Figure_Frame.winfo_children():
            widget.destroy()

        try:

            path = os.path.join(self.controller.Data_Directory, self.file_selected.get())
        except:
            # Fallback in case of an exception
            path = os.path.join(self.controller.Data_Directory, self.selected_file)


        segments = h5py.File(path, 'r')['segments']

        try:
            # Get the selected event ID from the GUI and convert it to integer
            event = int(self.event_id_selected.get())
        except:
            # Fallback in case of an exception
            event = int(self.event_id_selected)

        # Define the coordinate bounds (rounded values)
        min_x = round(-350)
        max_x = round(350)
        min_y = round(-216.7)
        max_y = round(82.9)
        min_z = round(415.8)
        max_z = round(918.2)
        
        # Define granularity for pixels and coarse bins in centimeters
        # pixel_granularity = 0.4  # Fine granularity (commented out)
        pixel_granularity = 1.5  # Fine granularity
        coarse_granularity = 20.0  # Coarse bin size

        # Generate coordinate ranges for Z, Y, and X axes based on granularity
        Cube_Coords_Xs = np.arange(min_x, max_x, pixel_granularity)
        Cube_Coords_Zs = np.arange(min_z, max_z, pixel_granularity)
        Cube_Coords_Ys = np.arange(min_y, max_y, pixel_granularity)

        # Initialize dictionaries to hold pixel data
        Pixel_Dict = {}

        # Initialize Pixel_Dict with keys as (z, y) tuples and empty lists as values
        for _, cube_z in enumerate(Cube_Coords_Zs):
            for cube_y in Cube_Coords_Ys:
                bottom_left_red = (cube_z, cube_y)
                Pixel_Dict[bottom_left_red] = []

        # Filter segments for the selected event and energy threshold
        segments_event = segments[segments['event_id'] == event]
        segments_event = segments_event[segments_event['dE'] > 2]

        # If a vertex ID is selected, further filter the segments
        if self.vertex_id_selected:
            segments_event = segments_event[segments_event['vertex_id'] == self.vertex_id_selected]

        # Convert the filtered segments to a pandas DataFrame for easier manipulation
        segments_event = pd.DataFrame(segments_event)

        # Extract Z and Y values from Pixel_Dict keys to determine overall range
        z_values = [coord[0] for coord in Pixel_Dict.keys()]
        y_values = [coord[1] for coord in Pixel_Dict.keys()]

        # Generate coarse bin coordinates based on coarse granularity
        coarse_z_coords = np.arange(min_z, max_z, coarse_granularity)
        coarse_y_coords = np.arange(min_y, max_y, coarse_granularity)

        # Update min and max values based on actual data
        min_z, max_z = min(z_values), max(z_values)
        min_y, max_y = min(y_values), max(y_values)

        # Create bin edges for fine granularity
        z_bins = np.arange(min_z, max_z + pixel_granularity, pixel_granularity)
        y_bins = np.arange(min_y, max_y + pixel_granularity, pixel_granularity)

        # Assign each segment to a fine Z-bin and Y-bin using digitize
        segments_event['z_bin'] = np.digitize(segments_event['z'], bins=z_bins) - 1
        segments_event['y_bin'] = np.digitize(segments_event['y'], bins=y_bins) - 1

        # Create bin edges for coarse granularity
        coarse_z_bins = np.arange(min_z, max_z + coarse_granularity, coarse_granularity)
        coarse_y_bins = np.arange(min_y, max_y + coarse_granularity, coarse_granularity)

        # Assign each segment to a coarse Z-bin and Y-bin
        segments_event['coarse_z_bin'] = np.digitize(segments_event['z'], bins=coarse_z_bins) - 1
        segments_event['coarse_y_bin'] = np.digitize(segments_event['y'], bins=coarse_y_bins) - 1

        # Group segments by their fine (z_bin, y_bin) combination
        grouped = segments_event.groupby(['z_bin', 'y_bin']).apply(lambda df: df.index.to_list())

        # Get the total number of pixels for progress tracking
        pixel_count = len(Pixel_Dict.keys())
        
        # Initialize progress value
        self.progress_value = 0  # Ensure progress bar reaches 100%

        # Iterate over each pixel and map segments to pixels based on bin indices
        for i, pixel_bottom_left in enumerate(Pixel_Dict.keys()):
            self.progress_value = (i / pixel_count) * 100  # Update progress percentage

            z, y = pixel_bottom_left
            # Compute which fine bin this pixel corresponds to
            z_bin = int((z - min_z) / pixel_granularity)
            y_bin = int((y - min_y) / pixel_granularity)

            # Retrieve the list of segment indices for the current bin, or empty list if none
            Pixel_Dict[pixel_bottom_left] = grouped.get((z_bin, y_bin), [])

        # Create a new matplotlib figure and axis for plotting
        self.fig, self.ax = plt.subplots()

        # Embed the matplotlib figure into the Tkinter Figure_Frame
        canvas = FigureCanvasTkAgg(self.fig, master=self.Figure_Frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Check if a custom colormap is selected in the GUI
        if hasattr(self, 'cmap_yes_no'):
            if (str(self.cmap_yes_no.get()) == 'Yes' and str(self.cmap_selection_combobox.get()) != ''):
                cmap = colormaps[self.cmap_selection_combobox.get()]  # Get selected colormap
                norm = plt.Normalize(vmin=0, vmax=75)  # Normalize energy values
                sm = ScalarMappable(cmap=cmap, norm=norm)  # Create scalar mappable for colorbar
                sm.set_array([])
                plt.colorbar(sm, ax=self.ax)  # Add colorbar to the plot
        else:
            # Use default 'plasma' colormap if no custom colormap is selected
            cmap = colormaps['plasma']
            norm = plt.Normalize(vmin=0, vmax=75)
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=self.ax)

        # Group segments by their coarse (z_bin, y_bin) combination
        coarse_grouped = segments_event.groupby(['coarse_z_bin', 'coarse_y_bin'])

        # Iterate over each coarse bin to subdivide into fine bins and plot
        for cz_idx, cz in enumerate(coarse_z_coords):
            for cy_idx, cy in enumerate(coarse_y_coords):
                # Check if the current coarse bin has any segments
                if (cz_idx, cy_idx) in coarse_grouped.groups:
                    bin_hits = coarse_grouped.get_group((cz_idx, cy_idx))

                    # Define fine bin coordinates within the current coarse bin
                    fine_z_coords = np.arange(cz, cz + coarse_granularity, pixel_granularity)
                    fine_y_coords = np.arange(cy, cy + coarse_granularity, pixel_granularity)

                    # Create fine bin edges for digitization
                    fine_z_bins = np.arange(cz, cz + coarse_granularity + pixel_granularity, pixel_granularity)
                    fine_y_bins = np.arange(cy, cy + coarse_granularity + pixel_granularity, pixel_granularity)

                    # Create a copy of bin_hits to avoid SettingWithCopyWarning
                    bin_hits = bin_hits.copy()

                    # Assign segments to fine bins within the coarse bin
                    bin_hits['z_bin'] = np.digitize(bin_hits['z'], bins=fine_z_bins) - 1
                    bin_hits['y_bin'] = np.digitize(bin_hits['y'], bins=fine_y_bins) - 1

                    # Group segments by their fine (z_bin, y_bin) combination within the coarse bin
                    grouped_fine = bin_hits.groupby(['z_bin', 'y_bin']).apply(lambda df: df.index.to_list())

                    # Iterate over each fine bin within the current coarse bin
                    for fz_idx, fz in enumerate(fine_z_coords):
                        for fy_idx, fy in enumerate(fine_y_coords):
                            pixel_bottom_left = (fz, fy)  # Coordinates of the current fine bin
                            # Retrieve segment indices for the current fine bin
                            segment_indices = grouped_fine.get((fz_idx, fy_idx), [])
                            Pixel_Dict[pixel_bottom_left] = segment_indices

                            if segment_indices:
                                # Get the segments corresponding to the current fine bin
                                good_ones = bin_hits.loc[segment_indices]
                                total_energy = good_ones['dE'].sum()  # Sum of energy deposits
                                norm_value = norm(total_energy)  # Normalize the total energy
                                color = cmap(norm_value)  # Get color based on normalized energy

                                # Create a rectangle patch representing the current pixel
                                pixel = patches.Rectangle(
                                    pixel_bottom_left,
                                    pixel_granularity,
                                    pixel_granularity,
                                    facecolor=color
                                )
                                self.ax.add_patch(pixel)  # Add the patch to the plot
                else:
                    # If the coarse bin is empty, skip plotting its fine bins
                    continue

        # Set the plot's X and Y axis limits to match the coordinate ranges
        self.ax.set_xlim(min_z, max_z)
        self.ax.set_ylim(min_y, max_y)

        # Render the updated plot on the canvas
        canvas.draw()

        # Add a navigation toolbar to the Figure_Frame for interactive plot features
        toolbar = NavigationToolbar2Tk(canvas, self.Figure_Frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.LEFT, fill=tk.X)

        return  # End of plot method

    def Save_For_ML(self, DF, Save_Path):
        """
        Saves the processed data as an image suitable for machine learning applications.

        Args:
            DF (pd.DataFrame): DataFrame containing segment data.
            Save_Path (str): Path where the image will be saved.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib import cm

        # Define the coordinate bounds (rounded values)
        min_x = round(-350)
        max_x = round(350)
        min_y = round(-216.7)
        max_y = round(82.9)
        min_z = round(415.8)
        max_z = round(918.2)

        # Define granularity for pixels and coarse bins in centimeters
        # pixel_granularity = 0.4  # Fine granularity (commented out)
        pixel_granularity = 1.5  # Fine granularity
        coarse_granularity = 40.0  # Coarse bin size

        # Generate coarse bin coordinates based on coarse granularity
        coarse_z_coords = np.arange(min_z, max_z, coarse_granularity)
        coarse_y_coords = np.arange(min_y, max_y, coarse_granularity)

        # Initialize or reset the pixel_array_dict to store segment indices per pixel
        if not hasattr(self, 'pixel_array_dict'):
            self.pixel_array_dict = {}
        else:
            self.pixel_array_dict = {k: [] for k in self.pixel_array_dict.keys()}

        # Create bin edges for coarse granularity
        coarse_z_bins = np.arange(min_z, max_z + coarse_granularity, coarse_granularity)
        coarse_y_bins = np.arange(min_y, max_y + coarse_granularity, coarse_granularity)

        # Assign each segment to a coarse Z-bin and Y-bin
        DF['coarse_z_bin'] = np.digitize(DF['z'], bins=coarse_z_bins) - 1
        DF['coarse_y_bin'] = np.digitize(DF['y'], bins=coarse_y_bins) - 1

        # Group segments by their coarse (z_bin, y_bin) combination
        coarse_grouped = DF.groupby(['coarse_z_bin', 'coarse_y_bin'])

        dots_per_inch = 400  # Define resolution for the saved figure

        # Prepare the matplotlib figure with specified DPI
        self.fig, self.ax = plt.subplots(dpi=dots_per_inch)
        cmap = cm.plasma  # Choose the 'plasma' colormap
        norm = plt.Normalize(vmin=0, vmax=75)  # Normalize energy values

        # Iterate over each coarse bin to subdivide into fine bins and plot
        for cz_idx, cz in enumerate(coarse_z_coords):
            for cy_idx, cy in enumerate(coarse_y_coords):
                # Check if the current coarse bin has any segments
                if (cz_idx, cy_idx) in coarse_grouped.groups:
                    bin_hits = coarse_grouped.get_group((cz_idx, cy_idx))

                    # Define fine bin coordinates within the current coarse bin
                    fine_z_coords = np.arange(cz, cz + coarse_granularity, pixel_granularity)
                    fine_y_coords = np.arange(cy, cy + coarse_granularity, pixel_granularity)

                    # Create fine bin edges for digitization
                    fine_z_bins = np.arange(cz, cz + coarse_granularity + pixel_granularity, pixel_granularity)
                    fine_y_bins = np.arange(cy, cy + coarse_granularity + pixel_granularity, pixel_granularity)

                    # Create a copy of bin_hits to avoid SettingWithCopyWarning
                    bin_hits = bin_hits.copy()

                    # Assign segments to fine bins within the coarse bin
                    bin_hits['z_bin'] = np.digitize(bin_hits['z'], bins=fine_z_bins) - 1
                    bin_hits['y_bin'] = np.digitize(bin_hits['y'], bins=fine_y_bins) - 1

                    # Group segments by their fine (z_bin, y_bin) combination within the coarse bin
                    grouped_fine = bin_hits.groupby(['z_bin', 'y_bin']).apply(lambda df: df.index.to_list())

                    # Iterate over each fine bin within the current coarse bin
                    for fz_idx, fz in enumerate(fine_z_coords):
                        for fy_idx, fy in enumerate(fine_y_coords):
                            pixel_bottom_left = (fz, fy)  # Coordinates of the current fine bin
                            # Retrieve segment indices for the current fine bin
                            segment_indices = grouped_fine.get((fz_idx, fy_idx), [])
                            self.pixel_array_dict[pixel_bottom_left] = segment_indices

                            if segment_indices:
                                # Get the segments corresponding to the current fine bin
                                good_ones = bin_hits.loc[segment_indices]
                                total_energy = good_ones['dE'].sum()  # Sum of energy deposits
                                norm_value = norm(total_energy)  # Normalize the total energy
                                color = cmap(norm_value)  # Get color based on normalized energy

                                # Create a rectangle patch representing the current pixel
                                pixel = patches.Rectangle(
                                    pixel_bottom_left,
                                    pixel_granularity,
                                    pixel_granularity,
                                    facecolor=color
                                )
                                self.ax.add_patch(pixel)  # Add the patch to the plot
                else:
                    # If the coarse bin is empty, skip plotting its fine bins
                    continue

        # Set the plot's X and Y axis limits to match the coordinate ranges
        self.ax.set_xlim(min_z, max_z)
        self.ax.set_ylim(min_y, max_y)
        self.ax.axis('off')  # Hide the axes for a cleaner image

        # Save the figure as an image file with tight layout and no padding
        plt.savefig(Save_Path, bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the figure to free memory

        return  # End of Save_For_ML method