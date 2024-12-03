import os
import tkinter as tk
import h5py
import argparse
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg , NavigationToolbar2Tk)

from  Main_Run_File import *


class Custom_Plot():

    """ Add Custom plots within this Custom_Plot class.

    Make sure the function has the 'self' and 'args' arguments

    for plotting :
        'self.controller.Data_Directory' contains the directory that was passed through when the code was ran, 
        'self.file_selected.get()' will get the file name that is selected by the user. 

        'args' is a dictionary containing the keys 'Plot_Mode' , 'canvas' and 'fig'

        'fig', is the normal fig you have when using matplotlib.pylplot.
        'canvas', is what you need to draw to to make it show up in the app window

    This should be all you to create plots :) 

      """

    def Track_dE_Analysis(self , args):


        if args['Plot_Mode'] == 'Single_Plot':

            fig ,  canvas = args['fig']  , args['canvas']    

            path = os.path.join(  self.controller.Data_Directory , self.file_selected.get() )


            sim_h5 = h5py.File(path , 'r')['segments']


            segments_selected = pd.DataFrame( sim_h5[ (  sim_h5['event_id'] ==  int(self.event_id_selected.get()) ) &  (  sim_h5['vertex_id'] ==  int(self.vertex_id_selected.get()) )  ] ) 

            segments_selected = segments_selected[ segments_selected['dE'] > 2] 

            gs = fig.add_gridspec(3, 2)  # Define a grid with 3 rows and 2 columns

            track = np.unique( segments_selected['traj_id'] )[0]

            track_selected = segments_selected[ segments_selected['traj_id'] == track ]

            norm = plt.Normalize(vmin=0, vmax=max(segments_selected['dE']))


            # Left plot (dE vs index)

            ax_left = fig.add_subplot(gs[0, 0])
            ax_left.plot(np.arange(track_selected['dE'].shape[0]), track_selected['dE'], marker='.', alpha=0.2)
            scatter_left = ax_left.scatter( np.arange(track_selected['dE'].shape[0]),  track_selected['dE'],  c=track_selected['dE'],  cmap= cm.plasma , norm = norm )
            fig.colorbar(scatter_left, ax=ax_left, shrink=0.5, aspect=10)
            ax_left.set_ylim(bottom=0)
            ax_left.set_xlabel('Index')
            ax_left.set_ylabel('dE [MeV]')



            # Right plot (dEdx vs index)
            ax_right = fig.add_subplot(gs[0, 1])
            ax_right.plot(np.arange(track_selected['dEdx'].shape[0]), track_selected['dEdx'], marker='.')
            ax_right.set_xlabel('Index')
            ax_right.set_ylabel('dEdx  [MeV/cm]')

            # 3D scatter plot (x, y, z)
            ax_bottom_left = fig.add_subplot(gs[1 , 0], projection='3d')  # Span rows 2 and 3
            ax_bottom_left.scatter( track_selected['z'],  track_selected['y'],  track_selected['x'] ,c = track_selected['dE'] , cmap= cm.plasma , norm = norm , s = 7)

            ax_bottom_right = fig.add_subplot(gs[1 , 1], projection='3d')  # Span rows 2 and 3
            ax_bottom_right.scatter( segments_selected['z'],  segments_selected['y'], segments_selected['x'], c = segments_selected['dE'] , cmap=cm.plasma , norm = norm , s = 7 )

            for ax_i in [ax_bottom_left , ax_bottom_right ]:

                ax_i.set_xlabel('Z')
                ax_i.set_ylabel('Y')
                ax_i.set_zlabel('X')

                ax_i.set_xlim( self.controller.min_z_for_plot , self.controller.max_z_for_plot )
                ax_i.set_ylim( self.controller.min_y_for_plot , self.controller.max_y_for_plot )
                ax_i.set_zlim( self.controller.min_x_for_plot , self.controller.max_x_for_plot )

            particle_name  = self.controller.pdg_id_map[ str(np.unique(track_selected['pdg_id'] )[0]) ]
            fig.suptitle(f" file  : { self.file_selected.get() } -- event_id : {self.event_id_selected.get()} -- vertex_id : {self.vertex_id_selected.get()}-- traj_id : {track} -- particle : { particle_name }" , fontsize = 6)


            canvas.draw()

            toolbar = NavigationToolbar2Tk(canvas, self.Custom_Figure_Frame, pack_toolbar=False)
            toolbar.update()
            toolbar.pack(side=tk.LEFT, fill=tk.X)




            fig.tight_layout()
            
            if hasattr(self , 'next_previous_frame') == False:
                self.next_previous_frame = tk.Frame(self)
                self.next_previous_frame.pack( anchor = 's' , side= tk.BOTTOM)
                previous_button = tk.Button( self.next_previous_frame , text="previous")
                previous_button.pack()
                next_button = tk.Button( self.next_previous_frame , text="next")
                next_button.pack()

            return


        elif args['Plot_Mode'] == 'Download_Dir':
    
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
            for widget in self.Custom_Figure_Frame.winfo_children():
                widget.destroy()


            pdf_name = f'Test_Track_Analysis_{ self.particle_select.get() }.pdf'
            norm = plt.Normalize(vmin=0, vmax=75 )
            cmap = cm.plasma


            with PdfPages(pdf_name) as output:

                for location , event in enumerate(unique_event_ids , start=1):

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
