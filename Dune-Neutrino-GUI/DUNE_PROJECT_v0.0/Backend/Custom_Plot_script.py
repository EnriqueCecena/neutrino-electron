
from Imports.common_imports import *



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


            ax_left = fig.add_subplot(gs[0, 0])
            ax_left.plot(np.arange(track_selected['dE'].shape[0]), track_selected['dE'], marker='.', alpha=0.2)
            scatter_left = ax_left.scatter( np.arange(track_selected['dE'].shape[0]),  track_selected['dE'],  c=track_selected['dE'],  cmap= cm.plasma , norm = norm )
            fig.colorbar(scatter_left, ax=ax_left, shrink=0.5, aspect=10)
            ax_left.set_ylim(bottom=0)
            ax_left.set_xlabel('Index')
            ax_left.set_ylabel('dE [MeV]')


            ax_right = fig.add_subplot(gs[0, 1])
            ax_right.plot(np.arange(track_selected['dEdx'].shape[0]), track_selected['dEdx'], marker='.')
            ax_right.set_xlabel('Index')
            ax_right.set_ylabel('dEdx  [MeV/cm]')


            ax_bottom_left = fig.add_subplot(gs[1 , 0], projection='3d')  
            ax_bottom_left.scatter( track_selected['z'],  track_selected['y'],  track_selected['x'] ,c = track_selected['dE'] , cmap= cm.plasma , norm = norm , s = 7)

            ax_bottom_right = fig.add_subplot(gs[1 , 1], projection='3d')
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

            if hasattr(self, 'fig'):
                plt.close(self.fig)  
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
                            ax_bottom_left = fig.add_subplot(gs[1 , 0], projection='3d') 
                            ax_bottom_left.scatter( vertex_df_track['z'],  vertex_df_track['y'],  vertex_df_track['x'] ,c = vertex_df_track['dE'] , cmap=cmap , norm = norm , s = 7)

                            ax_bottom_right = fig.add_subplot(gs[1 , 1], projection='3d') 
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

            print('\n\n Complete')
        return 
    

    def Track_dE_Analysis_Thesis(self, args):

        if args['Plot_Mode'] == 'Single_Plot':

            fig, canvas = args['fig'], args['canvas']

            path = os.path.join(self.controller.Data_Directory, self.file_selected.get())
            sim_h5 = h5py.File(path, 'r')['segments']

            particle_selected = self.particle_select.get()
            particle_name = particle_selected
            particle_pdg_id = int(self.controller.pdg_id_map_reverse[particle_name])

            segments_selected = pd.DataFrame(
                sim_h5[
                    (sim_h5['event_id'] == int(self.event_id_selected.get())) &
                    (sim_h5['vertex_id'] == int(self.vertex_id_selected.get()))
                ]
            )
            segments_selected = segments_selected[segments_selected['dE'] > 2]

            # Create a 3x2 grid
            gs = fig.add_gridspec(3, 2)

            # Select the desired track
            desired_track = segments_selected[segments_selected['pdg_id'] == particle_pdg_id]
            track = np.random.choice(np.unique(desired_track['traj_id']))
            track_selected = segments_selected[segments_selected['traj_id'] == track]

            norm = plt.Normalize(vmin=0, vmax=max(segments_selected['dE']))
            if particle_name == 'Muon-':
                particle_name = r'$μ^{-}$'

            # Top-left: dE vs index
            ax_left = fig.add_subplot(gs[0, 0])
            scatter_left = ax_left.scatter( np.arange(track_selected['dE'].shape[0]), track_selected['dE'], c=track_selected['dE'], cmap=cm.plasma, norm=norm )
            fig.colorbar(scatter_left, ax=ax_left, shrink=0.5, aspect=10)
            ax_left.set_ylim(bottom=0)
            ax_left.set_xlabel('Index')
            ax_left.set_ylabel('dE [MeV]')
            ax_left.set_title(f'dE Along {particle_name} track')

            # Top-right: dEdx vs index
            ax_right = fig.add_subplot(gs[0, 1])
            ax_right.scatter( np.arange(track_selected['dEdx'].shape[0]), track_selected['dEdx'], marker='.' )
            ax_right.set_xlabel('Index')
            ax_right.set_ylabel('dEdx [MeV/cm]')
            ax_right.set_title(f'dEdx Along {particle_name} track')

            # Bottom-left: isolated track, spanning two rows for extra size
            ax_bottom_left = fig.add_subplot(gs[1:, 0], projection='3d')
            ax_bottom_left.scatter( track_selected['z'], track_selected['y'], track_selected['x'], c=track_selected['dE'], cmap=cm.plasma, norm=norm, s=7 )
            ax_bottom_left.set_title(f'Isolated {particle_name} track')

            # Bottom-right: whole vertex, spanning two rows and custom colors
            ax_bottom_right = fig.add_subplot(gs[1:, 1], projection='3d')
            other_segments = segments_selected[segments_selected['traj_id'] != track]
            # Plot other hits in blue
            ax_bottom_right.scatter(  other_segments['z'], other_segments['y'], other_segments['x'], color='blue', alpha=0.5, s=5, label='other' )
            # Plot desired track hits in orange
            ax_bottom_right.scatter( track_selected['z'], track_selected['y'], track_selected['x'], color='orange', s=7, label=f'{particle_name}' )
            ax_bottom_right.legend()
            ax_bottom_right.set_title('Whole Interaction Vertex')

            # Label and limit settings
            for ax_i in [ax_bottom_left, ax_bottom_right]:
                ax_i.set_xlabel('Z')
                ax_i.set_ylabel('Y')
                ax_i.set_zlabel('X')
                ax_i.set_xlim(self.controller.min_z_for_plot, self.controller.max_z_for_plot)
                ax_i.set_ylim(self.controller.min_y_for_plot, self.controller.max_y_for_plot)
                ax_i.set_zlim(self.controller.min_x_for_plot, self.controller.max_x_for_plot)

            canvas.draw()
            toolbar = NavigationToolbar2Tk(canvas, self.Custom_Figure_Frame, pack_toolbar=False)
            toolbar.update()
            toolbar.pack(side=tk.LEFT, fill=tk.X)

            fig.tight_layout()

            if not hasattr(self, 'next_previous_frame'):
                self.next_previous_frame = tk.Frame(self)
                self.next_previous_frame.pack(anchor='s', side=tk.BOTTOM)
                previous_button = tk.Button(self.next_previous_frame, text="previous")
                previous_button.pack()
                next_button = tk.Button(self.next_previous_frame, text="next")
                next_button.pack()

            return



    def Specific_Vertex(self, args):

        if args['Plot_Mode'] == 'Single_Plot':

            if self.particle_select.get() != '':

                fig, canvas = args['fig'], args['canvas']

                path = os.path.join(self.controller.Data_Directory, self.file_selected.get())
                sim_h5 = h5py.File(path, 'r')['segments']

                mc_hdr = h5py.File(path, 'r')['mc_hdr']


                mc_hdr_selected =    mc_hdr[ (mc_hdr['event_id'] == int(self.event_id_selected.get())) & (mc_hdr['vertex_id'] == int(self.vertex_id_selected.get()) ) ]

                reaction = mc_hdr_selected['reaction']
                nu_pdg   = mc_hdr_selected['nu_pdg']
                isCC     = mc_hdr_selected['isCC']

                if reaction == 7:
                    interaction_label = r"$\nu$-$e^{-}$ scattering"

                    
                elif nu_pdg == 12 and isCC:
                    interaction_label = r"$\nu_{e}$-CC"


                elif nu_pdg == 12 and not isCC:
                    interaction_label = r"$\nu_{e}$-NC"

                else:
                    interaction_label = r"Other"



                segments_selected = pd.DataFrame(
                    sim_h5[ (sim_h5['event_id'] == int(self.event_id_selected.get())) & (sim_h5['vertex_id'] == int(self.vertex_id_selected.get()) ) ] )
                segments_selected = segments_selected[segments_selected['dE'] > 1.5]

                ax = fig.add_subplot()

                cmap = colormaps['plasma']
                norm = plt.Normalize(vmin=0, vmax= max( segments_selected['dE'] ))
                c = segments_selected['dE']

                ax_scatter = ax.scatter( segments_selected['z'] , segments_selected['y'] , c = c , cmap = cmap , norm = norm   )

                ax.set_xlabel('Z')
                ax.set_ylabel('y')

                cbar = fig.colorbar(ax_scatter, ax= ax, shrink=0.5, aspect=10)
                cbar.set_label('dE [MeV]')

                ax.set_title(rf'Event_ID = {self.event_id_selected.get()} | Vertex = {self.vertex_id_selected.get()} | Interaction = {interaction_label}')



                canvas.draw()
                toolbar = NavigationToolbar2Tk(canvas, self.Custom_Figure_Frame, pack_toolbar=False)
                toolbar.update()
                toolbar.pack(side=tk.LEFT, fill=tk.X)

                fig.tight_layout()

                if not hasattr(self, 'next_previous_frame'):
                    self.next_previous_frame = tk.Frame(self)
                    self.next_previous_frame.pack(anchor='s', side=tk.BOTTOM)
                    previous_button = tk.Button(self.next_previous_frame, text="previous")
                    previous_button.pack()
                    next_button = tk.Button(self.next_previous_frame, text="next")
                    next_button.pack()

                return
            

            else:
                path = os.path.join(self.controller.Data_Directory, self.file_selected.get())
                
                fig, canvas = args['fig'], args['canvas']

                ax = fig.add_subplot()


                sim_h5 = h5py.File(path, 'r')['segments']

                DF = pd.DataFrame(
                    sim_h5[ (sim_h5['event_id'] == int(self.event_id_selected.get())) & (sim_h5['vertex_id'] == int(self.vertex_id_selected.get()) ) ] )
                

                min_x = round(-350)
                max_x = round(350)
                min_y = round(-216.7)
                max_y = round(82.9)
                min_z = round(415.8)
                max_z = round(918.2)

                # Define granularity for pixels and coarse bins in centimeters
                # pixel_granularity = 0.4  # Fine granularity 
                pixel_granularity = 1.5  # Fine granularity

                coarse_granularity_z = 40.0  # Coarse bin size
                coarse_granularity_y = 40.0  # Coarse bin size


                # Generate coarse bin coordinates based on coarse granularity
                coarse_z_coords = np.arange(min_z, max_z, coarse_granularity_z)
                coarse_y_coords = np.arange(min_y, max_y, coarse_granularity_y)


                # Initialize or reset the pixel_array_dict to store segment indices per pixel
                if not hasattr(self, 'pixel_array_dict'):
                    self.pixel_array_dict = {}
                else:
                    self.pixel_array_dict = {k: [] for k in self.pixel_array_dict.keys()}

                # Create bin edges for coarse granularity
                coarse_z_bins = np.arange(min_z, max_z + coarse_granularity_z, coarse_granularity_z)
                coarse_y_bins = np.arange(min_y, max_y + coarse_granularity_y, coarse_granularity_y)


                # Assign each segment to a coarse Z-bin and Y-bin
                DF['coarse_z_bin'] = np.digitize(DF['z'], bins=coarse_z_bins) - 1
                DF['coarse_y_bin'] = np.digitize(DF['y'], bins=coarse_y_bins) - 1

                # Group segments by their coarse (z_bin, y_bin) combination
                coarse_grouped = DF.groupby(['coarse_z_bin', 'coarse_y_bin'])



                dots_per_inch = 1

                z_range = max_z - min_z
                y_range = max_y - min_y

                n_z = int(z_range / pixel_granularity)
                n_y = int(y_range / pixel_granularity)


                width_in_inches = n_z / dots_per_inch
                height_in_inches = n_y / dots_per_inch

                cmap = cm.plasma  # Choose the 'plasma' colormap
                norm = plt.Normalize(vmin=0, vmax=75)  # Normalize energy values


                # Compute background color correctly
                background_color = cmap(norm(0))  # Get color from colormap

                # self.fig, self.ax = plt.subplots( figsize=(width_in_inches, height_in_inches) , dpi=dots_per_inch )


                # Iterate over each coarse bin to subdivide into fine bins and plot
                for cz_idx, cz in enumerate(coarse_z_coords):
                    for cy_idx, cy in enumerate(coarse_y_coords):
                        # Check if the current coarse bin has any segments
                        if (cz_idx, cy_idx) in coarse_grouped.groups:
                            bin_hits = coarse_grouped.get_group((cz_idx, cy_idx))


                            fine_z_coords = np.arange(cz, cz + coarse_granularity_z, pixel_granularity)
                            fine_y_coords = np.arange(cy, cy + coarse_granularity_y, pixel_granularity)


                            fine_z_bins = np.arange(cz, cz + coarse_granularity_z + pixel_granularity, pixel_granularity)
                            fine_y_bins = np.arange(cy, cy + coarse_granularity_y + pixel_granularity, pixel_granularity)

                            # Create a copy of bin_hits to avoid SettingWithCopyWarning
                            bin_hits = bin_hits.copy()


                            bin_hits['z_bin'] = np.digitize(bin_hits['z'], bins=fine_z_bins) - 1
                            bin_hits['y_bin'] = np.digitize(bin_hits['y'], bins=fine_y_bins) - 1


                            grouped_fine = bin_hits.groupby(['z_bin', 'y_bin']).apply(lambda df: df.index.to_list())


                            for fz_idx, fz in enumerate(fine_z_coords):
                                for fy_idx, fy in enumerate(fine_y_coords):
                                    pixel_bottom_left = (fz, fy) 

                                    segment_indices = grouped_fine.get((fz_idx, fy_idx), [])
                                    self.pixel_array_dict[pixel_bottom_left] = segment_indices

                                    if segment_indices:

                                        good_ones = bin_hits.loc[segment_indices]
                                        total_energy = good_ones['dE'].sum()  # Sum of energy deposits
                                        norm_value = norm(total_energy)  # Normalize the total energy
                                        color = cmap(norm_value)  # Get color based on normalized energy


                                        pixel = patches.Rectangle( pixel_bottom_left, pixel_granularity, pixel_granularity, facecolor=color )
                                        ax.add_patch(pixel)  # Add the patch to the plot
                        else:
                            # If the coarse bin is empty, skip plotting its fine bins
                            continue


                ax.set_facecolor( background_color )


                ax.axis('off')  # Hide the axes for a cleaner image

                # Save the figure as an image file with tight layout and no padding
                # plt.savefig(Save_Path, bbox_inches='tight', pad_inches=0)
                plt.savefig(Save_Path, bbox_inches= None , pad_inches=0 , facecolor = background_color )
                plt.close()  # Close the figure to free memory


                ax.set_title(rf'Event_ID = {self.event_id_selected.get()} | Vertex = {self.vertex_id_selected.get()} | Interaction = {interaction_label}')



                canvas.draw()
                toolbar = NavigationToolbar2Tk(canvas, self.Custom_Figure_Frame, pack_toolbar=False)
                toolbar.update()
                toolbar.pack(side=tk.LEFT, fill=tk.X)

                fig.tight_layout()

                if not hasattr(self, 'next_previous_frame'):
                    self.next_previous_frame = tk.Frame(self)
                    self.next_previous_frame.pack(anchor='s', side=tk.BOTTOM)
                    previous_button = tk.Button(self.next_previous_frame, text="previous")
                    previous_button.pack()
                    next_button = tk.Button(self.next_previous_frame, text="next")
                    next_button.pack()

                return
            
