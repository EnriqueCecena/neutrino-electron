from Imports.common_imports import *

from Helpers.Frame_Manager_Script import Frame_Manager
from Helpers.Scrollable_Frame_Script import ScrollableFrame
# from Core_Fucnctions import Pixel_Array_Script


class Create_Dataset_Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        self.controller = controller
        self.selected_files = []
        self.file_vars = []

        # instantiate your manager here (so setup_process/cancel_process work)
        self.manager = Frame_Manager(self)

        self.header_frame = tk.Frame(self)
        self.header_frame.pack(anchor='w', padx=10, pady=20)

        back_button = tk.Button(
            self.header_frame,
            text='Back',
            command=lambda: controller.show_frame("Dataset_Page")
        )
        back_button.pack(side=tk.LEFT)

        header_label = tk.Label(
            self.header_frame,
            text="Create Dataset",
            font=("Helvetica", 16)
        )
        header_label.pack(side=tk.LEFT, padx=150)

        self.progressive_frame = tk.Frame(self)
        self.progressive_frame.pack(anchor='w', padx=10, pady=(0, 20))

        self.progress = ttk.Progressbar(
            self.progressive_frame,
            orient="horizontal",
            length=600,
            mode="determinate"
        )
        self.progress_label = tk.Label(
            self.progressive_frame,
            text='',
            font=("Arial", 12)
        )
        self.progress.pack(anchor='w', side=tk.LEFT)
        self.progress_label.pack(anchor='w', side=tk.LEFT)

        self.file_select_frame = tk.Frame(self)
        self.file_select_frame.pack(anchor='w', padx=10, pady=(0, 20))

        tk.Label(self.file_select_frame, text="Select Files:").pack(anchor='w')

        # Scrollable frame for the file list
        scroll_frame = ScrollableFrame(self.file_select_frame)
        scroll_frame.pack(fill="both", expand=True, pady=5)

        # Determine which files to show
        allowed_files = getattr(controller, 'Allowed_Files', [])
        if not allowed_files:
            allowed_files = os.listdir(controller.Data_Directory)

        for file in sorted(allowed_files):
            var = tk.IntVar()
            cb = tk.Checkbutton(
                scroll_frame.scrollable_frame,
                text=file,
                variable=var,
                anchor='w',
                width=200
            )
            cb.pack(fill='x', padx=5, pady=2)
            self.file_vars.append((var, file))

        self.button_frame = tk.Frame(self)
        self.button_frame.pack(anchor='w', padx=10, pady=(0, 20))

        tk.Button(
            self.button_frame,
            text="Select All",
            command=self.select_all
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            self.button_frame,
            text="Deselect All",
            command=self.deselect_all
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            self.button_frame,
            text="Confirm Selection",
            command=self.confirm_selection
        ).pack(side=tk.LEFT, padx=5)

        # Interact Frame for Preview and Create
        self.Interact_Frame = tk.Frame(self)
        self.Interact_Frame.pack(anchor='w', padx=10, pady=(0, 20))

        self.Preview_Button = tk.Button(
            self.button_frame,
            text='Preview',
            command=self.Preview_Interaction
        )
        self.Preview_Button.pack(side=tk.LEFT, anchor='w', padx=5)

        self.ZY_CheckButton_Value = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self.button_frame,
            variable=self.ZY_CheckButton_Value,
            text='ZY',
            command=lambda: print('ZY', self.ZY_CheckButton_Value.get())
        ).pack(side=tk.LEFT, padx=5)

        self.ZX_CheckButton_Value = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.button_frame,
            variable=self.ZX_CheckButton_Value,
            text='ZX',
            command=lambda: print('ZX', self.ZX_CheckButton_Value.get())
        ).pack(side=tk.LEFT, padx=5)

        self.XY_CheckButton_Value = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.button_frame,
            variable=self.XY_CheckButton_Value,
            text='XY',
            command=lambda: print('XY', self.XY_CheckButton_Value.get())
        ).pack(side=tk.LEFT, padx=5)

        tk.Label(self.button_frame, text="Rotations :").pack(
            anchor='w', side=tk.LEFT, padx=10
        )
        self.Text_Box_Rotation_Value = tk.IntVar(value=0)
        tk.Entry(
            self.button_frame,
            bg='black',
            fg='white',
            font=("Arial", 12),
            width=4,
            textvariable=self.Text_Box_Rotation_Value
        ).pack(anchor='w', side=tk.LEFT, padx=(0, 10), pady=5)

        tk.Label(self.Interact_Frame, text="ML Dataset Name :").pack(
            anchor='w', side=tk.LEFT, padx=10
        )
        self.Text_Box_ML_Dataset_Name = tk.Text(
            self.Interact_Frame,
            bg='black',
            fg='white',
            font=("Arial", 12),
            width=20,
            height=1,
            padx=10,
            pady=10
        )
        self.Text_Box_ML_Dataset_Name.pack(
            anchor='w', side=tk.LEFT, padx=(0, 10), pady=5
        )

        tk.Label(self.Interact_Frame, text="File Tag :").pack(
            anchor='w', side=tk.LEFT, padx=10
        )
        self.Text_Box_File_Tag = tk.Text(
            self.Interact_Frame,
            bg='black',
            fg='white',
            font=("Arial", 12),
            width=10,
            height=1,
            padx=10,
            pady=10
        )
        self.Text_Box_File_Tag.pack(
            anchor='w', side=tk.LEFT, padx=(0, 10), pady=5
        )

        self.Create_Dataset_Button = tk.Button(
            self.Interact_Frame,
            text='Create',
            command=self.manager.setup_process
        )
        self.Create_Dataset_Button.pack(side=tk.LEFT, anchor='w')

        self.Cancel_Creation = tk.Button(
            self.Interact_Frame,
            text='Cancel',
            command=self.manager.cancel_process,
            state='disabled'
        )
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
        self.selected_files = [
            file for var, file in self.file_vars if var.get()
        ]

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
        random_vertex_id = np.random.choice(
            temp_mc_hdr[temp_mc_hdr['event_id'] == random_event_id]['vertex_id']
        )

        self.event_id_selected = random_event_id
        self.vertex_id_selected = random_vertex_id

        self.controller.Use_Pixel_Array.plot(self)


    def Create_ML_Dataset_2(self):

        if not self.selected_files:
            print("No files selected for dataset creation!")
            self.progress_value = 100
            self.controller.running = False
            self.Create_Dataset_Button.config(state='normal')
            return

        # Read user parameters
        Test_directory = self.Text_Box_ML_Dataset_Name.get('1.0', tk.END).strip()
        File_tag       = self.Text_Box_File_Tag.get('1.0', tk.END).strip()
        Rotation_Step  = float(self.Text_Box_Rotation_Value.get() or 0)
        Projection_Dict = {
            'ZY': self.ZY_CheckButton_Value.get(),
            'ZX': self.ZX_CheckButton_Value.get(),
            'XY': self.XY_CheckButton_Value.get(),
        }
        # Determine active projections
        proj_list = [p for p, enabled in Projection_Dict.items() if enabled]
        if not proj_list:
            proj_list = ['ZY']  # default projection

        # Disable UI and prepare folders
        print(f"Output directory: {Test_directory}")
        self.Text_Box_ML_Dataset_Name.config(state='disabled')
        self.Cancel_Creation.config(state='normal')
        os.makedirs(Test_directory, exist_ok=True)
        # Create subfolders for each projection and physics class

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

        for prj in proj_list:
            for subdir in Directory_Name_Map.values():
                os.makedirs(os.path.join(Test_directory, prj, subdir), exist_ok=True)

        # Counters per class & projection
        Dir_File_Name_Counter = {
            label: {prj: 0 for prj in proj_list}
            for label in Directory_Name_Map.keys()
        }

        # Gather all unique event IDs
        all_event_ids = []
        for fname in self.selected_files:
            path = os.path.join(self.controller.Data_Directory, fname)
            with h5py.File(path, 'r') as sim_h5:
                all_event_ids.extend(
                    np.unique(sim_h5['mc_hdr']['event_id']).tolist()
                )
        all_event_ids = list(set(all_event_ids))
        num_events = len(all_event_ids)

      
        min_z, max_z = self.controller.min_z_for_plot, self.controller.max_z_for_plot
        min_y, max_y = self.controller.min_y_for_plot, self.controller.max_y_for_plot
        min_x, max_x = self.controller.min_x_for_plot, self.controller.max_x_for_plot

        self.controller.running = True
        self.Create_Dataset_Button.config(state='disabled')
        
        cnter = 0
        for fname in self.selected_files:
            if not self.controller.running:
                break
            path = os.path.join(self.controller.Data_Directory, fname)
            with h5py.File(path, 'r') as sim_h5:
                seg_ds = sim_h5['segments']
                seg_mask = seg_ds['dE'] > 1.5
                temp_segments = seg_ds[seg_mask]
                hdr_ds = sim_h5['mc_hdr']

                for event_id in np.unique(temp_segments['event_id']):
                    cnter += 1
                    self.progress_value = (cnter / num_events) * 100
                    if not self.controller.running:
                        break

                    # build a DataFrame of all surviving segments for this event
                    seg_df = pd.DataFrame(temp_segments[temp_segments['event_id'] == event_id])
                  
                    # grab the mc_hdr rows for this event as a numpy structured array
                    hdr_array = hdr_ds[hdr_ds['event_id'] == event_id]
                    mc_hdr_vertex_ids = np.unique(hdr_array['vertex_id']).tolist()
                  
                    # compute which vertex_ids are purely noise
                    noise_ids = list(set(seg_df['vertex_id']) - set(mc_hdr_vertex_ids))
                    
                    for true_v in mc_hdr_vertex_ids:
                        hdr_rows = hdr_array[hdr_array['vertex_id'] == true_v]
                        if hdr_rows.shape[0] != 1:
                            continue
                          
                        x0, y0, z0 = hdr_rows['x_vert'][0], hdr_rows['y_vert'][0], hdr_rows['z_vert'][0]

                        if not (min_z < z0 < max_z and min_y < y0 < max_y and min_x < x0 < max_x):
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
                        elif not any((isCC, isQES, isMEC, isDIS, isCOH)):
                            interaction_label = r"$\nu$-NC"
                        elif isCC:
                            interaction_label = r"$\nu$-CC"
                        else:
                            interaction_label = 'Other'

                        # Skip any vertex with no real segments dE>1.5
                        real_df = seg_df[seg_df['vertex_id'] == true_v]
                        if real_df.empty:
                            continue

                        # collect noise if present
                        noise_df = seg_df[seg_df['vertex_id'].isin(noise_ids)] if noise_ids else pd.DataFrame(columns=seg_df.columns)

                        if noise_df.empty:
                            full_df = real_df.copy()
                        else:
                            full_df = pd.concat([real_df, noise_df], ignore_index=True)

                        # Determine angles
                        angles = (
                            np.arange(0, 360, Rotation_Step)
                            if Rotation_Step > 0 else np.array([0.0])
                        )
                        
                        # final DataFrame + save
                        for angle in angles:
                            if angle:
                                rad = np.deg2rad(angle)
                                dx = full_df['x'] - x0 
                                dy = full_df['y'] - y0
                                df_rot = full_df.copy()
                                df_rot['x'] = dx * np.cos(rad) - dy * np.sin(rad) + x0
                                df_rot['y'] = dx * np.sin(rad) + dy * np.cos(rad) + y0
                            else:
                                df_rot = full_df

                            for prj in proj_list:
                                out_dir = os.path.join(Test_directory, prj, Directory_Name_Map.get(interaction_label, 'Other'))
                                out_name = (
                                    f"{File_tag}_{event_id}_{true_v}_{prj}_{ Directory_Name_Map[interaction_label] }_"
                                    f"{prj}_ANG_{angle:.1f}.png"
                                )
                                out_path = os.path.join(out_dir, out_name)
                                # Call helper to project & rotate
                                Pixel_Array_Script.Use_Pixel_Array.Save_For_ML_Testing( self, df_rot, out_path, prj, angle, min_z, max_z, min_y, max_y, min_x, max_x )
                                Dir_File_Name_Counter[interaction_label][prj] += 1

            print(f"{fname} (finished)")

        # Finalize
        self.progress_value = 100
        self.controller.running = False
        self.Create_Dataset_Button.config(state='normal')
        self.Text_Box_ML_Dataset_Name.config(state='normal')
        self.Cancel_Creation.config(state='disabled')

