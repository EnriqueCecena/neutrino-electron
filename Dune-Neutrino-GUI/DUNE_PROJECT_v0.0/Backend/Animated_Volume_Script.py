from Imports.common_imports import *

class Animated_Volume:

    # Class to run a 3D animation on a Tkinter Frame.

    def __init__(self,parent_frame : tk.Frame, h5_file_path:str, event_id:int, vertex_id:int ,energy_cut = 1.5, playback_speed = 1.0, fade_duration = 2.0,interval_ms= 100 , limit_dict = {},  animation_page = None , controller = None) :

        self.parent_frame = parent_frame
        self.h5_path = h5_file_path
        self.event_id = event_id
        self.vertex_id = vertex_id
        self.energy_cut = energy_cut
        self.playback_speed = playback_speed
        self.fade_duration = fade_duration
        self.interval_ms = interval_ms
        self.animation_page = animation_page
        self.controller = controller

        self.min_x_for_plot = limit_dict['min_x']
        self.max_x_for_plot = limit_dict['max_x']
        self.min_y_for_plot = limit_dict['min_y']
        self.max_y_for_plot = limit_dict['max_y']
        self.min_z_for_plot = limit_dict['min_z']
        self.max_z_for_plot = limit_dict['max_z']


        self.dE_min = 0.0
        self.dE_max = 75.0

        self.current_rotation_angle = 0


        self.df = None
        self.df_mc_hdr = None
        self.params = None
        self._visible_indices = []

        # These will be set during setup
        self.fig = None
        self.ax = None
        self.canvas_widget = None
        self.ani = None



    def _load_dataframe(self) -> pd.DataFrame:

        with h5py.File(self.h5_path, "r") as hf:
            segments = hf["segments"][()]

        df = pd.DataFrame(segments)
        # Filter on event_id and vertex_id
        # df = df[(df["event_id"] == self.event_id) & (df["vertex_id"] == self.vertex_id)].copy()

        # Apply energy cut
        df = df[df["dE"] > self.energy_cut].copy()

        # Ensure t0_end >= t0_start
        df["t0_end"] = np.maximum(df["t0_start"], df["t0_end"])

        if df.empty:
            raise ValueError(  f"No hits found for event {self.event_id}, vertex {self.vertex_id} with dE > {self.energy_cut}." )

        return df.reset_index(drop=True)
    


    def _load_mc_hdr(self):
        df_mc_hdr = pd.DataFrame.from_records( h5py.File(self.h5_path)['mc_hdr'], columns=np.dtype(  h5py.File(self.h5_path)['mc_hdr'] ).names )

        return df_mc_hdr

    def _make_cuboid_faces(self, x0: float, y0: float, z0: float, dx: float, dy: float, dz: float):

        # Build one cuboid (6 faces) in plot-coordinates (Z, X, Y).

        # eight corners in original coords (x_orig, y_orig, z_orig):
        v000 = (x0, y0, z0)
        v100 = (x0 + dx, y0, z0)
        v110 = (x0 + dx, y0, z0 + dz)
        v010 = (x0, y0, z0 + dz)
        v001 = (x0, y0 + dy, z0)
        v101 = (x0 + dx, y0 + dy, z0)
        v111 = (x0 + dx, y0 + dy, z0 + dz)
        v011 = (x0, y0 + dy, z0 + dz)

        original_faces = [
            [v000, v100, v110, v010],  # bottom (y = y0)
            [v001, v011, v111, v101],  # top    (y = y0 + dy)
            [v000, v001, v101, v100],  # front  (z = z0)
            [v010, v110, v111, v011],  # back   (z = z0 + dz)
            [v100, v101, v111, v110],  # right  (x = x0 + dx)
            [v000, v010, v011, v001],  # left   (x = x0)
        ]

        faces_plot = []
        for face in original_faces:
            face_plot = []
            for (xo, yo, zo) in face:
                # Map original coords to plotting coords:
                x_plot = zo        
                y_plot = xo      
                z_plot = yo       
                face_plot.append((x_plot, y_plot, z_plot))
            faces_plot.append(face_plot)

        return faces_plot

    def _initialize_plot(self, df: pd.DataFrame):

        # Create figure & 3D axes
        self.fig = plt.Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")



        NX, NZ = 5, 7                         # number of modules along X and Z
        DX = (self.max_x_for_plot - self.min_x_for_plot) / NX      # 700 / 5  = 140.0
        DY = (self.max_y_for_plot - self.min_y_for_plot )           # ≈300
        DZ = (self.max_z_for_plot - self.min_z_for_plot) / NZ      # ≈71.77

        # dE_min = 0.0
        # dE_max = 75.0

        self.ax.set_xlim(self.min_z_for_plot, self.max_z_for_plot)
        self.ax.set_ylim(self.min_x_for_plot, self.max_x_for_plot)
        self.ax.set_zlim(self.min_y_for_plot, self.max_y_for_plot)

        self.ax.set_xlabel("Z")
        self.ax.set_ylabel("X")
        self.ax.set_zlabel("Y")

        # Build the grid of 35 cuboids (5 × 7 modules)
        all_faces = []
        x0_base = self.min_x_for_plot
        y0_base = self.min_y_for_plot
        z0_base = self.min_z_for_plot


        for i in range(NX):
            for j in range(NZ):
                x0 = x0_base + i * DX
                y0 = y0_base
                z0 = z0_base + j * DZ
                faces = self._make_cuboid_faces(x0, y0, z0, DX, DY, DZ)
                all_faces.extend(faces)

        cuboid_collection = Poly3DCollection( all_faces, facecolors=(0.8, 0.8, 0.8, 0.2),edgecolors="red",linewidths=0.8 ) 
        self.ax.add_collection3d(cuboid_collection)

        # Prepare empty scatter
        self.scatter = self.ax.scatter([], [], [], c=[], marker="o", s=7)

        # Title placeholder
        self.title = self.ax.set_title("")

        # Embed figure into Tk frame
        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=self.parent_frame)
        self.canvas_widget.draw()
        self.canvas_widget.get_tk_widget().pack(fill="both", expand=True)



    def _compute_animation_params(self, df: pd.DataFrame):

        # Given df sorted by t0_start, compute arrays and frame count for animation.

        t_start_arr = df["t0_start"].to_numpy()
        t_end_arr = df["t0_end"].to_numpy()
        x_arr = df["x"].to_numpy()
        y_arr = df["y"].to_numpy()
        z_arr = df["z"].to_numpy()
        dE_arr = df["dE"].to_numpy()

        norm = mcolors.Normalize(vmin= self.dE_min, vmax= self.dE_max)
        cmap = cm.plasma

        t_min = t_start_arr.min()
        t_max = t_end_arr.max()
        t_final = t_max + self.fade_duration

        dt_real = self.interval_ms / 1000.0
        dt_data = self.playback_speed * dt_real
        total_duration = t_final - t_min
        num_frames = int(np.ceil(total_duration / dt_data)) + 1

        return {"t_start_arr": t_start_arr, "t_end_arr": t_end_arr, "x_arr": x_arr, "y_arr": y_arr, "z_arr": z_arr, "dE_arr": dE_arr, "norm": norm, "cmap": cmap, "t_min": t_min, "num_frames": num_frames, "dt_data": dt_data}



    def _animate_update(self, frame_idx: int, params: dict):

        # Update function called by FuncAnimation for each frame.

        t_current = params["t_min"] + frame_idx * params["dt_data"]

        t_start_arr = params["t_start_arr"]
        t_end_arr = params["t_end_arr"]
        x_arr = params["x_arr"]
        y_arr = params["y_arr"]
        z_arr = params["z_arr"]
        dE_arr = params["dE_arr"]
        norm = params["norm"]
        cmap = params["cmap"]

        if self.fade_duration > 0:
            mask_alive = ( (t_current >= t_start_arr) & (t_current <= (t_end_arr + self.fade_duration)) )
        else:
            mask_alive = ( (t_current >= t_start_arr) & (t_current <= t_end_arr) )

        if not mask_alive.any():
            # No visible points
            self._visible_indices = []

            self.scatter._offsets3d = ([], [], [])
            self.scatter.set_facecolors([])
            self.scatter.set_edgecolors([])
            self.title.set_text(f"t = {t_current:.3f}")
            return (self.scatter, self.title)

        idxs = np.where(mask_alive)[0]
        alphas = np.zeros_like(idxs, dtype=float)

        for out_i, i in enumerate(idxs):
            ts = t_start_arr[i]
            te = t_end_arr[i]
            if t_current < ts:
                alpha = 0.0
            elif ts <= t_current <= te:
                alpha = 1.0
            else:  # t_current > te
                if self.fade_duration > 0:
                    alpha = 1.0 - (t_current - te) / self.fade_duration
                    alpha = np.clip(alpha, 0.0, 1.0)
                else:
                    alpha = 0.0
            alphas[out_i] = alpha

        keep = (alphas > 0.0)
        if keep.any():
            idxs = idxs[keep]
            alphas = alphas[keep]

            idxs_visible = idxs[keep]
            self._visible_indices = idxs_visible 

            xs_orig = x_arr[idxs]
            ys_orig = y_arr[idxs]
            zs_orig = z_arr[idxs]

            xs_plot = zs_orig
            ys_plot = xs_orig
            zs_plot = ys_orig

            dE_vals = dE_arr[idxs]
            normed = norm(dE_vals)
            rgba = cmap(normed)[:]  # shape = (M,4)
            rgba[:, 3] = alphas

            self.scatter._offsets3d = (xs_plot, ys_plot, zs_plot)
            self.scatter.set_facecolors(rgba)
            self.scatter.set_edgecolors(rgba)
        else:

            self._visible_indices = [] 
            self.scatter._offsets3d = ([], [], [])
            self.scatter.set_facecolors([])
            self.scatter.set_edgecolors([])

        self.title.set_text(f"t = {t_current:.3f}")
        return (self.scatter, self.title)


    def _animate_update_roations(self, frame_idx: int, params: dict):

        # t_current = params["t_min"] + frame_idx * params["dt_data"]
        t_current = frame_idx * params["dt_data"]

        x_arr = params["x_arr"]
        y_arr = params["y_arr"]
        z_arr = params["z_arr"]
        dE_arr = params["dE_arr"]

        self._visible_indices = np.arange(len(x_arr))

        angle = self.current_rotation_angle
        rad = np.deg2rad(angle)


        dx = x_arr - self.pivot_x
        dy = y_arr - self.pivot_y
        new_x = dx * np.cos(rad) - dy * np.sin(rad) + self.pivot_x
        new_y = dx * np.sin(rad) + dy * np.cos(rad) + self.pivot_y
        new_z = z_arr  # z stays fixed for a z‐axis rotation


        xs_plot = new_z
        ys_plot = new_x
        zs_plot = new_y


        normed = params["norm"](dE_arr)
        colors = params["cmap"](normed)
        colors[:, 3] = 1.0


        self.scatter._offsets3d = (xs_plot, ys_plot, zs_plot)
        self.scatter.set_facecolors(colors)
        self.scatter.set_edgecolors(colors)


        self.current_rotation_angle = (angle + 10 * params["dt_data"]) % 360
        self.title.set_text(f"t = {t_current:.3f}s   rot = {self.current_rotation_angle:.1f}°")

        return (self.scatter, self.title)





    def start(self):

        # Main entrypoint: loads data, initializes plot, and launches the animation.

        
        # if self.controller.Advanced_Evaluation_Page.Type_of_Animation_Dropdown_selected.get() == 'Full Run':
        df = self._load_dataframe()
        df = df.sort_values("t0_start").reset_index(drop=True)

        df_mc_hdr =  self._load_mc_hdr()

        self.df = df



        if self.controller.Setup_Animation_Page.Type_of_Animation_Dropdown_selected.get() == 'Full Run':

            self._initialize_plot(df)

            params = self._compute_animation_params(df)

            self.ani = FuncAnimation(self.fig ,lambda idx: self._animate_update(idx, params),frames=params["num_frames"],interval=self.interval_ms,blit=False,repeat=False, )

        else:
            current_event_id = self.controller.Setup_Animation_Page.event_id_selected.get()
            current_vertex_id = self.controller.Setup_Animation_Page.vertex_id_selected.get()

            clean_df = df[ (df['event_id'] == int(current_event_id)) & (df['vertex_id'] == int(current_vertex_id)) ]
            clean_df_mc_hdr = df_mc_hdr[ (df_mc_hdr['event_id'] == int(current_event_id)) & (df_mc_hdr['vertex_id'] == int(current_vertex_id)) ]


            if clean_df.empty:
                print(  "No Data",
                        f"No hits found for event {current_event_id}, "
                        f"vertex {current_vertex_id} above dE > {self.energy_cut}"
                )
                return


            # remember to replace the buggy self_df_mc_hdr line with:
            self.df_mc_hdr = clean_df_mc_hdr

            print( clean_df['t0'] )
            # grab the first (and only) pivot for this event/vertex:
            pivot = self.df_mc_hdr.iloc[0]
            self.pivot_x = float(pivot['x_vert'])
            self.pivot_y = float(pivot['y_vert'])
            self.pivot_z = float(pivot['z_vert'])
            self.df = clean_df

            self._initialize_plot(self.df)
            params = self._compute_animation_params(self.df)



            # self.ani = FuncAnimation(self.fig ,lambda idx: self._animate_update_roations(idx, params),frames=params["num_frames"],interval=self.interval_ms,blit=False,repeat=False, )
            self.ani = FuncAnimation(self.fig ,lambda idx: self._animate_update_roations(idx, params),frames=itertools.count(),interval=self.interval_ms,blit=False,repeat=False, )




        self.ani.event_source.start()
    
        # else:
        #     print(self.controller.Advanced_Evaluation_Page.Type_of_Animation_Dropdown_selected.get() )



    def play(self):
        if self.ani is not None:
            self.ani.event_source.start()




    def pause_1(self):

        if self.ani is not None:
            self.ani.event_source.stop()


        temp_df = self.df.iloc[self._visible_indices].copy()

        if self.controller.Setup_Animation_Page.Type_of_Animation_Dropdown_selected.get() == "Rotations":
            angle = self.current_rotation_angle
            rad   = np.deg2rad(angle)
            dx    = temp_df['x'] - self.pivot_x
            dy    = temp_df['y'] - self.pivot_y

            temp_df['x'] = dx * np.cos(rad) - dy * np.sin(rad) + self.pivot_x
            temp_df['y'] = dx * np.sin(rad) + dy * np.cos(rad) + self.pivot_y


        for i in range(4):
            dropdown = getattr(self.animation_page, f"dropdown_{i+1}")
            proj     = dropdown.get()

            if proj in ("ZY", "ZX", "XY"):
                self.controller.Use_Pixel_Array.plot_testing( self=self, DF=temp_df,  plot_canvas=getattr(self.animation_page, f"plot_frame_{i+1}"), projection=proj )


    def pause(self):
        if self.ani is not None:
            self.ani.event_source.stop()
            print( self.ani )

            
 


    def replay(self):
        if self.ani is not None:
            self.ani.event_source.stop()

            self.ani.frame_seq = self.ani.new_frame_seq()

            self._animate_update(0, self._compute_animation_params(self._load_dataframe()))
            self.canvas_widget.draw()
            self.ani.event_source.start()




