from Imports.common_imports import *

class Advanced_Animation_Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # Title and Navigation
        title_frame = tk.Frame(self)
        title_frame.pack(fill='x', pady=(10, 5), padx=10)

        back_button = tk.Button(title_frame, text="Back", command=lambda: (self.controller.attributes("-fullscreen", False ) , controller.show_frame("Setup_Animation_Page")))
        back_button.pack(side='left')

        title_label = tk.Label(title_frame, text="Animation Page", font=("Helvetica", 16))
        title_label.pack(side='left', padx=(20, 0))

        # control_frame = tk.Frame(self)
        # control_frame.pack(fill='x', pady=(5, 10), padx=10)

        self.play_button = tk.Button(title_frame, text="Play", command=self.play)
        self.play_button.pack(side=tk.LEFT, padx=5)

        self.pause_button = tk.Button(title_frame, text="Pause", command=self.pause)
        self.pause_button.pack(side=tk.LEFT, padx=5)

        self.replay_button = tk.Button(title_frame, text="Replay", command=self.replay)
        self.replay_button.pack(side=tk.LEFT, padx=5)


        self.content_frame = tk.Frame(self)
        self.content_frame.pack(fill='both', expand=True)

        self.first_row = tk.Frame(self.content_frame)
        self.first_row.pack(pady=10)


        self.canvas1_frame = tk.Frame(self.first_row)
        self.canvas1_frame.pack(side=tk.LEFT, padx=20)
        self.canvas1 = tk.Canvas(self.canvas1_frame, width=800, height=500, bg="white")
        self.canvas1.pack()


        self.canvas2_frame = tk.Frame(self.first_row)
        self.canvas2_frame.pack(side=tk.LEFT, padx=20)
        self.canvas2 = tk.Canvas(self.canvas2_frame, width=800, height=500, bg="white")
        self.canvas2.pack()

        self.second_row = tk.Frame(self.content_frame)
        self.second_row.pack(pady=20)

        dropdown_options = ['ZY' ,'ZX' , 'XY', 'ZY (heatmap)' , 'ZX (heatmap)'  , 'XY (heatmap)' ,] 
        # for i in range(4):
        #     frame = tk.Frame(self.second_row)
        #     frame.pack(side=tk.LEFT, padx=10)


        #     var = tk.StringVar(value=str(i+1))
        #     # dropdown = ttk.Combobox(frame, textvariable=var, values=[str(i+1)], state='readonly', width=5)
        #     dropdown = ttk.Combobox(frame, textvariable=var, values= dropdown_options, state='readonly', width=15)
        #     dropdown.pack(pady=(0, 5))

        #     canvas = tk.Canvas(frame, width=400, height=300, bg="white")
        #     canvas.pack()

        #     setattr(self, f"dropdown_{i+1}", dropdown)
        #     setattr(self, f"canvas_{i+1}", canvas)


        for i in range(4):
            # 1) make a frame of a known size and turn off “auto‐shrink”
            frame = tk.Frame(self.second_row, width=400, height=350)
            frame.pack(side=tk.LEFT, padx=10)
            frame.pack_propagate(False)   # ← lock its size to exactly 400×350

            # 2) put the dropdown in the top-left
            var = tk.StringVar(value=dropdown_options[0])
            dropdown = ttk.Combobox(frame,
                                    textvariable=var,
                                    values=dropdown_options,
                                    state='readonly',
                                    width=15)
            dropdown.pack(side='top', anchor='nw', padx=5, pady=(5,3))

            # 3) reserve the rest of the frame for your plot
            plot_container = tk.Frame(frame, width=400, height=300 , bg = "white")
            plot_container.pack(side='top', anchor='nw')
            plot_container.pack_propagate(False)

            setattr(self, f"dropdown_{i+1}", dropdown)
            setattr(self, f"plot_frame_{i+1}", plot_container)

        self.adv_eval = None
            

    def play(self):

        setup_page = self.controller.frames["Setup_Animation_Page"]
        animation_page = self.controller.frames.get("Advanced_Animation_Page")
        

        h5_path = os.path.join( self.controller.Data_Directory , setup_page.file_selected.get() )
        print( 123 , setup_page.file_selected.get() )
        print( 3455 , self.controller.Data_Directory )
        print( setup_page.event_id_selected.get() )
        event_id = int(setup_page.event_id_selected.get())
        vertex_id = int(setup_page.vertex_id_selected.get())
        energy_cut = float(setup_page.Energy_cut_selected.get())
        playback_speed = float(setup_page.Playback_Speed_selected.get())

        limit_dict = {'min_x': self.controller.min_x_for_plot,
                      'max_x': self.controller.max_x_for_plot,
                      'min_y': self.controller.min_y_for_plot,
                      'max_y': self.controller.max_y_for_plot,
                      'min_z': self.controller.min_z_for_plot,
                      'max_z': self.controller.max_z_for_plot  } 

        if self.controller.Setup_Animation_Page.Type_of_Animation_Dropdown_selected.get() == 'Full Run':
            
            Live_canvas = self.canvas1_frame


        else:
            Live_canvas = self.canvas2_frame

        for widget in Live_canvas.winfo_children():
            widget.destroy()

        self.adv_eval = self.controller.Animated_Volume(
            parent_frame= Live_canvas,
            h5_file_path=h5_path,
            event_id=event_id,
            vertex_id=vertex_id,
            energy_cut=energy_cut,
            playback_speed=playback_speed,
            fade_duration=setup_page.fade_duration if hasattr(setup_page, 'fade_duration') else 2.0,
            interval_ms=int(setup_page.interval_ms if hasattr(setup_page, 'interval_ms') else 100),
            limit_dict=limit_dict,
            animation_page = animation_page,
            controller = self.controller
        )
        self.adv_eval.start()

        # Advanced_Evaluation(
        #     parent_frame=self.canvas1_frame,
        #     h5_file_path=h5_path,
        #     event_id=event_id,
        #     vertex_id=vertex_id,
        #     energy_cut=energy_cut,
        #     playback_speed=playback_speed,
        #     fade_duration=setup_page.fade_duration if hasattr(setup_page, 'fade_duration') else 2.0,
        #     interval_ms=int(setup_page.interval_ms if hasattr(setup_page, 'interval_ms') else 100)
        # ).start()

    def pause(self):
        if self.adv_eval is not None:
            self.adv_eval.pause_1()
            # self.adv_eval.pause_plot_updates()

    def replay(self):
        if self.adv_eval is not None:
            self.adv_eval.replay()

