from Imports.common_imports import *

from Helpers.Frame_Manager_Script import Frame_Manager
from Helpers.Scrollable_Frame_Script import ScrollableFrame

from Backend import Custom_Plot_script






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
        self.Back_Button = tk.Button(   self.back_frame, text='Back to Figure Selection' , command= lambda: controller.show_frame("Plot_Selection_Page")   )
        self.Back_Button.pack(anchor='w', side=tk.LEFT)

        # Select custom plot 

        self.custom_fig_select_frame = tk.Frame(self)
        self.custom_fig_select_frame.pack(anchor='w', pady=20)

        self.custom_fig_seleceted = tk.StringVar()

        tk.Label(self.custom_fig_select_frame , text="Custom Figure: ").pack(side=tk.LEFT)
        
    
        Custom_Plot_Names = [attr for attr in dir(Custom_Plot_script.Custom_Plot) if callable(getattr(Custom_Plot_script.Custom_Plot, attr)) and not attr.startswith("__")]

        self.custom_combobox = ttk.Combobox( self.custom_fig_select_frame , textvariable = self.custom_fig_seleceted , values = Custom_Plot_Names , state='readonly' , width=20 )

        self.custom_fig_seleceted.trace_add('write', self.on_custom_selected)

        self.custom_combobox.pack(anchor='w' , side = tk.LEFT)

        # Create file selection Frame 

        self.file_select_frame = tk.Frame(self)
        self.file_select_frame.pack(anchor='w', pady=20)

        self.file_selected = tk.StringVar()

        tk.Label(self.file_select_frame , text="file : ").pack(side=tk.LEFT)
        self.file_combobox = ttk.Combobox( self.file_select_frame , textvariable = self.file_selected , values = controller.Allowed_Files , state='readonly' , width=55 )
        self.file_combobox.pack(anchor='w' , side = tk.LEFT)


        self.particle_select = tk.StringVar()

        self.particle_frame = tk.Frame(self)
        self.particle_frame.pack(anchor='w' )
        tk.Label( self.particle_frame , text = "particle :" ).pack(anchor='w', side=tk.LEFT)

        self.particle_combobox = ttk.Combobox( self.particle_frame, textvariable = self.particle_select ,  values = list( self.controller.pdg_id_map.values() ) )
        self.particle_combobox.pack(anchor='w', side=tk.LEFT)


        self.plot_button_frame = tk.Frame(self)
        self.plot_button_frame.pack(anchor='w', pady = 5)
        self.Plot_Button = tk.Button( self.plot_button_frame, text='Plot' , command= lambda : self.Custom_Selection()   )

        self.Plot_Button.pack(anchor='w', side=tk.LEFT)

        self.Custom_Figure_Frame = tk.Frame(self)
        self.Custom_Figure_Frame.pack(anchor='w', side= tk.LEFT ,pady=5)



    def Custom_Selection(self):
        # Close the old figure (if any) and clear the frame
        if hasattr(self, 'custom_fig'):
            plt.close(self.custom_fig)
        for widget in self.Custom_Figure_Frame.winfo_children():
            widget.destroy()

        # Create the custom figure if the selected type is one of the Track_dE_Analysis types
        if self.custom_fig_seleceted.get() in ['Track_dE_Analysis', 'Track_dE_Analysis_Thesis']:
            self.custom_fig = plt.figure(figsize=(9, 6))
            canvas = FigureCanvasTkAgg(self.custom_fig, master=self.Custom_Figure_Frame)
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            if not hasattr(self, "Download_dir"):
                self.Download_dir = tk.Button(
                    self.plot_button_frame,
                    text='Download All File Plots',
                    command=lambda: Custom_Plot_script.Custom_Plot.Track_dE_Analysis( self, {'Plot_Mode': 'Download_Dir', 'canvas': canvas, 'fig': self.custom_fig}) )
                self.Download_dir.pack(anchor='w', side=tk.LEFT)

            # Call the plotting function based on the selection
            getattr(Custom_Plot_script.Custom_Plot, self.custom_fig_seleceted.get())(
                self,
                {'Plot_Mode': 'Single_Plot', 'canvas': canvas, 'fig': self.custom_fig}
            )

        elif self.custom_fig_seleceted.get() == 'Specific_Vertex':

            self.custom_fig = plt.figure(figsize=(9, 6))
            canvas = FigureCanvasTkAgg(self.custom_fig, master=self.Custom_Figure_Frame)
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            getattr(Custom_Plot_script.Custom_Plot, self.custom_fig_seleceted.get())(
                self,
                {'Plot_Mode': 'Single_Plot', 'canvas': canvas, 'fig': self.custom_fig}
            )


    def on_custom_selected(self, *args):
        """Callback triggered when a new custom figure is selected from the dropdown."""
        if self.custom_fig_seleceted.get() in ['Track_dE_Analysis', 'Track_dE_Analysis_Thesis' , 'Specific_Vertex']:
            self.file_selected.trace_add('write', self.on_file_selected)


    def on_file_selected(self, *args):
        path = os.path.join(self.controller.Data_Directory, self.file_selected.get())
        with h5py.File(path, 'r') as sim_h5:
            unique_event_ids = list(np.unique(sim_h5["segments"]['event_id']))

        # Create a new StringVar and add the trace
        self.event_id_selected = tk.StringVar()
        self.event_id_selected.trace_add('write', self.on_event_selected)

        if hasattr(self, 'event_combobox') and self.event_combobox:
            self.event_combobox['values'] = unique_event_ids
            self.event_combobox.config(textvariable=self.event_id_selected)
            if hasattr(self, 'vertex_combobox'):
                self.vertex_combobox['values'] = []
                if hasattr(self, 'vertex_id_selected'):
                    self.vertex_id_selected.set('')
        else:
            tk.Label(self.file_select_frame, text="event id :").pack(padx=(10, 10), side=tk.LEFT)
            self.event_combobox = ttk.Combobox(
                self.file_select_frame,
                textvariable=self.event_id_selected,
                values=unique_event_ids,
                width=5,
                state='readonly'
            )
            self.event_combobox.pack(anchor='w', side=tk.LEFT)


    def on_event_selected(self, *args):
        """Callback triggered when a new event is selected from the dropdown."""
        print('Im getting called')
        path = os.path.join(self.controller.Data_Directory, self.file_selected.get())
        with h5py.File(path, 'r') as sim_h5:
            segments = sim_h5["segments"]
            event_segment = segments[segments['event_id'] == int(self.event_id_selected.get())]
            unique_vertex_ids = list(np.unique(event_segment['vertex_id']))

        # If the vertex combobox exists, update its values and reuse the same StringVar.
        if hasattr(self, 'vertex_combobox') and self.vertex_combobox:
            if not hasattr(self, 'vertex_id_selected'):
                self.vertex_id_selected = tk.StringVar()
                self.vertex_combobox.config(textvariable=self.vertex_id_selected)
            self.vertex_combobox['values'] = unique_vertex_ids
            self.vertex_id_selected.set('')
        else:
            # Create the vertex combobox along with its StringVar.
            self.vertex_id_selected = tk.StringVar()
            tk.Label(self.file_select_frame, text="vertex id :").pack(padx=(10, 10), side=tk.LEFT)
            self.vertex_combobox = ttk.Combobox(
                self.file_select_frame,
                textvariable=self.vertex_id_selected,
                values=unique_vertex_ids,
                width=10,
                state='readonly'
            )
            self.vertex_combobox.pack(anchor='w', side=tk.LEFT)

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#