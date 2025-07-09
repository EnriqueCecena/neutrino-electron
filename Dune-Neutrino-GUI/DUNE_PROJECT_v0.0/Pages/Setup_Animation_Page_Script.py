from Imports.common_imports import *


class Setup_Animation_Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # Title and Navigation Frame
        title_frame = tk.Frame(self)
        title_frame.pack(fill='x', pady=(10, 5), padx=10)

        # back_button = tk.Button( title_frame ,  text="Back" , command=lambda: controller.show_frame("Training_And_Eval_Options_Page") )
        back_button = tk.Button( title_frame ,  text="Back" , command=lambda: controller.show_frame("Plot_Selection_Page") )
        back_button.pack(side='left')

        title_label = tk.Label(  title_frame  ,  text="Setup Animation"  ,  font=("Helvetica", 16) )
        title_label.pack(side='left', padx=(20, 0))

        self.DF = None

        self.Type_of_Run = tk.Frame(self )
        self.Type_of_Run.pack(anchor='w', pady=20)

        self.Type_of_Animation_Dropdown_selected = tk.StringVar(value = 'Full Run')

        tk.Label( self.Type_of_Run , text="Type of Sim:" ).pack(anchor = 'w', side=tk.LEFT , pady=10)
        self.Type_of_Animation_Dropdown     = ttk.Combobox( self.Type_of_Run , textvariable = self.Type_of_Animation_Dropdown_selected , values=[ 'Full Run' , 'Rotations']  )
        self.Type_of_Animation_Dropdown.pack(anchor = 'w' ,side=tk.LEFT,  pady=10)
        
        # Create file selection Frame 
        self.file_select_frame = tk.Frame(self)
        self.file_select_frame.pack(anchor='w', pady=10)


        self.file_selected = tk.StringVar()
        tk.Label(self.file_select_frame , text="file : ").pack(side=tk.LEFT)
        self.file_combobox = ttk.Combobox( self.file_select_frame , textvariable = self.file_selected , values = controller.Allowed_Files , state='readonly' , width=60 )
        self.file_combobox.pack(anchor='w' , side = tk.LEFT)

        self.spill_identity_frame =  tk.Frame(self)
        self.spill_identity_frame.pack(anchor='w', pady=5)


        self.event_id_selected = tk.StringVar()
        tk.Label(self.spill_identity_frame , text="Event ID: ").pack(side=tk.LEFT)
        self.event_combobox = ttk.Combobox( self.spill_identity_frame , textvariable = self.event_id_selected , values = [] , state='readonly' , width=10 )
        self.event_combobox.pack(anchor='w' , side = tk.LEFT)

        self.vertex_id_selected = tk.StringVar()
        tk.Label(self.spill_identity_frame , text="Vertex ID: ").pack(side=tk.LEFT)
        self.vertex_combobox = ttk.Combobox( self.spill_identity_frame , textvariable = self.vertex_id_selected , values = [] , state='readonly' , width=10 )
        self.vertex_combobox.pack(anchor='w' , side = tk.LEFT)

        self.settings_frame =  tk.Frame(self)
        self.settings_frame.pack(anchor='w', pady=5)

        self.Energy_cut_selected = tk.DoubleVar( value= 1.5)
        tk.Label(self.settings_frame , text="Energy Cut: ").pack(side=tk.LEFT)
        self.Energy_cut_Entry = tk.Entry(self.settings_frame , textvariable = self.Energy_cut_selected  , width=10    )
        self.Energy_cut_Entry.pack(anchor='w' , side = tk.LEFT)


        self.Playback_Speed_selected = tk.DoubleVar( value= 0.1)
        tk.Label(self.settings_frame , text="Playback: ").pack(side=tk.LEFT)
        self.Playback_Speed_Entry = tk.Entry(self.settings_frame , textvariable = self.Playback_Speed_selected  ,  width=10 )
        self.Playback_Speed_Entry.pack(anchor='w' , side = tk.LEFT)



        self.Button_frame =  tk.Frame(self)
        self.Button_frame.pack(anchor='w', pady=20)

        self.Animate_Button = tk.Button( self.Button_frame , text='RUN', command= lambda:  ( self.controller.attributes("-fullscreen", True ) , controller.show_frame("Advanced_Animation_Page") )  )
        self.Animate_Button.pack( side = 'left' )

        # Bind the dropdown selection to update the "event_id" selection dropdown
        self.file_selected.trace_add('write', self.on_file_selected)

        self.event_id_selected.trace_add('write', self.on_event_id_selected)




    def on_file_selected(self, *args):
        """Callback triggered when a new file is selected from the dropdown."""

        path = os.path.join( self.controller.Data_Directory , self.file_selected.get() )
        # self.event_id_selected = tk.StringVar()
        with h5py.File(path , 'r') as sim_h5:
                unique_event_ids = list(np.unique( sim_h5["segments"]['event_id'] ))

        if hasattr(self, 'event_combobox') and self.event_combobox:
            self.event_combobox['values'] = unique_event_ids  
            self.event_combobox.set('')  

        else:
            tk.Label(self.file_select_frame , text= "event id :").pack(padx=(10,10) , side=tk.LEFT ) 
            self.event_combobox = ttk.Combobox( self.file_select_frame, textvariable=self.event_id_selected, values=unique_event_ids, state='readonly' )
            self.event_combobox.pack(anchor='w', side=tk.LEFT)


    def on_event_id_selected( self, *args ):

        if self.event_id_selected.get() == '':
            return
        
        path = os.path.join( self.controller.Data_Directory , self.file_selected.get() )

        with h5py.File(path , 'r') as sim_h5:

            segments = sim_h5['segments']
            sim_h5_vertex = segments[( segments['event_id'] == int(self.event_id_selected.get())  )]
            unique_vertex_ids = list(np.unique( sim_h5_vertex['vertex_id'] ))
        
        if hasattr(self, 'vertex_combobox') :
            self.vertex_combobox['values'] = unique_vertex_ids  
            self.vertex_combobox.set('')  

        

        return 