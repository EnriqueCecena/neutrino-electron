from Imports.common_imports import *

class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        label = tk.Label(self, text="Dune Data Analysis 📊", font=("Helvetica", 16))
        label.pack(pady=10, padx=10, anchor='w')
        self.controller = controller

        # Navigation buttons
        tk.Button(self, text="View Datasets",  command=lambda: controller.show_frame("Dataset_View_Page")).pack(anchor='w')

        tk.Button(self, text="Go to Plot Page", command=lambda: controller.show_frame("Plot_Selection_Page")).pack(anchor='w')

        tk.Button(self, text="Create ML Dateset", command=lambda: controller.show_frame("Dataset_Page")).pack(anchor='w')

        tk.Button(self, text="Train & Evaluate ML Model",  command=lambda: controller.show_frame("Training_And_Eval_Options_Page")).pack(anchor='w')
        
        tk.Button( self , text= "Load Datasets" , command= lambda: self.Load_Datasets() ).pack(anchor='w' , pady=(20,0))
        tk.Button(self, text="Go to Settings", command=lambda: controller.show_frame("Settings_Page")).pack(anchor='w')

        # tk.Button( self ,text="Ouuu" , command=lambda:  print( dir(controller))).pack(anchor='w' ,pady=(20,20))
        # Dataframe_Loader_Frame = tk.Frame(self)
        # Dataframe_Loader_Frame.pack( anchor='w')

        # ttk.Entry( Dataframe_Loader_Frame , width=30 , state='disabled' ).pack(anchor='w' , side= tk.LEFT)

    def Load_Datasets(self):
        directory = tk.filedialog.askdirectory( title="Select a Data Directory", initialdir=os.getcwd() )

        if directory: 
            self.controller.Data_Directory = directory

            try:
                File_Names = os.listdir(self.controller.Data_Directory)

                Temp_File_Names_Dict = { }
                for _file_name_ in File_Names:
                    try:
                        Temp_File_Names_Dict.update({ int(_file_name_.split('.')[3]) : _file_name_ })
                    except:
                        pass
                # Temp_File_Names_Dict = {int(i.split('.')[3]): i for i in File_Names } 
                sorted_keys = sorted(Temp_File_Names_Dict.keys())
                File_Names = [Temp_File_Names_Dict[i] for i in sorted_keys]
                self.controller.File_Names = File_Names
                self.controller.Allowed_Files = File_Names

            except:
                print("something has gone wrong")

                    # Reinitialize frames that rely on Allowed_Files
            self.controller.reinitialize_frame("View_Segments_Page")
            self.controller.reinitialize_frame("View_mc_hdr_Page")
            self.controller.reinitialize_frame("View_traj_Page")

            self.controller.reinitialize_frame("Figure_Creation_Page")
            self.controller.reinitialize_frame("Scatter_Creation_Page")

            self.controller.reinitialize_frame("Custom_Figure_Page")
            self.controller.reinitialize_frame("Create_Dataset_Page")
            self.controller.reinitialize_frame("Advanced_Evaluation_Page")
            self.controller.reinitialize_frame("File_Selection_Page")


            self.controller.show_frame("StartPage")

        return