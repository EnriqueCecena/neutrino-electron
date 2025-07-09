from Imports.common_imports import *

class Evaluate_Model_Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # Title and Navigation Frame
        title_frame = tk.Frame(self)
        title_frame.pack(fill='x', pady=(10, 5), padx=10)

        back_button = tk.Button( title_frame ,  text="Back" , command=lambda: controller.show_frame("Training_And_Eval_Options_Page") )
        back_button.pack(side='left')

        title_label = tk.Label(  title_frame  ,  text="Evaluate Model"  ,  font=("Helvetica", 16) )
        title_label.pack(side='left', padx=(20, 0))

        # Directory Selection Frame
        dir_frame_input_1 = tk.Frame(self)
        dir_frame_input_1.pack(fill='x', pady=(5, 10), padx=10)

        self.selected_dir_1 = tk.StringVar()
        dir_entry_1 = tk.Entry( dir_frame_input_1  ,  textvariable=self.selected_dir_1  ,   bg='black' ,  fg='white',  font=("Arial", 12) ,  width=90,  state='disabled' )
        dir_entry_1.pack(side='left', fill='x')
        self.selected_dir_1.trace_add('write', lambda *args: self.Add_Dir_Classes( self.selected_dir_1 ) )

        select_button = tk.Button( dir_frame_input_1  ,  text="Select Directory (1)" ,  command=lambda: self.controller.Frame_Manager.select_directory_window(self , dir_entry_1) )
        select_button.pack(side='left', padx=(5, 0))


        dir_frame_input_2 = tk.Frame(self)
        dir_frame_input_2.pack(fill='x', pady=(5, 10), padx=10)

        self.selected_dir_2 = tk.StringVar()
        dir_entry_2 = tk.Entry( dir_frame_input_2  ,  textvariable=self.selected_dir_2  ,   bg='black' ,  fg='white',  font=("Arial", 12) ,  width=90,  state='disabled' )
        dir_entry_2.pack(side='left', fill='x')
        # self.selected_dir_3.trace_add('write', lambda *args: self.Add_Dir_Classes( self.selected_dir_2) )

        select_button = tk.Button( dir_frame_input_2  ,  text="Select Directory (2)" ,  command=lambda: self.controller.Frame_Manager.select_directory_window(self , dir_entry_2) )
        select_button.pack(side='left', padx=(5, 0))

        dir_frame_input_3 = tk.Frame(self)
        dir_frame_input_3.pack(fill='x', pady=(5, 10), padx=10)

        self.selected_dir_3 = tk.StringVar()
        dir_entry_3 = tk.Entry( dir_frame_input_3  ,  textvariable=self.selected_dir_3  ,   bg='black' ,  fg='white',  font=("Arial", 12) ,  width=90,  state='disabled' )
        dir_entry_3.pack(side='left', fill='x')
        # self.selected_dir_3.trace_add('write', lambda *args: self.Add_Dir_Classes( self.selected_dir_1 ) )

        select_button = tk.Button( dir_frame_input_3  ,  text="Select Directory (3)" ,  command=lambda: self.controller.Frame_Manager.select_directory_window(self , dir_entry_3) )
        select_button.pack(side='left', padx=(5, 0))

        dropdown_frame = tk.Frame( self )
        dropdown_frame.pack(fill='x', pady=(10, 0), padx=10)

        self.Class_dropdown_var = tk.StringVar()
        self.Image_dropdown_var = tk.StringVar()

        tk.Label(dropdown_frame , text= 'Class : ' ).pack(side = tk.LEFT , anchor= 'w')
        self.Class_dropdown = ttk.Combobox( dropdown_frame  , textvariable= self.Class_dropdown_var , state='disabled', width= 30 )
        self.Class_dropdown.pack(side=tk.LEFT , anchor='w')
        self.Class_dropdown_var.trace_add('write', lambda *args: self.Add_Image_Options( self.selected_dir_1 ) )



        tk.Label(dropdown_frame , text= 'Image : ' ).pack(side = tk.LEFT , anchor= 'w')
        self.Image_dropdown = ttk.Combobox( dropdown_frame, textvariable = self.Image_dropdown_var , state='disabled' , width= 15 )
        self.Image_dropdown.pack(side=tk.LEFT , anchor='w')


        # Evaluation Options Frame
        eval_options_frame = tk.Frame(self)
        eval_options_frame.pack(fill='x', pady=(10, 0), padx=10)

        eval_options_frame_2 = tk.Frame(self)
        eval_options_frame_2.pack(fill='x', pady=(10, 0), padx=10)

        heatmap_button = tk.Button( eval_options_frame , state='disabled' ,  text='Heatmap' , command= lambda : self.Heatmap_Pressed( path = os.path.join( self.selected_dir_1.get() , self.Class_dropdown_var.get() , self.Image_dropdown_var.get() ))  )
        heatmap_button.pack(side='left', padx=(0, 5))

        heatmap_button = tk.Button( eval_options_frame , state='disabled', text='Heatmap (Random 100)' , command= lambda : self.Heatmap_Pressed( path = os.path.join( self.selected_dir_1.get() , self.Class_dropdown_var.get() ) , random_100 = True) )
        heatmap_button.pack(side='left', padx=(0, 5))

        correlation_button = tk.Button( eval_options_frame  ,state='disabled' ,  text='Correlation' , command= lambda : self.Correlation_Pressed( path = os.path.join( self.selected_dir_1.get() , self.Class_dropdown_var.get() , self.Image_dropdown_var.get() ))  )
        correlation_button.pack(side='left')


        correlation_button = tk.Button( eval_options_frame  ,  text='Metrics' , command= lambda : self.Mertrics_Pressed( path = os.path.join( self.selected_dir_1.get() , self.Class_dropdown_var.get() , self.Image_dropdown_var.get() ))  )
        correlation_button.pack(side='left')

        Single_Pred_Button = tk.Button( eval_options_frame ,state='disabled' ,  text='Single Prediction' , command= lambda : self.Prob_image_func( path = os.path.join( self.selected_dir_1.get() , self.Class_dropdown_var.get() , self.Image_dropdown_var.get() ))  )
        Single_Pred_Button.pack(side='left')

        Big_Pred_Button = tk.Button( eval_options_frame  ,state='disabled',  text='Class Probs' , command= lambda : self.Whole_Class_prediction( path = os.path.join( self.selected_dir_1.get() , self.Class_dropdown_var.get() ))  )
        Big_Pred_Button.pack(side='left')

        Deposite_button = tk.Button( eval_options_frame  ,  text='Image_to_Array' , command= lambda : self.Important_Regions( path = os.path.join( self.selected_dir_1.get()  ))  )
        Deposite_button.pack(side='left')
        
        Energy_Hist_button = tk.Button( eval_options_frame  ,state='disabled',  text='Energy_Hist' , command= lambda : self.Energy_Prediction_Hist( path = os.path.join( self.selected_dir_1.get()  ))  )
        Energy_Hist_button.pack(side='left')

        Pixel_Hist_button = tk.Button( eval_options_frame_2  ,state='disabled',  text='Pixel_Count_Hist' , command= lambda : self.Pixel_Count_Prediction_Hist( path = os.path.join( self.selected_dir_1.get()  ))  )
        Pixel_Hist_button.pack(side='left')

        Pixel_PDF = tk.Button( eval_options_frame_2  ,state='disabled',  text='Download_Pixel_PDF' , command= lambda : self.Pixel_Predictions_PDF( path = os.path.join( self.selected_dir_1.get()  ))  )
        Pixel_PDF.pack(side='left')

        FP_PDF = tk.Button( eval_options_frame_2  ,state='disabled',  text='False_Positve_NES' , command= lambda : self.Scattering_Softmax_PDF( path = os.path.join( self.selected_dir_1.get()  ))  )
        FP_PDF.pack(side='left')

        FP_2_Class_PDF = tk.Button( eval_options_frame_2 ,state='disabled' ,  text='Two_Class_Threshold' , command= lambda : self.Two_Class_Binary_Threshold_PDF( path = os.path.join( self.selected_dir_1.get()  ))  )
        FP_2_Class_PDF.pack(side='left')

        Lepton_Angle_PDF = tk.Button( eval_options_frame_2 ,state='disabled' ,  text='Lepton_Angle' , command= lambda : self.Two_Class_Lepton_Angle( path = os.path.join( self.selected_dir_1.get()  ))  )
        Lepton_Angle_PDF.pack(side='left')

        FP_Hist = tk.Button( eval_options_frame_2  ,state='disabled',  text='FP Hist' , command= lambda : self.False_Positive_Hist( path = os.path.join( self.selected_dir_1.get()  ))  )
        FP_Hist.pack(side='left')

        FP_PDF = tk.Button( eval_options_frame_2  ,state='disabled',  text='FP PDF' , command= lambda : self.False_Positive_PDF( path = os.path.join( self.selected_dir_1.get()  ))  )
        FP_PDF.pack(side='left')


        self.Figure_Canvas_Frame = tk.Frame( self )
        self.Figure_Canvas_Frame.pack( anchor='w' , side= tk.TOP , pady= 10 )


    def Add_Dir_Classes(self , Entry_Box):
        possible_classes = sorted ( os.listdir( Entry_Box.get() ) ) 
        self.Class_dropdown_var.set('')
        self.Class_dropdown.config(state='readonly' , values= possible_classes )

        return

    def Add_Image_Options(self, Entry_Box):
        path = os.path.join( Entry_Box.get()  , self.Class_dropdown_var.get() )
        self.Image_dropdown_var.set('')
        possible_images = sorted ( os.listdir( path) ) 

        # sorted_possible_images = sorted(possible_images, key=lambda x: int(x.split('_')[2].split('.')[0]))
        self.Image_dropdown.config(state='readonly' , values= possible_images)

    def Heatmap_Pressed(self , path  , random_100 = False):
        
        if random_100 == False:
            # Model_Evaluation_script.Evaluating_Model.Heatmap_func( self , path = path)
            # self.controller.Evaluating_Model.Heatmap_func( self , path = path)
            self.controller.Heat_Map_Class.Heatmap_func( self , path = path)

        else:
            # Model_Evaluation_script.Evaluating_Model.Heatmap_func( self , path = path , random_100 = True)
            # self.controller.Evaluating_Model.Heatmap_func( self , path = path , random_100 = True)
            self.controller.Heat_Map_Class.Heatmap_func( self , path = path , random_100 = True)

        return  

    def Prob_image_func(self, path ):
        
        print(path)

        # Model_Evaluation_script.Evaluating_Model.Plot_Prob_Single_Image_func( self , path = path)
        self.controller.Evaluating_Model.Plot_Prob_Single_Image_func( self , path = path)



    def Correlation_Pressed(self , path ):

        # Model_Evaluation_script.Evaluating_Model.Correlation_func( self)
        self.controller.Evaluating_Model.Correlation_func( self)

        return  
    

    def Mertrics_Pressed(self , path ):

        # Model_Evaluation_script.Evaluating_Model.Plot_metrics_func( self)
        self.controller.Evaluating_Model.Plot_metrics_func( self)

        return  

    def Whole_Class_prediction(self , path):

        # Model_Evaluation_script.Evaluating_Model.Whole_Class_Predictinos( self , path = path)
        self.controller.Evaluating_Model.Whole_Class_Predictinos( self , path = path)

        return

    def Important_Regions(self, path):

        print( path )
        # Model_Evaluation_script.Evaluating_Model.Detector_Important_Regions( self , path = path)
        self.controller.Evaluating_Model.Detector_Important_Regions( self , path = path)

        return


    def Energy_Prediction_Hist(self, path):

        # Model_Evaluation_script.Evaluating_Model.energy_prediction_distribution( self , path = path)
        self.controller.Evaluating_Model.energy_prediction_distribution( self , path = path)

        return


    def Pixel_Count_Prediction_Hist(self, path):

        # Model_Evaluation_script.Evaluating_Model.Active_Pixel_Count( self , path = path)
        self.controller.Evaluating_Model.Active_Pixel_Count( self , path = path)

        return


    def Pixel_Predictions_PDF(self, path):

        # Model_Evaluation_script.Evaluating_Model.Download_Pixel_Eval_Plots( self , path = path)
        self.controller.Evaluating_Model.Download_Pixel_Eval_Plots( self , path = path)

        return


    def Scattering_Softmax_PDF(self, path):

        # Model_Evaluation_script.Evaluating_Model.Scattering_False_Positve_Analysis( self , path = path)
        self.controller.Evaluating_Model.Scattering_False_Positve_Analysis( self , path = path)

        return


    def Two_Class_Binary_Threshold_PDF(self, path):

        # Model_Evaluation_script.Evaluating_Model.Two_Class_Scattering_False_Positve_Analysis( self , path = path)
        self.controller.Evaluating_Model.Two_Class_Scattering_False_Positve_Analysis( self , path = path)

        return

    def Two_Class_Lepton_Angle(self, path):

        # Model_Evaluation_script.Evaluating_Model.two_class_lepton_angel_PDF( self , path = path)
        self.controller.Evaluating_Model.two_class_lepton_angel_PDF( self , path = path)

        return

    def False_Positive_Hist(self, path):

        # Model_Evaluation_script.Evaluating_Model.FP_Positive_Prob_Hist( self , path = path)
        self.controller.Evaluating_Model.FP_Positive_Prob_Hist( self , path = path)

        return

    def False_Positive_PDF(self, path):

        # Model_Evaluation_script.Evaluating_Model.New_Comparison_Func( self , path = path)
        self.controller.Evaluating_Model.New_Comparison_Func( self , path = path)

        return
