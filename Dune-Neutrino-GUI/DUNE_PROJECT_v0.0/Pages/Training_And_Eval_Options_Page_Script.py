from Imports.common_imports import *
from Helpers.Frame_Manager_Script import Frame_Manager

class Training_And_Eval_Options_Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        page_title_frame = tk.Frame(self)
        page_title_frame.pack( anchor= 'w' , pady=10)

        tk.Button(page_title_frame , text = "back", command= lambda: controller.show_frame("StartPage") ).pack( padx=10 , anchor='w', side = tk.LEFT)
        tk.Label(page_title_frame, text="Select_Option", font=("Helvetica", 16)).pack( padx=50 , anchor='w' , side = tk.LEFT)

        tk.Button(self, text="Architecture Configuration",
                  command=lambda: ( self.controller.attributes("-fullscreen", True )  , controller.show_frame("Model_Architecture_Page") ) ).pack( padx = 10 ,pady=(20,0) ,anchor='w')
        


        tk.Button(self, text="Train Model",
                  command=lambda: controller.show_frame("Model_Training_Page")).pack(  padx=10  ,anchor='w')

        tk.Button(self, text="Evaluate Model",
                  command=lambda: controller.show_frame("Evaluate_Model_Page")).pack( padx=10   ,anchor='w')

        # tk.Button(self, text="Advanced Evaluation",
        #           command=lambda: controller.show_frame("Advanced_Evaluation_Page")).pack( padx=10   ,anchor='w')

