from Imports.common_imports import *

class Show_Confusion_Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller


        page_title_frame = tk.Frame(self)
        page_title_frame.pack( anchor= 'w' , pady=10 ,fill='x')

        # tk.Button(page_title_frame, text="Back", command = lambda:  self.exit_button_event() ).pack(anchor='w', padx=10 , side= tk.LEFT)
        tk.Button(page_title_frame, text="Back", command = lambda:  self.controller.show_frame("Monitor_Training_Page") ).pack(anchor='w', padx=10 , side= tk.LEFT)

        tk.Label(page_title_frame, text="Confusion Matrix ", font=("Helvetica", 16)).pack( padx=50, anchor='w' , side = tk.LEFT)

        # 'Full' button to toggle full-screen
        self.full_button = tk.Button( page_title_frame, text='Full', command= lambda: self.controller.attributes("-fullscreen" , not self.controller.attributes("-fullscreen")  ) )
        self.full_button.pack(anchor='w' , side=tk.LEFT)


        self.Confusion_Canvas_Frame = tk.Frame( self )
        self.Confusion_Canvas_Frame.pack( anchor='w' , side= tk.TOP , pady= 10 )

        self.Confusion_Matrix = tk.Frame( self.Confusion_Canvas_Frame )
        self.Confusion_Matrix.pack( anchor='w' , side= tk.TOP , pady= 10 )


        self.confusion_fig_ax = None
        self.confusion_fig_canvas = None