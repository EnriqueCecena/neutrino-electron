from Imports.common_imports import *

class Settings_Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        page_title_frame = tk.Frame(self)
        page_title_frame.pack( anchor= 'w' , pady=(10, 30))

        tk.Button(page_title_frame , text = "back", command= lambda: controller.show_frame("StartPage") ).pack( padx=10 , anchor='w', side = tk.LEFT)
        tk.Label(page_title_frame, text="Settings", font=("Helvetica", 16)).pack( padx=50 , anchor='w' , side = tk.LEFT)

        tk.Button(self, text='Select Files', command= lambda: controller.show_frame("File_Selection_Page")).pack( anchor='w' , padx=10  )
        

        tk.Button(self, text='Cleaning Method', command= lambda: controller.show_frame("Cleaning_Method_Select_Page") , state='disabled' ).pack( anchor='w' , padx=10 )
        