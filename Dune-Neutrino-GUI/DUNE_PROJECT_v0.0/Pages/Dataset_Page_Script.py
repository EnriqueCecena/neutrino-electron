from Imports.common_imports import *

from Helpers.Frame_Manager_Script import Frame_Manager


class Dataset_Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)


        title_frame = tk.Frame(self)
        title_frame.pack(anchor='w', pady=5 )
        tk.Button(title_frame, text="Back", command=lambda: controller.show_frame("StartPage")).pack(anchor='w', side=tk.LEFT, pady=2)
        tk.Label(title_frame, text="Dataset Creation", font=("Helvetica", 16)).pack(  anchor='w' , side=tk.LEFT, pady=10, padx=10,)


        tk.Button(self, text="Create",
                  command=lambda: controller.show_frame("Create_Dataset_Page")).pack( anchor='w')

        tk.Button(self, text="Load",
                  command=lambda: controller.show_frame("Load_Dataset_Page")).pack(anchor='w')

