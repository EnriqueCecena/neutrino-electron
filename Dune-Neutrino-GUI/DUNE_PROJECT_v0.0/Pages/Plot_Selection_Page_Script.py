from Imports.common_imports import *
from Helpers.Frame_Manager_Script import Frame_Manager


class Plot_Selection_Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        self.manager = Frame_Manager(self)

        title_frame = tk.Frame(self)
        title_frame.pack(anchor='w', pady=5 )
        tk.Button(title_frame, text="Back", command=lambda: controller.show_frame("StartPage")).pack(anchor='w', side=tk.LEFT, pady=2)
        tk.Label(title_frame, text="Plot Figures", font=("Helvetica", 16)).pack(  anchor='w' , side=tk.LEFT, pady=10, padx=10,)

        
        tk.Button(self, text="Create Scatter Plot",
                    command=lambda: controller.show_frame("Scatter_Creation_Page")).pack(anchor='w', pady=2)

                    

        tk.Button(self, text="Create Line Plot",
                    command=lambda: controller.show_frame("Line_Creation_Page")).pack(anchor='w', pady=2)

        

        tk.Button(self, text="Create Histogram",
                    # command=lambda: self.manager.set_plot_type('hist')).pack(anchor='w', pady=2)
                    command=lambda: controller.show_frame("Hist_Creation_Page")).pack(anchor='w', pady=2)
        

        tk.Button(self, text="Custom Plot",
                    # command=lambda: self.manager.set_plot_type('custom') ).pack(anchor='w', pady=2)    
                    command=lambda: controller.show_frame("Custom_Figure_Page")).pack(anchor='w', pady=2)

        
        tk.Button(self, text="Event Player",
                    command=lambda: controller.show_frame('Setup_Animation_Page') ).pack(anchor='w', pady=2)    

        tk.Button(self, text='Plot Settings',
                    command=lambda: controller.show_frame("Dataset_View_Page") , state= 'disabled').pack(anchor='w', pady=2)
