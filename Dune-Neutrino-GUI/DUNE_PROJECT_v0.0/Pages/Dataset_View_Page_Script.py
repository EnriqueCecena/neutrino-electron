from Imports.common_imports import *



class Dataset_View_Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        title_frame = tk.Frame(self)
        title_frame.pack(anchor='w', pady=5 )
        tk.Button(title_frame, text="Back", command=lambda: controller.show_frame("StartPage")).pack(anchor='w', side=tk.LEFT, pady=2)
        tk.Label(title_frame, text="View Datasets", font=("Helvetica", 16)).pack(  anchor='w' , side=tk.LEFT, pady=10, padx=10,)


        tk.Button(self, text="View segments", command=lambda: controller.show_frame("View_Segments_Page")).pack( anchor='w')
        tk.Button(self, text="View mc_hdr", command=lambda: controller.show_frame("View_mc_hdr_Page") ).pack( anchor='w')
        tk.Button(self, text="View Trajectories", command=lambda: controller.show_frame("View_traj_Page")  ).pack( anchor='w' )
