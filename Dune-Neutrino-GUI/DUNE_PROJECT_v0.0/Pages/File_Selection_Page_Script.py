from Imports.common_imports import *


class File_Selection_Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.file_vars = []  # List to hold IntVar for each file Checkbutton

        # Title and Back Navigation
        page_title_frame = tk.Frame(self)
        page_title_frame.pack( anchor= 'w' , pady=(10, 30))

        tk.Button(page_title_frame , text = "back", command= lambda: controller.show_frame("Settings_Page") ).pack( padx=10 , anchor='w', side = tk.LEFT)
        tk.Label(page_title_frame, text="File Selection", font=("Helvetica", 16)).pack( padx=50 , anchor='w' , side = tk.LEFT)

        # Frame for Select/Deselect buttons
        button_frame = tk.Frame(self)
        button_frame.pack(pady=5, padx=10, anchor='w')

        # Select All Button
        select_all_button = tk.Button(button_frame, text="Select All", command=self.select_all)
        select_all_button.pack(side=tk.LEFT, padx=(0, 5))

        # Deselect All Button
        deselect_all_button = tk.Button(button_frame, text="Deselect All", command=self.deselect_all)
        deselect_all_button.pack(side=tk.LEFT)

        # Access Data_Directory via controller
        if hasattr(controller, 'Data_Directory') and controller.Data_Directory:
            try:
                # Assuming controller.File_Names is a list of filenames
                File_Names = controller.File_Names


                if File_Names:
                    tk.Label(self, text="Files:", font=("Helvetica", 12)).pack(pady=5, anchor='w', padx=10)
                    
                    # Create a scrollable frame
                    scrollable_frame = self.controller.ScrollableFrame(self)
                    # scrollable_frame = self.helpers.ScrollableFrame(self)

                    # scrollable_frame.pack(pady=5, padx=10, fill="both", expand=True)
                    scrollable_frame.pack(anchor='w' , pady=5, padx=10  )


                    # Add file names to the scrollable frame
                    for file in File_Names:
                        var = tk.IntVar()
                        file_check = tk.Checkbutton(
                            scrollable_frame.scrollable_frame,
                            text=file,
                            anchor="w",
                            variable=var
                        )
                        file_check.pack(fill="x", padx=10, pady=2)
                        
                        self.file_vars.append(var)
                else:
                    tk.Label(self, text="No files found in the directory.").pack(pady=5, padx=10, anchor='w')
            except FileNotFoundError:
                tk.Label(self, text="Data Directory not found.").pack(pady=5, padx=10, anchor='w')
            except Exception as e:
                tk.Label(self, text=f"Error: {e}").pack(pady=5, padx=10, anchor='w')
        else:
            tk.Label(self, text="No Data Directory provided.", fg="red").pack(pady=5, padx=10, anchor='w')

        # "Confirm Selection" button to update the Allowed_Files list
        confirm_button = tk.Button(self, text="Confirm Selection", command=self.confirm_selection)
        confirm_button.pack(pady=10, padx=10, anchor='w')



    def select_all(self):
        """Select all file checkboxes."""
        for var in self.file_vars:
            var.set(1)

    def deselect_all(self):
        """Deselect all file checkboxes."""
        for var in self.file_vars:
            var.set(0)

    def get_selected_files(self):
        """Retrieve the list of selected files."""
        selected_files = [file for file, var in zip(self.controller.File_Names, self.file_vars) if var.get()]
        return selected_files

    def confirm_selection(self):
        """Update the Allowed_Files in the controller based on user selection."""

        selected = self.get_selected_files()
        if not selected:
            # If no specific files selected, show all files
            self.controller.Allowed_Files = sorted(self.controller.File_Names)
        else:
            self.controller.Allowed_Files = sorted(selected) # Update the Allowed_Files list


        # Reinitialize frames that rely on Allowed_Files
        self.controller.reinitialize_frame("View_Segments_Page")
        self.controller.reinitialize_frame("View_mc_hdr_Page")
        self.controller.reinitialize_frame("View_traj_Page")

        self.controller.reinitialize_frame("Figure_Creation_Page")
        self.controller.reinitialize_frame("Scatter_Creation_Page")

        self.controller.reinitialize_frame("Custom_Figure_Page")
        self.controller.reinitialize_frame("Create_Dataset_Page")
        self.controller.reinitialize_frame("Advanced_Evaluation_Page")

        self.controller.show_frame(File_Selection_Page)


        # print("Allowed Files Updated:", self.controller.Allowed_Files)
        print("Allowed Files Updated:")

