from Imports.common_imports import *

# from Helpers.Frame_Manager_Script import Frame_Manager
# from Helpers.Scrollable_Frame_Script import ScrollableFrame
# from Backend.Generic_Plot_script import Generic_Plot

class Frame_Manager():

    def __init__(self, frame):
        self.frame = frame
        self.controller = frame.controller  # Access the controller from the frame

    # def update_dropdown(self):
    #     """Refresh the dropdown menu with the latest Allowed_Files."""
    #     menu = self.frame.files_drop_down["menu"]
    #     menu.delete(0, "end")  # Clear existing options

    def refresh_content(self):
        """Refresh dropdown and DataFrame display when the frame is shown."""
        self.update_dropdown()
        self.Display_DF_in_Frame(self.frame.selected_file.get())

    def refresh_frame(self):
        """Clear all widgets in the display_frame."""
        for widget in self.frame.display_frame.winfo_children():
            widget.destroy()

    def Display_DF_in_Frame(self, dropdown_file_name, h5_data_name=str):
        """Display the DataFrame based on the selected file."""
        self.refresh_frame()  # Clear previous content

        # Display the selected file name
        tk.Label( self.frame.display_frame, text=f"Selected File: {dropdown_file_name}", font=("Helvetica", 12) ).pack(anchor='w')

        # Construct the file path
        File_path = os.path.join(  self.controller.Data_Directory, dropdown_file_name ) 

        try:
            if h5_data_name == 'mc_hdr':
                h5_DataFrame = pd.DataFrame.from_records( h5py.File(File_path)[str(h5_data_name)], columns=np.dtype( h5py.File(File_path)[str(h5_data_name)] ).names )
            else:
                h5_DataFrame = pd.DataFrame(   h5py.File(File_path)[str(h5_data_name)][()] )

            self.frame.Event_IDs = np.unique(h5_DataFrame['event_id'])

            if not self.frame.Event_IDs.size:
                raise ValueError("No Event IDs found in the selected file.")

            # Ensure Event_ID_selection is within bounds
            self.frame.Event_ID_selection = max( 0, min( self.frame.Event_ID_selection,len(self.frame.Event_IDs) - 1  ) )

            # Get the current Event ID
            current_event_id = self.frame.Event_IDs[self.frame.Event_ID_selection]
            h5_DataFrame_event = h5_DataFrame[ h5_DataFrame['event_id'] == current_event_id ]
            os.system('cls||clear')
            h5_DataFrame_event = pd.DataFrame(h5_DataFrame_event)

            # Update Event Counter Label
            self.frame.event_counter_label.config(
                text=f"Event {self.frame.Event_ID_selection + 1} of {len(self.frame.Event_IDs)}"
            )

            # Update navigation buttons' state
            self.update_navigation_buttons()

            # Create a Treeview widget to display the DataFrame
            self.frame.tree = ttk.Treeview(
                self.frame.display_frame,
                columns=list(h5_DataFrame_event.columns),
                show="headings"
            )

            # Configure Treeview Style
            style = ttk.Style()
            style.configure("Treeview", font=("Helvetica", 7))
            style.configure("Treeview.Heading", font=("Helvetica", 8, "bold"))

            self.frame.tree.pack(fill="both", expand=True)

            # define a callback to resize all columns equally whenever the Treeview resizes
            def _stretch_cols(event):
                total_width = event.widget.winfo_width()
                cols = event.widget["columns"]
                if cols:
                    w = total_width // len(cols)
                    for c in cols:
                        event.widget.column(c, width=w)

            # bind the callback to the Treeview's <Configure> event
            self.frame.tree.bind("<Configure>", _stretch_cols)

            # re-configure headings & enable stretching (no fixed width needed here)
            for col in h5_DataFrame_event.columns:
                self.frame.tree.heading(col, text=col)
                self.frame.tree.column(col, anchor="center", stretch=True)

            # # Define columns and headings
            # for col in h5_DataFrame_event.columns:
            #     self.frame.tree.heading(col, text=col)
            #     self.frame.tree.column(col, width=52, anchor="center" , stretch= True )

            # Insert DataFrame rows into the Treeview
            for _, row in h5_DataFrame_event.iterrows():
                self.frame.tree.insert("", "end", values=list(row))

            # Add a vertical scrollbar to the Treeview
            self.frame.scrollbar = ttk.Scrollbar( self.frame.display_frame, orient="vertical", command=self.frame.tree.yview )
            self.frame.tree.configure(yscrollcommand=self.frame.scrollbar.set)
            self.frame.scrollbar.pack(side="right", fill="y")

            # Pack the Treeview
            self.frame.tree.pack(fill="both", expand=True)

        except Exception as e:
            tk.Label(  self.frame.display_frame, text=f"Error loading file: {e}", fg="red" ).pack(anchor='w')

    def go_back(self):
        """Navigate to the previous event."""
        if self.frame.Event_ID_selection > 0:
            self.frame.Event_ID_selection -= 1
            self.Display_DF_in_Frame(self.frame.selected_file.get())

    def go_next(self):
        """Navigate to the next event."""
        if self.frame.Event_ID_selection < len(self.frame.Event_IDs) - 1:
            self.frame.Event_ID_selection += 1
            self.Display_DF_in_Frame(self.frame.selected_file.get())

    def update_navigation_buttons(self):
        """Enable or disable navigation buttons based on the current event selection."""
        if self.frame.Event_ID_selection <= 0:
            self.frame.back_button.config(state=tk.DISABLED)
        else:
            self.frame.back_button.config(state=tk.NORMAL)

        if self.frame.Event_ID_selection >= len(self.frame.Event_IDs) - 1:
            self.frame.next_button.config(state=tk.DISABLED)
        else:
            self.frame.next_button.config(state=tk.NORMAL)

    def on_file_selected(self, h5_data_name):
        """Callback triggered when a new file is selected from the dropdown."""
        selected_file = self.frame.selected_file.get()
        self.frame.Event_ID_selection = 0
        self.Display_DF_in_Frame(selected_file, h5_data_name)

    def update_dropdown(self):
        """Refresh the dropdown menu with the latest Allowed_Files."""
        menu = self.frame.files_drop_down["menu"]
        menu.delete(0, "end")

        if hasattr(self.controller, 'Allowed_Files') and self.controller.Allowed_Files:
            files = self.controller.Allowed_Files
        elif hasattr(self.controller, 'File_Names') and self.controller.File_Names:
            files = self.controller.File_Names
        else:
            files = ["No Files Available"]

        for file in files:
            menu.add_command( label=file, command=lambda value=file: self.frame.selected_file.set(value) )

        if files and files[0] != "No Files Available":
            self.frame.selected_file.set(files[0])
        else:
            self.frame.selected_file.set("No Files Available")

    def select_directory_window(self, Text_Box, select_class_setting_button=None):
        directory_path = tk.filedialog.askdirectory(  initialdir=str(os.getcwd()), title="Select a Directory" )
        if directory_path:
            Text_Box.config(state='normal')
            Text_Box.delete(0, tk.END)
            Text_Box.insert(0, directory_path)
            Text_Box.config(state='disabled')

            if select_class_setting_button is not None:
                select_class_setting_button.config(state='normal')

            self.controller.selected_directory = directory_path

            if str(self.__class__.__name__) == 'Model_Training_Page':
                training_class_page = self.controller.frames[ Advance_Class_Selection_Page ]
                training_class_page.Update_Page_With_Class()
        else:
            Text_Box.config(state='normal')
            Text_Box.delete("1.0", tk.END)
            Text_Box.insert("1.0", directory_path)
            Text_Box.config(state='disabled')


    def setup_process(self):
        if not self.controller.running:
            try:
                self.frame.progress_value = 0
                self.frame.progress['value'] = 0
                self.frame.progress.config(maximum=100)
                self.controller.running = True
            except:
                pass

            if str(self.__class__.__name__) == 'Create_Dataset_Page':
                self.frame.Create_Dataset_Button.config(state='disabled')
                self.check_progress()
                threading.Thread( target=self.frame.Create_ML_Dataset_2 ).start()

            elif str(self.__class__.__name__) == 'Model_Training_Page':
                self.controller.running = True
                # threading.Thread( target=lambda: self.controller.Model_Training.Train_Model(self.frame)  ).start()
                threading.Thread( target=lambda: self.controller.Model_Training.Train_Model(self)  ).start()

            else:
                self.check_progress()
                # Frame_Manager.check_progress(self )
                # threading.Thread( target=Pixel_Array_Script.Use_Pixel_Array.plot, args=(self.frame,) ).start()
                print( dir(self.controller) ) 
                threading.Thread( target=self.controller.Use_Pixel_Array.plot, args=(self.frame,) ).start()

    def cancel_process(self):
        if self.controller.running:
            try:
                self.frame.progress['value'] = 100
            except:
                pass
            self.controller.running = False
            print('Cancelled', self.controller.running)

    def check_progress(self):
        self.frame.progress['value'] = self.frame.progress_value
        self.frame.progress_label.config( text=f"{self.frame.progress_value:.2f}%" )
        if self.controller.running:
            self.frame.after(100, self.check_progress)




    # def set_plot_type(self, plot_type):
    #     self.plot_type = plot_type

    #     if self.plot_type == 'custom':

    #         self.reinitialize_frame(Custom_Figure_Page)  # Re-initialize the frame
    #         self.show_frame(Custom_Figure_Page)          # Show the re-initialized frame

        
    #     else:
    #         self.controller.reinitialize_frame("Figure_Creation_Page")  # Re-initialize the frame
    #         self.controller.show_frame("Figure_Creation_Page")          # Show the re-initialized frame

    # def set_plot_type(self, plot_type):
    #     self.controller.plot_type = plot_type

        # if plot_type == 'custom':
        #     # use the class object, not its name
        #     CustomPage = self.controller.Custom_Figure_Page
        #     self.controller.reinitialize_frame(CustomPage)
        #     self.controller.show_frame(CustomPage)
        # else:
        #     print(dir(self.controller))
        #     FigurePage = self.controller.Figure_Creation_Page
        #     self.controller.reinitialize_frame(FigurePage)
        #     self.controller.show_frame(FigurePage)


    def set_plot_type(self, plot_type):
        self.controller.plot_type = plot_type

        # pick the class, not the instance
        if plot_type == 'custom':
            PageClass = self.controller.Custom_Figure_Page
        else:
            PageClass = self.controller.Figure_Creation_Page

        # destroy old, re‐make new, then raise it
        self.controller.reinitialize_frame(PageClass)
        self.controller.show_frame(PageClass)