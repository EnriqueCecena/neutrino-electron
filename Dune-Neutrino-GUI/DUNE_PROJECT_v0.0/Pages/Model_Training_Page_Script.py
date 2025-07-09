from Imports.common_imports import *
class Model_Training_Page(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        # self.loss_fn = SparseCategoricalFocalLoss( gamma=2.0 )

        # self.optimizer = tf.keras.optimizers.Adam( learning_rate  = 0.00001 )
        self.controller = controller


        page_title_frame = tk.Frame(self)
        page_title_frame.pack( anchor= 'w' , pady=10)


        tk.Button(page_title_frame, text="Back", command=lambda: controller.show_frame("Training_And_Eval_Options_Page") ).pack(anchor='w', padx=10 , side= tk.LEFT)
        tk.Label(page_title_frame, text="Model Training ", font=("Helvetica", 16)).pack( padx=50, anchor='w' , side = tk.LEFT)


        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        dir_select_frame = tk.Frame(self)
        # dir_select_frame.pack(anchor='w', pady=(10, 50))
        dir_select_frame.pack(anchor='w', pady=10)

        self.selected_dir = tk.StringVar()

        text_box = tk.Entry(  dir_select_frame  , textvariable=self.selected_dir, bg='black', fg='white', font=("Arial", 12),  width=90, state='disabled')
        text_box.pack(anchor='w', side=tk.LEFT, padx=10, pady=10)

        select_dir_button = tk.Button(  dir_select_frame,  text="Select Directory (1)", command=lambda: self.select_directory_window(text_box))
        select_dir_button.pack(anchor='w', side=tk.LEFT)

        dir_select_frame_2 = tk.Frame(self)
        # dir_select_frame_ZX.pack(anchor='w', pady=(10, 50))
        dir_select_frame_2.pack(anchor='w', pady=10)

        self.selected_dir_2 = tk.StringVar()

        text_box_2 = tk.Entry(  dir_select_frame_2  , textvariable=self.selected_dir_2, bg='black', fg='white', font=("Arial", 12),  width=90, state='disabled')
        text_box_2.pack(anchor='w', side=tk.LEFT, padx=10, pady=10)

        select_dir_button_2 = tk.Button(  dir_select_frame_2,  text="Select Directory (2)", command=lambda: self.select_directory_window(text_box_2 , False))
        select_dir_button_2.pack(anchor='w', side=tk.LEFT)

        # tk.Button( dir_select_frame_2 , text= "touch me" , command = lambda: print( len(self.controller.model.inputs) ) ).pack(anchor='w', side=tk.LEFT)



        dir_select_frame_3 = tk.Frame(self)
        dir_select_frame_3.pack(anchor='w', pady=10)

        self.selected_dir_3 = tk.StringVar()

        text_box_3 = tk.Entry(  dir_select_frame_3  , textvariable=self.selected_dir_3, bg='black', fg='white', font=("Arial", 12),  width=90, state='disabled')
        text_box_3.pack(anchor='w', side=tk.LEFT, padx=10, pady=10)

        select_dir_button_3 = tk.Button(  dir_select_frame_3,  text="Select Directory (3)", command=lambda: self.select_directory_window(text_box_3 , False))
        select_dir_button_3.pack(anchor='w', side=tk.LEFT)


        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


        # self.class_setting_button = tk.Button( dir_select_frame  , text='Class Settings' , command=lambda: controller.show_frame(Advance_Class_Selection_Page), state='disabled' )
        self.class_setting_button = tk.Button( dir_select_frame  , text='Class Settings' , command=lambda:( self.controller.attributes("-fullscreen", True) , self.controller.show_frame("Advance_Class_Selection_Page") ) , state='disabled' )

        self.class_setting_button.pack(anchor='w', side=tk.LEFT)

        train_split_params_frame = tk.Frame(self)
        train_split_params_frame.pack( anchor='w' , pady = (10 , 20) )

        self.train_size_text_selected = tk.IntVar()
        tk.Label(train_split_params_frame , text = "train_size :" ).pack(anchor='w' , side = tk.LEFT )
        train_size_text_box = tk.Entry( train_split_params_frame , textvariable= self.train_size_text_selected  , bg = 'black', fg='white', font=("Arial", 12), state = 'normal' )
        train_size_text_box.pack( anchor='w', side = tk.LEFT  )
        train_size_text_box.delete(0, tk.END)
        train_size_text_box.insert(0,40)
        

        self.val_size_text_selected = tk.IntVar()
        tk.Label(train_split_params_frame , text = "val_size :" ).pack(anchor='w' , side = tk.LEFT )
        val_size_text_box = tk.Entry( train_split_params_frame , textvariable= self.val_size_text_selected, bg='black',  fg='white', font=("Arial", 12), state = 'normal' )
        val_size_text_box.pack( anchor='w' ,side = tk.LEFT  )
        val_size_text_box.delete(0, tk.END)
        val_size_text_box.insert(0,40)


        self.test_size_text_selected = tk.IntVar()
        tk.Label(train_split_params_frame , text = "test_size :" ).pack(anchor='w' , side = tk.LEFT )
        test_size_text_box = tk.Entry( train_split_params_frame, textvariable= self.test_size_text_selected,  bg='black',  fg='white', font=("Arial", 12), state = 'normal')
        test_size_text_box.pack( anchor='w' ,side = tk.LEFT  )
        test_size_text_box.delete(0, tk.END)
        test_size_text_box.insert(0,20)

        Model_Training_Params_Frame = tk.Frame(self)
        Model_Training_Params_Frame.pack( anchor='w' , pady = (10 , 20) )

        self.Epoch_size_text_selected = tk.IntVar()
        tk.Label(Model_Training_Params_Frame , text = "Num Epoches :" ).pack(anchor='w' , side = tk.LEFT )
        Epoch_size_text_box = tk.Entry( Model_Training_Params_Frame , textvariable= self.Epoch_size_text_selected  , bg = 'black', fg='white', font=("Arial", 12), state = 'normal' )
        Epoch_size_text_box.pack( anchor='w', side = tk.LEFT  )
        Epoch_size_text_box.delete(0, tk.END)
        Epoch_size_text_box.insert(0,100)


        self.Batch_size_text_selected = tk.IntVar()
        tk.Label(Model_Training_Params_Frame , text = "batch_size :" ).pack(anchor='w' , side = tk.LEFT )
        Batch_size_text_box = tk.Entry( Model_Training_Params_Frame , textvariable= self.Batch_size_text_selected  , bg = 'black', fg='white', font=("Arial", 12), state = 'normal' )
        Batch_size_text_box.pack( anchor='w', side = tk.LEFT  )
        Batch_size_text_box.delete(0, tk.END)
        Batch_size_text_box.insert(0,25)




        Control_Training_Frame = tk.Frame(self)
        Control_Training_Frame.pack( anchor='w' , pady = (10 , 20) )
        # tk.Button(Control_Training_Frame, text='Train', command=lambda: self.Train_Model()).pack(anchor='w' , side= tk.LEFT)
        # tk.Button(Control_Training_Frame, text='Train', command=lambda: Frame_Manager.setup_process(self) ).pack(anchor='w' , side= tk.LEFT)
        # tk.Button(Control_Training_Frame, text='Stop', command=lambda: Frame_Manager.cancel_process(self) ).pack(anchor='w' , side= tk.LEFT)


        tk.Button(Control_Training_Frame, text='Train', command=lambda: self.controller.Frame_Manager.setup_process(self) ).pack(anchor='w' , side= tk.LEFT)
        # tk.Button(Control_Training_Frame, text='Stop', command=lambda: self.controller.cancel_process(self) ).pack(anchor='w' , side= tk.LEFT)
        tk.Button(Control_Training_Frame, text='Stop', command=lambda: self.controller.Frame_Manager.cancel_process(self) ).pack(anchor='w' , side= tk.LEFT)

        # tk.Button(Control_Training_Frame, text='Train', command=lambda: Model_Training_script.Model_Training.Train_Model(self) ).pack(anchor='w' , side= tk.LEFT)


        # Model_Training_script.Model_Training.Train_Model


        # tk.Button(Control_Training_Frame, text='Monitor', command= lambda : controller.show_frame(Monitor_Training_Page) ).pack(anchor='w' ,  side= tk.LEFT)
        # tk.Button(Control_Training_Frame, text='Monitor', command= lambda :  self.Monitor_Launch_func() ).pack(anchor='w' ,  side= tk.LEFT)
        tk.Button(Control_Training_Frame, text='Monitor', command= lambda : (self.controller.attributes("-fullscreen", True), self.controller.show_frame("Monitor_Training_Page") ) ).pack(anchor='w' ,  side= tk.LEFT)
        # tk.Button(Control_Training_Frame, text='Monitor', command = self.Monitor_Launch_func ).pack(anchor='w' ,  side= tk.LEFT)

        tk.Button(Control_Training_Frame, text='Save Model', command= lambda :  self.Save_Trained_Model() ).pack(anchor='w' ,  side= tk.LEFT)
        tk.Button(Control_Training_Frame, text='Test Batch Sizes', command=lambda: self.Test_Batch_Sizes() ).pack(anchor='w' , side= tk.LEFT)



    def select_directory_window(self, text_box , page_update = True):
        directory_path = tk.filedialog.askdirectory( initialdir=os.getcwd(), title="Select a Directory" )
        if directory_path:
            text_box.config(state='normal')
            text_box.delete(0, tk.END)
            text_box.insert(0, directory_path)
            text_box.config(state='disabled')

            if page_update:
                self.class_setting_button.config(state='normal')

                self.controller.selected_directory = directory_path

                # training_class_page = self.controller.frames["Advance_Class_Selection_Page"]
                training_class_page = self.controller.Advance_Class_Selection_Page
                training_class_page.Update_Page_With_Class(directory_path)



    def Show_Advanced_Class_Page(self):
        self.controller.show_frame("Advance_Class_Selection_Page")
        Monitor_Training_Page.toggle_fullscreen(self)

    def Monitor_Launch_func(self):

        self.controller.enter_fullscreen_event()
        self.controller.show_frame("Monitor_Training_Page")
        return
    
    def Save_Trained_Model(self):


        model_path_input = tk.filedialog.asksaveasfilename( title="Save file as...", defaultextension=".keras",  filetypes=[("Text Files", "*.keras"), ("All Files", "*.*")] , initialdir=os.getcwd()  )

        if model_path_input:
            print("File will be saved to:", model_path_input)
        else:
            print("Save cancelled.")

        # try:
        self.controller.model.save(model_path_input)

        # except:
        #     print("ERROR SAVING MODEL")

        dict_path = os.path.dirname(model_path_input)
        Model_hist_file = open(f"{dict_path}/Test.txt", "w")
        try:
            Model_hist_file.write( str(self.controller.model.history_dict) )
        except:
            Model_hist_file.write("Error But we go on")

        Model_hist_file.close()


    def Test_Batch_Sizes(self):


        self.output_dir = tk.filedialog.askdirectory( title="Select Output Directory for Batch Size Test Results" , initialdir=os.getcwd() )
        if not self.output_dir:
            print("No output directory selected. Aborting Test_Batch_Sizes.")
            return


        self.batch_sizes_to_test = [10, 25, 50, 100, 250]
        self.current_batch_index = 0


        self.original_model = self.controller.model
        self.original_batch = self.Batch_size_text_selected.get()


        self.run_next_batch()


    def run_next_batch(self):


        if self.current_batch_index >= len(self.batch_sizes_to_test):
            print("All batch size tests completed.")
            self.Batch_size_text_selected.set(self.original_batch)
            self.controller.model = self.original_model
            return


        bs = self.batch_sizes_to_test[self.current_batch_index]
        print(f"Starting training for batch size {bs}...")


        self.Batch_size_text_selected.set(bs)


        new_model = tf.keras.models.clone_model(self.original_model)

        new_model.compile(
            optimizer=Adam(learning_rate=self.controller.model_learning_rate),
            loss='SparseCategoricalCrossentropy',
            # loss='SparseCategoricalFocalLoss',
            metrics=['accuracy']
        )

        # model.compile(
        #     optimizer=Adam(learning_rate=self.controller.model_learning_rate),
        #     loss=SparseCategoricalCrossentropy(from_logits=True),              # or use the string 'sparse_categorical_crossentropy'
        #     metrics=[Precision(name='precision')]
        # )

        self.controller.model = new_model

        # Start training using existing process.
        self.controller.Frame_Manager.setup_process(self)
        # self.manager.setup_process()

        # Periodically check whether training has finished.
        self.check_training_finished()


    def check_training_finished(self):
        # Use after() so that the UI remains responsive.
        if self.controller.running:
            self.after(1000, self.check_training_finished)
        else:
            # When training is finished, first save the training history and figures.
            bs = self.batch_sizes_to_test[self.current_batch_index]
            self.save_results_for_batch(bs)
            # Move on to the next batch size after a brief pause.
            self.current_batch_index += 1
            self.after(1000, self.run_next_batch)


    def save_results_for_batch(self, batch_size):

        # Create a nested directory for this batch size.
        batch_dir = os.path.join(self.output_dir, f"Batch_size_{batch_size}")
        os.makedirs(batch_dir, exist_ok=True)

        # Save the training history.
        history = self.controller.model.history_dict
        history_file = os.path.join(batch_dir, "history.txt")
        try:
            with open(history_file, "w") as f:
                f.write(str(history))
            print(f"History for batch size {batch_size} saved to {history_file}")
        except Exception as e:
            print(f"Error saving history for batch size {batch_size}: {e}")

        # Save metric plots from the Monitor Training page.
        monitor_page = self.controller.frames.get("Monitor_Training_Page")
        if monitor_page is not None:
            # Example: Save the class accuracy plot.
            if hasattr(monitor_page, "class_accuracy_ax"):
                try:
                    monitor_page.class_accuracy_ax.figure.savefig(
                        os.path.join(batch_dir, "class_accuracy.png")
                    )
                    print("Class accuracy plot saved.")
                except Exception as e:
                    print("Error saving class accuracy plot:", e)
            # Save class precision plot.
            if hasattr(monitor_page, "class_precision_ax"):
                try:
                    monitor_page.class_precision_ax.figure.savefig(
                        os.path.join(batch_dir, "class_precision.png")
                    )
                    print("Class precision plot saved.")
                except Exception as e:
                    print("Error saving class precision plot:", e)
            # Save class recall plot.
            if hasattr(monitor_page, "class_recall_ax"):
                try:
                    monitor_page.class_recall_ax.figure.savefig(
                        os.path.join(batch_dir, "class_recall.png")
                    )
                    print("Class recall plot saved.")
                except Exception as e:
                    print("Error saving class recall plot:", e)
            # Save loss plot if available.
            if hasattr(monitor_page, "loss_fig_ax"):
                try:
                    monitor_page.loss_fig_ax.figure.savefig(
                        os.path.join(batch_dir, "loss.png")
                    )
                    print("Loss plot saved.")
                except Exception as e:
                    print("Error saving loss plot:", e)
        else:
            print("Monitor_Training_Page not found.")

        # Save the confusion matrix from Show_Confusion_Page.
        confusion_page = self.controller.frames.get("Show_Confusion_Page")
        if confusion_page is not None:
            if hasattr(confusion_page, "confusion_fig_ax"):
                try:
                    confusion_page.confusion_fig_ax.figure.savefig(
                        os.path.join(batch_dir, "confusion_matrix.png")
                    )
                    print("Confusion matrix image saved.")
                except Exception as e:
                    print("Error saving confusion matrix image:", e)
        else:
            print("Show_Confusion_Page not found.")

        

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#