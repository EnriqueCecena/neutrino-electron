from Imports.common_imports import *

from Helpers.Frame_Manager_Script import Frame_Manager

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#


class Load_Dataset_Page( tk.Frame  ):

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller


        page_title_frame = tk.Frame(self)
        page_title_frame.pack( anchor= 'w' , pady=10)

        tk.Button(page_title_frame, text="Back", command=lambda: controller.show_frame("Training_And_Eval_Options_Page")).pack(anchor='w', padx=10 , side= tk.LEFT)
        tk.Label(page_title_frame, text="Load Image ", font=("Helvetica", 16)).pack( padx=50, anchor='w' , side = tk.LEFT)

        dir_select_frame = tk.Frame(self)
        dir_select_frame.pack( anchor='w' , pady = (10 , 50) )

        self.selected_dir = tk.StringVar()

        self.entry_box = tk.Entry(dir_select_frame, textvariable=self.selected_dir ,  bg='black',  fg='white' , width=80 , state='readonly')
        self.entry_box.pack(anchor='w', side=tk.LEFT, padx=10, pady=10)

        self.selected_dir.trace_add('write', lambda *args: self.Class_Select_Dropdown_Func() )

        select_dir_buttnon = tk.Button( dir_select_frame , text= "Dataset Directory" , command= lambda: Frame_Manager.select_directory_window(self , self.entry_box))
        select_dir_buttnon.pack( anchor='w' , side= tk.LEFT )

        self.View_Dataset_Control_Frame = tk.Frame(self)
        self.View_Dataset_Control_Frame.pack( anchor= 'w' , pady=10)

        tk.Label(self.View_Dataset_Control_Frame, text="Class :" ).pack(anchor='w', side=tk.LEFT) 


        self.Class_selected = tk.StringVar()
        self.Class_dropdown = ttk.Combobox( self.View_Dataset_Control_Frame , textvariable= self.Class_selected , state= 'readonly'  )
        self.Class_dropdown.pack(anchor='w', side=tk.LEFT) 

        self.Class_selected.trace_add('write', lambda *args: self.Image_Select_Dropdown_Func() )

        tk.Label(self.View_Dataset_Control_Frame, text="Image :" ).pack(anchor='w', side=tk.LEFT) 

        self.Image_Selected = tk.StringVar()
        self.Image_dropdown = ttk.Combobox( self.View_Dataset_Control_Frame , textvariable= self.Image_Selected , state= 'readonly'  )
        self.Image_dropdown.pack(anchor='w', side=tk.LEFT) 

        self.Image_Selected.trace_add( 'write' , lambda *args: self.Load_Image()  )

        self.Image_Frame = tk.Frame(self)
        self.Image_Frame.pack(anchor= 'w' , pady=10)


    def Class_Select_Dropdown_Func(self):

        Class_Dir_Names  = os.listdir(self.entry_box.get())

        self.Class_dropdown.config( values = sorted(Class_Dir_Names) )


    def Image_Select_Dropdown_Func(self):
        
        path = os.path.join( str(self.entry_box.get()) , str(self.Class_selected.get() ) )
        Image_File_Names  = os.listdir( path )

        sorted_Image_File_Names = sorted(Image_File_Names, key=lambda x: int(x.split('_')[2].split('.')[0]))

        # sorted_Image_File_Names = sorted( Image_File_Names )

        self.Image_dropdown.config( values = sorted_Image_File_Names )
        self.Image_dropdown.set('')

    def Load_Image(self):

        try:
            photo_path = os.path.join(  str(self.selected_dir.get()) , str(self.Class_selected.get())   ) 
            photo_path = os.path.join(  photo_path , str(self.Image_Selected.get())  ) 


            img = Image.open(photo_path)
            original_width, original_height = img.size
            new_size = (original_width * 2, original_height * 2)
            img_resized = img.resize(new_size, Image.Resampling.LANCZOS) 
            # img_resized = img.resize( (255,255), Image.Resampling.LANCZOS) 
            Photo = ImageTk.PhotoImage(img_resized)

            # Clear previous images in the frame
            for widget in self.Image_Frame.winfo_children():
                widget.destroy()


            label = tk.Label(self.Image_Frame, image=Photo)
            label.image = Photo  # Keep a reference so the image is not garbage collected
            label.pack(anchor='w', pady=10)

        except:
            pass

        return

        

        
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#