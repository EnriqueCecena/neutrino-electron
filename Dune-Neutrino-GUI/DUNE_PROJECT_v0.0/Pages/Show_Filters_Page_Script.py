from Imports.common_imports import *


class Show_Model_Filters_Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        page_title_frame = tk.Frame(self)
        page_title_frame.pack( anchor= 'w' , pady=10 ,fill='x')

        # tk.Button(page_title_frame, text="Back", command = lambda:  self.exit_button_event() ).pack(anchor='w', padx=10 , side= tk.LEFT)
        tk.Button(page_title_frame, text="Back", command = lambda:  self.controller.show_frame("Monitor_Training_Page") ).pack(anchor='w', padx=10 , side= tk.LEFT)


        tk.Label(page_title_frame, text="Model_Filters ", font=("Helvetica", 16)).pack( padx=50, anchor='w' , side = tk.LEFT)

        # 'Full' button to toggle full-screen
        self.full_button = tk.Button( page_title_frame, text='Full', command= lambda: self.controller.attributes("-fullscreen" , not self.controller.attributes("-fullscreen")  ) ) 
        self.full_button.pack(anchor='w' , side=tk.LEFT)

        self.test_print = tk.Button( page_title_frame , text= 'Print Layers' , command= lambda : self.Update_Model_display() )
        self.test_print.pack( anchor = 'w' , side = tk.LEFT )

        self.Filter_Canvas_Frame = tk.Frame( self )
        self.Filter_Canvas_Frame.pack( anchor='w' , side= tk.TOP , pady= 10 )

        self.Filter_Plot = tk.Frame( self.Filter_Canvas_Frame )
        self.Filter_Plot.pack( anchor='w' , side= tk.TOP , pady= 10 )




    def Update_Model_display(self):

        for layer in self.controller.model.layers :

            try:
                print(layer , layer.activation)
            except:
                print(layer , '< No ativation >')

    def Model_Filters_Ready(self):

        # Build the list of layers from the current model.
        # For each layer a label is shown. In the case of Conv2D layers, a “Show” button is added that will call Plot_Model_Filters for that layer.
        # Any previously added widgets (labels/buttons) are cleared first.

        # Clear all children in Filter_Canvas_Frame except the Filter_Plot (which is used for plotting)
        # Filter_Page = next( (frame for cls, frame in self.controller.frames.items() if cls.__name__ == "Show_Model_Filters_Page"), None  )
        Filter_Page = self.controller.frames.get("Show_Model_Filters_Page")
        for widget in Filter_Page.Filter_Canvas_Frame.winfo_children():
            # Do not remove the plot area frame
            if widget != Filter_Page.Filter_Plot:
                widget.destroy()
        # Also clear any plot already in the Filter_Plot area.
        for widget in Filter_Page.Filter_Plot.winfo_children():
            widget.destroy()

        model = self.controller.model
        if model is None:
            tk.Label(Filter_Page.Filter_Canvas_Frame, text="No model loaded.").pack(anchor='w')
            return

        # Add a title label for the architecture
        tk.Label(Filter_Page.Filter_Canvas_Frame, text="Model Architecture:",   font=("Helvetica", 14, "bold") ).pack(anchor='w', pady=5)

        # Loop through each layer in the model and create a small frame
        # that contains a label with the layer’s description.
        # If the layer is a Conv2D layer, add a button that calls Plot_Model_Filters.
        for layer in model.layers:
            layer_frame = tk.Frame(Filter_Page.Filter_Canvas_Frame)
            layer_frame.pack(anchor='w', fill='x', pady=2)

            # Create a label that shows the layer name and its class type.
            layer_desc = f"{layer.name}: {layer.__class__.__name__}"
            tk.Label(layer_frame, text=layer_desc).pack(side='left')

            # For Conv2D layers, add a "Show" button.
            if isinstance(layer, tf.keras.layers.Conv2D):
                tk.Button(layer_frame, text="Show", command=lambda l=layer: Show_Model_Filters_Page.Plot_Model_Filters( self = self , conv_layer = l) ).pack(side='left', padx=5)

    def Plot_Model_Filters(self, conv_layer):

        # Clear the current plot area and display a new plot of the filters from the given conv_layer. If the button is pressed again (or another conv layer’s button is clicked) the previous plot is removed.

        # Clear the plot area first.
        # Filter_Page = next( (frame for cls, frame in self.controller.frames.items() if cls.__name__ == "Show_Model_Filters_Page"), None  )
        Filter_Page = self.controller.frames.get("Show_Model_Filters_Page")

        for widget in Filter_Page.Filter_Plot.winfo_children():
            widget.destroy()

        # Attempt to get the weights for the layer.
        try:
            filters, biases = conv_layer.get_weights()
        except Exception as e:
            tk.messagebox.showerror("Error", f"Could not retrieve filters: {e}")
            return

        n_filters = filters.shape[-1]
        n_channels = filters.shape[-2]

        # Determine a grid size so that all filters can be displayed
        grid_r = int(np.ceil(np.sqrt(n_filters)))
        grid_c = int(np.ceil(n_filters / grid_r))

        # Create a new matplotlib Figure object.
        fig = plt.Figure( dpi = 100)
        # fig = plt.Figure(figsize=(grid_c * 2, grid_r * 2) , dpi = 100)
        axes = fig.subplots(grid_r, grid_c)
        # Ensure axes is a flat list.
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]


        for i in range(n_filters):

            f = filters[:, :, :, i]

            f_min, f_max = f.min(), f.max()
            if f_max - f_min != 0:
                f = (f - f_min) / (f_max - f_min)
            else:
                f = f - f_min


            if n_channels == 3:
                axes[i].imshow(f)
            else:
                axes[i].imshow(f[:, :, 0], cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f"Filter {i+1}", fontsize=8)


        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        fig.suptitle(f"Filters from layer: {conv_layer.name}", fontsize=12)
        fig.tight_layout()


        canvas = FigureCanvasTkAgg(fig, master=Filter_Page.Filter_Plot)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        toolbar_filter= NavigationToolbar2Tk(canvas, Filter_Page.Filter_Plot , pack_toolbar=False)
        toolbar_filter.update()
        toolbar_filter.pack(side=tk.LEFT, fill=tk.X)
            



#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#