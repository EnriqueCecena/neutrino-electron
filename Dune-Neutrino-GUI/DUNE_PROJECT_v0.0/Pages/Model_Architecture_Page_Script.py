from Imports.common_imports import *

from Helpers.Frame_Manager_Script import Frame_Manager

class Model_Architecture_Page(tk.Frame):
    def __init__(self, parent, controller=None):
        super().__init__(parent)
        self.controller = controller
        self.current_inputs = 1

        self.ACTIVATIONS = ("relu", "tanh", "sigmoid", "linear", "leaky_relu", "softmax")

        # Title and Navigation with Load Model button
        page_title_frame = tk.Frame(self)
        page_title_frame.pack(anchor='w', pady=10)
        tk.Button(page_title_frame, text="Back", command=lambda: (controller.attributes("-fullscreen", False )  , controller.show_frame("Training_And_Eval_Options_Page"))).pack(side=tk.LEFT, padx=10)

        self.Learning_Rate_Value = tk.DoubleVar( value= 0.00001)
        tk.Label(page_title_frame, text="Learning Rate:").pack(side=tk.LEFT, padx=5)
        Learning_Rate_Entry = tk.Entry( page_title_frame , textvariable = self.Learning_Rate_Value ,  width=6 )
        Learning_Rate_Entry.pack(side=tk.LEFT, padx=5 )


        tk.Label(page_title_frame, text="Model Architecture Configuration", font=("Helvetica", 16)).pack(side=tk.LEFT, padx=50)

        tk.Label(page_title_frame, text="Inputs:").pack(side=tk.LEFT, padx=5)
        self.num_inputs_var = tk.IntVar(value=1)
        input_selector = ttk.Combobox(page_title_frame, textvariable=self.num_inputs_var, values=[1, 2, 3], state="readonly", width=3)
        input_selector.pack(side=tk.LEFT, padx=5)
        input_selector.bind("<<ComboboxSelected>>", self._on_num_inputs_changed)

        tk.Button(page_title_frame, text="Load Model", command=self.load_model).pack(side=tk.LEFT, padx=50)

        # Notebook for input tabs
        self.notebook = ttk.Notebook(self , width= 700 , height= 300)
        # self.notebook.pack(fill=tk.X, pady=5)
        self.notebook.pack( anchor='w',pady=5 )


        self.input_frames = []          
        self.inner_input_containers = []
        self.conv_layers_frames = []
        self.conv_lists = []
        self.input_shapes = []
        self.shared_dense_layers = []


        plane_names = ["ZY", "ZX", "XY"]
        # default_shapes = [(334, 200), (334, 466), (466, 200)]
        default_shapes = [(334, 200, 3), (334, 466, 3), (466, 200, 3)]
        for i in range(3):
            # Outer tab frame
            tab = tk.Frame(self.notebook)
            # Scrollable area inside tab
            scroller = tk.Frame(tab)
            scroller.pack(fill='both', expand=True)
            canvas = tk.Canvas(scroller, highlightthickness=0 )
            scrollbar = ttk.Scrollbar(scroller, orient='vertical', command=canvas.yview)
            scrollbar.pack(side='right', fill='y')
            canvas.pack(side='left', fill='both', expand=True)
            canvas.configure(yscrollcommand=scrollbar.set)
            inner = tk.Frame(canvas)
            win_id = canvas.create_window((0, 0), window=inner, anchor='nw')
            inner.bind('<Configure>', lambda e, c=canvas: c.configure(scrollregion=c.bbox('all')))

            # Input shape controls
            shape_frame = tk.Frame(inner)
            shape_frame.pack(anchor='w', pady=2)
            tk.Label(shape_frame, text=f"Input {i+1} shape:").pack(side=tk.LEFT)
            h, w , c = default_shapes[i]
            height_var = tk.IntVar(value=h)
            width_var = tk.IntVar(value=w)
            channels_var = tk.IntVar(value=c)
            tk.Entry(shape_frame, textvariable=height_var, width=5).pack(side=tk.LEFT, padx=(5, 0))
            tk.Label(shape_frame, text="x").pack(side=tk.LEFT)
            tk.Entry(shape_frame, textvariable=width_var, width=5).pack(side=tk.LEFT, padx=(0, 5))
            self.input_shapes.append((height_var, width_var ,channels_var))

            # Convolutional layers list
            conv_list_frame = tk.Frame(inner)
            conv_list_frame.pack(anchor='w', pady=5)
            self.conv_layers_frames.append(conv_list_frame)
            self.conv_lists.append([])
            tk.Button(inner, text="Add Layer", command=lambda idx=i: self._add_conv_layer(idx)).pack(anchor='w', pady=2)

            self.input_frames.append(tab)
            self.inner_input_containers.append(inner)

        # Add the first tab by default
        self.notebook.add(self.input_frames[0], text=f"Input 1 ({plane_names[0]})")

        # Shared dense & dropout layers section
        dense_frame = tk.Frame(self)
        dense_frame.pack(anchor='w', pady=(15,5))
        self.dense_list_frame = tk.Frame(dense_frame)
        self.dense_list_frame.pack(anchor='w')
        # tk.Button(dense_frame, text="Add Dense Layer", command=self._add_dense_layer).pack(anchor='w', pady=2)
        tk.Button(dense_frame, text="Add Dense Layer", command=self._add_dense_layer).pack(anchor='w' , side=tk.LEFT)

        tk.Button( dense_frame , text= "Add Dropout Layer" , command= self._add_dropout_layer ).pack(anchor='w', side=tk.LEFT)

        # Output & compile section
        output_frame = tk.Frame(self)
        output_frame.pack(anchor='w', pady=5)
        tk.Label(output_frame, text="Output Classes:").pack(side=tk.LEFT, padx=5)
        self.output_classes_var = tk.IntVar(value=2)
        tk.Entry(output_frame, textvariable=self.output_classes_var, width=5).pack(side=tk.LEFT)

        bottom_frame = tk.Frame(self)
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        tk.Button(bottom_frame, text="Build/Compile Model", command=self._compile_model).pack(anchor='w', pady=2)

        text_frame = tk.Frame(bottom_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        self.summary_text = tk.Text(text_frame, height=10, width=140, font=("Courier", 10), wrap='none' , state='normal' )
        scroll_y = tk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.summary_text.yview)
        scroll_x = tk.Scrollbar(bottom_frame, orient=tk.HORIZONTAL, command=self.summary_text.xview)
        self.summary_text.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        # self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # self.summary_text.pack(anchor='w' , side=tk.LEFT, expand=True)
        self.summary_text.pack(pady=5 , anchor='w', fill='y' , expand=True)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_x.pack(fill=tk.X)

    def _on_num_inputs_changed(self, event=None):
        num = self.num_inputs_var.get()
        plane_names = ["ZY", "ZX", "XY"]
        if num > self.current_inputs:
            for idx in range(self.current_inputs, num):
                self.notebook.add(self.input_frames[idx], text=f"Input {idx+1} ({plane_names[idx]})")
        elif num < self.current_inputs:
            for j in range(self.current_inputs - 1, num - 1, -1):
                self.notebook.forget(j)
        self.current_inputs = num


    def _add_conv_layer(self, input_index):
        container = self.conv_layers_frames[input_index]
        layer_frame = tk.Frame(container)
        layer_frame.pack(anchor='w', pady=2)

        # 1) now includes "Dropout"
        type_var = tk.StringVar(value="Conv2D")
        layer_frame.type_var = type_var
        ttk.Combobox(layer_frame, textvariable=type_var,
                    values=["Conv2D", "MaxPool2D", "Dropout", "Flatten", "BatchNormalization", "GlobalAvgPool"],
                    state="readonly", width=12).pack(side=tk.LEFT, padx=5)

        fields_frame = tk.Frame(layer_frame)
        fields_frame.pack(side=tk.LEFT, anchor='w')




        def update_fields(*args):
            for w in fields_frame.winfo_children():
                w.destroy()
            t = type_var.get()

            if t == "Conv2D":
                tk.Label(fields_frame, text="Filters:").pack(side=tk.LEFT)
                filters_var = tk.IntVar(value=32)
                tk.Entry(fields_frame, textvariable=filters_var, width=5).pack(side=tk.LEFT, padx=5)

                tk.Label(fields_frame, text="Kernel:").pack(side=tk.LEFT)
                kernel_var = tk.StringVar(value="3x3")
                tk.Entry(fields_frame, textvariable=kernel_var, width=5).pack(side=tk.LEFT, padx=5)

                tk.Label(fields_frame, text="Act:").pack(side=tk.LEFT)
                activation_var = tk.StringVar(value="leaky_relu")
                ttk.Combobox(fields_frame, textvariable=activation_var, values=self.ACTIVATIONS, state="readonly", width=7).pack(side=tk.LEFT, padx=5)

                # keep references
                layer_frame.filters_var    = filters_var
                layer_frame.kernel_var     = kernel_var
                layer_frame.activation_var = activation_var

            elif t == "MaxPool2D":
                tk.Label(fields_frame, text="Pool:").pack(side=tk.LEFT)
                pool_var = tk.StringVar(value="2x2")
                tk.Entry(fields_frame, textvariable=pool_var, width=5).pack(side=tk.LEFT, padx=5)
                layer_frame.pool_var = pool_var

            elif t == "Dropout":
                tk.Label(fields_frame, text="Rate:").pack(side=tk.LEFT)
                dropout_var = tk.DoubleVar(value=0.1)
                tk.Entry(fields_frame, textvariable=dropout_var, width=5).pack(side=tk.LEFT, padx=5)
                layer_frame.dropout_var = dropout_var

            # Flatten/GlobalAvgPool need no extra params

        type_var.trace_add('write', update_fields)
        update_fields()

        tk.Button(layer_frame, text="X", command=lambda lf=layer_frame: self._remove_conv_layer(input_index, lf) ).pack(side=tk.RIGHT, padx=5)

        self.conv_lists[input_index].append(layer_frame)


    def _remove_conv_layer(self, input_index, layer_frame):
        layer_frame.destroy()
        self.conv_lists[input_index].remove(layer_frame)

    def _add_dense_layer(self):
        layer_frame = tk.Frame(self.dense_list_frame)
        layer_frame.pack(anchor='w', pady=2)
        tk.Label(layer_frame, text="Units:").pack(side=tk.LEFT)
        units_var = tk.IntVar(value=128)
        tk.Entry(layer_frame, textvariable=units_var, width=6).pack(side=tk.LEFT, padx=5)

        layer_frame.units_var = units_var
        tk.Label(layer_frame, text="Activation:").pack(side=tk.LEFT)

        activation_var = tk.StringVar(value="relu")
        ttk.Combobox(layer_frame, textvariable=activation_var, values=self.ACTIVATIONS, state="readonly", width=7).pack(side=tk.LEFT, padx=5)

        layer_frame.activation_var = activation_var
        tk.Button(layer_frame, text="X",  command=lambda lf=layer_frame: self._remove_dense_layer(lf)).pack(side=tk.RIGHT, padx=5)
        self.shared_dense_layers.append(layer_frame)

    def _add_dropout_layer(self):
        layer_frame = tk.Frame(self.dense_list_frame)
        layer_frame.pack(anchor='w', pady=2)

        # Label + entry for the rate
        tk.Label(layer_frame, text="Dropout:").pack(side=tk.LEFT)
        dropout_var = tk.DoubleVar(value=0.2)
        tk.Entry( layer_frame, textvariable=dropout_var, width=5 ).pack(side=tk.LEFT, padx=5)

        # store the var on the frame
        layer_frame.dropout_var = dropout_var

        # remove button re-uses your dense removal logic
        tk.Button( layer_frame, text="X", command=lambda lf=layer_frame: self._remove_dense_layer(lf) ).pack(side=tk.RIGHT, padx=5)

        # append into the same shared list
        self.shared_dense_layers.append(layer_frame)

    def _remove_dense_layer(self, layer_frame):
        layer_frame.destroy()
        self.shared_dense_layers.remove(layer_frame)

    def _compile_model(self):
        inputs = []
        branch_outputs = []
        for i in range(self.current_inputs):
            h = self.input_shapes[i][0].get()
            w = self.input_shapes[i][1].get()
            c = self.input_shapes[i][2].get()
            inp = Input(shape=(h, w, c), name=f"Input{i+1}")
            x = inp
            needs_flatten = True
            for layer_frame in self.conv_lists[i]:
                t = layer_frame.type_var.get()
                if t == "Conv2D":

                    if  hasattr(layer_frame, 'filters_var'):
                        filters = layer_frame.filters_var.get()
                    else:
                        filters = 32

                    if hasattr(layer_frame, 'kernel_var'):
                        kernel_str = layer_frame.kernel_var.get()
                    else:
                        kernel_str ='3x3'

                    parts = kernel_str.replace('x', ' ').replace(',', ' ').split()
                    if len(parts) == 1:
                        try:
                            k = int(parts[0])
                        except:
                            k = 3
                        kernel_size = (k, k)
                    else:
                        try:
                            k1, k2 = map(int, parts[:2])
                        except:
                            k1, k2 = 3, 3
                        kernel_size = (k1, k2)

                    x = Conv2D(filters, kernel_size, activation='relu', padding='same')(x)

                elif t == "MaxPool2D":
                    if hasattr(layer_frame, 'pool_var'):
                        pool_str = layer_frame.pool_var.get()
                    else:
                        pool_str = "2x2"

                    parts = pool_str.replace('x', ' ').replace(',', ' ').split()
                    if len(parts) == 1:
                        p = int(parts[0]) if parts[0].isdigit() else 2
                        pool_size = (p, p)
                    else:
                        try:
                            p1, p2 = map(int, parts[:2])
                        except:
                            p1, p2 = 2, 2
                        pool_size = (p1, p2)
                    x = MaxPooling2D(pool_size)(x)

                elif t == "Dropout":
                    rate = layer_frame.dropout_var.get()
                    x = Dropout(rate)(x)

                elif t == "BatchNormalization":
                    x = BatchNormalization()(x)

                elif t == "Flatten":
                    x = Flatten()(x)
                    needs_flatten = False

                elif t == "GlobalAvgPool":
                    x = GlobalAveragePooling2D()(x)
                    needs_flatten = False
                    
            if needs_flatten:
                x = Flatten()(x)
            inputs.append(inp)
            branch_outputs.append(x)

        if len(branch_outputs) > 1:
            x = concatenate(branch_outputs)
        else:
            x = branch_outputs[0]


        # for layer_frame in self.shared_dense_layers:
        #     if hasattr(layer_frame, 'units_var'):
        #         units = layer_frame.units_var.get()
        #     else:
        #         units = 0 

        #     if hasattr(layer_frame, 'activation_var'):
        #         activation = layer_frame.activation_var.get()
        #     else:
        #         activation = 'relu'

        #     if units and units > 0:
        #         x = Dense(units, activation=activation)(x)

        for layer_frame in self.shared_dense_layers:
            # Dense
            if hasattr(layer_frame, 'units_var'):
                units     = layer_frame.units_var.get()
                activation= layer_frame.activation_var.get()
                if units > 0:
                    x = Dense(units, activation=activation)(x)
            # Dropout
            elif hasattr(layer_frame, 'dropout_var'):
                rate = layer_frame.dropout_var.get()
                x = Dropout(rate)(x)

        classes = self.output_classes_var.get()

        if classes == 1:
            activation_last = 'sigmoid'
            loss = 'binary_crossentropy'
        else:
            activation_last = 'softmax'
            loss = 'sparse_categorical_crossentropy'

        classes = max(classes, 1)
        out = Dense(classes, activation=activation_last)(x)

        model = Model(inputs=inputs, outputs=out)

        optimizer = tf.keras.optimizers.Adam( learning_rate  = self.Learning_Rate_Value.get() )    
        # model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
        model.compile( optimizer = optimizer ,loss=loss, metrics=['accuracy'])
        buf = StringIO()
        model.summary(print_fn=lambda line: buf.write(line + "\n"))
        summary_str = buf.getvalue()
        self.summary_text.delete('1.0', tk.END)
        self.summary_text.insert(tk.END, summary_str)
        
        self.controller.model = model
        self.model_learning_rate = self.Learning_Rate_Value.get()




    def load_model(self):
        # Prompt user to select model file
        file_path = filedialog.askopenfilename(filetypes=[("Keras Model", "*.keras"), ("TensorFlow SavedModel", "*"), ("All Files", "*.*")])
        if not file_path:
            return

        # Load the model
        loaded_model = keras_load_model(file_path)

        # Clear existing convolutional and dense UI elements
        for conv_frames in self.conv_lists:
            for frame in conv_frames:
                frame.destroy()
            conv_frames.clear()
        for frame in self.shared_dense_layers:
            frame.destroy()
        self.shared_dense_layers.clear()

        # Set number of inputs and refresh tabs
        num_inputs = len(loaded_model.inputs)
        self.num_inputs_var.set(num_inputs)
        self._on_num_inputs_changed()

        # Populate input shape fields
        for i, inp in enumerate(loaded_model.inputs):
            shape = inp.shape
            if hasattr(shape, 'as_list'):
                dims = shape.as_list()
            else:
                dims = list(shape)
            _, h, w, c = dims
            self.input_shapes[i][0].set(h)
            self.input_shapes[i][1].set(w)
            self.input_shapes[i][2].set(c)

        # Separate branch-specific layers vs shared
        all_layers = [l for l in loaded_model.layers if not isinstance(l, InputLayer)]
        merge_idx = next((idx for idx, l in enumerate(all_layers) if isinstance(l, Concatenate)), None)
        if merge_idx is not None:
            branch_layers = all_layers[:merge_idx]
            shared_layers = all_layers[merge_idx + 1:]
        else:
            branch_layers = all_layers
            shared_layers = []

        if shared_layers and isinstance(shared_layers[-1], Dense):
            output_layer = shared_layers.pop(-1)
            self.output_classes_var.set(output_layer.units)

        # Map each InputLayer to branch index
        input_map = {}
        for idx, inp in enumerate(loaded_model.inputs):
            hist_layer = inp._keras_history[0]
            input_map[hist_layer] = idx

        # Determine branch membership for each branch-specific layer
        layer_branch = {}
        for l in branch_layers:
            branch_idx = None
            for node in getattr(l, '_inbound_nodes', []):
                input_tensors = getattr(node, 'input_tensors', None)
                if not input_tensors:
                    continue
                sources = set()
                for tensor in input_tensors:
                    hist = getattr(tensor, '_keras_history', None)
                    if not hist:
                        continue
                    pred = hist[0]
                    if pred in input_map:
                        sources.add(input_map[pred])
                    elif pred in layer_branch:
                        sources.add(layer_branch[pred])
                if sources:
                    branch_idx = next(iter(sources))
                    break
            if branch_idx is None:
                branch_idx = 0
            layer_branch[l] = branch_idx

        # Group layers per branch
        branch_lists = [[] for _ in range(num_inputs)]
        for l in branch_layers:
            idx = layer_branch.get(l, 0)
            branch_lists[idx].append(l)

        # Populate convolutional layers UI for each branch
        for i, layers in enumerate(branch_lists):
            for l in layers:
                cls = l.__class__.__name__
                if cls == "Conv2D":
                    type_str = "Conv2D"
                elif cls == "MaxPooling2D":
                    type_str = "MaxPool2D"
                elif cls == "Dropout":
                    type_str = "Dropout"
                elif cls == "BatchNormalization":
                    type_str = "BatchNormalization"
                elif cls == "Flatten":
                    type_str = "Flatten"
                elif cls == "GlobalAveragePooling2D":
                    type_str = "GlobalAvgPool"
                else:
                    continue
                self._add_conv_layer(i)
                frame = self.conv_lists[i][-1]
                frame.type_var.set(type_str)
                if type_str == "Conv2D":
                    frame.filters_var.set(l.filters)
                    k1, k2 = l.kernel_size
                    frame.kernel_var.set(f"{k1}x{k2}")
                    frame.activation_var.set(l.get_config().get('activation'))
                elif type_str == "MaxPool2D":
                    p1, p2 = l.pool_size
                    frame.pool_var.set(f"{p1}x{p2}")
                elif type_str == "Dropout":
                    frame.dropout_var.set(l.rate)

        # Populate shared dense/dropout layers
        for l in shared_layers:
            if isinstance(l, Dense):
                self._add_dense_layer()
                frame = self.shared_dense_layers[-1]
                frame.units_var.set(l.units)
                frame.activation_var.set(l.get_config().get('activation'))
            elif isinstance(l, Dropout):
                self._add_dropout_layer()
                frame = self.shared_dense_layers[-1]
                frame.dropout_var.set(l.rate)

        # Assign to controller
        self.controller.model = loaded_model

