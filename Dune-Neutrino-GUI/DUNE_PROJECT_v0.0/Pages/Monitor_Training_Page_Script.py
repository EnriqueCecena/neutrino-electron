from Imports.common_imports import *


class Monitor_Training_Page(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # Initialize legend toggle flag
        self.legend_visible = True

        # ─── Title & Controls ────────────────────────────────────────
        page_title_frame = tk.Frame(self)
        page_title_frame.pack(anchor='w', pady=10, fill='x')

        # Back
        # tk.Button(page_title_frame, text="Back", command=self.exit_button_event).pack(anchor='w', padx=10, side=tk.LEFT)
        tk.Button(page_title_frame, text="Back", command= lambda: (self.controller.attributes("-fullscreen", False) ,  self.controller.show_frame("Model_Training_Page") )).pack(anchor='w', padx=10, side=tk.LEFT)

        # Model Filters
        tk.Button(page_title_frame, text='Model Filters',
                  command=lambda: controller.show_frame("Show_Model_Filters_Page"))\
          .pack(anchor='w', side=tk.LEFT)

        # Title Label
        tk.Label(page_title_frame, text="AI Command Center", font=("Helvetica", 16))\
          .pack(padx=50, anchor='w', side=tk.LEFT)

        # Confusion Matrix
        tk.Button(page_title_frame, text='Confusion Matrix',
                  command=lambda: controller.show_frame("Show_Confusion_Page"))\
          .pack(anchor='w', side=tk.LEFT)

        # Toggle Legend
        self.Toggle_legend_button = tk.Button( page_title_frame, text='Toggle Legend', command=self.toggle_legend)
        self.Toggle_legend_button.pack(anchor='w', side=tk.LEFT)

        # Fullscreen
        # self.full_button = tk.Button(page_title_frame, text='Full', command=self.toggle_fullscreen)
        self.full_button = tk.Button(page_title_frame, text='Full', command= lambda: self.controller.attributes("-fullscreen" , not self.controller.attributes("-fullscreen") ))
        self.full_button.pack(anchor='w', side=tk.LEFT)

        # ─── Classes ▾ Dropdown with Checkboxes ───────────────────────
        self.class_filter_mb = tk.Menubutton(
            page_title_frame, text='Classes ▾', relief=tk.RAISED)
        self.class_filter_menu = tk.Menu(self.class_filter_mb, tearoff=False)
        self.class_filter_mb.config(menu=self.class_filter_menu)
        self.class_filter_mb.pack(anchor='w', side=tk.LEFT, padx=(10,0))

        # Holds label → BooleanVar
        self.class_filter_vars = {}

        # ─── Plot Frames ──────────────────────────────────────────────
        self.First_Canvas_Row = tk.Frame(self); self.First_Canvas_Row.pack(anchor='w', side=tk.TOP, pady=10)
        self.First_Canvas_Row_Accuracy  = tk.Frame(self.First_Canvas_Row); self.First_Canvas_Row_Accuracy.pack(side=tk.LEFT, padx=10)
        self.First_Canvas_Row_Precision = tk.Frame(self.First_Canvas_Row); self.First_Canvas_Row_Precision.pack(side=tk.LEFT, padx=10)
        self.First_Canvas_Row_Recall    = tk.Frame(self.First_Canvas_Row); self.First_Canvas_Row_Recall.pack(side=tk.LEFT, padx=10)
        self.First_Canvas_Row_Loss      = tk.Frame(self.First_Canvas_Row); self.First_Canvas_Row_Loss.pack(side=tk.LEFT, padx=10)

        self.Second_Canvas_Row = tk.Frame(self); self.Second_Canvas_Row.pack(anchor='w', side=tk.TOP, pady=10)
        self.Second_Canvas_Row_Class_Accuracy  = tk.Frame(self.Second_Canvas_Row); self.Second_Canvas_Row_Class_Accuracy.pack(side=tk.LEFT, padx=10)
        self.Second_Canvas_Row_Class_Precision = tk.Frame(self.Second_Canvas_Row); self.Second_Canvas_Row_Class_Precision.pack(side=tk.LEFT, padx=10)
        self.Second_Canvas_Row_Class_Recall    = tk.Frame(self.Second_Canvas_Row); self.Second_Canvas_Row_Class_Recall.pack(side=tk.LEFT, padx=10)
        self.Second_Canvas_delta_Loss          = tk.Frame(self.Second_Canvas_Row); self.Second_Canvas_delta_Loss.pack(side=tk.LEFT, padx=10)

        # ─── Placeholders for Axes & Canvases ────────────────────────
        self.class_accuracy_ax      = None
        self.class_accuracy_canvas  = None
        self.class_precision_ax     = None
        self.class_precision_canvas = None
        self.class_recall_ax        = None
        self.class_recall_canvas    = None
        self.loss_fig_ax            = None
        self.loss_fig_canvas        = None


    def setup_class_filter(self, *args):

        # Extract the labels list from the last argument
        if not args:
            return
        translated_labels = args[-1]

        # Clear old menu
        self.class_filter_menu.delete(0, 'end')
        self.class_filter_vars.clear()

        for lbl in translated_labels:
            var = tk.BooleanVar(value=True)
            self.class_filter_vars[lbl] = var
            self.class_filter_menu.add_checkbutton(
                label=lbl,
                variable=var,
                command=self.update_class_visibility
            )


    def update_class_visibility(self):

        for lbl, var in self.class_filter_vars.items():
            vis = var.get()
            for ax in (self.class_accuracy_ax,
                       self.class_precision_ax,
                       self.class_recall_ax):
                if ax:
                    for line in ax.get_lines():
                        if line.get_label() == lbl:
                            line.set_visible(vis)


        for ax, canvas in (
            (self.class_accuracy_ax,   self.class_accuracy_canvas),
            (self.class_precision_ax,  self.class_precision_canvas),
            (self.class_recall_ax,     self.class_recall_canvas),
        ):
            if ax and canvas:

                vis_lines = [l for l in ax.get_lines() if l.get_visible()]
                if vis_lines:
                    handles = vis_lines
                    labels  = [l.get_label() for l in vis_lines]
                    ax.legend(handles, labels, loc="upper left", prop={'size':7})
                else:

                    if ax.legend_:
                        ax.legend_.remove()
                canvas.draw_idle()


    def toggle_legend(self):

        self.legend_visible = not self.legend_visible
        for ax, canvas in (
            (self.class_accuracy_ax,   self.class_accuracy_canvas),
            (self.class_precision_ax,  self.class_precision_canvas),
            (self.class_recall_ax,     self.class_recall_canvas),
        ):
            if ax and canvas and ax.get_legend():
                ax.get_legend().set_visible(self.legend_visible)
                canvas.draw_idle()

