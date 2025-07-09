from Imports.common_imports import *

from Helpers.Frame_Manager_Script import Frame_Manager

class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        # Create a canvas
        canvas = tk.Canvas(self, borderwidth=0 , width=600 , height=100)
        canvas.pack(side="left", fill="both", expand=True)

        # Create a scrollbar
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")

        # Frame that will hold the actual widgets
        self.scrollable_frame = tk.Frame(canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")   # Update scrollregion to encompass all widgets
            )
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)


#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#



class SliderSection:
    def __init__(self, parent, section_name, allocated_var, update_flag, default_max=100):

        self.parent = parent
        self.section_name = section_name
        self.allocated_var = allocated_var
        self.update_flag = update_flag
        self.default_max = default_max

        self.labels = []

        # Create a labeled frame for all sliders in this section.
        self.frame = ttk.LabelFrame(parent, text=f"{self.section_name} Sliders")
        # self.frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.frame.pack(anchor = 'w', side=tk.TOP, expand=False, padx=10, pady=5)


        # Entry to set all slider values in this section
        self.all_slider_entry = ttk.Entry(self.frame, width=6)
        self.all_slider_entry.pack(side=tk.RIGHT, padx=5)
        self.all_slider_entry.insert(0, "")
        self.all_slider_entry.bind("<Return>", self.set_all_sliders)
        self.all_slider_entry.bind("<FocusOut>", self.set_all_sliders)

        # Scrollable frame for slider rows.
        self.scrollable_frame = ScrollableFrame_2(self.frame)
        self.scrollable_frame.pack(fill="both", expand=True)

        # Containers for slider widgets and values.
        self.slider_vars = []
        self.slider_widgets = []
        self.entry_vars = []
        self.percent_labels = []
        self.slider_max_values = []
        self.slider_original_counts = []

        # Avoid recursive updates.
        self.allocated_var.trace_add('write', self.on_allocation_change)

    def add_slider(self, slider_label=None, max_value=None, initial_value=None, original_count=None):
        
        if self.update_flag.get(): return
        self.update_flag.set(True)
        try:
            max_value = max_value if max_value is not None else self.default_max
            initial_value = initial_value if initial_value is not None else max_value

            slider_var = tk.DoubleVar(value=initial_value)
            idx = len(self.slider_vars)

            row = ttk.Frame(self.scrollable_frame.scrollable_frame)
            row.pack(fill='x', padx=10, pady=2)

            ttk.Label(row, text=slider_label or f"{self.section_name} {idx+1}", width=30).grid(row=0, column=0, sticky='w')
            self.labels.append(slider_label or f"{self.section_name} {idx+1}")

            slider = ttk.Scale(
                row, from_=0, to=max_value, orient='horizontal',
                variable=slider_var, command=lambda v, var=slider_var, i=idx: self._update_percentage(var, i)
            )
            slider.grid(row=0, column=1, sticky='ew')
            row.columnconfigure(1, weight=1)

            entry_var = tk.StringVar(value=str(int(initial_value)))
            entry = ttk.Entry(row, textvariable=entry_var, width=6)
            entry.grid(row=0, column=2, padx=5)
            entry.bind('<Return>', lambda e, i=idx: self._entry_changed(i))
            entry.bind('<FocusOut>', lambda e, i=idx: self._entry_changed(i))

            percent = (initial_value / max_value * 100) if max_value else 0
            pct_lbl = ttk.Label(row, text=f"{percent:.1f}%")
            pct_lbl.grid(row=0, column=3, padx=5)

            self.slider_vars.append(slider_var)
            self.slider_widgets.append(slider)
            self.entry_vars.append(entry_var)
            self.percent_labels.append(pct_lbl)
            self.slider_max_values.append(max_value)
            self.slider_original_counts.append(original_count if original_count is not None else max_value)

            slider_var.trace_add('write', lambda *a, var=slider_var, i=idx: self._update_percentage(var, i))
        finally:
            self.update_flag.set(False)

    def _update_percentage(self, var, idx):
        val = var.get()
        max_v = self.slider_max_values[idx]
        self.entry_vars[idx].set(str(int(val)))
        pct = (val/max_v*100) if max_v else 0
        self.percent_labels[idx].config(text=f"{pct:.1f}%")

    def _entry_changed(self, idx):
        if self.update_flag.get(): return
        self.update_flag.set(True)
        try:
            try: val = float(self.entry_vars[idx].get())
            except ValueError: return
            max_v = self.slider_widgets[idx].cget('to')
            val = min(val, float(max_v))
            self.slider_vars[idx].set(val)
            self._update_percentage(self.slider_vars[idx], idx)
        finally:
            self.update_flag.set(False)

    def set_all_sliders(self, event=None):
        try: new = float(self.all_slider_entry.get())
        except ValueError: return
        for i, var in enumerate(self.slider_vars):
            var.set(min(new, self.slider_max_values[i]))

    def on_allocation_change(self, *args):
        pass

class ScrollableFrame_2(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self, width=600, height=250, borderwidth=0)
        canvas.pack(side='left', fill='both', expand=True)
        sb = ttk.Scrollbar(self, orient='vertical', command=canvas.yview)
        sb.pack(side='right', fill='y')
        self.scrollable_frame = tk.Frame(canvas)
        self.scrollable_frame.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0,0), window=self.scrollable_frame, anchor='nw')
        canvas.configure(yscrollcommand=sb.set)