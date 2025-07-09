from Imports.common_imports import *

import Backend
import Core_Functions
import Helpers
import Pages

from Pages import *
from Backend import *
from Core_Functions import *
from Helpers import *

from Helpers.Frame_Manager_Script import Frame_Manager

matplotlib.use('agg')

class App(tk.Tk):
    def _destroy_frame(self, frame_class):
        frame = self.frames.get(frame_class)
        if frame:
            frame.destroy()
            del self.frames[frame_class]


    def _reinitialize_frame(self, frame_or_name):
        # if a string was given, look up the class in Pages
        if isinstance(frame_or_name, str):
            try:
                FrameClass = getattr(Pages, frame_or_name)
            except AttributeError:
                raise KeyError(f"No page named {frame_or_name!r}")
        # if an instance was given, pull its class
        elif not isinstance(frame_or_name, type):
            FrameClass = frame_or_name.__class__
        # otherwise assume it's already a class
        else:
            FrameClass = frame_or_name

        # destroy old
        self._destroy_frame(FrameClass)

        # create new
        new_frame = FrameClass(self.container, self)
        self.frames[FrameClass] = new_frame
        setattr(self, FrameClass.__name__, new_frame)  # keep as attribute if you like
        new_frame.grid(row=0, column=0, sticky="nsew")


    def __init__(self, Data_Directory='', input_type='edep', det_complex='2x2'):
    # def __init__(self, input_type='edep', det_complex='2x2'):
        super().__init__()
        self.title("Msci Project App")
        # self.geometry("400x300")

        self.min_width = 400
        self.min_height = 300 

        # ─── Container for pages ─────────────────────────────────────
        self.container = tk.Frame(self)
        self.container.pack()                    # don't fill/expand
        self.container.grid_propagate(True)

        os.system('cls||clear')
        print("RUNNING")

        # ─── Colour-blind setup ───────────────────────────────────────
        colors = plt.get_cmap('tab20').colors
        plt.rcParams['axes.prop_cycle'] = cycler('color', colors)
        self.cmap = cm.plasma

        # expose helpers
        self.destroy_frame = self._destroy_frame
        self.reinitialize_frame = self._reinitialize_frame

        # ─── Core attributes ─────────────────────────────────────────
        self.Data_Directory = Data_Directory
        # self.Data_Directory = ''
        self.pdg_id_map = Backend.pdg_id_script.pdg_id_map
        self.pdg_id_map_reverse = {v: k for k, v in self.pdg_id_map.items()}
        self.plot_type = 'scatter'
        self.selected_directory = ''
        self.running = False
        self.model = None
        self.model_learning_rate = 1e-4
        self.test_images = None
        self.is_fullscreen = False

        # plot bounds
        self.max_z_for_plot = round(918.2)
        self.min_z_for_plot = round(415.8)
        self.max_y_for_plot = round(82.9)
        self.min_y_for_plot = round(-216.7)
        self.max_x_for_plot = round(350)
        self.min_x_for_plot = round(-350)

        # ─── File lists ──────────────────────────────────────────────
        try:
            File_Names = os.listdir(Data_Directory)
            Temp_File_Names_Dict = {}
            for _file_name_ in File_Names:
                try:
                    Temp_File_Names_Dict.update({ int(_file_name_.split('.')[3]) : _file_name_ })
                except:
                    pass
            # Temp_File_Names_Dict = {int(i.split('.')[3]): i for i in File_Names }
            sorted_keys = sorted(Temp_File_Names_Dict.keys())
            File_Names = [Temp_File_Names_Dict[i] for i in sorted_keys]
            if File_Names == []:
                print("\n\nNO hdf5 FILES FOUND\n\n")
            self.File_Names = File_Names
            self.Allowed_Files = File_Names
        except:
            self.File_Names = ['']
            self.Allowed_Files = ['']

        self.input_type = input_type
        self.det_complex = det_complex


        # ─── expose Backend modules as attributes ────────────────────
        for name in Backend.__all__:
            setattr(self, name, getattr(Backend, name))
            # print("Backend", name)


        # ─── expose Core_Functions modules as attributes ─────────────
        for name in Core_Functions.__all__:
            setattr(self, name, getattr(Core_Functions, name))
            # print("Core_Functions", name)

        # ─── expose Helpers modules/classes as attributes ────────────
        for name in Helpers.__all__:
            setattr(self, name, getattr(Helpers, name))
            # print("Helpers", name)





        # ─── instantiate pages dynamically ──────────────────────────
        self.frames = {}
        for page_name in Pages.__all__:
            PageClass = getattr(Pages, page_name)
            frame = PageClass(self.container, self)
            # key by class
            self.frames[PageClass] = frame

            self.frames[page_name] = frame
            setattr(self, page_name, frame)   
            frame.grid(row=0, column=0, sticky="nsew")





        # show the start page
        self.show_frame('StartPage')

    def show_frame(self, page_identifier):
        """Raise a page given its class *or* its name."""
        if isinstance(page_identifier, str):
            try:
                PageClass = getattr(Pages, page_identifier)
            except AttributeError:
                raise KeyError(f"No page named {page_identifier!r}")
        else:
            PageClass = page_identifier

        print( PageClass )
        frame = self.frames[PageClass]
        frame.tkraise()

        if hasattr(frame, 'refresh_content'):
            frame.refresh_content()


        self.after_idle(lambda: self._resize_to(frame))

    def _resize_to(self, frame):
        # force layout update
        self.update_idletasks()
        
        w = frame.winfo_reqwidth()
        h = frame.winfo_reqheight()


        w = max(w, self.min_width)
        h = max(h, self.min_height)
        self.geometry(f"{w}x{h}")

if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    directory = tk.filedialog.askdirectory( title="Select a Data Directory", initialdir=os.getcwd() )
    root.destroy()

    if directory:
        app = App(Data_Directory=directory)
        app.mainloop()


    # app = App()
    # app.mainloop()
