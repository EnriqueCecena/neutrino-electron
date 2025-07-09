import h5py
import tkinter as tk
import tensorflow as tf
import sklearn
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import Image, ImageTk
import os
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm


import h5py
from matplotlib.patches import Patch


import keras

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import backend as K

import matplotlib.colors as colors
from scipy.spatial import cKDTree

class Evaluating_Model:
    # We assume self.controller is set externally and:
    # - self.controller.model is a trained Keras model.
    # - self.controller.frames contains the Evaluate_Model_Page frame.
    # - self.controller.test_images is available for other functions.

    def Correlation_func(self):
        evaluate_page = next((frame for cls, frame in self.controller.frames.items()
                              if cls.__name__ == "Evaluate_Model_Page"), None)
        if evaluate_page is None:
            print("Evaluate_Model_Page frame not found.")
            return

        for child in evaluate_page.Figure_Canvas_Frame.winfo_children():
            child.destroy()

        test_images = self.controller.test_images
        predictions = self.controller.model.predict(test_images)
        if predictions.ndim == 1:
            predictions = np.expand_dims(predictions, axis=0)
        if predictions.shape[0] < 2:
            print("Need at least two test images to compute a correlation matrix.")
            return

        correlation_matrix = np.corrcoef(predictions, rowvar=False)
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax)
        num_classes = predictions.shape[1]
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xticklabels([self.controller.model.class_name_dict[i] for i in range(num_classes)], rotation=90)
        ax.set_yticklabels([self.controller.model.class_name_dict[i] for i in range(num_classes)])
        ax.set_xlabel("Classes")
        ax.set_ylabel("Classes")
        ax.set_title("Correlation Matrix between Classes")
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, f"{correlation_matrix[i, j]:.2f}",
                        ha="center", va="center", color="black")
        canvas = FigureCanvasTkAgg(fig, master=evaluate_page.Figure_Canvas_Frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas=canvas, window=evaluate_page.Figure_Canvas_Frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.LEFT, fill=tk.X)

    def Plot_metrics_func(self):
        evaluate_page = next((frame for cls, frame in self.controller.frames.items()
                              if cls.__name__ == "Evaluate_Model_Page"), None)
        if evaluate_page is None:
            print("Evaluate_Model_Page frame not found.")
            return

        for child in evaluate_page.Figure_Canvas_Frame.winfo_children():
            child.destroy()

        test_images = self.controller.test_images
        all_test_true_labels = []
        all_test_pred_labels = []

        for x_val_batch, y_val_batch in test_images:
            predictions = self.controller.model.predict(x_val_batch)
            rounded_preds = np.argmax(predictions, axis=-1)
            all_test_true_labels.extend(y_val_batch.numpy())
            all_test_pred_labels.extend(rounded_preds)

        all_test_true_labels = np.array(all_test_true_labels)
        all_test_pred_labels = np.array(all_test_pred_labels)
        plot_dict = {}
        for i, class_name in self.controller.model.class_name_dict.items():
            support = np.sum(all_test_true_labels == i)
            if support == 0:
                continue

            # Compute TP, FP, and FN manually for class i
            TP = np.sum((all_test_true_labels == i) & (all_test_pred_labels == i))
            FP = np.sum((all_test_true_labels != i) & (all_test_pred_labels == i))
            FN = np.sum((all_test_true_labels == i) & (all_test_pred_labels != i))
            
            # Manual calculations based on the referenced formulas:
            acc_manual = TP / support if support > 0 else 0.0
            precision_manual = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall_manual = TP / (TP + FN) if (TP + FN) > 0 else 0.0

            print(f'{self.controller.model.class_name_dict[i]} acc :', acc_manual)
            print(f'{self.controller.model.class_name_dict[i]} pre :', precision_manual)
            print(f'{self.controller.model.class_name_dict[i]} rec :', recall_manual)
            plot_dict[self.controller.model.class_name_dict[i]] = {
                'accuracy': acc_manual,
                'precision': precision_manual,
                'recall': recall_manual
            }
        fig, ax = plt.subplots(figsize=(8, 6))
        metrics = ['accuracy', 'precision', 'recall']
        x = np.arange(len(metrics))
        class_names = list(plot_dict.keys())
        num_classes = len(class_names)
        bar_width = 0.8 / num_classes

        for i, cname in enumerate(class_names):
            metric_values = [plot_dict[cname][m] for m in metrics]
            offset = (i - num_classes / 2) * bar_width + bar_width / 2
            ax.bar(x + offset, metric_values, bar_width, label=cname)

        ax.set_ylabel('Score', fontsize=15, fontweight="bold")
        ax.set_title('Per-Class Metrics' , fontsize=15, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend( fontsize = 10 , loc='upper left', bbox_to_anchor=(1.05, 1) )
        canvas = FigureCanvasTkAgg(fig, master=evaluate_page.Figure_Canvas_Frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas=canvas, window=evaluate_page.Figure_Canvas_Frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.LEFT, fill=tk.X)




    def load_and_preprocess_image(self, path, image_size=(256, 256)):

        Model_input_height, Model_input_width = ( self.controller.model.input_shape[1], self.controller.model.input_shape[2]  )

        image_size = (Model_input_height, Model_input_width)
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=3)  # force 3 channels
        image = tf.image.resize(image, image_size)
        image = image / 255.0  # normalize to [0, 1]
        return image

    def Plot_Prob_Single_Image_func(self, path):

        # Display a single image alongside a bar chart of class probabilities.

        evaluate_page = next( (frame for cls, frame in self.controller.frames.items() if cls.__name__ == "Evaluate_Model_Page"),  None  )
        if evaluate_page is None:
            print("Evaluate_Model_Page frame not found.")
            return

        for child in evaluate_page.Figure_Canvas_Frame.winfo_children():
            child.destroy()


        input_h, input_w = self.controller.model.input_shape[1:3]
        pil_img = load_img(path, target_size=(input_h, input_w))
        img_array = img_to_array(pil_img)  # shape (h, w, 3), dtype float32
        model_input = np.expand_dims(img_array / 255.0, axis=0)  # normalize if your model expects it


        try:
            probs = self.controller.model.predict(model_input, verbose=0)[0]
        except Exception as e:
            print(f"Prediction error: {e}")
            return


        class_dict = self.controller.model.class_name_dict
        labels = [class_dict.get(i, str(i)) for i in range(len(probs))]

        paired = list(zip(labels, probs))

        top_idx = int(np.argmax(probs))
        top_label = class_dict.get(top_idx, str(top_idx))


        fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(10, 5))

        ax_img.imshow(img_array.astype("uint8"))
        ax_img.axis("off")
        ax_img.set_title("Input Image", fontsize=12, fontweight="bold")


        x = np.arange(len(labels))
        heights = [p for _, p in paired]
        ax_bar.bar(x, heights)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(labels, rotation=90, fontsize=8)
        ax_bar.set_ylabel("Probability", fontsize=10)
        ax_bar.set_title(f"Prediction Probabilities\nTop: {top_label}", fontsize=12, fontweight="bold")

        fig.tight_layout()


        canvas = FigureCanvasTkAgg(fig, master=evaluate_page.Figure_Canvas_Frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

        toolbar = NavigationToolbar2Tk( canvas=canvas,  window=evaluate_page.Figure_Canvas_Frame, pack_toolbar=False )
        toolbar.update()
        toolbar.pack(side=tk.LEFT, fill=tk.X)

        plt.close(fig)


    def Whole_Class_Predictinos(self, path: str, FOR_ALL_FILES: bool = True):

        evaluate_page = next( (f for cls, f in self.controller.frames.items() if cls.__name__ == "Evaluate_Model_Page"), None )
        if evaluate_page is None:
            print("Evaluate_Model_Page not found.")
            return
        for widget in evaluate_page.Figure_Canvas_Frame.winfo_children():
            widget.destroy()


        import glob
        patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif")
        files = []
        for pat in patterns:
            files.extend(glob.glob(os.path.join(path, pat)))
        if not files:
            print(f"No image files found in {path!r}.")
            return


        if FOR_ALL_FILES:
            selected = files
        else:
            num = min(500, len(files))
            selected = np.random.choice(files, size=num, replace=False)


        all_probs = []
        for img_path in selected:
            try:
                img_t = Evaluating_Model.load_and_preprocess_image( self, img_path)  # H×W×3 in [0,1]
                batch = tf.expand_dims(img_t, axis=0)
                probs = self.controller.model.predict(batch, verbose=0)[0]
                all_probs.append(probs)
            except Exception as e:
                print(f"Skipping {img_path!r}: {e}")
                continue

        if not all_probs:
            print("No successful predictions to average.")
            return


        all_probs = np.stack(all_probs, axis=0)
        avg_probs = all_probs.mean(axis=0)


        class_dict = getattr(self.controller.model, "class_name_dict", {})
        labels = [class_dict.get(i, str(i)) for i in range(len(avg_probs))]


        idx = np.argsort(avg_probs)[::-1]
        labels = [labels[i]   for i in idx]
        values = [avg_probs[i] for i in idx]


        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(labels))
        ax.bar(x, values)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Average Probability", fontsize=12)
        ax.set_title(
            f"{'All' if FOR_ALL_FILES else 'Sampled'} Images → Avg Softmax Probabilities [{   os.path.basename(os.path.normpath(path))  }]",
            fontsize=14
        )
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=evaluate_page.Figure_Canvas_Frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(
            canvas=canvas, window=evaluate_page.Figure_Canvas_Frame, pack_toolbar=False
        )
        toolbar.update()
        toolbar.pack(side=tk.LEFT, fill=tk.X)

        plt.close(fig)






    def Heatmap_func(self, path: str , random_100 = False ):
        evaluate_page = next((f for cls,f in self.controller.frames.items()
                              if cls.__name__=="Evaluate_Model_Page"), None)


        if evaluate_page is None:
            print("Evaluate_Model_Page not found."); return

        # Clear previous
        for w in evaluate_page.Figure_Canvas_Frame.winfo_children():
            w.destroy()

        print( path )


        def Heat_map_steps( self , evaluate_page , path: str , random_100 = False , Note = ''):

            # Load + preprocess
            target_h, target_w = self.controller.model.input_shape[1:3]
            pil_img = load_img(path, target_size=(target_h, target_w))
            img_arr = img_to_array(pil_img)/255.0
            batch = img_arr[None,...]

            # Build a new Input → model graph
            inp = tf.keras.Input(shape=self.controller.model.input_shape[1:])
            x = inp
            last_conv_output = None
            for layer in self.controller.model.layers:
                x = layer(x)
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_output = x
            if last_conv_output is None:
                print("No Conv2D layer found."); return
            preds = x

            grad_model = tf.keras.Model(inputs=inp, outputs=[last_conv_output, preds])

            # Compute gradients
            with tf.GradientTape() as tape:
                conv_out, predictions = grad_model(batch)
                loss = tf.reduce_max(predictions, axis=1)
            grads = tape.gradient(loss, conv_out)
            pooled = tf.reduce_mean(grads, axis=(0,1,2))
            heatmap = tf.squeeze(tf.maximum(conv_out[0] @ pooled[...,None], 0))
            heatmap /= tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
            heatmap = np.uint8(255 * heatmap.numpy())
            heatmap = Image.fromarray(heatmap).resize((target_w, target_h))

            # Overlay
            cmap = cm.jet(np.array(heatmap))[:,:,:3]
            overlay = (0.5*img_arr + 0.5*cmap).clip(0,1)

            # Plot
            fig, (ax0,ax1) = plt.subplots(1,2,figsize=(10,5))
            ax0.imshow(pil_img); ax0.axis('off'); ax0.set_title("Original")
            ax1.imshow(overlay); ax1.axis('off'); ax1.set_title("Heatmap Overlay")

            prediction_probs = predictions.numpy().flatten().tolist() 


            pred_pos = np.argmax( prediction_probs  )
            try:
                plt.suptitle(f"True Label : {Note} | Pred Label : { self.controller.model.class_name_dict.get(pred_pos, str(pred_pos)) } ")
            except:
                plt.suptitle(f"True Label : {Note} | Pred Label : { pred_pos } ")


            if random_100 == False: 
                canvas = FigureCanvasTkAgg(fig, master=evaluate_page.Figure_Canvas_Frame)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                canvas.draw()
                toolbar = NavigationToolbar2Tk(canvas, evaluate_page.Figure_Canvas_Frame, pack_toolbar=False)
                toolbar.update(); toolbar.pack(side=tk.LEFT, fill=tk.X)
                plt.close(fig)

            else: 
                print("complete")



        if random_100 == False:
            path_dir = path.split('/')[-2]
            Heat_map_steps( self = self , evaluate_page = evaluate_page , path = path  , random_100 = False , Note = path_dir )

        else:
            path_dir = path.split('/')[-1]

            pdf_name = f'Heatmap_predictions_{path_dir}.pdf'
            with PdfPages(pdf_name) as output:
                random_image_selections  = np.random.choice( os.listdir(path), size = 100 )

                for file in random_image_selections:
                    loop_path = os.path.join( path , file  )

                    Heat_map_steps( self = self , evaluate_page = evaluate_page , path = loop_path  , random_100 = True , Note = path_dir )

                    output.savefig( dpi = 600 )
                    plt.close()



    def Detector_Important_Regions(self, path):

        evaluate_page = next((frame for cls, frame in self.controller.frames.items() if cls.__name__ == "Evaluate_Model_Page"), None)
        if evaluate_page is None:
            print("Evaluate_Model_Page frame not found.")
            return


        for child in evaluate_page.Figure_Canvas_Frame.winfo_children():
            child.destroy()


        test_ds = self.controller.test_images


        for batch in test_ds.take(1):
            images, labels = batch  # Expect images to have shape (batch, height, width, channels)
            height, width = images.shape[1], images.shape[2]
            break


        Prediction_History_Martrix = np.zeros((height, width), dtype=np.float32)
        Counter_Matrix = np.zeros((height, width), dtype=np.float32)


        def recover_matrix_from_colormap_with_kdtree(colored_img, cmap_name='plasma', vmin=0, vmax=75, num_colors=256):

            cmap = cm.get_cmap(cmap_name, num_colors)
            color_table = (cmap(np.linspace(0, 1, num_colors))[:, :3] * 255).astype(np.uint8)

            tree = cKDTree(color_table.astype(np.float32))
            H, W, _ = colored_img.shape

            pixels = colored_img.reshape(-1, 3).astype(np.float32)

            _, indices = tree.query(pixels, k=1)

            norm_vals = indices.astype(np.float32) / (num_colors - 1)
            recovered = vmin + norm_vals * (vmax - vmin)
            return recovered.reshape(H, W)


        for batch in test_ds:
            images, labels = batch
            for i in range(images.shape[0]):
                image_tensor = images[i]
                true_label = labels[i]

                img_np = image_tensor.numpy()

                if img_np.dtype != np.uint8:
                    img_np = np.uint8(255 * img_np)

                recovered_matrix = recover_matrix_from_colormap_with_kdtree( img_np, cmap_name='plasma', vmin=0, vmax=75 )

                scaler_binary_matrix = (recovered_matrix > 0).astype(np.float32)
                

                pred = self.controller.model.predict(np.expand_dims(image_tensor.numpy(), axis=0))
                predicted_class = np.argmax(pred, axis=-1)[0]
                

                if int(true_label.numpy()) == int(predicted_class):
                    Prediction_History_Martrix += scaler_binary_matrix
                

                Counter_Matrix += scaler_binary_matrix


        Prediction_Prob_Matrix = np.divide(
            Prediction_History_Martrix,
            Counter_Matrix,
            out=np.zeros_like(Prediction_History_Martrix),
            where=Counter_Matrix != 0
        )


        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        heatmap = ax.imshow(Prediction_Prob_Matrix, cmap='hot', vmin=0, vmax=1)
        fig.colorbar(heatmap, ax=ax)
        ax.set_title("Prediction Probability Heatmap")
        ax.axis('off')


        canvas = FigureCanvasTkAgg(fig, master=evaluate_page.Figure_Canvas_Frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.draw()

        toolbar = NavigationToolbar2Tk(canvas=canvas, window=evaluate_page.Figure_Canvas_Frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.LEFT, fill=tk.X)



    def energy_prediction_distribution(self , path = ''):

        evaluate_page = next((frame for cls, frame in self.controller.frames.items() if cls.__name__ == "Evaluate_Model_Page"), None)
        if evaluate_page is None:
            print("Evaluate_Model_Page frame not found.")
            return


        for child in evaluate_page.Figure_Canvas_Frame.winfo_children():
            child.destroy()


        prediction_history = {"Correct": [], "Incorrect": []}


        def recover_matrix_from_colormap_with_kdtree(colored_img, cmap_name='plasma', vmin=0, vmax=75, num_colors=256):

            cmap = cm.get_cmap(cmap_name, num_colors)
            color_table = (cmap(np.linspace(0, 1, num_colors))[:, :3] * 255).astype(np.uint8)

            tree = cKDTree(color_table.astype(np.float32))
            H, W, _ = colored_img.shape

            pixels = colored_img.reshape(-1, 3).astype(np.float32)

            _, indices = tree.query(pixels, k=1)

            norm_vals = indices.astype(np.float32) / (num_colors - 1)
            recovered = vmin + norm_vals * (vmax - vmin)
            return recovered.reshape(H, W)


        test_ds = self.controller.test_images


        for batch in test_ds:
            images, labels = batch  # Expect images to have shape (batch, height, width, channels)
            for i in range(images.shape[0]):
                image_tensor = images[i]
                true_label = labels[i]


                img_np = image_tensor.numpy()

                if img_np.dtype != np.uint8:
                    img_np = np.uint8(255 * img_np)


                scalar_matrix = recover_matrix_from_colormap_with_kdtree( img_np, cmap_name='plasma', vmin=0, vmax=75 )

                total_energy = np.sum(scalar_matrix)


                pred = self.controller.model.predict(np.expand_dims(image_tensor.numpy(), axis=0))
                predicted_class = np.argmax(pred, axis=-1)[0]

                if int(true_label.numpy()) == int(predicted_class):
                    prediction_history["Correct"].append(total_energy)
                else:
                    prediction_history["Incorrect"].append(total_energy)


        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

        combined_data = np.concatenate((prediction_history["Correct"], prediction_history["Incorrect"]))
        if combined_data.size == 0:
            print("No energy data available for histogram.")
            return
        bins = np.histogram_bin_edges(combined_data, bins='auto')
        ax.hist([prediction_history["Correct"], prediction_history["Incorrect"]],
                bins=bins, stacked=True, label=["Correct", "Incorrect"] , range=(0, 15_000))
        ax.set_xlabel("Energy [MeV]" , fontsize=15, fontweight="bold")
        ax.set_ylabel("Frequency" , fontsize=15, fontweight="bold")
        ax.set_yscale("log")
        ax.set_title("Energy Prediction Distribution")
        ax.legend()


        canvas = FigureCanvasTkAgg(fig, master=evaluate_page.Figure_Canvas_Frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.draw()

        toolbar = NavigationToolbar2Tk(canvas=canvas, window=evaluate_page.Figure_Canvas_Frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.LEFT, fill=tk.X)





    def Active_Pixel_Count(self , path = ''):


        evaluate_page = next( (frame for cls, frame in self.controller.frames.items() if cls.__name__ == "Evaluate_Model_Page"),  None )
        if evaluate_page is None:
            print("Evaluate_Model_Page frame not found.")
            return


        for child in evaluate_page.Figure_Canvas_Frame.winfo_children():
            child.destroy()


        prediction_history = {"Correct": [], "Incorrect": []}


        def recover_matrix_from_colormap_with_kdtree(colored_img, cmap_name='plasma', vmin=0, vmax=75, num_colors=256):

            cmap = cm.get_cmap(cmap_name, num_colors)
            color_table = (cmap(np.linspace(0, 1, num_colors))[:, :3] * 255).astype(np.uint8)
            tree = cKDTree(color_table.astype(np.float32))
            H, W, _ = colored_img.shape
            pixels = colored_img.reshape(-1, 3).astype(np.float32)
            _, indices = tree.query(pixels, k=1)
            norm_vals = indices.astype(np.float32) / (num_colors - 1)
            recovered = vmin + norm_vals * (vmax - vmin)
            return recovered.reshape(H, W)


        test_ds = self.controller.test_images


        for batch in test_ds:
            images, labels = batch  # images shape: (batch_size, height, width, channels)
            for i in range(images.shape[0]):
                image_tensor = images[i]
                true_label = labels[i]


                img_np = image_tensor.numpy()
                if img_np.dtype != np.uint8:
                    img_np = np.uint8(255 * img_np)


                scalar_matrix = recover_matrix_from_colormap_with_kdtree( img_np, cmap_name='plasma', vmin=0, vmax=75 )


                active_pixel_count = np.count_nonzero(scalar_matrix)


                pred = self.controller.model.predict(np.expand_dims(image_tensor.numpy(), axis=0))
                predicted_class = np.argmax(pred, axis=-1)[0]


                if int(true_label.numpy()) == int(predicted_class):
                    prediction_history["Correct"].append(active_pixel_count)
                else:
                    prediction_history["Incorrect"].append(active_pixel_count)


        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

        combined_data = np.concatenate((prediction_history["Correct"], prediction_history["Incorrect"]))
        if combined_data.size == 0:
            print("No active pixel count data available for histogram.")
            return
        bins = np.histogram_bin_edges(combined_data, bins='auto')
        ax.hist([prediction_history["Correct"], prediction_history["Incorrect"]],
                bins=bins, stacked=True, label=["Correct", "Incorrect"] )
        ax.set_xlabel("Active Pixels" ,  fontsize=15, fontweight="bold")
        ax.set_ylabel("Frequency" , fontsize=15, fontweight="bold")
        ax.set_title("Active Pixel Count Distribution")
        ax.set_yscale('log')
        ax.legend()


        canvas = FigureCanvasTkAgg(fig, master=evaluate_page.Figure_Canvas_Frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.draw()

        toolbar = NavigationToolbar2Tk(canvas=canvas, window=evaluate_page.Figure_Canvas_Frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.LEFT, fill=tk.X)


    def Download_Pixel_Eval_Plots(self, path=""):
        """
        Creates a PDF in the current working directory containing:
         1. Prediction probability heatmap (global)
         1a. Stacked histogram of energy distribution (all images)
         1b. % correct vs. energy-bin (all images)
         1c. Stacked histogram of active-pixel counts (all images)
         1d. % correct vs. active-pixel-bin (all images)
         Then, for each class:
         2. Stacked histogram of energy distribution for that class
         3. Stacked histogram of active-pixel counts for that class
         4. Scatter of % correct per energy bin for that class
         5. Scatter of % correct per active-pixel-count bin for that class
        Each page is laid out with tight margins so labels are never cut off.
        """

        # Helper to invert the 'plasma' colormap
        def _recover_scalar(colored_img, cmap_name='plasma', vmin=0, vmax=75, num_colors=256):
            cmap = cm.get_cmap(cmap_name, num_colors)
            table = (cmap(np.linspace(0,1,num_colors))[:,:3] * 255).astype(np.uint8)
            tree = cKDTree(table.astype(np.float32))
            H, W, _ = colored_img.shape
            pix = colored_img.reshape(-1,3).astype(np.float32)
            _, idx = tree.query(pix, k=1)
            norm = idx.astype(np.float32)/(num_colors-1)
            vals = vmin + norm*(vmax-vmin)
            return vals.reshape(H,W)


        test_ds = self.controller.test_images
        for imgs, _ in test_ds.take(1):
            H, W = imgs.shape[1], imgs.shape[2]
            break


        pred_hist = np.zeros((H,W), dtype=np.float32)
        count_hist = np.zeros((H,W), dtype=np.float32)


        class_dict    = self.controller.model.class_name_dict
        energy_corr   = {cls: [] for cls in class_dict}
        energy_incorr = {cls: [] for cls in class_dict}
        pix_corr      = {cls: [] for cls in class_dict}
        pix_incorr    = {cls: [] for cls in class_dict}


        global_energy_corr   = []
        global_energy_incorr = []
        global_pix_corr      = []
        global_pix_incorr    = []


        for imgs, labels in test_ds:
            for img_t, true_lbl in zip(imgs, labels):
                img_np = img_t.numpy()
                if img_np.dtype != np.uint8:
                    img_np = (255*img_np).astype(np.uint8)

                scalar       = _recover_scalar(img_np)
                total_energy = scalar.sum()
                active_pix   = np.count_nonzero(scalar)

                p = self.controller.model.predict( np.expand_dims(img_t.numpy(), axis=0), verbose=0 )
                pred_cls = np.argmax(p, axis=-1)[0]
                correct  = (int(true_lbl.numpy()) == int(pred_cls))


                mask = (scalar>0).astype(np.float32)
                if correct:
                    pred_hist += mask
                count_hist += mask


                if correct:
                    global_energy_corr.append(total_energy)
                    global_pix_corr.append(active_pix)
                else:
                    global_energy_incorr.append(total_energy)
                    global_pix_incorr.append(active_pix)


                cls = int(true_lbl.numpy())
                if correct:
                    energy_corr[cls].append(total_energy)
                    pix_corr[cls].append(active_pix)
                else:
                    energy_incorr[cls].append(total_energy)
                    pix_incorr[cls].append(active_pix)


        prob_matrix = np.divide( pred_hist, count_hist, out=np.zeros_like(pred_hist), where=count_hist!=0 )


        def _bin_pct(corr, all_, bins):
            n = len(bins)-1
            idx_all  = np.clip(np.digitize(all_,  bins) - 1, 0, n-1)
            idx_corr = np.clip(np.digitize(corr, bins) - 1, 0, n-1)
            total = np.bincount(idx_all, minlength=n)
            corrn = np.bincount(idx_corr, minlength=n)
            pct = np.zeros(n)
            nz = total>0
            pct[nz] = 100.0 * corrn[nz] / total[nz]
            centers = (bins[:-1] + bins[1:]) / 2.0
            return centers, pct


        pdf_path = os.path.join(os.getcwd(), "Pixel_Evaluation_Plots_By_Class.pdf")
        with PdfPages(pdf_path) as pdf:

            fig, ax = plt.subplots(figsize=(6,6), dpi=100)
            im = ax.imshow(prob_matrix, cmap='hot', vmin=0, vmax=1)
            fig.colorbar(im, ax=ax)
            ax.set_title("Prediction Probability Heatmap")
            ax.axis('off')
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


            e_all = np.array(global_energy_corr + global_energy_incorr)
            if e_all.size > 0:
                e_bins = np.histogram_bin_edges(e_all, bins='auto')
                fig, ax = plt.subplots(figsize=(6,6), dpi=100)
                ax.hist(
                    [global_energy_corr, global_energy_incorr],
                    bins=e_bins, stacked=True, label=["Correct","Incorrect"]
                )
                ax.set_xlabel("Energy [MeV]")
                ax.set_ylabel("Frequency")
                ax.set_yscale("log")
                ax.set_title("Global Energy Distribution")
                ax.legend()
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)


                e_cent, e_pct = _bin_pct(global_energy_corr, e_all, e_bins)
                fig, ax = plt.subplots(figsize=(6,6), dpi=100)
                ax.scatter(e_cent, e_pct)
                ax.set_xlabel("Energy bin center [MeV]")
                ax.set_ylabel("% Correct")
                ax.set_ylim(0,102)
                ax.set_title("Global Energy Bin Accuracy")
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)


            p_all = np.array(global_pix_corr + global_pix_incorr)
            if p_all.size > 0:
                p_bins = np.histogram_bin_edges(p_all, bins='auto')
                fig, ax = plt.subplots(figsize=(6,6), dpi=100)
                ax.hist(
                    [global_pix_corr, global_pix_incorr],
                    bins=p_bins, stacked=True, label=["Correct","Incorrect"]
                )
                ax.set_xlabel("Active Pixels")
                ax.set_ylabel("Frequency")
                ax.set_yscale("log")
                ax.set_title("Global Active-Pixel Distribution")
                ax.legend()
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)


                p_cent, p_pct = _bin_pct(global_pix_corr, p_all, p_bins)
                fig, ax = plt.subplots(figsize=(6,6), dpi=100)
                ax.scatter(p_cent, p_pct)
                ax.set_xlabel("Active-pixel bin center")
                ax.set_ylabel("% Correct")
                ax.set_ylim(0,102)
                ax.set_title("Global Active-Pixel Bin Accuracy")
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)


            for cls_idx, cls_name in class_dict.items():

                e_all = np.array(energy_corr[cls_idx] + energy_incorr[cls_idx])
                p_all = np.array(pix_corr[cls_idx]    + pix_incorr[cls_idx])
                if e_all.size == 0 or p_all.size == 0:
                    continue

                e_bins = np.histogram_bin_edges(e_all, bins='auto')
                p_bins = np.histogram_bin_edges(p_all, bins='auto')


                fig, ax = plt.subplots(figsize=(6,6), dpi=100)
                ax.hist(
                    [energy_corr[cls_idx], energy_incorr[cls_idx]],
                    bins=e_bins, stacked=True, label=["Correct","Incorrect"]
                )
                ax.set_xlabel("Energy [MeV]")
                ax.set_ylabel("Frequency")
                ax.set_yscale("log")
                ax.set_title(f"Energy Prediction Distribution ({cls_name})")
                ax.legend()
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)


                fig, ax = plt.subplots(figsize=(6,6), dpi=100)
                ax.hist( [pix_corr[cls_idx], pix_incorr[cls_idx]], bins=p_bins, stacked=True, label=["Correct","Incorrect"] )
                ax.set_xlabel("Active Pixels")
                ax.set_ylabel("Frequency")
                ax.set_yscale("log")
                ax.set_title(f"Active Pixel Count Distribution ({cls_name})")
                ax.legend()
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)


                e_cent, e_pct = _bin_pct(energy_corr[cls_idx], e_all, e_bins)
                fig, ax = plt.subplots(figsize=(6,6), dpi=100)
                ax.scatter(e_cent, e_pct)
                ax.set_xlabel("Energy bin center [MeV]")
                ax.set_ylabel("% Correct")
                ax.set_ylim(0,102)
                ax.set_title(f"Energy Bin Accuracy ({cls_name})")
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

                p_cent, p_pct = _bin_pct(pix_corr[cls_idx], p_all, p_bins)
                fig, ax = plt.subplots(figsize=(6,6), dpi=100)
                ax.scatter(p_cent, p_pct)
                ax.set_xlabel("Active-pixel bin center")
                ax.set_ylabel("% Correct")
                ax.set_ylim(0,102)
                ax.set_title(f"Active-Pixel Count Accuracy ({cls_name})")
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        print(f"Saved all pixel-evaluation plots by class to {pdf_path}")
        return pdf_path





    def Scattering_False_Positve_Analysis(self, path):


        class_dict = self.controller.model.class_name_dict
        nes_idx = next((i for i, n in class_dict.items() if n == r'$\nu - e$ (scattering)'), None)
        if nes_idx is None:
            raise ValueError("Class 'Neutrino_Electron_Scattering' not found in class_name_dict")


        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        counts_argmax = {cls: 0 for cls in class_dict}
        counts_thresh = { thr: {cls: 0 for cls in class_dict} for thr in thresholds }


        try:
            total_batches = tf.data.experimental.cardinality(
                self.controller.test_images).numpy()
            if total_batches < 0:
                total_batches = None
        except:
            total_batches = None

        for batch in tqdm(self.controller.test_images, total=total_batches, desc="Scanning test images", unit="batch"):
            images, labels = batch
            for img_t, true_lbl in zip(images, labels):
                probs = self.controller.model.predict(
                    np.expand_dims(img_t.numpy(), axis=0),
                    verbose=0
                )[0]
                pred = int(np.argmax(probs))
                true_cls = int(true_lbl.numpy())

                if pred == nes_idx and true_cls != nes_idx:
                    counts_argmax[true_cls] += 1
                    nes_conf = probs[nes_idx]
                    for thr in thresholds:
                        if nes_conf > thr:
                            counts_thresh[thr][true_cls] += 1


        pdf_path = os.path.join(os.getcwd(),
                                "False_Positive_Scattering_Analysis.pdf")
        with PdfPages(pdf_path) as pdf:

            cls_indices = [i for i in class_dict if i != nes_idx]
            cls_names   = [class_dict[i] for i in cls_indices]
            x = np.arange(len(cls_indices))

            fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
            ax.bar(x, [counts_argmax[i] for i in cls_indices])
            ax.set_xticks(x)
            ax.set_xticklabels(cls_names, rotation=90)
            ax.set_ylabel("False Positives", fontweight="bold")
            ax.set_title("False Positives per True Class (argmax → NUEEL)")
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)


            for thr in thresholds[1:]:
                pct = int(thr * 100)
                fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
                ax.bar(x, [counts_thresh[thr][i] for i in cls_indices])
                ax.set_xticks(x)
                ax.set_xticklabels(cls_names, rotation=90)
                ax.set_ylabel("False Positives", fontweight="bold")
                ax.set_title(f"False Positives per True Class (p(NUEEL) > {pct}%)")
                fig.tight_layout()
                pdf.savefig(fig); plt.close(fig)


            labels = ["argmax"] + [f"{int(s*100)}%" for s in thresholds]

            argmax_total = sum(counts_argmax[i] for i in cls_indices)
            totals = [argmax_total] + [ sum(counts_thresh[s][i] for i in cls_indices) for s in thresholds ]
            fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
            ax.bar(labels, totals)
            ax.set_xlabel("Method / Confidence Threshold", fontweight="bold")
            ax.set_ylabel("Total False Positives (→ NUEEL)", fontweight="bold")
            ax.set_title("False Positives: Argmax vs. Confidence Slices")
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        print(f"Saved false-positive analysis to {pdf_path}")
        return pdf_path


    def Two_Class_Scattering_False_Positve_Analysis(self, path):

        import pandas as pd   # ensure pd is the pandas module, not a dict


        class_dict = self.controller.model.class_name_dict
        if len(class_dict) != 2:
            raise ValueError("Model must have exactly two output classes for this analysis.")
        
        target = r'$\nu - e$ (scattering)'
        nes_idx = next((i for i, n in class_dict.items() if n == target), None)
        if nes_idx is None:
            raise ValueError(f"Class '{target}' not found in class_name_dict")
        other_idx = next(i for i in class_dict if i != nes_idx)


        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        methods    = ['argmax'] + thresholds


        false_pos = {m: 0 for m in methods}
        metrics   = {m: {'TP':0,'FP':0,'TN':0,'FN':0} for m in methods}
        confusion = {m: np.zeros((2,2), dtype=int) for m in methods}


        for imgs, _ in self.controller.test_images.take(1):
            H, W = imgs.shape[1], imgs.shape[2]
            break

        def _recover_scalar(colored_img, cmap_name='plasma', vmin=0, vmax=75, num_colors=256):
            cmap = cm.get_cmap(cmap_name, num_colors)
            table = (cmap(np.linspace(0,1,num_colors))[:,:3] * 255).astype(np.uint8)
            tree = cKDTree(table.astype(np.float32))
            H_, W_, _ = colored_img.shape
            pix = colored_img.reshape(-1,3).astype(np.float32)
            _, idx = tree.query(pix, k=1)
            norm = idx.astype(np.float32)/(num_colors-1)
            vals = vmin + norm*(vmax-vmin)
            return vals.reshape(H_,W_)

        def _bin_pct(corr, all_, bins):
            n = len(bins)-1
            idx_all  = np.clip(np.digitize(all_,  bins) - 1, 0, n-1)
            idx_corr = np.clip(np.digitize(corr, bins) - 1, 0, n-1)
            total = np.bincount(idx_all, minlength=n)
            corrn = np.bincount(idx_corr, minlength=n)
            pct = np.zeros(n)
            nz = total>0
            pct[nz] = 100.0 * corrn[nz] / total[nz]
            centers = (bins[:-1] + bins[1:]) / 2.0
            return centers, pct

        pixel_data = {}
        for m in methods:
            pixel_data[m] = {
                'pred_hist':    np.zeros((H,W), dtype=np.float32),
                'count_hist':   np.zeros((H,W), dtype=np.float32),
                'energy_corr':  [], 'energy_incorr':  [],
                'pix_corr':     [], 'pix_incorr':     [],
                'energy_corr_per_class':  {nes_idx:[], other_idx:[]},
                'energy_incorr_per_class':{nes_idx:[], other_idx:[]},
                'pix_corr_per_class':      {nes_idx:[], other_idx:[]},
                'pix_incorr_per_class':    {nes_idx:[], other_idx:[]},
            }


        try:
            total_batches = tf.data.experimental.cardinality(
                self.controller.test_images).numpy()
            if total_batches < 0:
                total_batches = None
        except:
            total_batches = None

        for batch in tqdm(self.controller.test_images, total=total_batches, desc="Analyzing test images", unit="batch"):
            images, labels = batch
            for img_t, true_lbl in zip(images, labels):
                true_cls = int(true_lbl.numpy())
                probs = self.controller.model.predict(
                    np.expand_dims(img_t.numpy(), axis=0),
                    verbose=0
                )[0]
                pred_arg = int(np.argmax(probs))
                preds_thr = { thr: (nes_idx if probs[nes_idx] > thr else other_idx) for thr in thresholds }


                if pred_arg == nes_idx and true_cls == other_idx:
                    false_pos['argmax'] += 1
                for thr in thresholds:
                    if preds_thr[thr] == nes_idx and true_cls == other_idx:
                        false_pos[thr] += 1


                for m in methods:
                    p = pred_arg if m == 'argmax' else preds_thr[m]
                    if p == nes_idx and true_cls == nes_idx:
                        metrics[m]['TP'] += 1
                    elif p == nes_idx and true_cls != nes_idx:
                        metrics[m]['FP'] += 1
                    elif p != nes_idx and true_cls == nes_idx:
                        metrics[m]['FN'] += 1
                    else:
                        metrics[m]['TN'] += 1
                    row = 0 if true_cls == nes_idx else 1
                    col = 0 if p == nes_idx else 1
                    confusion[m][row, col] += 1


                img_np = img_t.numpy()
                colored = np.uint8(img_np * 255) if img_np.dtype != np.uint8 else img_np
                scalar = _recover_scalar(colored)
                mask = (scalar > 0).astype(np.float32)
                energy = scalar.sum()
                pix_count = np.count_nonzero(scalar)

                for m in methods:
                    p = pred_arg if m == 'argmax' else preds_thr[m]
                    pdict = pixel_data[m]
                    pdict['count_hist'] += mask
                    if p == true_cls:
                        pdict['pred_hist'] += mask
                        pdict['energy_corr'].append(energy)
                        pdict['pix_corr'].append(pix_count)
                        pdict['energy_corr_per_class'][true_cls].append(energy)
                        pdict['pix_corr_per_class'][true_cls].append(pix_count)
                    else:
                        pdict['energy_incorr'].append(energy)
                        pdict['pix_incorr'].append(pix_count)
                        pdict['energy_incorr_per_class'][true_cls].append(energy)
                        pdict['pix_incorr_per_class'][true_cls].append(pix_count)


        pdf_path = os.path.join(os.getcwd(), "2_Class_Scattering_False_Positve_Analysis.pdf")
        with PdfPages(pdf_path) as pdf:

            other_name = class_dict[other_idx]
            x = np.arange(1)

            fig, ax = plt.subplots(figsize=(6,4), dpi=100)
            ax.bar(x, [false_pos['argmax']])
            ax.set_xticks(x)
            ax.set_xticklabels([other_name], rotation=90)
            ax.set_ylabel("False Positives", fontweight="bold")
            ax.set_title("False Positives per True Class (argmax → NUEEL)")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


            for thr in thresholds:
                fig, ax = plt.subplots(figsize=(6,4), dpi=100)
                ax.bar(x, [false_pos[thr]])
                ax.set_xticks(x)
                ax.set_xticklabels([other_name], rotation=90)
                ax.set_ylabel("False Positives", fontweight="bold")
                ax.set_title(f"False Positives per True Class (p(NUEEL) > {int(thr*100)}%)")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)


            labels = ["argmax"] + [f"{int(t*100)}%" for t in thresholds]
            totals = [false_pos['argmax']] + [false_pos[t] for t in thresholds]
            fig, ax = plt.subplots(figsize=(6,4), dpi=100)
            ax.bar(labels, totals)
            ax.set_xlabel("Method / Threshold", fontweight="bold")
            ax.set_ylabel("Total False Positives (→ NUEEL)", fontweight="bold")
            ax.set_title("False Positives: Argmax vs. Confidence Slices")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


            metric_names = ['accuracy','precision','recall']
            n_methods   = len(methods)
            x = np.arange(n_methods)
            bar_w = 0.8 / len(metric_names)

            fig, ax = plt.subplots(
                figsize=(10,4),    # wider so legend fits
                dpi=100,
                constrained_layout=True
            )
            for i, mname in enumerate(metric_names):
                vals = []
                for m in methods:
                    tp = metrics[m]['TP']; fp = metrics[m]['FP']
                    tn = metrics[m]['TN']; fn = metrics[m]['FN']
                    total = tp+fp+tn+fn
                    acc = (tp+tn)/total if total else 0
                    pre = tp/(tp+fp) if (tp+fp) else 0
                    rec = tp/(tp+fn) if (tp+fn) else 0
                    vals.append({'accuracy':acc,'precision':pre,'recall':rec}[mname])
                offset = (i - len(metric_names)/2)*bar_w + bar_w/2
                ax.bar(x + offset, vals, bar_w, label=mname)

            ax.set_xticks(x)
            ax.set_xticklabels(
                ["argmax"] + [f"{int(t*100)}%" for t in thresholds],
                rotation=90
            )
            ax.set_ylabel("Score", fontweight="bold")
            ax.set_title("Accuracy, Precision & Recall vs. Threshold")
            ax.legend(
                fontsize='small',
                loc='upper left',
                bbox_to_anchor=(1.05, 1)
            )
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


            idx_labels = [r'$\nu - e$ (scattering)', "Other"]
            col_labels = [r'$\nu - e$ (scattering)', "Other"]
            for m in methods:
                cm_raw = confusion[m]
                with np.errstate(divide='ignore', invalid='ignore'):
                    row_sums = cm_raw.sum(axis=1, keepdims=True)
                    cm_norm = np.divide(cm_raw, row_sums, where=row_sums!=0)
                    cm_norm = np.nan_to_num(cm_norm)
                df_cm = pd.DataFrame(cm_norm, index=idx_labels, columns=col_labels)
                fig, ax = plt.subplots(figsize=(5,4), dpi=100)
                sn.heatmap(
                    df_cm, annot=True, fmt=".2f", annot_kws={'size':7},
                    cbar_kws={"shrink":0.7}, ax=ax
                )
                title = "Confusion Matrix (argmax)" if m=='argmax' \
                        else f"Confusion Matrix (p(NUEEL) > {int(m*100)}%)"
                ax.set_title(title)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)


            for m in methods:
                label_str = "argmax" if m=='argmax' else f"p(NUEEL) > {int(m*100)}%"
                pdict = pixel_data[m]


                with np.errstate(divide='ignore', invalid='ignore'):
                    prob_mat = np.divide(
                        pdict['pred_hist'], pdict['count_hist'],
                        out=np.zeros_like(pdict['pred_hist']),
                        where=pdict['count_hist']!=0
                    )
                fig, ax = plt.subplots(figsize=(6,6), dpi=100)
                im = ax.imshow(prob_mat, cmap='hot', vmin=0, vmax=1)
                fig.colorbar(im, ax=ax)
                ax.set_title(f"Prediction Probability Heatmap ({label_str})")
                ax.axis('off')
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)


                e_c, e_i = pdict['energy_corr'], pdict['energy_incorr']
                if e_c or e_i:
                    all_e = np.array(e_c + e_i)
                    bins = np.histogram_bin_edges(all_e, bins='auto')
                    fig, ax = plt.subplots(figsize=(6,6), dpi=100)
                    ax.hist([e_c, e_i], bins=bins, stacked=True, label=["Correct","Incorrect"])
                    ax.set_xlabel("Energy [MeV]")
                    ax.set_ylabel("Frequency")
                    ax.set_yscale("log")
                    ax.set_title(f"Global Energy Distribution ({label_str})")
                    ax.legend()
                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)


                    centers, pct = _bin_pct(e_c, all_e, bins)
                    fig, ax = plt.subplots(figsize=(6,6), dpi=100)
                    ax.scatter(centers, pct)
                    ax.set_xlabel("Energy bin center [MeV]")
                    ax.set_ylabel("% Correct")
                    ax.set_ylim(0,102)
                    ax.set_title(f"Global Energy Bin Accuracy ({label_str})")
                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)


                p_c, p_i = pdict['pix_corr'], pdict['pix_incorr']
                if p_c or p_i:
                    all_p = np.array(p_c + p_i)
                    bins = np.histogram_bin_edges(all_p, bins='auto')
                    fig, ax = plt.subplots(figsize=(6,6), dpi=100)
                    ax.hist([p_c, p_i], bins=bins, stacked=True, label=["Correct","Incorrect"])
                    ax.set_xlabel("Active Pixels")
                    ax.set_ylabel("Frequency")
                    ax.set_yscale("log")
                    ax.set_title(f"Global Active-Pixel Distribution ({label_str})")
                    ax.legend()
                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)


                    centers, pct = _bin_pct(p_c, all_p, bins)
                    fig, ax = plt.subplots(figsize=(6,6), dpi=100)
                    ax.scatter(centers, pct)
                    ax.set_xlabel("Active-pixel bin center")
                    ax.set_ylabel("% Correct")
                    ax.set_ylim(0,102)
                    ax.set_title(f"Global Active-Pixel Bin Accuracy ({label_str})")
                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)


                for cls in (nes_idx, other_idx):
                    cname = class_dict[cls]
                    ec = pdict['energy_corr_per_class'][cls]
                    ei = pdict['energy_incorr_per_class'][cls]
                    pc = pdict['pix_corr_per_class'][cls]
                    pi = pdict['pix_incorr_per_class'][cls]
                    if (ec or ei) and (pc or pi):

                        all_ec = np.array(ec + ei)
                        ebins = np.histogram_bin_edges(all_ec, bins='auto')
                        fig, ax = plt.subplots(figsize=(6,6), dpi=100)
                        ax.hist([ec, ei], bins=ebins, stacked=True, label=["Correct","Incorrect"])
                        ax.set_xlabel("Energy [MeV]")
                        ax.set_ylabel("Frequency")
                        ax.set_yscale("log")
                        ax.set_title(f"Energy Distribution ({cname}) {label_str}")
                        ax.legend()
                        fig.tight_layout()
                        pdf.savefig(fig)
                        plt.close(fig)


                        all_pc = np.array(pc + pi)
                        pbins = np.histogram_bin_edges(all_pc, bins='auto')
                        fig, ax = plt.subplots(figsize=(6,6), dpi=100)
                        ax.hist([pc, pi], bins=pbins, stacked=True, label=["Correct","Incorrect"])
                        ax.set_xlabel("Active Pixels")
                        ax.set_ylabel("Frequency")
                        ax.set_yscale("log")
                        ax.set_title(f"Active Pixel Count ({cname}) {label_str}")
                        ax.legend()
                        fig.tight_layout()
                        pdf.savefig(fig)
                        plt.close(fig)


                        centers, pct = _bin_pct(ec, all_ec, ebins)
                        fig, ax = plt.subplots(figsize=(6,6), dpi=100)
                        ax.scatter(centers, pct)
                        ax.set_xlabel("Energy bin center [MeV]")
                        ax.set_ylabel("% Correct")
                        ax.set_ylim(0,102)
                        ax.set_title(f"Energy Bin Accuracy ({cname}) {label_str}")
                        fig.tight_layout()
                        pdf.savefig(fig)
                        plt.close(fig)


                        centers, pct = _bin_pct(pc, all_pc, pbins)
                        fig, ax = plt.subplots(figsize=(6,6), dpi=100)
                        ax.scatter(centers, pct)
                        ax.set_xlabel("Active-pixel bin center")
                        ax.set_ylabel("% Correct")
                        ax.set_ylim(0,102)
                        ax.set_title(f"Pixel Bin Accuracy ({cname}) {label_str}")
                        fig.tight_layout()
                        pdf.savefig(fig)
                        plt.close(fig)

        print(f"Saved two-class analysis to {pdf_path}")
        return pdf_path





    def two_class_lepton_angel_PDF(self, path=None):

        h5_root = (
            "/Users/jordan/Documents/QMUL_Physics/Year_4_2024_2025/"
            "Semester_1/MSci_Project/Code/_Main/Datasets/Test_File"
        )
        img_root = (
            "/Users/jordan/Documents/QMUL_Physics/Year_4_2024_2025/"
            "Semester_1/MSci_Project/Code/_Main/Image_Datasets/"
            "LQ/Custom/_FINAL_TEST_IMAGES"
        )

        mc_dfs = {}
        for fn in os.listdir(h5_root):
            if not fn.endswith(".hdf5"):
                continue
            parts = fn.split(".")
            if len(parts) < 4:
                continue
            code = parts[-3]
            try:
                code = str(int(code))
            except ValueError:
                continue

            hdr_ds = h5py.File(os.path.join(h5_root, fn), "r")["mc_hdr"]
            raw = hdr_ds[:]  # structured array of shape (N,)
            mc_dfs[code] = pd.DataFrame({ "event_id": raw["event_id"],  "vertex_id": raw["vertex_id"],  "lep_ang": raw["lep_ang"], "Enu": raw["Enu"] })


        img_dirs = {}
        for d in os.listdir(img_root):
            full = os.path.join(img_root, d)
            if not os.path.isdir(full):
                continue
            prefix = d.split("_", 1)[0]
            if prefix in mc_dfs:
                img_dirs[prefix] = full

        if not img_dirs:
            print("No image folders found under", img_root)
            return


        cd = self.controller.model.class_name_dict
        nes_name = r'$\nu - e$ (scattering)'
        nes_idx = next(i for i, n in cd.items() if n == nes_name)
        other_idx = next(i for i in cd if i != nes_idx)


        scenarios = {"Argmax": None, "80%": 0.80, "99%": 0.99}
        results = { name: {"E_corr": [], "θ_corr": [], "E_inc": [], "θ_inc": []} for name in scenarios }


        for code, df in mc_dfs.items():
            if code not in img_dirs:
                continue
            for class_dir in os.listdir(img_dirs[code]):
                full_cls = os.path.join(img_dirs[code], class_dir)
                if not os.path.isdir(full_cls):
                    continue
                true_idx = nes_idx if "Neutrino" in class_dir else other_idx

                for fname in tqdm(os.listdir(full_cls),
                                  desc=f"{code}/{class_dir}", unit="img"):
                    if not fname.endswith(".png"):
                        continue
                    parts = fname[:-4].rsplit("_", 2)
                    if len(parts) != 3:
                        continue
                    try:
                        eid = int(parts[-2])
                        vid = int(parts[-1])
                    except ValueError:
                        continue

                    row = df[(df.event_id == eid) & (df.vertex_id == vid)]
                    if row.empty:
                        continue
                    lep_ang = float(row.lep_ang.iloc[0])
                    Enu     = float(row.Enu.iloc[0])

                    img_t = Evaluating_Model.load_and_preprocess_image( self, 
                        os.path.join(full_cls, fname)
                    )
                    probs = self.controller.model.predict(
                        tf.expand_dims(img_t, 0), verbose=0
                    )[0]
                    arg = int(np.argmax(probs))

                    for name, thr in scenarios.items():
                        if thr is None:
                            pred = arg
                        else:
                            pred = other_idx if (arg == nes_idx and probs[nes_idx] < thr) else arg

                        R = results[name]
                        if pred == true_idx:
                            R["E_corr"].append(Enu)
                            R["θ_corr"].append(lep_ang)
                        else:
                            R["E_inc"].append(Enu)
                            R["θ_inc"].append(lep_ang)


        all_E = []
        all_θ = []
        for R in results.values():
            all_E.extend(R["E_corr"])
            all_E.extend(R["E_inc"])
            all_θ.extend(R["θ_corr"])
            all_θ.extend(R["θ_inc"])

        E_arr = np.array(all_E)

        mask = (E_arr >= 0) & (E_arr <= 25000)
        E_fit = E_arr[mask]
        if E_fit.size > 0:
            bins_x = np.histogram_bin_edges(E_fit, bins="auto", range=(0, 25000))
        else:
            bins_x = np.linspace(0, 25000, 51)

        θ_arr = np.array(all_θ)
        if θ_arr.size > 0:
            bins_y = np.histogram_bin_edges(θ_arr, bins="auto")
        else:
            bins_y = 50


        out_pdf = os.path.join(os.getcwd(), "TwoClass_LeptonAngle_vs_Energy.pdf")
        with PdfPages(out_pdf) as pdf:
            for name in scenarios:
                E_c = np.array(results[name]["E_corr"])
                θ_c = np.array(results[name]["θ_corr"])
                E_i = np.array(results[name]["E_inc"])
                θ_i = np.array(results[name]["θ_inc"])

                fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

                if E_c.size > 0:
                    ax.hist2d( E_c, θ_c, bins=[bins_x, bins_y], range=[[0, 25000], [bins_y[0], bins_y[-1]]], cmap="Blues", alpha=1.0 )

                if E_i.size > 0:
                    ax.hist2d( E_i, θ_i,  bins=[bins_x, bins_y], range=[[0, 25000], [bins_y[0], bins_y[-1]]], cmap="Reds", alpha=0.5 )

                ax.set_xlim(0, 25000)
                ax.set_xlabel("Neutrino Energy MeV", fontsize=12)
                ax.set_ylabel(r"$\theta_{\ell}$", fontsize=12)
                ax.set_title(f"Lepton Angle vs Energy — {name}", fontsize=14)

                legend_elems = [ Patch(facecolor="blue", alpha=1.0, label="Correct"), Patch(facecolor="red",  alpha=0.5, label="Incorrect") ]
                ax.legend(handles=legend_elems, loc="upper right")
                fig.tight_layout()

                pdf.savefig(fig)
                plt.close(fig)

        print(f"Saved two-class lepton-angle PDF to {out_pdf}")
        return out_pdf
