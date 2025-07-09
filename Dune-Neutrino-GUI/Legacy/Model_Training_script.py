import tkinter as tk
import tensorflow as tf
import numpy as np
import seaborn as sn
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg , NavigationToolbar2Tk)


from PIL import Image, ImageTk # Used For displaying embedded images (png files)


import sklearn.metrics
from sklearn.utils import class_weight
import tensorflow as tf
# from tensorflow.keras.losses import CategoricalFocalCrossentropy, Reduction
from focal_loss import SparseCategoricalFocalLoss

import Main_Run_File

import os

class Model_Training:

    def __init__(self):
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()



    @tf.function
    def Model_training_Step(self, model, x_batch, y_batch , sample_weight):
        # forward
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            # per-example loss
            per_example = tf.keras.losses.sparse_categorical_crossentropy( y_batch, logits, from_logits=False )   # shape=(batch_size,)
            # per_example = self.loss_fn(y_true=y_batch, y_pred=logits)          # shape=(batch_size,)
            # apply sample_weight
            weighted = per_example * sample_weight
            loss = tf.reduce_mean(weighted)

        # backward
        grads = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # accuracy for logging
        pred_classes = tf.argmax(logits, axis=-1, output_type=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(pred_classes, y_batch), tf.float32))
        return loss, acc



    def Model_val_Step(self, model, x_batch, y_batch):
        # return model.test_on_batch(x_batch, y_batch)
        return model.evaluate(x_batch, y_batch)

    # @tf.function
    # def Model_val_Step(self, model, x_batch, y_batch,
    #                 sample_weight=None):
    #     logits = model(x_batch, training=False)
    #     # if no sample_weight passed, fall back to all‑ones
    #     if sample_weight is None:
    #         sample_weight = tf.ones_like(
    #             tf.cast(y_batch, tf.float32)
    #         )
    #     per_example = self.loss_fn(
    #         y_true=y_batch,
    #         y_pred=logits
    #     )
    #     loss = tf.reduce_mean(per_example * sample_weight)
    #     preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
    #     acc = tf.reduce_mean(
    #         tf.cast(tf.equal(preds, y_batch), tf.float32)
    #     )
    #     return loss, acc




    def Advance_Image_Set_Selection(self, replace_train=False, free_File_Names_Dict=None, train_File_value_Dict=None):



        def load_and_preprocess_image(path):
            H, W = self.controller.model.input_shape[1:3]
            img = tf.io.read_file(path)
            img = tf.image.decode_png(img, channels=3)
            img = tf.image.resize(img, (H, W)) / 255.0
            return img

        def create_dataset(file_dict, label_map, batch_size=32, base_dir=None):
            root = base_dir or self.selected_dir_got
            file_paths, labels = [], []
            for cls, files in file_dict.items():
                for f in files:
                    file_paths.append(os.path.join(root, cls, f))
                    labels.append(label_map[cls])
            ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
            ds = ds.shuffle(len(file_paths))
            ds = ds.map(lambda p, y: (load_and_preprocess_image(p), y),
                        num_parallel_calls=tf.data.AUTOTUNE)
            return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # ── Grab UI  ──────────────────────────────────────────────────
        Advance_Page   = next(f for cls, f in self.controller.frames.items() if cls.__name__ == "Advance_Class_Selection_Page")
        classes        = Advance_Page.different_classes
        custom_test_dir = Advance_Page.test_set_dir
        Train_sec      = Advance_Page.slider_sections['Train']
        Validate_sec   = Advance_Page.slider_sections['Validate']
        Test_sec       = Advance_Page.slider_sections['Test']

        # ── On refresh, only rebuild train_ds ─────────────────────────────────
        if replace_train:
            # derive the active class_names
            class_names = [c for c, cnt in train_File_value_Dict.items() if cnt > 0]
            label_map   = {c: i for i, c in enumerate(class_names)}

            # pick the next `cnt` files from the free pool for each class
            new_train_files = {}
            for cls in class_names:
                cnt   = train_File_value_Dict[cls]
                free_list = free_File_Names_Dict.get(cls, [])
                new_train_files[cls] = free_list[:cnt]

            # rebuild only the train dataset
            train_ds = create_dataset(  new_train_files, label_map,  batch_size=self.Batch_size_text_selected.get(),  base_dir=self.selected_dir_got  )
            return train_ds

        # ── Otherwise do the full split ────────────────────────────────────────

        Raw_File_Names_Dict   = {}
        train_File_Names_Dict = {}
        val_File_Names_Dict   = {}
        test_File_Names_Dict  = {}
        free_File_Names_Dict  = {}
        train_File_value_Dict = {}
        val_File_value_Dict   = {}
        test_File_value_Dict  = {}
        class_names           = []
        class_weight_dict     = {}

        # TRAIN split
        for i, cls in enumerate(classes):
            cnt = int(Train_sec.entry_vars[i].get())
            cls_p = os.path.join(self.selected_dir_got, cls)
            files = [f for f in os.listdir(cls_p) if f.lower().endswith('.png')]
            np.random.shuffle(files)
            Raw_File_Names_Dict[cls]   = files
            train_File_value_Dict[cls] = cnt
            if cnt > 0:
                class_names.append(cls)


        label_map = {c: i for i, c in enumerate(class_names)}
        labels = []
        for cls, cnt in train_File_value_Dict.items():
            if cls in label_map:
                labels += [label_map[cls]] * cnt
        labels = np.array(labels)
        if labels.size:
            weights = class_weight.compute_class_weight('balanced',
                                                        classes=np.unique(labels),
                                                        y=labels)
            class_weight_dict = {int(idx): float(w)
                                for idx, w in zip(np.unique(labels), weights)}
        else:
            class_weight_dict = {i: 1.0 for i in range(len(class_names))}


        for i, cls in enumerate(classes):
            val_File_value_Dict[cls] = int(Validate_sec.entry_vars[i].get())


        if custom_test_dir:
            for cls in classes:
                p = os.path.join(custom_test_dir, cls)
                files = [f for f in os.listdir(p) if f.lower().endswith('.png')] \
                        if os.path.isdir(p) else []
                test_File_Names_Dict[cls] = files
                test_File_value_Dict[cls] = len(files)
        else:
            for i, cls in enumerate(classes):
                test_File_value_Dict[cls] = int(Test_sec.entry_vars[i].get())


        for cls, all_files in Raw_File_Names_Dict.items():
            t = train_File_value_Dict.get(cls, 0)
            v = val_File_value_Dict.get(cls, 0)
            train_File_Names_Dict[cls] = all_files[:t]
            val_File_Names_Dict[cls]   = all_files[t:t+v]

            if custom_test_dir:
                tst  = test_File_Names_Dict.get(cls, [])
                free = all_files[t+v:]
            else:
                x    = test_File_value_Dict.get(cls, 0)
                tst  = all_files[t+v:t+v+x]
                free = train_File_Names_Dict[cls] + all_files[t+v+x:]

            test_File_Names_Dict[cls] = tst
            free_File_Names_Dict[cls] = free

        # BUILD the three datasets
        label_map = {c: i for i, c in enumerate(class_names)}
        train_ds  = create_dataset(train_File_Names_Dict, label_map,  batch_size=self.Batch_size_text_selected.get(),  base_dir=self.selected_dir_got)
        val_ds    = create_dataset(val_File_Names_Dict, label_map,  batch_size=self.Batch_size_text_selected.get(),  base_dir=self.selected_dir_got)
        test_base = custom_test_dir or self.selected_dir_got
        test_ds   = create_dataset(test_File_Names_Dict, label_map,  batch_size=self.Batch_size_text_selected.get(),  base_dir=test_base)

        Raw_Data = train_ds.concatenate(val_ds).concatenate(test_ds)
        Raw_Data.class_names = class_names

        return ( Raw_Data, train_ds, val_ds, test_ds, train_File_value_Dict, val_File_value_Dict, test_File_value_Dict, free_File_Names_Dict, class_weight_dict )

    def Basic_Image_Set_Selection(self):

        Model_input_height, Model_input_width = ( self.controller.model.input_shape[1], self.controller.model.input_shape[2]  )


        Raw_Data = tf.keras.utils.image_dataset_from_directory( self.selected_dir_got , shuffle=True , batch_size=self.Batch_size_text_selected.get() , label_mode='int',  image_size=(Model_input_height, Model_input_width) )

        total_size = len(Raw_Data.file_paths) 

        train_percent = self.train_size_text_selected.get()
        val_percent = self.val_size_text_selected.get()
        test_percent = self.test_size_text_selected.get()


        if (train_percent + val_percent + test_percent) != 100:
            print("Train, validation, and test sizes do not add up to 100%.")
            return

        train_size = int(total_size * (train_percent / 100))
        val_size = int(total_size * (val_percent / 100))
        test_size = int(total_size * (test_percent / 100))


        train = Raw_Data.take(train_size)
        val = Raw_Data.skip(train_size).take(val_size)
        test = Raw_Data.skip(train_size + val_size).take(test_size)

        return Raw_Data , train , val , test





    def Train_Model(self):

        os.system('cls||clear')
        self.selected_dir_got = self.selected_dir.get()
        trainer = Model_Training()
        trainer.optimizer.learning_rate = self.controller.model_learning_rate

        def print_dataset_counts(dataset, class_names, name):

            labels_ds = dataset.unbatch().map(lambda x, y: y)
            all_labels = np.array(list(labels_ds.as_numpy_iterator()), dtype=int)
            

            unique_labels, counts = np.unique(all_labels, return_counts=True)
            

            print(f"\n{name} set:")
            for lbl, cnt in zip(unique_labels, counts):
                print(f"  Class {class_names[lbl]}: {cnt} images")


        def find_first_image(directory):
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith(('.png','.jpg','jpeg','bmp','gif')):
                        return os.path.join(root, file)
            return None

        first_image = find_first_image(self.selected_dir_got)
        if not first_image:
            print("No images found.")
            return
        try:
            with Image.open(first_image): pass
        except Exception as e:
            print(f"Bad image: {e}")
            return


        Advance_Class_Page = next( frame for cls, frame in self.controller.frames.items() if cls.__name__ == "Advance_Class_Selection_Page" )
        use_advanced = Advance_Class_Page.Enable_Value.get()
        custom_test_dir = ( Advance_Class_Page.test_set_dir if Advance_Class_Page.test_set_dir and os.path.isdir(Advance_Class_Page.test_set_dir) else None )

        # ——— build datasets & class_weight_dict ———
        if use_advanced:
            (Raw_Data, train, val, test, train_File_value_Dict, val_File_value_Dict, test_File_value_Dict, free_File_Names_Dict, class_weight_dict) = Model_Training.Advance_Image_Set_Selection(self)
            print("✔ Dataset split complete.")

            # only recompute on test if custom dir is valid
            if custom_test_dir:
                test_labels_ds = test.unbatch().map(lambda x, y: y)
                all_test = np.array(list(test_labels_ds.as_numpy_iterator()), dtype=int)
                n_classes = len(Raw_Data.class_names)
                classes = np.arange(n_classes)
                if all_test.size > 0:
                    weights = class_weight.compute_class_weight( 'balanced', classes=classes, y=all_test )
                    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
                else:
                    class_weight_dict = {i: 1.0 for i in range(n_classes)}
                
                class_weight_dict = {i: 1.0 for i in range(n_classes)}

                print("✔ Recomputed weights from TEST set:", class_weight_dict)

        else:
            Raw_Data, train, val, test = Model_Training.Basic_Image_Set_Selection(self)
            train_labels_ds = train.unbatch().map(lambda x, y: y)
            all_train = np.array(list(train_labels_ds.as_numpy_iterator()), dtype=int)
            n_classes = len(Raw_Data.class_names)
            classes = np.arange(n_classes)
            if all_train.size > 0:
                weights = class_weight.compute_class_weight( 'balanced', classes=classes, y=all_train  )
                class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
            else:
                class_weight_dict = {i: 1.0 for i in range(n_classes)}
            print("✔ Computed weights from TRAIN set:", class_weight_dict)

        # ——— print summary info ———
        class_names = Raw_Data.class_names
        num_classes = len(class_names)
        print("\n\n\nTraining :")
        print("Class Names:", class_names)
        print("Number of classes =", num_classes)

        print_dataset_counts(train, class_names, "Training")
        print_dataset_counts(val,   class_names, "Validation")
        print_dataset_counts(test,  class_names, "Test")

        print(self.controller.model.summary())

        num_epochs = int(self.Epoch_size_text_selected.get())
        history = { 'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': [], 'per_class': {} }

        refresh_train_cnter = 0

        if num_epochs > 0:
            for epoch in range(num_epochs):
                if not self.controller.running:
                    break

                # ——— refresh train set if requested ———
                if use_advanced:
                    ebfr = int(Advance_Class_Page.Epochs_Before_Refresh.get())
                    if ebfr != 0 and refresh_train_cnter >= ebfr:
                        train = Model_Training.Advance_Image_Set_Selection( self, replace_train=True, free_File_Names_Dict=free_File_Names_Dict, train_File_value_Dict=train_File_value_Dict )
                        refresh_train_cnter = 0
                        print("✔ Training set replaced")
                refresh_train_cnter += 1

                # ——— training loop ———
                train_loss_total = 0
                train_acc_total = 0
                train_steps = 0
                for i, (x_batch, y_batch) in enumerate(train):
                    sw = tf.gather( tf.constant( [class_weight_dict[j] for j in range(num_classes)], dtype=tf.float32 ), y_batch )

                    batch_loss, batch_acc = trainer.Model_training_Step( self.controller.model, x_batch, y_batch, sample_weight=sw )
                    train_loss_total += batch_loss
                    train_acc_total += batch_acc
                    train_steps += 1
                    print(f"{i}/{len(train)} - loss: {batch_loss:.4f} - acc: {batch_acc:.4f}")

                avg_train_loss = train_loss_total / max(train_steps, 1)
                avg_train_acc = train_acc_total / max(train_steps, 1)

                # ——— validation loop ———
                val_loss_total = 0
                val_steps = 0
                all_true = []
                all_pred = []
                for x_val, y_val in val:
                    preds = self.controller.model.predict(x_val)
                    rounded = np.argmax(preds, axis=-1)
                    all_true.extend(y_val.numpy())
                    all_pred.extend(rounded)
                    batch_loss, _ = Model_Training.Model_val_Step(
                        self, self.controller.model, x_val, y_val
                    )
                    val_loss_total += batch_loss
                    val_steps += 1

                avg_val_loss = val_loss_total / max(val_steps, 1)
                all_true = np.array(all_true)
                all_pred = np.array(all_pred)
                avg_val_acc = sklearn.metrics.accuracy_score(all_true, all_pred)
                avg_val_pre = sklearn.metrics.precision_score(
                    all_true, all_pred, average='macro', zero_division=0
                )
                avg_val_rec = sklearn.metrics.recall_score(
                    all_true, all_pred, average='macro', zero_division=0
                )

                # per‐class metrics
                for idx, cname in enumerate(class_names):
                    if cname not in history['per_class']:
                        history['per_class'][cname] = { 'val_accuracy': [], 'val_precision': [], 'val_recall': [] }
                    TP = np.sum((all_true == idx) & (all_pred == idx))
                    FP = np.sum((all_true != idx) & (all_pred == idx))
                    FN = np.sum((all_true == idx) & (all_pred != idx))
                    sup = np.sum(all_true == idx)

                    history['per_class'][cname]['val_accuracy'].append( TP / sup if sup else 0.0 )
                    history['per_class'][cname]['val_precision'].append( TP / (TP + FP) if (TP + FP) else 0.0 )
                    history['per_class'][cname]['val_recall'].append( TP / (TP + FN) if (TP + FN) else 0.0 )

                # record
                history['loss'].append(float(avg_train_loss.numpy()))
                history['accuracy'].append(float(avg_train_acc.numpy()))
                history['val_loss'].append(avg_val_loss)
                history['val_accuracy'].append(avg_val_acc)
                history['val_precision'].append(avg_val_pre)
                history['val_recall'].append(avg_val_rec)

                print(
                    f"Epoch {epoch+1}/{num_epochs}"
                    f" - loss: {avg_train_loss:.4f}"
                    f" - val_loss: {avg_val_loss:.4f}"
                    f" - accuracy: {avg_train_acc:.2f}"
                    f" - val_accuracy: {avg_val_acc:.2f}"
                    f" - val_precision: {avg_val_pre:.2f}"
                    f" - val_recall: {avg_val_rec:.2f}"
                )

                # # --- Confusion Matrix on Validation Set each epoch ---
                # Confusion_Page = next( (f for cls, f in self.controller.frames.items() if cls.__name__ == "Show_Confusion_Page"), None )
                # if Confusion_Page:
                #     # clear old widgets
                #     for w in Confusion_Page.Confusion_Canvas_Frame.winfo_children():
                #         w.destroy()

                #     # compute and display
                #     cm = sklearn.metrics.confusion_matrix( all_true_labels, all_pred_labels , labels = list(range(len(class_names))) )
                #     labels = [Model_Training.translate_label(self, c) for c in class_names]
                #     fig_cm, ax_cm = plt.subplots(figsize=(6, 6), dpi=120)
                #     disp = sklearn.metrics.ConfusionMatrixDisplay( confusion_matrix=cm , display_labels=labels )
                #     disp.plot(ax=ax_cm, cmap='Blues', values_format='.2f')
                #     ax_cm.set_title(f"Validation Confusion Matrix\nEpoch {epoch+1}")
                #     ax_cm.tick_params(axis='both', which='major', labelsize=6)
                #     fig_cm.tight_layout()

                #     canvas_cm = FigureCanvasTkAgg(
                #         fig_cm, master=Confusion_Page.Confusion_Canvas_Frame
                #     )
                #     canvas_cm.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                #     canvas_cm.draw()
                #     NavigationToolbar2Tk(canvas_cm, Confusion_Page.Confusion_Canvas_Frame, pack_toolbar=False).pack(side=tk.LEFT, fill=tk.X)
                #     plt.close(fig_cm)
    
                if not hasattr(self, 'live_plots_initialized'):
                    monitor_page = next( (frame for cls, frame in self.controller.frames.items() if cls.__name__ == "Monitor_Training_Page") , None )
                    if monitor_page is None:
                        print("Monitor_Training_Page not found in frames.")
                        return
                    self.live_plots_initialized = True
                    self.monitor_page = monitor_page
                    self.train_fig_acc, self.train_ax_acc = plt.subplots(figsize=(4, 4), dpi=100)
                    self.canvas_acc = FigureCanvasTkAgg(self.train_fig_acc, master=monitor_page.First_Canvas_Row_Accuracy)
                    self.canvas_acc.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                    self.train_fig_pre, self.train_ax_pre = plt.subplots(figsize=(4, 4), dpi=100)
                    self.canvas_pre = FigureCanvasTkAgg(self.train_fig_pre, master=monitor_page.First_Canvas_Row_Precision)
                    self.canvas_pre.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                    self.train_fig_recall, self.train_ax_recall = plt.subplots(figsize=(4, 4), dpi=100)
                    self.canvas_recall = FigureCanvasTkAgg(self.train_fig_recall, master=monitor_page.First_Canvas_Row_Recall)
                    self.canvas_recall.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                    self.train_fig_loss, self.train_ax_loss = plt.subplots(figsize=(4, 4), dpi=100)
                    self.canvas_loss = FigureCanvasTkAgg(self.train_fig_loss, master=monitor_page.First_Canvas_Row_Loss)
                    self.canvas_loss.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                    self.train_delta_loss_fig, self.train_delta_loss_ax = plt.subplots(figsize=(4, 4), dpi=100)
                    self.canvas_delta_loss = FigureCanvasTkAgg(self.train_delta_loss_fig, master=monitor_page.Second_Canvas_delta_Loss)
                    self.canvas_delta_loss.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                if not hasattr(self, 'live_class_plots_initialized'):
                    self.live_class_plots_initialized = True
                    self.train_class_fig_acc, self.train_class_ax_acc = plt.subplots(figsize=(4, 4), dpi=100)
                    self.canvas_class_acc = FigureCanvasTkAgg(self.train_class_fig_acc, master=monitor_page.Second_Canvas_Row_Class_Accuracy)
                    self.canvas_class_acc.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                    self.train_class_fig_pre, self.train_class_ax_pre = plt.subplots(figsize=(4, 4), dpi=100)
                    self.canvas_class_pre = FigureCanvasTkAgg(self.train_class_fig_pre, master=monitor_page.Second_Canvas_Row_Class_Precision)
                    self.canvas_class_pre.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                    self.train_class_fig_rec, self.train_class_ax_rec = plt.subplots(figsize=(4, 4), dpi=100)
                    self.canvas_class_rec = FigureCanvasTkAgg(self.train_class_fig_rec, master=monitor_page.Second_Canvas_Row_Class_Recall)
                    self.canvas_class_rec.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                self.train_ax_acc.cla()
                self.train_ax_acc.plot(history['accuracy'], color='teal', label='accuracy')
                self.train_ax_acc.plot(history['val_accuracy'], color='orange', label='val_accuracy')
                self.train_ax_acc.set_xlabel("Epoch")
                self.train_ax_acc.set_ylabel("Accuracy")
                self.train_ax_acc.set_title("Accuracy", fontsize=17)
                self.train_ax_acc.set_ylim(0, 1.1)
                self.train_ax_acc.legend(loc="upper left")
                self.canvas_acc.draw()
                self.train_ax_pre.cla()
                self.train_ax_pre.plot(history['val_precision'], color='orange', label='val_precision')
                self.train_ax_pre.set_xlabel("Epoch")
                self.train_ax_pre.set_ylabel("Precision")
                self.train_ax_pre.set_title("Precision", fontsize=14)
                self.train_ax_pre.set_ylim(0, 1.1)
                self.train_ax_pre.legend(loc="upper left", prop={'size': 7})
                self.canvas_pre.draw()
                self.train_ax_recall.cla()
                self.train_ax_recall.plot(history['val_recall'], color='orange', label='val_recall')
                self.train_ax_recall.set_xlabel("Epoch")
                self.train_ax_recall.set_ylabel("Recall")
                self.train_ax_recall.set_title("Recall", fontsize=14)
                self.train_ax_recall.set_ylim(0, 1.1)
                self.train_ax_recall.legend(loc="upper left", prop={'size': 7})
                self.canvas_recall.draw()
                self.train_ax_loss.cla()
                self.train_ax_loss.plot(history['loss'], color='blue', label='loss')
                self.train_ax_loss.plot(history['val_loss'], color='orange', label='val_loss')
                self.train_ax_loss.set_xlabel("Epoch")
                self.train_ax_loss.set_ylabel("Loss")
                self.train_ax_loss.set_title("Loss", fontsize=14)
                self.train_ax_loss.legend(loc="upper left")
                self.canvas_loss.draw()
                self.train_delta_loss_ax.cla()
                self.train_delta_loss_ax.plot(np.array(history['val_loss']) - np.array(history['loss']))
                self.train_delta_loss_ax.hlines(y=0, xmin=0, xmax=len(history['val_loss']), colors='black', linestyles='--')
                self.train_delta_loss_ax.set_xlabel("Epoch")
                self.train_delta_loss_ax.set_ylabel(r"$\Delta$ Loss")
                self.train_delta_loss_ax.set_title(r"$\Delta$ Loss", fontsize=14)
                self.canvas_delta_loss.draw()
                default_line_styles = ['-', '--', ':', '-.']
                possible_line_styles = default_line_styles.copy()
                for _ in range(len(class_names) - len(default_line_styles)):
                    possible_line_styles.append((np.random.randint(0,10),(np.random.randint(0,10),np.random.randint(0,10),np.random.randint(0,10),np.random.randint(0,10))))
                self.train_class_ax_acc.cla()
                Model_Training.per_class_plot(self, self.train_class_ax_acc, class_names, history['per_class'], 'val_accuracy', possible_line_styles)
                self.train_class_ax_acc.set_xlabel("Epoch")
                self.train_class_ax_acc.set_ylabel("val_Accuracy")
                self.train_class_ax_acc.legend(loc="upper left", prop={'size': 7})
                self.train_class_ax_acc.set_title("per class accuracy", fontsize=14)
                self.train_class_ax_acc.set_ylim(0, 1.1)
                self.canvas_class_acc.draw()
                self.train_class_ax_pre.cla()
                Model_Training.per_class_plot(self, self.train_class_ax_pre, class_names, history['per_class'], 'val_precision', possible_line_styles)
                self.train_class_ax_pre.set_xlabel("Epoch")
                self.train_class_ax_pre.set_ylabel("val_Precision")
                self.train_class_ax_pre.legend(loc="upper left", prop={'size': 7})
                self.train_class_ax_pre.set_title("per class precision", fontsize=14)
                self.train_class_ax_pre.set_ylim(0, 1.1)
                self.canvas_class_pre.draw()
                self.train_class_ax_rec.cla()
                Model_Training.per_class_plot(self, self.train_class_ax_rec, class_names, history['per_class'], 'val_recall', possible_line_styles)
                self.train_class_ax_rec.set_xlabel("Epoch")
                self.train_class_ax_rec.set_ylabel("val_Recall")
                self.train_class_ax_rec.legend(loc="upper left", prop={'size': 7})
                self.train_class_ax_rec.set_title("per class recall", fontsize=14)
                self.train_class_ax_rec.set_ylim(0, 1.1)
                self.canvas_class_rec.draw()
                # self.controller.root.update_idletasks()
                # self.controller.root.update()
            # -------------------- PLOTTING ------------------------


            # Dynamically find the Monitor_Training_Page frame by class name
            monitor_page = next( (frame for cls, frame in self.controller.frames.items() if cls.__name__ == "Monitor_Training_Page") , None  )

            if monitor_page is None:
                print("Monitor_Training_Page not found in frames.")
                return


            #Destroy all children in each sub-frame to clear old plots
            for child in monitor_page.First_Canvas_Row_Accuracy.winfo_children():
                child.destroy()
            for child in monitor_page.First_Canvas_Row_Precision.winfo_children():
                child.destroy()
            for child in monitor_page.First_Canvas_Row_Recall.winfo_children():
                child.destroy()
            for child in monitor_page.First_Canvas_Row_Loss.winfo_children():
                child.destroy()

            for child in monitor_page.Second_Canvas_Row_Class_Accuracy.winfo_children():
                child.destroy()
            for child in monitor_page.Second_Canvas_Row_Class_Precision.winfo_children():
                child.destroy()
            for child in monitor_page.Second_Canvas_Row_Class_Recall.winfo_children():
                child.destroy()
            for child in monitor_page.Second_Canvas_delta_Loss.winfo_children():
                child.destroy()



            self.train_fig_acc, self.train_ax_acc = plt.subplots(figsize=(4, 4), dpi=100)

            canvas_2 = FigureCanvasTkAgg(self.train_fig_acc, master=monitor_page.First_Canvas_Row_Accuracy)
            canvas_2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            self.train_ax_acc.plot(history['accuracy'], color='teal', label='accuracy')
            self.train_ax_acc.plot(history['val_accuracy'], color='orange', label='val_accuracy')
            self.train_ax_acc.set_xlabel("Epoch")
            self.train_ax_acc.set_ylabel("Accuracy")
            self.train_ax_acc.set_title("Accuracy", fontsize=17)
            self.train_ax_acc.set_ylim(0, 1.1)
            self.train_ax_acc.legend(loc="upper left" )

            canvas_2.draw()

            toolbar_2 = NavigationToolbar2Tk(canvas_2, monitor_page.First_Canvas_Row_Accuracy, pack_toolbar=False)
            toolbar_2.update()
            toolbar_2.pack(side=tk.LEFT, fill=tk.X)
            plt.close(self.train_fig_acc)


            self.train_fig_pre , self.train_ax_pre = plt.subplots(figsize=(4, 4), dpi=100)

            canvas_3 = FigureCanvasTkAgg(self.train_fig_pre, master=monitor_page.First_Canvas_Row_Precision)
            canvas_3.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            self.train_ax_pre.plot(history['val_precision'], color='orange', label='val_precision')
            self.train_ax_pre.set_xlabel("Epoch")
            self.train_ax_pre.set_ylabel("Precision")
            self.train_ax_pre.set_title("Precision", fontsize=14)
            self.train_ax_pre.set_ylim(0, 1.1)
            self.train_ax_pre.legend(loc="upper left"   , prop={'size': 7})

            canvas_3.draw()

            toolbar_3 = NavigationToolbar2Tk(canvas_3, monitor_page.First_Canvas_Row_Precision, pack_toolbar=False)
            toolbar_3.update()
            toolbar_3.pack(side=tk.LEFT, fill=tk.X)
            plt.close(self.train_fig_pre)


            self.train_fig_recall , self.train_ax_recall = plt.subplots(figsize=(4, 4), dpi=100)

            canvas_4 = FigureCanvasTkAgg(self.train_fig_recall, master=monitor_page.First_Canvas_Row_Recall)
            canvas_4.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            self.train_ax_recall.plot(history['val_recall'], color='orange', label='val_recall')
            self.train_ax_recall.set_xlabel("Epoch")
            self.train_ax_recall.set_ylabel("Recall")
            self.train_ax_recall.set_title("Recall", fontsize=14)
            self.train_ax_recall.set_ylim(0, 1.1)
            self.train_ax_recall.legend(loc="upper left" ,  prop={'size': 7})

            canvas_4.draw()

            toolbar_4 = NavigationToolbar2Tk(canvas_4, monitor_page.First_Canvas_Row_Recall, pack_toolbar=False)
            toolbar_4.update()
            toolbar_4.pack(side=tk.LEFT, fill=tk.X)
            plt.close(self.train_fig_recall)


            self.train_fig_loss , self.train_ax_loss = plt.subplots(figsize=(4, 4), dpi=100)

            canvas_5 = FigureCanvasTkAgg(self.train_fig_loss, master=monitor_page.First_Canvas_Row_Loss)
            canvas_5.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            self.train_ax_loss.plot(history['loss'], color='blue', label='loss')
            self.train_ax_loss.plot(history['val_loss'], color='orange', label='val_loss')
            self.train_ax_loss.set_xlabel("Epoch")
            self.train_ax_loss.set_ylabel("Loss")
            self.train_ax_loss.set_title("Loss", fontsize=14)
            self.train_ax_loss.legend(loc="upper left")

            monitor_page.loss_fig_ax = self.train_ax_loss
            monitor_page.loss_fig_canvas = canvas_5

            canvas_5.draw()

            toolbar_5 = NavigationToolbar2Tk(canvas_5, monitor_page.First_Canvas_Row_Loss, pack_toolbar=False)
            toolbar_5.update()
            toolbar_5.pack(side=tk.LEFT, fill=tk.X)
            plt.close(self.train_fig_loss)



            self.train_class_fig_acc , self.train_class_ax_acc = plt.subplots(figsize=(4, 4), dpi=100)

            canvas_6 = FigureCanvasTkAgg(self.train_class_fig_acc, master=monitor_page.Second_Canvas_Row_Class_Accuracy)
            canvas_6.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


            default_line_styles = ['-', '--' , ':' , '-.'] 
            possible_line_styles = [ i for i in default_line_styles ]
            

            for _ in range( len(class_names) - len(default_line_styles) ):
                possible_line_styles.append( (  np.random.randint(0,10) , ( np.random.randint(0,10) ,  np.random.randint(0,10) ,  np.random.randint(0,10) ,  np.random.randint(0,10)) )  )



            Model_Training.per_class_plot( self = self , ax_i =  self.train_class_ax_acc , class_names = class_names , plot_data = history['per_class'] , plot_data_key = 'val_accuracy' , possible_line_styles = possible_line_styles )


            self.train_class_ax_acc.set_xlabel("Epoch")
            self.train_class_ax_acc.set_ylabel("val_Accuracy")
            self.train_class_ax_acc.legend(loc="upper left" , prop={'size': 7})
            self.train_class_ax_acc.set_title("per class accuracy", fontsize=14)
            self.train_class_ax_acc.set_ylim(0, 1.1)

            monitor_page.class_accuracy_ax = self.train_class_ax_acc
            monitor_page.class_accuracy_canvas = canvas_6

            canvas_6.draw()

            toolbar_6 = NavigationToolbar2Tk(canvas_6, monitor_page.Second_Canvas_Row_Class_Accuracy, pack_toolbar=False)
            toolbar_6.update()
            toolbar_6.pack(side=tk.LEFT, fill=tk.X)
            plt.close(self.train_class_fig_acc)



            self.train_class_fig_pre , self.train_class_ax_pre = plt.subplots(figsize=(4, 4), dpi=100)

            canvas_7 = FigureCanvasTkAgg(self.train_class_fig_pre, master=monitor_page.Second_Canvas_Row_Class_Precision)
            canvas_7.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)




            Model_Training.per_class_plot( self = self , ax_i =  self.train_class_ax_pre , class_names = class_names , plot_data = history['per_class'] , plot_data_key = 'val_precision' , possible_line_styles = possible_line_styles )

            self.train_class_ax_pre.set_xlabel("Epoch")
            self.train_class_ax_pre.set_ylabel("val_Precision")
            self.train_class_ax_pre.legend( loc="upper left" ,prop=dict(size=7) ) 
            self.train_class_ax_pre.set_title("per class precision", fontsize=14)
            self.train_class_ax_pre.set_ylim(0, 1.1)

            monitor_page.class_precision_ax = self.train_class_ax_pre
            monitor_page.class_precision_canvas = canvas_7
            canvas_7.draw()

            toolbar_7 = NavigationToolbar2Tk(canvas_7, monitor_page.Second_Canvas_Row_Class_Precision, pack_toolbar=False)
            toolbar_7.update()
            toolbar_7.pack(side=tk.LEFT, fill=tk.X)
            plt.close(self.train_class_fig_pre)



            self.train_class_fig_rec , self.train_class_ax_rec = plt.subplots(figsize=(4, 4), dpi=100)

            canvas_8 = FigureCanvasTkAgg(self.train_class_fig_rec, master=monitor_page.Second_Canvas_Row_Class_Recall)
            canvas_8.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            Model_Training.per_class_plot( self = self , ax_i =  self.train_class_ax_rec , class_names = class_names , plot_data = history['per_class'] , plot_data_key = 'val_recall', possible_line_styles = possible_line_styles )


            self.train_class_ax_rec.set_xlabel("Epoch")
            self.train_class_ax_rec.set_ylabel("val_Recall")
            self.train_class_ax_rec.legend(loc="upper left" ,  prop=dict(size=7) ) 
            self.train_class_ax_rec.set_title("per class recall", fontsize=14)
            self.train_class_ax_rec.set_ylim(0, 1.1)

            monitor_page.class_recall_ax = self.train_class_ax_rec
            monitor_page.class_recall_canvas = canvas_8
            canvas_8.draw()

            toolbar_8 = NavigationToolbar2Tk(canvas_8, monitor_page.Second_Canvas_Row_Class_Recall, pack_toolbar=False)
            toolbar_8.update()
            toolbar_8.pack(side=tk.LEFT, fill=tk.X)
            plt.close(self.train_class_fig_rec)



            self.train_delta_loss_fig , self.train_delta_loss_ax = plt.subplots(figsize=(4, 4), dpi=100)

            canvas_9 = FigureCanvasTkAgg(self.train_delta_loss_fig, master=monitor_page.Second_Canvas_delta_Loss)
            canvas_9.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            self.train_delta_loss_ax.plot( (np.array( history['val_loss'] ) - np.array( history['loss'] )) ,  )
            self.train_delta_loss_ax.hlines( y = 0 , xmin = 0 , xmax= len(  history['val_loss'] )  , colors='black' , linestyles='--')

            self.train_delta_loss_ax.set_xlabel("Epoch")
            self.train_delta_loss_ax.set_ylabel(r"$\Delta$ Loss")
            self.train_delta_loss_ax.set_title(r"$\Delta$ Loss", fontsize=14)

            canvas_9.draw()

            toolbar_9 = NavigationToolbar2Tk(canvas_9, monitor_page.Second_Canvas_delta_Loss, pack_toolbar=False)
            toolbar_9.update()
            toolbar_9.pack(side=tk.LEFT, fill=tk.X)
            plt.close(self.train_delta_loss_fig)

        
        if int(num_epochs) != 0 :
            translated = [ Model_Training.translate_label(self, cls) for cls in class_names ]
            monitor_page.setup_class_filter( self, translated)



        all_test_true_labels = []
        all_test_pred_labels = []

        for x_val_batch, y_val_batch in test:
            predictions = self.controller.model.predict(x_val_batch)
            rounded_preds = np.argmax(predictions, axis=-1)

            all_test_true_labels.extend(y_val_batch.numpy())
            all_test_pred_labels.extend(rounded_preds)

        self.confusion_fig, self.confusion_ax = plt.subplots(figsize=(6, 6), dpi=160)
        plt.subplots_adjust(left=0.216, right=1, top=0.96, bottom=0.212, wspace=0.4, hspace=0.3)

        Confusion_Page = next( (frame for cls, frame in self.controller.frames.items() if cls.__name__ == "Show_Confusion_Page"), None  )

        if Confusion_Page is None:
            print("Show_Confusion_Page not found in frames.")
            return

        for child in Confusion_Page.Confusion_Canvas_Frame.winfo_children():
            child.destroy()

        Confusion_Canvas = FigureCanvasTkAgg(self.confusion_fig, master=Confusion_Page.Confusion_Canvas_Frame)
        Confusion_Canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        all_test_true_labels_cm = list(map( lambda i : Model_Training.translate_label( self = self, label = class_names[i] )  , all_test_true_labels))
        all_test_pred_labels_cm = list(map( lambda i : Model_Training.translate_label( self = self, label = class_names[i] )  , all_test_pred_labels))

        Confusion_dict_data = pd.DataFrame( data = {'Actual Interaction' : all_test_true_labels_cm , 'Predicted Interaction' : all_test_pred_labels_cm } )

        confusion_matrix = pd.crosstab( Confusion_dict_data['Actual Interaction'] , Confusion_dict_data['Predicted Interaction'] ,
                                        rownames= ['Actual Interaction'] , colnames= ['Predicted Interaction'] , 
                                        normalize= 'index')

        sn.heatmap( confusion_matrix, annot=True , ax=self.confusion_ax  , xticklabels = True, yticklabels = True , fmt=".2f" , annot_kws={'size': 7} , cbar_kws={"shrink": 0.7})

        self.confusion_ax.tick_params( axis='both', which='major', labelsize= 6 )
        self.confusion_ax.set_xlabel('Predicted Interaction')
        self.confusion_ax.set_ylabel('Actual Interaction')
        # self.confusion_ax.set_title('Confusion Matrix', fontsize=16)

        Confusion_Canvas.draw()

        toolbar_confusion = NavigationToolbar2Tk(Confusion_Canvas, Confusion_Page.Confusion_Canvas_Frame, pack_toolbar=False)
        toolbar_confusion.update()
        toolbar_confusion.pack(side=tk.LEFT, fill=tk.X)

        plt.close(self.confusion_fig)

        Confusion_Page.confusion_fig_ax = self.confusion_ax
        Confusion_Page.confusion_fig_canvas = Confusion_Canvas


        Main_Run_File.Show_Model_Filters_Page.Model_Filters_Ready(self)

        Main_Run_File.Frame_Manager.cancel_process(self)

        self.controller.model.class_name_dict = { i : Model_Training.translate_label(self ,j)  for i , j in enumerate(class_names)} 


        self.controller.model.history_dict = history

        self.controller.test_images = test


        return




    def per_class_plot(self, ax_i , class_names , plot_data , plot_data_key , possible_line_styles ):

        cnter  = 0 
        default_line_styles = ['-', '--' , ':' , '-.'] 
        for c_name  in  class_names :

            label = Model_Training.translate_label( self= self , label = c_name)

            ax_i.plot( plot_data[c_name][ plot_data_key ] , label = label , linestyle = possible_line_styles[ cnter ] )
            
            cnter +=1
        return
    

    def translate_label(self,label):

        if label == 'Neutrino_Electron_Scattering':
            translated_label = r'$\nu - e$ (scattering)'
        
        elif label == 'Neutrino_Electron_Scattering_NC':
            translated_label = r'$\nu - e$ (scattering) (NC)'

        elif label == 'Neutrino_Electron_Scattering_CC':
            translated_label = r'$\nu - e$ (scattering) (CC)'
        
        elif label == 'Anti_Electron_Neutrino_CC':
            translated_label = r'$\bar{\nu_e} - e$ (CC)'

        elif label == 'Electron_Neutrino':
            translated_label = r'$\nu_{e}$ - (NC/CC)'
        
        elif label == 'Electron_Neutrino_NC':
            translated_label = r'$\nu_{e}$ - NC'

        elif label == 'Electron_Neutrino_CC':
            translated_label = r'$\nu_{e}$ - CC'

        elif label == 'Neutrino_NC':
            translated_label = r'$\nu$ - NC'

        elif label == 'Neutrino_CC_Other':
            translated_label = r'$\nu$ - CC (Other)'

        elif label == 'Neutrino_NC_Other':
            translated_label = r'$\nu$ - NC (Other)'

        elif label == 'Other':
            translated_label = 'Other'

        else:
            translated_label = label


        return translated_label
