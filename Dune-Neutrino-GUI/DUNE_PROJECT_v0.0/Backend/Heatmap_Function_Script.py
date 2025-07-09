from Imports.common_imports import *


class Heat_Map_Class:

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
