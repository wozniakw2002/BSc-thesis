import keras
import tensorflow as tf
import matplotlib as mpl
import numpy as np
from IPython.display import display

class GradCam:
    '''
    This class contains methods for visualise what CNN is actually looking at, using the Grad-CAM method.

    Methods:
    --------
    create_gradcam_heatmap(img_path, model, last_conv_layer_index=4, last_index=None)
        Method for creating Grad-CAM heatmap.
    
    display_gradcam(img_path, heatmap, alpha=0.4)
        Method for displaying Grad-Cam heatmap.

    create_and_display_gradcam(img_path, model, last_conv_layer_index=4, alpha=0.4)
        Method for both creating and displaying Grad-Cam heatmap.
    '''
    
    @staticmethod
    def create_gradcam_heatmap(image, model: tf.keras.models.Sequential, last_conv_layer_index:int = 4, last_index: int=None) -> tf.Tensor:
        '''
        Method for creating Grad-CAM heatmap.

        Arguments:
        ----------
        img_path: str -> path to a photo.
        model: tf.keras.models.Sequential -> model on which we are performing Grad-CAM.
        last_conv_layer_index:int = 4 -> index of last concolutional layer
        last_index: int=None -> index of layer performing prediction. If None then it's last layer.

        Returns:
        --------
        heatmap: tf.Tensor -> Grad-CAM heatmap.
        '''

        new_model = tf.keras.models.Model(inputs=model.inputs, outputs = model.outputs)
        new_model.get_layer(index=-1).activation = None
        if last_index is None:
            grad_model = keras.models.Model(
                new_model.inputs, [new_model.get_layer(index = last_conv_layer_index).output, new_model.output]
            )
        
            with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
                last_conv_layer_output, preds = grad_model(image)
                class_channel = preds
        else:
            grad_model = keras.models.Model(
                new_model.inputs, [new_model.get_layer(index = last_conv_layer_index).output, new_model.get_layer(index=last_index).output]
            )
        
            with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
                last_conv_layer_output, preds = grad_model(image)
                pred_index = tf.argmax(preds[0])
                class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        reduced_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output_0 = last_conv_layer_output[0]
        heatmap = last_conv_layer_output_0 @ reduced_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        if float(tf.math.reduce_max(heatmap)) == 0:
            return GradCam.create_gradcam_heatmap(image, new_model, last_conv_layer_index=last_conv_layer_index, last_index=-4)
        else:
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            return heatmap
    
    @staticmethod
    def overlap_gradcam(image, heatmap: tf.Tensor ,alpha: float=0.4) -> None:
        '''
        Method for overlaying a heatmap on an image.

        Arguments:
        ----------
        img_path: str -> path to a photo.
        heatmap: tf.Tensor -> Grad-CAM heatmap.
        alpha: float=0.4 -> intensivity of heatmap.

        Returns:
        --------
        heatmap: tf.Tensor -> Grad-CAM heatmap.
        '''
        heatmap = np.uint8(255 * heatmap)
        jet_colors = mpl.colormaps["jet"](np.arange(256))[:, :3]
        colored_heatmap = jet_colors[heatmap]
        resized_colored_heatmap = keras.utils.img_to_array(keras.utils.array_to_img(colored_heatmap).resize((224, 224)))
        composed_image = keras.utils.array_to_img(resized_colored_heatmap * alpha + image * 255)
        return composed_image
    


    @staticmethod
    def create_and_overlap_gradcam(image_h,image_o, model: tf.keras.models.Sequential, last_conv_layer_index:int = 4, last_index: int=None, alpha: float=0.4) -> None:
        '''
        Method for creating and overlapping Grad-CAM heatmap.

        Arguments:
        ----------
        img_path: str -> path to a photo.
        model: tf.keras.models.Sequential -> model on which we are creating Grad-CAM.
        last_conv_layer_index:int = 4 -> index of last concolutional layer
        alpha: float=0.4 -> intensivity of heatmap.
        '''

        heatmap = GradCam.create_gradcam_heatmap(image_h, model, last_conv_layer_index, last_index)
        image = GradCam.overlap_gradcam(image_o, heatmap)
        return image