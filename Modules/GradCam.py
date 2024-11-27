import keras
import tensorflow as tf
import matplotlib as mpl
from Modules.ModelModule import DataPreparation
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
    def create_gradcam_heatmap(img_path: str, model: tf.keras.models.Sequential, last_conv_layer_index:int = 4, last_index: int=None) -> tf.Tensor:
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

        img_array =  DataPreparation.single_photo_preparation(img_path)
        new_model = tf.keras.models.Model(inputs=model.inputs, outputs = model.outputs)
        new_model.get_layer(index=-1).activation = None
        if last_index is None:
            grad_model = keras.models.Model(
                new_model.inputs, [new_model.get_layer(index = last_conv_layer_index).output, new_model.output]
            )
        
            with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
                last_conv_layer_output, preds = grad_model(img_array)
                class_channel = preds
        else:
            grad_model = keras.models.Model(
                new_model.inputs, [new_model.get_layer(index = last_conv_layer_index).output, new_model.get_layer(index=last_index).output]
            )
        
            with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
                last_conv_layer_output, preds = grad_model(img_array)
                pred_index = tf.argmax(preds[0])
                class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        reduced_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output_0 = last_conv_layer_output[0]
        heatmap = last_conv_layer_output_0 @ reduced_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        if float(tf.math.reduce_max(heatmap)) == 0:
            return GradCam.create_gradcam_heatmap(img_path, new_model, last_conv_layer_index=last_conv_layer_index, last_index=-4)
        else:
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            return heatmap
    
    @staticmethod
    def display_gradcam(img_path: str, heatmap: tf.Tensor, alpha: float=0.4) -> None:
        '''
        Method for displaying Grad-CAM heatmap.

        Arguments:
        ----------
        img_path: str -> path to a photo.
        heatmap: tf.Tensor -> Grad-CAM heatmap.
        alpha: float=0.4 -> intensivity of heatmap.
        '''

        img = keras.utils.load_img(img_path)
        img = keras.utils.img_to_array(img)

        heatmap = np.uint8(255 * heatmap)

        jet = mpl.colormaps["jet"]

        jet_colors = jet(np.arange(256))[:, :3]
        colored_heatmap = jet_colors[heatmap]

        colored_heatmap = keras.utils.array_to_img(colored_heatmap)
        colored_heatmap = colored_heatmap.resize((img.shape[1], img.shape[0]))
        colored_heatmap = keras.utils.img_to_array(colored_heatmap)

        composed_image = colored_heatmap * alpha + img
        composed_image = keras.utils.array_to_img(composed_image)

        display(composed_image)
    
    @staticmethod
    def create_and_display_gradcam(img_path: str, model: tf.keras.models.Sequential, last_conv_layer_index: int=4, alpha: float=0.4) -> None:
        '''
        Method for creating and displaying Grad-CAM heatmap.

        Arguments:
        ----------
        img_path: str -> path to a photo.
        model: tf.keras.models.Sequential -> model on which we are creating Grad-CAM.
        last_conv_layer_index:int = 4 -> index of last concolutional layer
        alpha: float=0.4 -> intensivity of heatmap.
        '''

        heatmap = GradCam.create_gradcam_heatmap(img_path, model, last_conv_layer_index)
        GradCam.display_gradcam(img_path, heatmap)