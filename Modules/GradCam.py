import keras
import tensorflow as tf
import matplotlib as mpl
import numpy as np
from IPython.display import display
import copy

class GradCam:
    '''
    This class contains methods to visualize what a Convolutional Neural Network (CNN) is focusing on using the Grad-CAM method.

    Methods:
    --------
    create_gradcam_heatmap(image, model, last_conv_layer_index=4, last_index=None) -> tf.Tensor:
        Creates a Grad-CAM heatmap for a given image and model.

    overlap_gradcam(image, heatmap, alpha=0.4) -> tf.Tensor:
        Overlays a Grad-CAM heatmap onto the input image.

    create_and_overlap_gradcam(image_h, image_o, model, last_conv_layer_index=4, last_index=None, alpha=0.4) -> np.array:
        Creates and overlays a Grad-CAM heatmap onto the input image in a single step.
    '''

    @staticmethod
    def create_gradcam_heatmap(image, model: tf.keras.models.Sequential, last_conv_layer_index: int = 4, last_index: int = None) -> tf.Tensor:
        '''
        Creates a Grad-CAM heatmap.

        Arguments:
        ----------
        image: tf.Tensor
            Input image tensor for which Grad-CAM is being generated.
        model: tf.keras.models.Sequential
            Model to be used for Grad-CAM generation.
        last_conv_layer_index: int, default=4
            Index of the last convolutional layer in the model.
        last_index: int, default=None
            Index of the layer used for predictions. If None, the last layer is used.

        Returns:
        --------
        heatmap: tf.Tensor
            Grad-CAM heatmap tensor.
        '''
        new_model = tf.keras.models.clone_model(model)
        new_model = tf.keras.models.Model(inputs=new_model.inputs, outputs = new_model.outputs)
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
    def overlap_gradcam(image, heatmap: tf.Tensor, alpha: float = 0.4) -> tf.Tensor:
        '''
        Overlays a Grad-CAM heatmap onto an image.

        Arguments:
        ----------
        image: tf.Tensor
            Original image tensor.
        heatmap: tf.Tensor
            Grad-CAM heatmap tensor.
        alpha: float, default=0.4
            Intensity factor for overlaying the heatmap.

        Returns:
        --------
        composed_image: tf.Tensor
            Image with the heatmap overlay applied.
        '''

        heatmap = np.uint8(255 * heatmap)
        jet_colors = mpl.colormaps["jet"](np.arange(256))[:, :3]
        colored_heatmap = jet_colors[heatmap]
        resized_colored_heatmap = keras.utils.img_to_array(keras.utils.array_to_img(colored_heatmap).resize((224, 224)))
        composed_image = keras.utils.array_to_img(resized_colored_heatmap * alpha + image * 255)
        return composed_image
    


    @staticmethod
    def create_and_overlap_gradcam(image_h, image_o, model: tf.keras.models.Sequential, last_conv_layer_index: int = 4, last_index: int = None, alpha: float = 0.4) -> np.array:
        '''
        Combines Grad-CAM heatmap generation and overlaying in a single step.

        Arguments:
        ----------
        image_h: tf.Tensor
            Image tensor used for heatmap generation.
        image_o: tf.Tensor
            Original image tensor for overlaying the heatmap.
        model: tf.keras.models.Sequential
            Model to be used for Grad-CAM generation.
        last_conv_layer_index: int, default=4
            Index of the last convolutional layer in the model.
        last_index: int, default=None
            Index of the layer used for predictions. If None, the last layer is used.
        alpha: float, default=0.4
            Intensity factor for overlaying the heatmap.

        Returns:
        --------
        image: np.array
            Image with the heatmap overlay applied.
        '''

        heatmap = GradCam.create_gradcam_heatmap(image_h, model, last_conv_layer_index, last_index)
        image = GradCam.overlap_gradcam(image_o, heatmap, alpha)
        return image