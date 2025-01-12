import keras
import tensorflow as tf
import matplotlib as mpl
import numpy as np
from IPython.display import display
import copy
import cv2
from Modules.ModelModule import Model

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
    def create_and_overlap_gradcam(img, model: tf.keras.models.Sequential, last_conv_layer_name: str = 'conv2d_1') -> np.array:
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

        x = np.expand_dims(np.array(img.tolist(), dtype=np.float32), axis=0)

        preds = Model.predict(model, x)
        preds = preds[0]
        model(tf.keras.Input((224, 224, 1)))

        last_conv_layer = model.get_layer(last_conv_layer_name)

        grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.get_layer(index=-1).output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(x)
            loss = predictions[:, 0]
            loss_mod = -tf.math.log(1/loss - 1 -1e-8)

        grads = tape.gradient(loss_mod, conv_outputs)[0]
        grads = (grads+1e-8) / (np.max(np.abs(grads)) + 1e-8)
        weights = np.mean(grads, axis=(0, 1))
        cam = np.mean(weights * conv_outputs[0], axis=-1)
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam / cam.max()

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        image = keras.utils.array_to_img(heatmap)
        return image