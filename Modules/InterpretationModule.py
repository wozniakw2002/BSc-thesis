from Modules.DataLoaderModule import DataLoader
import shap
import numpy as np
from lime import lime_image
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import keras
import io
import numpy as np
from PIL import Image


class Interpretation:
    def show_shap(model, photo):
        background = DataLoader.load_pickle_file('background.pickle')
        e = shap.DeepExplainer(model, background)
        shap_values = e.shap_values(photo)
        

        plt.figure() 
        shap.image_plot(shap_values, photo, show=False) 
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight') 
        buf.seek(0)
        plt.close()

        image = Image.open(buf)
        image_array = np.array(image)
        buf.close()
        
        return keras.utils.array_to_img(image_array)
    
    

    def get_class_probabilities(input_images, model):
        if len(np.shape(input_images)) == 4:
            images = np.empty(len(input_images), dtype=object)
            for i in range(len(input_images)):
                input_image = input_images[i,:,:,0]
                images[i] = input_image

            images = np.expand_dims(np.array(images.tolist(), dtype=np.float32), axis=-1)
        else:
            images = np.expand_dims(input_images, axis=0)[:,:,:,0]
            images = np.expand_dims(images,axis=3)

        pred = model.predict(images)
        num_classes = 2 
        probabilities = np.zeros((len(pred), num_classes))
        for i in range(len(pred)):
            probabilities[i, 1] = pred[i]  
            probabilities[i, 0] = 1 - pred[i]
        return probabilities
    
    def show_lime_interpretation(model, photo, label, num_samples=100):
        rgb_img = np.stack([photo[0].squeeze()] * 3, axis=-1)
        explainer = lime_image.LimeImageExplainer(random_state=0)
        explanation = explainer.explain_instance(
                        rgb_img, 
                        lambda x: Interpretation.get_class_probabilities(x, model),
                        labels=[0,1], num_samples=num_samples)

        image, mask = explanation.get_image_and_mask(label=label,positive_only=True, hide_rest=False)
        return keras.utils.array_to_img(mark_boundaries(image, mask))