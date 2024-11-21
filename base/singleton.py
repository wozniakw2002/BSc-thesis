import tensorflow as tf
import os
from django.conf import settings

class ModelSingleton:
    _model = None

    @staticmethod
    def get_model():
        if ModelSingleton._model is None:
            model_path = os.path.join(settings.BASE_DIR, 'model.h5')
            ModelSingleton._model = tf.keras.models.load_model(model_path)
        return ModelSingleton._model
