import tensorflow as tf
import os
from django.conf import settings
from Modules.ModelModule import Model


class ModelSingleton:
    _model = None

    @staticmethod
    def get_model():
        if ModelSingleton._model is None:
            model_path = os.path.join(settings.BASE_DIR, 'model.h5')
            ModelSingleton._model = Model.example_model()
            ModelSingleton._model.load_weights(model_path)
        return ModelSingleton._model
