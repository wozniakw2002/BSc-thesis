from Modules.DataLoaderModule import DataLoader
from Modules.PreprocessingModule import Preprocessor
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import pickle


class DataPreparation:
    """
    A class for preparing and preprocessing data for training, validation, and testing.

    Methods:
    --------
    data_formatting(data_tuple):
        Formats the input data by reshaping and converting to numpy arrays.
    
    training_data_preparation(...):
        Loads and preprocesses data, optionally saves or loads from files.
    
    single_photo_preparation(str):
        Prepares a single photo for prediction (placeholder for implementation).
    """

    @staticmethod
    def data_formatting(data_tuple: tuple) -> tuple:
        """
        Format the data by converting it to numpy arrays and reshaping.

        Parameters:
        ----------
        data_tuple: tuple -> A tuple containing features (X) and labels (Y).
        
        Returns:
        -------
        tuple -> Reshaped features and labels.
        """

        X, Y = data_tuple
        X_formatted = np.expand_dims(np.array(X.tolist(), dtype=np.float32), axis=-1)
        Y_formatted = Y.astype(np.int32)
        return X_formatted, Y_formatted

    @staticmethod
    def training_data_preparation(
        full_augmentation: bool = True,
        save_data: bool = False,
        save_train_path: str = "",
        save_val_path: str = "",
        save_test_path: str = "",
        load_saved_data: bool = False,
        train_path: str = "",
        val_path: str = "",
        test_path: str = ""
    ) -> tuple:
        """
        Load and preprocess training, validation, and test data.

        Parameters:
        ----------
        full_augmentation: bool -> Whether to apply full data augmentation or only flipping.
        save_data: bool -> Whether to save the processed data to files.
        save_train_path: str, save_val_path: str, save_test_path: str -> Paths for saving processed training, validation, and test data.
        load_saved_data: bool -> Whether to load preprocessed data from files.
        train_path: str, val_path: str, test_path: str -> Paths for loading saved data.

        Returns:
        -------
        tuple -> Training, validation, and test features and labels.
        """

        if load_saved_data:
            X_train, Y_train = DataLoader.load_data_as_array(train_path)
            X_val, Y_val = DataLoader.load_data_as_array(val_path)
            X_test, Y_test = DataLoader.load_data_as_array(test_path)
        else:
            loader = DataLoader()
            X, Y = loader.load_data_from_folder_as_png('data')
            X_train, X_test, X_val, Y_train, Y_test, Y_val = Preprocessor.split_data(X, Y)

            X_train, Y_train = Preprocessor.preprocessing(X_train, Y_train)
            if full_augmentation:
                X_train, Y_train = Preprocessor.augmentation(X_train, Y_train)
            else:
                X_train, Y_train = Preprocessor.flip_photos(X_train, Y_train)

            X_train, Y_train = DataPreparation.data_formatting((X_train, Y_train))
            X_val, Y_val = DataPreparation.data_formatting(
                Preprocessor.preprocessing(X_val, Y_val)
            )
            X_test, Y_test = DataPreparation.data_formatting(
                Preprocessor.preprocessing(X_test, Y_test)
            )

        if save_data:
            DataLoader.save_data_as_array(save_train_path, X_train, Y_train)
            DataLoader.save_data_as_array(save_val_path, X_val, Y_val)
            DataLoader.save_data_as_array(save_test_path, X_test, Y_test)

        return X_train, X_test, X_val, Y_train, Y_test, Y_val
    
    @staticmethod
    def single_photo_preparation(path: str) -> np.ndarray:
        """
        Prepare a single photo for prediction.

        Parameters:
        ----------
        path: str -> Path to the photo.

        Returns:
        -------
        np.ndarray -> Preprocessed photo ready for prediction.
        """
        
        pass
        # loader = DataLoader()
        # x = loader.load_single_photo(path)
        # x = Preprocessor.preprocessing(x)
        # x = DataPreparation.data_formatting(x)
        # return x


class Model:
    """
    A class for defining, training, and evaluating CNN models.

    Methods:
    --------
    example_model(input_shape)
        Defines and compiles a simple CNN model.

    f1_score(precision, recall)
        Computes the F1 score given precision and recall.

    train_model(...)
        Trains the model and logs metrics.

    predict(model, X, threshold)
        Generates predictions and thresholded labels.
    
    predict_training(model, ...)
        Predicts results for train, validation, and test datasets.

    save_history(history, name)
        Saves the training history to a file.

    save_predictions(predictions, name, ...)
        Saves predictions and labels to a file.

    training_pipeline(...)
        Combines the full training pipeline, including data preparation, training, and evaluation.
    """

    @staticmethod
    def example_model(input_shape: tuple = (224, 224, 1)) -> Sequential:
        """
        Define and compile a simple CNN model.

        Parameters:
        ----------
        input_shape: tuple -> Shape of the input images.

        Returns:
        -------
        Sequential -> Compiled Keras model.
        """

        model = Sequential([
            Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC', 'Precision', 'Recall']
        )
        return model

    @staticmethod
    def f1_score(precision: np.ndarray, recall: np.ndarray) -> np.ndarray:
        """
        Compute F1 score from precision and recall.

        Parameters:
        ----------
        precision: np.ndarray -> Array of precision values.
        recall: np.ndarray -> Array of recall values.

        Returns:
        -------
        np.ndarray -> Array of F1 scores.
        """

        return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    

    @staticmethod
    def train_model(
        model: Sequential,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32
    ) -> tf.keras.callbacks.History:
        """
        Train the model and compute F1 score during training.

        Parameters:
        ----------
        model: Sequential -> Keras model to be trained.
        X_train, Y_train, X_val, Y_val: np.ndarray -> Training and validation datasets.
        epochs: int -> Number of training epochs.
        batch_size: int -> Batch size for training.

        Returns:
        -------
        tf.keras.callbacks.History -> Training history object.
        """

        history = model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=epochs, batch_size=batch_size
        )

        history.history['f1_score'] = Model.f1_score(
            np.array(history.history['Precision']),
            np.array(history.history['Recall'])
        ).tolist()

        history.history['val_f1_score'] = Model.f1_score(
            np.array(history.history['val_Precision']),
            np.array(history.history['val_Recall'])
        ).tolist()
        return history

    @staticmethod
    def predict(model: Sequential, X: np.ndarray, threshold: float = 0.5) -> tuple:
        """
        Predict and classify data based on a threshold.

        Parameters:
        ----------
        model: Sequential -> Keras model for generating predictions.
        X: np.ndarray -> Data to predict on.
        threshold: float -> Threshold for classification.

        Returns:
        -------
        tuple -> Raw predictions and thresholded labels.
        """

        predictions = model.predict(X)
        labels = (predictions > threshold).astype("int32")
        return predictions, labels

    @staticmethod
    def predict_training(
        model: Sequential,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        threshold: float = 0.5
    ) -> dict:
        """
        Generate predictions for training, validation, and test sets.

        Parameters:
        ----------
        model: Sequential -> Trained model for generating predictions.
        X_train, X_val, X_test: np.ndarray -> Datasets for predictions.
        threshold: float -> Classification threshold.

        Returns:
        -------
        dict -> Dictionary with predictions and labels for all datasets.
        """

        train_predictions = Model.predict(model, X_train, threshold)
        val_predictions = Model.predict(model, X_val, threshold)
        test_predictions = Model.predict(model, X_test, threshold)

        predictions = {
            "prediction_train": train_predictions[0],
            "prediction_label_train": train_predictions[1],
            "prediction_val": val_predictions[0],
            "prediction_label_val": val_predictions[1],
            "prediction_test": test_predictions[0],
            "prediction_label_test": test_predictions[1]
        }
        return predictions
    
    @staticmethod
    def save_history(history: tf.keras.callbacks.History, name: str) -> None:
        """
        Save training history to a pickle file.

        Parameters:
        ----------
        history: tf.keras.callbacks.History -> Training history object.
        name: str -> Name of the file to save history.
        """

        with open(f'{name}.pkl', 'wb') as pickle_file:
            pickle.dump(history.history, pickle_file)

    @staticmethod
    def save_predictions(
        predictions: dict,
        name: str,
        Y_train: np.ndarray,
        Y_val: np.ndarray,
        Y_test: np.ndarray
    ) -> None:
        """
        Save predictions and ground truth labels to a pickle file.

        Parameters:
        ----------
        predictions: dict -> Predictions and labels for training, validation, and test sets.
        name: str -> Name of the file to save predictions.
        Y_train, Y_val, Y_test: np.ndarray -> Ground truth labels for datasets.
        """

        predictions.update({
            "label_train": Y_train,
            "label_val": Y_val,
            "label_test": Y_test
        })
        with open(f'{name}.pkl', 'wb') as pickle_file:
            pickle.dump(predictions, pickle_file)

    @staticmethod
    def training_pipeline(
        model: Sequential,
        name: str,
        epochs: int = 10,
        batch_size: int = 32,
        full_augmentation: bool = True,
        save_data: bool = False,
        save_train_path: str = "",
        save_val_path: str = "",
        save_test_path: str = "",
        load_saved_data: bool = False,
        train_path: str = "",
        val_path: str = "",
        test_path: str = ""
    ) -> tuple:
        """
        Combines the full training pipeline.

        Parameters:
        ----------
        model: Sequential -> Model to be trained.
        name: str -> Prefix for saving results.
        epochs: int -> Number of training epochs.
        batch_size: int -> Batch size for training.
        full_augmentation: bool -> Whether to apply full augmentation.
        save_data: bool -> Whether to save processed data.
        load_saved_data: bool -> Whether to load preprocessed data from files.

        Returns:
        -------
        tuple, dict -> Training history and predictions.
        """
        
        X_train, X_test, X_val, Y_train, Y_test, Y_val = DataPreparation.training_data_preparation(
            full_augmentation, save_data, save_train_path, save_val_path, save_test_path,
            load_saved_data, train_path, val_path, test_path
        )

        history = Model.train_model(model, X_train, Y_train, X_val, Y_val, epochs, batch_size)
        predictions = Model.predict_training(model, X_train, X_val, X_test, threshold=0.5)

        Model.save_history(history, f'history_{name}')
        Model.save_predictions(predictions, f'predictions_{name}', Y_train, Y_val, Y_test)
        return history, predictions