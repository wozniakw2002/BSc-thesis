from Modules.DataLoaderModule import DataLoader
from Modules.PreprocessingModule import Preprocessor
from Modules.StatisticsModule import Statistics
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
import pickle

class DataPreparation:
    """
    A class to handle data preparation and preprocessing for machine learning tasks.

    Methods:
    --------
    data_formatting(data_tuple: tuple[np.array, np.array]) -> tuple[np.array, np.array]:
        Formats the input data by converting it into numpy arrays, expanding dimensions, and ensuring the correct dtype.

    training_data_preparation(...):
        Loads, preprocesses, and optionally augments data for training, validation, and testing.

    single_photo_preparation(path: str = None, image = None) -> np.ndarray:
        Prepares a single photo for prediction by preprocessing and reshaping.
    """

    @staticmethod
    def data_formatting(data_tuple: tuple[np.array, np.array]) -> tuple[np.array, np.array]:
        """
        Format the data by converting it to numpy arrays and reshaping.

        Parameters:
        ----------
        data_tuple: tuple[np.array, np.array]
            A tuple containing features (X) and labels (Y).

        Returns:
        -------
        tuple[np.array, np.array]
            Reshaped features and labels as numpy arrays.
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
        Load and preprocess data for training, validation, and testing.

        Parameters:
        ----------
        full_augmentation: bool
            Whether to apply full data augmentation or just flipping.

        save_data: bool
            Whether to save the processed data to files.

        save_train_path: str, save_val_path: str, save_test_path: str
            Paths to save processed training, validation, and test data.

        load_saved_data: bool
            Whether to load preprocessed data from saved files.

        train_path: str, val_path: str, test_path: str
            Paths to load saved training, validation, and test data.

        Returns:
        -------
        tuple:
            Training, validation, and test features and labels.
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
    def single_photo_preparation(path: str = None, image = None) -> np.ndarray:
        """
        Prepare a single photo for prediction.

        Parameters:
        ----------
        path: str
            Path to the photo (if image is not directly provided).

        image:
            PIL image object (optional).

        Returns:
        -------
        np.ndarray
            Preprocessed photo ready for prediction.
        """
        if image is None:
            x = DataLoader.load_single_photo(path)
        else:
            image = image.convert('L')
            x = [np.array(image)]
        x, _ = Preprocessor.preprocessing(np.array(x))
        x = np.expand_dims(np.array(x.tolist(), dtype=np.float32), axis=-1)
        return x


class Model:
    """
    A class to define, train, and evaluate Convolutional Neural Network (CNN) models.

    Methods:
    --------
    example_model(input_shape: tuple = (224, 224, 1)) -> Sequential:
        Defines and compiles a simple CNN model.

    f1_score(precision: np.ndarray, recall: np.ndarray) -> np.ndarray:
        Computes the F1 score given precision and recall arrays.

    train_model(...):
        Trains a CNN model and computes F1 scores during training.

    predict(...):
        Generates predictions and thresholded labels for input data.

    predict_training(...):
        Predicts results for training, validation, and test datasets.

    save_history(...):
        Saves training history to a file.

    save_predictions(...):
        Saves predictions and ground truth labels to a file.

    training_pipeline(...):
        Executes a full training pipeline from data preparation to evaluation.
    """

    @staticmethod
    def example_model(input_shape: tuple = (224, 224, 1)) -> Sequential:
        """
        Define and compile a simple Convolutional Neural Network (CNN) model.

        Parameters:
        ----------
        input_shape: tuple
            Shape of the input images.

        Returns:
        -------
        Sequential
            Compiled Keras model ready for training.
        """

        model = Sequential([
            Input(shape=input_shape),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
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
        Compute the F1 score from precision and recall values.

        Parameters:
        ----------
        precision: np.ndarray
            Array of precision values.

        recall: np.ndarray
            Array of recall values.

        Returns:
        -------
        np.ndarray
            Array of F1 scores.
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
        Train the model and compute the F1 score during training.

        Parameters:
        ----------
        model : Sequential
            The Keras model to be trained.
        X_train : np.ndarray
            Training data features.
        Y_train : np.ndarray
            Training data labels.
        X_val : np.ndarray
            Validation data features.
        Y_val : np.ndarray
            Validation data labels.
        epochs : int, optional
            Number of epochs for training. Default is 10.
        batch_size : int, optional
            Batch size for training. Default is 32.

        Returns:
        -------
        tf.keras.callbacks.History
            The history object containing details of the training process.
        """

        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=epochs, batch_size=batch_size,
            callbacks=[stop_early]
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
        Generate predictions and classify data based on a specified threshold.

        Parameters:
        ----------
        model : Sequential
            The Keras model used for generating predictions.
        X : np.ndarray
            Data to predict on.
        threshold : float, optional
            Threshold value for classification. Default is 0.5.

        Returns:
        -------
        tuple
            A tuple containing raw predictions and thresholded labels.
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
        Generate predictions for training, validation, and test datasets.

        Parameters:
        ----------
        model : Sequential
            The trained Keras model for generating predictions.
        X_train : np.ndarray
            Training data features.
        X_val : np.ndarray
            Validation data features.
        X_test : np.ndarray
            Test data features.
        threshold : float, optional
            Threshold value for classification. Default is 0.5.

        Returns:
        -------
        dict
            A dictionary containing predictions and labels for training,
            validation, and test datasets.
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
        Save the training history to a pickle file.

        Parameters:
        ----------
        history : tf.keras.callbacks.History
            The training history object.
        name : str
            The name of the file (without extension) to save the history.
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
        predictions : dict
            A dictionary containing predictions and labels for all datasets.
        name : str
            The name of the file (without extension) to save the predictions.
        Y_train : np.ndarray
            Ground truth labels for the training dataset.
        Y_val : np.ndarray
            Ground truth labels for the validation dataset.
        Y_test : np.ndarray
            Ground truth labels for the test dataset.
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
        test_path: str = "",
        execution_count: int = 1,
        threshold: float = 0.5
    ) -> tuple:
        """
        Combines the full training pipeline, including data preparation, model training, 
        prediction generation, and performance evaluation.

        Parameters:
        ----------
        model : Sequential
            The neural network model to be trained.
        name : str
            Prefix for saving results (e.g., reports, predictions).
        epochs : int, optional
            Number of training epochs (default is 10).
        batch_size : int, optional
            Batch size for training (default is 32).
        full_augmentation : bool, optional
            Whether to apply full data augmentation during training (default is True).
        save_data : bool, optional
            Whether to save the processed training, validation, and test data (default is False).
        save_train_path : str, optional
            Path to save the processed training data if `save_data` is True (default is "").
        save_val_path : str, optional
            Path to save the processed validation data if `save_data` is True (default is "").
        save_test_path : str, optional
            Path to save the processed test data if `save_data` is True (default is "").
        load_saved_data : bool, optional
            Whether to load preprocessed data from saved files (default is False).
        train_path : str, optional
            Path to load preprocessed training data if `load_saved_data` is True (default is "").
        val_path : str, optional
            Path to load preprocessed validation data if `load_saved_data` is True (default is "").
        test_path : str, optional
            Path to load preprocessed test data if `load_saved_data` is True (default is "").
        execution_count : int, optional
            Number of times to execute the training process for ensemble averaging (default is 1).
        threshold : float, optional
            Threshold for classifying predictions (default is 0.5).

        Returns:
        -------
        tuple
            Contains the final training history and predictions dictionary.
        """

        # Data preparation step: load, preprocess, and optionally save data
        X_train, X_test, X_val, Y_train, Y_test, Y_val = DataPreparation.training_data_preparation(
            full_augmentation, save_data, save_train_path, save_val_path, save_test_path,
            load_saved_data, train_path, val_path, test_path
        )

        # Initial training and prediction
        history = Model.train_model(model, X_train, Y_train, X_val, Y_val, epochs, batch_size)
        predictions = Model.predict_training(model, X_train, X_val, X_test, threshold)

        # Initialize averaging variables for repeated execution
        loss_mean = np.array(history.history['loss'])
        val_loss_mean = np.array(history.history['val_loss'])
        Y_test_prob = predictions['prediction_test'].flatten()

        # Repeat training process for ensemble averaging if execution_count > 1
        for _ in range(1, execution_count):
            # Clone and reinitialize the model for each execution
            model_test = tf.keras.models.clone_model(model)
            model_test.set_weights(model.get_weights())
            model_test.compile(optimizer='adam', loss='binary_crossentropy',
                            metrics=['accuracy', 'AUC', 'Precision', 'Recall'])

            # Train and predict with the cloned model
            history = Model.train_model(model_test, X_train, Y_train, X_val, Y_val, epochs, batch_size)
            predictions = Model.predict_training(model_test, X_train, X_val, X_test, threshold)

            # Accumulate results for averaging
            loss_mean += np.array(history.history['loss'])
            val_loss_mean += np.array(history.history['val_loss'])
            Y_test_prob += predictions['prediction_test'].flatten()

        # Compute average results
        loss_mean = loss_mean / execution_count
        val_loss_mean = val_loss_mean / execution_count
        Y_test_pred_prob = Y_test_prob / execution_count
        Y_test_pred = (Y_test_pred_prob > threshold).astype("int32")

        # Report statistics and performance metrics
        Statistics.report(Y_test, Y_test_pred, Y_test_pred_prob, loss_mean, val_loss_mean, name)

        # Optionally save history and predictions (commented out by default)
        # Model.save_history(history, f'history_{name}')
        # Model.save_predictions(predictions, f'predictions_{name}', Y_train, Y_val, Y_test)

        return history, predictions
