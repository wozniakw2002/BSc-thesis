import numpy as np
from sklearn.model_selection import train_test_split
import cv2

class Preprocessor:
    """
    This static class provides methodes useful during preprocessing.
    
    Methods:
    --------
    normalize_data(data)
        Normalizes data to the interval [0,1].
    
    split_data(X,Y)
        Splits data into training, validation and test datasets.

    split_wide_photos(X,Y)
        Splits photos with two knees into 2 photos with one knee on each.
    
    resize_photos(X)
        Resizes every photo to size (224, 224) pixels.
    
    flip_photos(X,Y)
        Makes a copy of every photo by flipping it by y-axis.
    """

    @staticmethod
    def normalize_data(data: np.array) -> np.array:
        """
        This static method normalizes data to the interval [0,1].

        Parametrs:
        ----------
        data: np.array -> Array of photos.
        
        Returns:
        --------
        normalized: np.array -> Array of dormalized data.
        """

        normalized = data/255
        return normalized
    
    @staticmethod
    def split_data(X: np.array, Y: np.array) -> tuple:
        """
        This static method splits data into training, test and validation sets.

        Parametrs:
        ----------
        X: np.array -> Array of photos.
        Y: np.array -> array of labels.

        Returns:
        --------
        tuple -> Arrays of splitaed data.
        """

        X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y,test_size=0.15, random_state=2024)
        X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=2024)
        return X_train, X_test, X_valid, Y_train, Y_test, Y_valid

    @staticmethod
    def split_wide_photos(X: np.array, Y: np.array) -> tuple:
        """
        This static method splits photos with two knees into 2 photos with one knee on each.

        Parametrs:
        ----------
        X: np.array -> array of photos.
        Y: np.array -> array of labels.

        Returns:
        --------
        tuple -> Arrays of splitaed data.
        """

        indexes = [i for i, v in enumerate(X) if np.shape(v) == (161,640)]
        X_wide = X[indexes]
        Y_wide = Y[indexes]
        X_split = np.empty(2 * len(indexes), dtype=object)
        Y_split = np.repeat(Y_wide, 2)
        for i in range(len(indexes)):
            X_split[2 * i] = X_wide[i][:, :320]
            X_split[2 * i + 1] = X_wide[i][:, 320:]
        
        excluded = [i for i in range(len(X)) if i not in indexes]
        X_final = np.append(X[excluded], X_split)
        Y_final = np.append(Y[excluded], Y_split)
        return X_final, Y_final
    
    @staticmethod
    def resize_photos(X: np.array) -> np.array:
        """
        This static method resizes photos to the size of (224,224) pixels.
        It uses cubic interpolation.

        Parametrs:
        ----------
        X: np.array -> array of photos.

        Returns:
        --------
        np.array -> array of resized photos.
        """

        X_res = np.empty(len(X), dtype=object)
        for i in range(len(X)):
            X_res[i] = cv2.resize(X[i], (224,224), interpolation=cv2.INTER_CUBIC)
        return X_res

    @staticmethod
    def flip_photos(X: np.array, Y: np.array) -> tuple:
        """
        This static method makes a copy of every photo by flipping it by y-axis.

        Parametrs:
        ----------
        X: np.array -> array of photos.
        Y: np.array -> array of labels.
        Returns:
        --------
        tuple-> array of concatenated original and copied photos, array of labels.
        """

        X_flipped = np.repeat(X,2)
        Y_flipped = np.repeat(Y, 2)
        for i in range(len(X)):
            X_flipped[2*i] = cv2.flip(X_flipped[2*i], 1)
        
        return X_flipped, Y_flipped
    
    @staticmethod
    def prepoccesing(X: np.array, Y: np.array) -> tuple:
        """
        This static method performs preprocessing on the input data X and Y by 
        splitting, resizing, and normalizing the images.

        Parametrs:
        ----------
        X: np.array -> array of photos.
        Y: np.array -> array of labels.

        Returns:
        --------
        tuple -> Arrays of preprocessed data.
        """
        X_splited, Y_splited = Preprocessor.split_wide_photos(X,Y)
        X_resized = Preprocessor.resize_photos(X_splited)
        X_normalized = Preprocessor.normalize_data(X_resized)
        return X_normalized, Y_splited

