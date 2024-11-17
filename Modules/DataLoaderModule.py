import cv2
import os
import numpy as np
import pickle
class DataLoader:
    """
    This class loades and saves data.
    
    Methods:
    --------
    load_data_from_folder_as_png(path: str)
        Loades data from folder and its subfolders recursively.
    
    load_single_photo(self, path: str)
        Loades single photo from path.

    save_data_as_array(self, path: str, X: np.array, Y: np.array)
        Saves photos and labels as arrays.
    
    load_data_as_array(self, path)
        Loades data saved as array.
    """

    __index = 0

    def __load_data(self, path: str, X: np.array, Y: np.array) -> tuple:
        """
        This is a private method for loading data from folder.

        Parametrs:
        ----------
        path: str -> relative path to data folder

        Returns:
        --------
        X,Y - arrays of data and labels (0 - healthy, 1 - ill)
        """

        os.chdir(os.path.join(os.getcwd(), path))
        directories = os.listdir(os.getcwd())
        for el in directories:
            path_to_subdirectory = os.path.join(os.getcwd(), el)
            if os.path.isfile(path_to_subdirectory):
                label = os.path.basename(os.path.normpath(os.getcwd()))
                X[self.__index] = cv2.imread(path_to_subdirectory, 0)
                if label == 'Normal':
                    Y[self.__index] = 0
                else:
                    Y[self.__index] = 1
                self.__index +=1
            else:
                self.__load_data(path_to_subdirectory, X,Y)
        os.chdir('..')
        return X, Y

    def load_data_from_folder_as_png(self, path: str) -> tuple:
        """
        This method takes relative path to data folder and loades data from its all subfolders.
        Photos have to be exacly in folders like Normal and Osteoarthritis but they don't have that folders can be subfolders of other.

        Parametrs:
        ----------
        path: str -> relative path to data folder

        Returns:
        --------
        X: np.array -> photos in a grey scale
        Y: np.array -> labels of photos, 0 - healthy, 1 - ill
        """

        num_files = 0
        for _, _, files in os.walk(os.path.join(os.getcwd(), path)):
            num_files += len(files)

        X = np.empty(num_files, dtype=object)
        Y = np.empty(num_files, dtype=object)
        X,Y = self.__load_data(path, X,Y)
        self.__index = 0
        return X,Y

    def load_single_photo(self, path: str) -> np.array:
        """
        This method takes relative path to photo and loads it.

        Parametrs:
        ----------
        path: str -> relative path to data

        Returns:
        --------
        photo: np.array -> photo in a gray scale
        """

        photo = cv2.imread(path, 0)
        return [photo]
    
    def save_data_as_array(path: str, X: np.array, Y: np.array) -> None:
        """
        This void method saves data nad labels to the file as a np.array.

        Parametrs:
        ----------
        path: str -> path to file in which data are going to be saved
        X: np.array -> array of photos
        Y: np.array -> array of labels

        """

        with open(path, 'wb') as outfile:
            pickle.dump([X,Y], outfile, pickle.HIGHEST_PROTOCOL)
        print('File saved')

    def load_data_as_array(path: str) -> np.array:
        """
        This method reads data and its labels, saved as an array.

        Parametrs:
        ----------
        path: str -> path to file in which data are saved

        Returns:
        --------
        X: np.array -> array with photos
        Y: np.array -> array with labels
        """

        with open(path, 'rb') as infile:
            data = pickle.load(infile)
        X = data[0]
        Y = data[1]
        return X,Y
    
    def load_pickle_file(path: str):
        """
        This method loads pickle file.

        Parametrs:
        ----------
        path: str -> path to file in which pickle file is saved

        Returns:
        --------
        
        """

        with open(path, 'rb') as pickle_file:
            loaded_file= pickle.load(pickle_file)
        return loaded_file
    


