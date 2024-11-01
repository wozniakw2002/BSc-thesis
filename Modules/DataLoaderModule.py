import cv2
import os
import numpy as np
class DataLoader:

    __index = 0

    def __load_data(self, path: str, X: np.array, Y: np.array) -> tuple:
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
        num_files = 0
        for _, _, files in os.walk(os.path.join(os.getcwd(), path)):
            num_files += len(files)
        X = np.empty(num_files, dtype=object)
        Y = np.empty(num_files, dtype=object)
        X,Y = self.__load_data(path, X,Y)
        self.__index = 0
        return X,Y

    
    


loader = DataLoader()
X,Y = loader.load_data_from_folder_as_png('data')