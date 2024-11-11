import numpy as np
from sklearn.model_selection import train_test_split
import cv2

class Preprocessor:

    @staticmethod
    def normalize_data(data):
        normalized = data/255
        return normalized
    
    @staticmethod
    def split_data(X,Y):
        X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y,test_size=0.15, random_state=2024)
        X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=2024)
        return X_train, X_test, X_valid, Y_train, Y_test, Y_valid

    @staticmethod
    def split_wide_photos(X,Y):
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
    def resize_photos(X):
        X_res = np.empty(len(X), dtype=object)
        for i in range(len(X)):
            X_res[i] = cv2.resize(X[i], (224,224), interpolation=cv2.INTER_CUBIC)
        return X_res

    @staticmethod
    def flip_photos(X,Y):
        X_flipped = np.repeat(X,2)
        Y_flipped = np.repeat(Y, 2)
        for i in range(len(X)):
            X_flipped[2*i] = cv2.flip(X_flipped[2*i], 1)
        
        return X_flipped, Y_flipped
    