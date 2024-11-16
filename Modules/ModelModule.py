from Modules.DataLoaderModule import DataLoader
from Modules.PreprocessingModule import Preprocessor
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.optimizers import Adam
import pickle

class DataPreparation:
    def resize_ar(array):
        m=[]
        for ar in array:
            m.append(ar)
        return np.array(m)



    def data_preparation():
        loader = DataLoader()
        X,Y = loader.load_data_from_folder_as_png('data')
        X_train, X_test, X_val, Y_train, Y_test, Y_val = Preprocessor.split_data(X, Y)
        X_train, Y_train = Preprocessor.prepoccesing(X_train, Y_train)
        X_train, Y_train = Preprocessor.flip_photos(X_train,Y_train)
        X_val, Y_val = Preprocessor.prepoccesing(X_val, Y_val)
        X_test, Y_test = Preprocessor.prepoccesing(X_test, Y_test)

        X_train = np.expand_dims(DataPreparation.resize_ar(X_train),axis=-1)
        X_val = np.expand_dims(DataPreparation.resize_ar(X_val),axis=-1)
        X_test = np.expand_dims(DataPreparation.resize_ar(X_test),axis=-1)

        Y_train = Y_train.astype(np.int32)
        Y_val= Y_val.astype(np.int32)
        Y_test= Y_test.astype(np.int32)

        return  X_train, X_test, X_val, Y_train, Y_test, Y_val



class Model:
    def model():
        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC', 'Precision', 'Recall'])
        return model
    
    def f1_score(precision, recall):
        precision = np.array(precision)
        recall = np.array(recall)
        return (2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())).tolist()
    
    
    def train_model(model, X_train, Y_train, X_val, Y_val):
        history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10, batch_size=32)
        train_f1_score = Model.f1_score(history.history['Precision'], history.history['Recall'])
        val_f1_score = Model.f1_score(history.history['val_Precision'], history.history['val_Recall'])
        history.history['f1_score'] = train_f1_score
        history.history['val_f1_score'] = val_f1_score
        return history
    
    def predict(model, X_train, X_val, X_test):
        predictions = {}
        predictions["prediction_train"] = model.predict(X_train)
        predictions["prediction_val"] = model.predict(X_val)
        predictions["prediction_test"] = model.predict(X_test)
        predictions["prediction_label_train"] = (predictions["prediction_train"]> 0.5).astype("int")
        predictions["prediction_label_val"] = (predictions["prediction_val"]> 0.5).astype("int")
        predictions["prediction_label_test"] = (predictions["prediction_test"]> 0.5).astype("int")
        return predictions
    
    def save_history(history,name):
        with open(f'{name}.pkl', 'wb') as pickle_file:
            pickle.dump(history.history, pickle_file)

    def save_predictions(predictions, name, Y_train, Y_val, Y_test):
        predictions["label_train"] = Y_train
        predictions["label_val"] = Y_val
        predictions["label_test"] = Y_test

        with open(f'{name}.pkl', 'wb') as pickle_file:
            pickle.dump(predictions, pickle_file)
