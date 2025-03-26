import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Embedding, Add
from tensorflow.keras.models import Model
from keras.utils import plot_model
from types import SimpleNamespace
from pathlib import Path
from src.image_sharing_plateform.constants import *
import os
import pickle

class CreateSqueezeModel(keras.Model):
    def __init__(self, dropout, activation, **kwargs):
        super().__init__(**kwargs)
        self.fe1 = Dropout(dropout)
        self.fe2 = Dense(256, activation=activation)

    def call(self, inputs, training=False):
        x = self.fe1(inputs, training=training)
        x = self.fe2(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "dropout": self.fe1.rate,
            "activation": self.fe2.activation.__name__,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CreateLSTMSequence(keras.Model):
    def __init__(self, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.embedding = Embedding(vocab_size, 256, mask_zero=True)
        self.drp = Dropout(0.5)
        self.lstm = LSTM(256)

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x = self.drp(x, training=training)
        x = self.lstm(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"vocab_size": self.embedding.input_dim})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



class ImageSharingModel:
    def __init__(self, trained_model_path,history_path,CreateSqueezeModel_config, CreateLSTMSequence_config):
        
        """
        Custom Data Generator for Image Captioning using RNN/LSTM.
        """
        self.trained_model_path = trained_model_path
        self.history_path = history_path
        self.CreateSqueezeModel_config = CreateSqueezeModel_config
        self.CreateLSTMSequence_config = CreateLSTMSequence_config
    

    def save_trained_model_history(self,model, history):
        os.makedirs(self.trained_model_path, exist_ok=True)
        
        model.save(os.path.join(self.trained_model_path, "trained_model.h5"))
        
        
        with open(os.path.join(self.history_path, "history.pkl"), "wb") as f:
            pickle.dump(history.history, f)

        # print(f"Model saved at {os.path.join(self.trained_model_path, 'trained_model.h5')}")


    def create_image_captioning_model(self):
        inputs1 = Input(shape=(self.CreateSqueezeModel_config.input_shape,))
        
        squeeze_model = CreateSqueezeModel(self.CreateSqueezeModel_config.dropout,
                                           self.CreateSqueezeModel_config.activation)
        
        fe2 = squeeze_model(inputs1)

        inputs2 = Input(shape=(self.CreateLSTMSequence_config.input_length_lstm,))
        
        lstm_model = CreateLSTMSequence(self.CreateLSTMSequence_config.vocab_size)
        
        se3 = lstm_model(inputs2)

        decoder1 = Add()([fe2, se3])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(self.CreateLSTMSequence_config.vocab_size, activation='softmax')(decoder2)

        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        model.compile(loss=self.CreateLSTMSequence_config.loss, optimizer=self.CreateLSTMSequence_config.optimizer)

        return model
    

    def start_model_training(self,model, train_data_generator, validation_data_generator):
        steps_per_epoch = len(train_data_generator)
        validation_steps = len(validation_data_generator)

        # Train the model and store the history

        history = model.fit(
                train_data_generator,
                epochs=1,  # Adjust based on your needs
                steps_per_epoch=steps_per_epoch,
                validation_data=validation_data_generator,
                validation_steps=validation_steps,
                verbose=1
                )

        self.save_trained_model_history(model, history)