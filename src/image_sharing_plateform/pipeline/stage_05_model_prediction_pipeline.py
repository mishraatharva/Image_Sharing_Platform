from src.image_sharing_plateform.config.configuration import ModelPredictionConfigurationManager
from tensorflow.keras.models import load_model
import pickle
import os
from pathlib import Path
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from flask import g
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from PIL import Image
import tensorflow as tf
from src.image_sharing_plateform.model.cnn_lstm_model.model import CreateSqueezeModel, CreateLSTMSequence



class PredictionPipeline:
    def __init__(self, PARAMS_FILE_PATH):
        self.params_file_path = PARAMS_FILE_PATH

    def return_trained_model(self):
        manager = ModelPredictionConfigurationManager(self.params_file_path)
        self.prediction_config = manager.get_model_prediction_config()
        model_path = self.prediction_config.trained_model_path + "/" + "trained_model.h5"
        
        custom_objects = {
            "CreateSqueezeModel":CreateSqueezeModel,
            "CreateLSTMSequence": CreateLSTMSequence,
        }
        print(model_path)
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        return model
        
    def load_vectorizer(self):
        vectorizer_path = self.prediction_config.vectorizer_path + "/" + "vectorizer"

        vectorizer = load_model(vectorizer_path)
        return vectorizer.layers[0]
    

    def extract_features(self,filename,base_model):
       
        model = tf.keras.Sequential([
            base_model,
            GlobalAveragePooling2D()
        ])
        
        image = Image.open(filename)
        image = image.resize((500,500))
        image = np.array(image) / 255.0  # Normalize correctly
        image = np.expand_dims(image, axis=0)  # Add batch dimension
            
        # Extract feature and flatten it
        feature = model(image)  # Shape: (1, 512)
        
        return feature
     

    def generate_caption(self, model, image_feature, vectorizer):
        """Generate a caption for the given image feature vector."""
    
        # Ensure `vectorizer` gets a tensor
        sequence = vectorizer(tf.convert_to_tensor(["<start>"]))  # Convert "<start>" to tensor
        sequence = sequence.numpy()[0]  # Extract numpy array for compatibility

        print("Initial Sequence Shape:", sequence.shape)

        caption = []

        for _ in range(15):
            # Pad the sequence before passing to the model
            sequence_padded = pad_sequences([sequence], maxlen=32, padding='post')
            print("Sequence Padded Shape:", sequence_padded.shape)
        
            # Convert inputs to tensors
            image_feature_tensor = tf.convert_to_tensor(image_feature)
            sequence_padded_tensor = tf.convert_to_tensor(sequence_padded)
        
            # Use tf.function for better performance
            @tf.function
            def predict_fn(img, seq):
                return model([img, seq], training=False)
        
            # Predict without verbose
            y_pred = predict_fn(image_feature_tensor, sequence_padded_tensor)
        
            predicted_index = np.argmax(y_pred)

            # Convert "<end>" token to tensor before comparison
            end_token = vectorizer(tf.convert_to_tensor(["<end>"])).numpy()[0][0]
        
            if predicted_index == end_token:  # Stop if "<end>" token is generated
                break

            caption.append(predicted_index)

            #  Ensure sequence remains within the correct shape
            sequence = np.append(sequence, predicted_index)[-15:]  # Keep last 15 tokens

        # Convert indices back to words using `vocab`
        vocab = vectorizer.get_vocabulary()
        final_caption = " ".join(vocab[idx] for idx in caption)

        return final_caption
