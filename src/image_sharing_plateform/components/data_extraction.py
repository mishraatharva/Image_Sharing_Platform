from src.image_sharing_plateform.entity import ExtractedFeatureConfig
from src.image_sharing_plateform.data.load_data import load_doc
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import string
import os

class FeatureExtraction:
    def __init__(self, config):
        self.config = config
    
    def extract_features(self):
        image_data_path = self.config.image_data_path

        # Load VGG16 model without top layers
        base_model = VGG16(input_shape=(self.config.resize_img_height, self.config.resize_img_width, 3), 
                           include_top=False, weights='imagenet')

        # Add GlobalAveragePooling to convert (None, 15, 15, 512) → (None, 512)
        model = tf.keras.Sequential([
            base_model,
            GlobalAveragePooling2D()  # Converts spatial dimensions to (batch_size, 512)
        ])

        features = {}
        
        for img in tqdm(os.listdir(image_data_path)):
            filename = os.path.join(image_data_path, img)
            
            # Load and preprocess the image
            image = Image.open(filename)
            image = image.resize((self.config.resize_img_height, self.config.resize_img_width))
            image = np.array(image) / 255.0  # Normalize correctly
            image = np.expand_dims(image, axis=0)  # Add batch dimension
            
            # Extract feature and flatten it
            feature = model.predict(image)  # Shape: (1, 512)

            features[img] = feature.flatten()  # Convert (1, 512) → (512,)
        
        return features
