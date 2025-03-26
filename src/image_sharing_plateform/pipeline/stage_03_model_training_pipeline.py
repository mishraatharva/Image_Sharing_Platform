from src.image_sharing_plateform.config.configuration import ModelTrainingConfigurationManager
from src.image_sharing_plateform.components.data_transformation import DataTransformation
from src.image_sharing_plateform.data.split_data import TrainTestSplit
import tensorflow as tf
import pickle
import os
from src.image_sharing_plateform.components.data_generator import ImageCaptionGenerator
from src.image_sharing_plateform.constants import *
from src.image_sharing_plateform.model.cnn_lstm_model.model import ImageSharingModel


# from src.image_sharing_plateform.data.load_data import create_photos,create_captions,create_features (delete if not needed)

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self,logger):
        
        try:
            try:
                manager = ModelTrainingConfigurationManager(PARAMS_FILE_PATH)
                model_training_config = manager.get_model_training_config()
    
                # # # # Access attributes
                train_data_path = model_training_config.train_data_path
                validation_data_path = model_training_config.validation_data_path
                trained_model_path = model_training_config.trained_model_path
                history_path = model_training_config.history_path

                with open(train_data_path + "/" + "train_data.pkl", "rb") as file:
                    train_data = pickle.load(file)
                    print(train_data.keys())

                with open(validation_data_path + "/" + "validation_data.pkl", "rb") as file:
                    validation_data = pickle.load(file)

                train_data_features = train_data["image_data"]
                train_caption_tokenized = train_data["caption_data"]

                validation_data_features = validation_data["image_data"]
                validation_caption_tokenized = validation_data["caption_data"]

            except Exception as e:
                print(f"Error: {e}")
    
    
            train_data_generator = ImageCaptionGenerator(
                image_features = train_data_features , 
                tokenized_captions = train_caption_tokenized, 
                batch_size = 8 , 
                shuffle=True)
    

            validation_data_generator = ImageCaptionGenerator(
                image_features = validation_data_features , 
                tokenized_captions = validation_caption_tokenized , 
                batch_size=10 , 
                shuffle=True)
    

            model_training = ImageSharingModel(
                model_training_config.trained_model_path, 
                model_training_config.history_path,
                model_training_config.CreateSqueezeModel_config, 
                model_training_config.CreateLSTMSequence_config)
    
    
            model = model_training.create_image_captioning_model()
    

            model_training.start_model_training(
                model, 
                train_data_generator, 
                validation_data_generator)

        except Exception as e:
            # logger.exception(e)
            raise e
        
# class mlflow_pipeline:
#     def __init__(self,trained_model_path, history_path):
#         self.config = PARAMS_FILE_PATH


#     def load_model(self,trained_model_path):
#         path = os.path.join(trained_model_path + "trained_model.h5")
        
#         with open(path, "rb") as f:
#             model = pickle.load(f)
#         return model

#     def load_history(self,history_path):
#         path = os.path.join(history_path + "history.pkl")
        
#         with open(path, "rb") as f:
#             history = pickle.load(f)
#         return history

#     def setup_mlflow(self):
#         print(self.config)
#         trained_model_path = self.config.model_training.trained_model_path
#         history_path  = self.config.model_training.history_path

#         model = self.load_model(trained_model_path)
#         history = self.load_history(history_path)