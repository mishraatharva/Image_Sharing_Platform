from src.image_sharing_plateform.entity import (ExtractedFeatureConfig,
                                                DataTransformationConfig,
                                                ModelTrainingConfig,
                                                MLFLOW_CONFIG,
                                                ModelPredictionConfig
                                                 )

from src.image_sharing_plateform.data.load_data import read_yaml,create_directories
from src.image_sharing_plateform.constants import *



##################################################################
# FeatureExtractionConfigurationManager
###################################################################

class FeatureExtractionConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH):
        self.config = read_yaml(config_filepath)

    
    def get_feature_extraction_config(self) -> ExtractedFeatureConfig:
        config = self.config.feature_extraction

        # create_directories([config.extracted_feature_path])

        feature_extraction_config = ExtractedFeatureConfig(
            image_data_path = config.image_data_path,
            
            extracted_feature_path = config.extracted_feature_path,
            
            resize_img_height = config.resize_img_height,
            
            resize_img_width = config.resize_img_width
        )

        return feature_extraction_config


##################################################################
# DataTransformationConfigurationManager
###################################################################


class DataTransformationConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH):
        self.config = read_yaml(config_filepath)

    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.preprocessed_data_path, config.training_data, config.validation_data])

        data_transformation_config = DataTransformationConfig(
            original_image_data_path = config.original_image_data_path,

            original_caption_data_path = config.original_caption_data_path,
            
            preprocessed_data_path = config.preprocessed_data_path,

            training_data = config.training_data,
            
            validation_data = config.validation_data,
            
            vectorize_path = config.vectorize_path,

            SEQ_LENGTH = config.SEQ_LENGTH
            
        )
        return data_transformation_config



##################################################################
# ModelTrainingConfigurationManager
###################################################################

class ModelTrainingConfigurationManager:
    def __init__(self,config_filepath):
        
        self.config = read_yaml(config_filepath)
    

    def get_model_training_config(self) -> ModelTrainingConfig:
        model_training_data_config = self.config.model_training

        create_directories([model_training_data_config.trained_model_path, model_training_data_config.history_path])

        model_training_config = ModelTrainingConfig(
            train_data_path = model_training_data_config.train_data_path,

            validation_data_path = model_training_data_config.validation_data_path,
            
            trained_model_path = model_training_data_config.trained_model_path,

            history_path = model_training_data_config.history_path,
            
            CreateSqueezeModel_config = self.config.CreateSqueezeModel_config,

            CreateLSTMSequence_config = self.config.CreateLSTMSequence_config

        )
        
        return model_training_config
    

##################################################################
# ML-FLOW-CONFIGURATION-MANAGER
###################################################################

class ML_FlowConfigurationManager:
    def __init__(self,config_filepath):
        
        self.config = read_yaml(config_filepath)
    

    def get_mlflow_config(self) -> MLFLOW_CONFIG:
        ml_flow_data_config = self.config.model_training
        print(ml_flow_data_config)

        ml_flow_config = MLFLOW_CONFIG(
            
            trained_model_path = ml_flow_data_config.trained_model_path,

            history_path = ml_flow_data_config.history_path,
        )
        
        return ml_flow_config
    

##################################################################
# ModelPrediction-Configuration-Manager
###################################################################

class ModelPredictionConfigurationManager:
    def __init__(self, config_path):
        self.config = read_yaml(config_path)

    def get_model_prediction_config(self) -> ModelPredictionConfig:
        model_prediction_config  = self.config.model_prediction_config

        prediction_config  = ModelPredictionConfig(
            trained_model_path = model_prediction_config.trained_model_path,
            vectorizer_path =   model_prediction_config.vectorizer_path,
            )

        return prediction_config
