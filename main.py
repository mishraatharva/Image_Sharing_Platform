from src.image_sharing_plateform.pipeline.stage_01_feature_extraction import FeatureExtractionPipeline
from src.image_sharing_plateform.pipeline.stage_02_data_transformation import DataTransformationPipeline
from src.image_sharing_plateform.pipeline.stage_03_model_training_pipeline import ModelTrainingPipeline
from src.image_sharing_plateform.pipeline.stage_04_mlflow_pipeline import mlflow_pipeline
from src.image_sharing_plateform.constants import *


from src.image_sharing_plateform.logging import logger

STAGE_NAME = "Feature Extraction Stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   feature_extraction_pipeline = FeatureExtractionPipeline()
   feature_extraction_pipeline.main(logger)
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Data Transformation Stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_transformation = DataTransformationPipeline()
   data_transformation.main(logger)
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Model Training Stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   model_training_pipeline = ModelTrainingPipeline()
   model_training_pipeline.main(logger)
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "SETUP MLFLOW"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   mlflow_pipeline = mlflow_pipeline()
   print("pipeline object created")
   mlflow_pipeline.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e