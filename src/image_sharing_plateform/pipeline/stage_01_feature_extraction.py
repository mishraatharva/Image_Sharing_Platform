from src.image_sharing_plateform.config.configuration import FeatureExtractionConfigurationManager
from src.image_sharing_plateform.components.data_extraction import FeatureExtraction
from pickle import dump, load

class FeatureExtractionPipeline():
    def __init__(self):
        pass

    def main(self,logger):
        try:
            config = FeatureExtractionConfigurationManager()
    
            feature_extraction_config = config.get_feature_extraction_config()
            
            logger.info(f"{feature_extraction_config}")
    
            feature_extraction = FeatureExtraction(config=feature_extraction_config)
    
            extracted_features = feature_extraction.extract_features(logger)

            feature_path = feature_extraction_config.extracted_feature_path + "/" + "extracted_features.p"
            
            dump(extracted_features, open(feature_path,"wb"))
            
            logger.info(f"extracted feature saved to {feature_path}")

        except Exception as e:
           raise e