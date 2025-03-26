from src.image_sharing_plateform.config.configuration import ConfigurationManager,FeatureExtractionConfigurationManager
from src.image_sharing_plateform.components.data_extraction import FeatureExtraction


class FeatureExtractionPipeline():
    def __init__(self):
        pass

    try:
        config = FeatureExtractionConfigurationManager()
    
        feature_extraction_config = config.get_feature_extraction_config()
        print(feature_extraction_config)
    
        data_transformation = FeatureExtraction(config=feature_extraction_config)
    
        extracted_features = data_transformation.extract_features()

    except Exception as e:
        raise e