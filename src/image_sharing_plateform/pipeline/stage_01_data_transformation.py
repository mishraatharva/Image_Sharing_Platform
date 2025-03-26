from src.image_sharing_plateform.config.configuration import DataTransformationConfigurationManager
from src.image_sharing_plateform.components.data_transformation import DataTransformation
from src.image_sharing_plateform.data.split_data import TrainTestSplit
import tensorflow as tf
# from src.image_sharing_plateform.data.load_data import create_photos,create_captions,create_features (delete if not needed)

class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self,logger):
        
        try:
            """
            STAGE1:
              step1: Create DataTransformationConfigurationManager() object.
              step2: get all configuration and values config.yaml file.
              step3: create a dictionary 'descriptions' containing image is a key : all caption in a list as a value.
              step4: clean all like :=  'converts to lowercase', 'remove punctuation from each token', 'remove hanging 's and a'.
              step5: split full data in train and validation data. i.e 'training_data' and 'validation_data'.
              step5: save this 'training_data' and 'validation_data' in .txt format in desired location.
    
            STAGE2:
              step1: read training_data.txt and validation_data.txt.
              step2: create a list of all image in image data and create a dictionary of image_id as key and his vgg generated feature of shape (512,).
              step3: create vectorizer on train_data
              step4: vectorize 'train_data' and create 'train_data_caption' which in vectorized data.
              step5: repeat step2, step3, step4 for validation data as well.
              step6: save 'train_data_captions, train_data_images, train_data_features'.
              step7: save 'validation_data_captions, validation_data_images, validation_data_features'.
              step8: save vectorizer object for further use.
            """
            config = DataTransformationConfigurationManager()
            logger.info(f"DataTransformationConfigurationManager object created")

    
            data_transformation_config = config.get_data_transformation_config()
    
            data_transformation = DataTransformation(config=data_transformation_config)
    
            descriptions = data_transformation.all_img_captions()
            
            logger.info(f"created a dictionary 'descriptions' containing image is a key : all caption in a list as a value.")


            full_data = data_transformation.cleaning_text(descriptions)

            logger.info(f"data cleaning completed")

            training_data, validation_data = TrainTestSplit.train_val_split(full_data)

            logger.info(f"training_data, validation_data data created seperately")
    
            data_transformation.save_descriptions(training_data,"training_data.txt")
            data_transformation.save_descriptions(validation_data,"validation_data.txt")

            logger.info(f"training_data, validation_data data saved at required location")


            # train_data_captions = data_transformation.create_captions("training_data.txt")
            train_data_images = data_transformation.create_photos("training_data.txt")
            train_data_features = data_transformation.create_features(train_data_images, "extracted_features.p")
            train_data_captions, train_data_images, train_data_features = data_transformation.clean_final_data(training_data, train_data_images, train_data_features)
            
            logger.info(f"'train_data_captions, train_data_images, train_data_features' created")
            
    
            vectorizer = data_transformation.get_vectorizer(training_data)
            logger.info(f"vectorizer object created")
            train_data_captions = data_transformation.vectorize_data(vectorizer,training_data)
    
            logger.info(f"vectorized training_data")

    
            # validation_data_captions = data_transformation.create_captions(r"validation_data.txt")
            validation_data_images = data_transformation.create_photos(r"validation_data.txt")
            validation_data_features = data_transformation.create_features(validation_data_images ,"extracted_features.p")
            validation_data_captions, validation_data_images, validation_data_features = data_transformation.clean_final_data(validation_data, validation_data_images, validation_data_features)
              
            logger.info(f"'validation_data_captions, validation_data_images, validation_data_features' created")
            
            validation_data_captions = data_transformation.vectorize_data(vectorizer,validation_data_captions)
            logger.info(f"vectorized validation_data")
            
            
            data_transformation.save_training_validation_data(train_data_captions, train_data_images, train_data_features, "train")
            logger.info(f"final train data saved containes train_data_captions, train_data_images, train_data_features in pickle file")

            data_transformation.save_training_validation_data(validation_data_captions, validation_data_images, validation_data_features, "validation")
            logger.info(f"final validation data saved containes validation_data_captions, validation_data_images, validation_data_features in pickle file")


            vectorizer = tf.keras.models.Sequential([vectorizer])
            vectorizer.save(r"U:\nlp_project\Image_Sharing_Plateform\data\processed\vectorizer")
            logger.info(f"vectorizer data saved for further use.")


        except Exception as e:
            raise e