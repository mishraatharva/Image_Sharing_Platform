# from src.image_sharing_plateform.config.configuration import ConfigurationManager
from src.image_sharing_plateform.entity import DataTransformationConfig
import string
from pathlib import Path
from src.image_sharing_plateform.data.load_data import load_doc
import pickle
from pickle import dump, load
import numpy as np
import tensorflow as tf

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def all_img_captions(self):
        """This function will take original caption data and return:

        'descriptions': a dictionary containing key value pair of image and list of all captions of each image.
        
        In original data image and there caption is repeating.
        """
        
        path = self.config.original_caption_data_path + "/" + "Flickr8k.token.txt" 

        file = load_doc(Path(path))
        captions = file.split('\n')
        descriptions ={}
        text_data = []
        for caption in captions:
            if '\t' in caption:
                img, caption = caption.split('\t')
                if img[:-2] not in descriptions:
                    descriptions[img[:-2]] = [ caption ]
                else:
                    descriptions[img[:-2]].append(caption)
                # text_data.append(caption)
        return descriptions
    
    
    def cleaning_text(self,captions):
        table = str.maketrans('','',string.punctuation)
        for img,caps in captions.items():
            for i,img_caption in enumerate(caps):
                img_caption.replace("-"," ")
                desc = img_caption.split()
                
                #converts to lowercase
                desc = [word.lower() for word in desc]
                
                #remove punctuation from each token
                desc = [word.translate(table) for word in desc]
                
                #remove hanging 's and a 
                desc = [word for word in desc if(len(word)>1)]
                
                #remove tokens with numbers in them
                desc = [word for word in desc if(word.isalpha())]
                
                #convert back to string
                img_caption = ' '.join(desc)
                img_caption = '<start> ' + " ".join(desc) + ' <end>'
                captions[img][i]= img_caption
        return captions
    
    def save_descriptions(self,descriptions, filename):
        lines = list()
        for key, desc_list in descriptions.items():
            for desc in desc_list:
                lines.append(key + '\t' + desc )
        data = "\n".join(lines)
        cleaned_data_path = self.config.preprocessed_data_path + "/" +  filename
        file = open(cleaned_data_path,"w")
        file.write(data)
        file.close()

    # def create_captions(self,filename):
    #     file_path = self.config.preprocessed_data_path + "/"+ filename
    #     print(file_path)
    #     file = load_doc(file_path)
    #     train_captions = {}
    #     texts = file.split("\n")
    #     for text in texts:
    #         text = text.split("\t")
    #         if text[0] not in train_captions:
    #             train_captions[text[0]] = []
    #         else:
    #             train_captions[text[0]].append(text[1])
    #     return train_captions
    
    def create_photos(self,filename):
        file_path = self.config.preprocessed_data_path + "/"+ filename
        file = load_doc(file_path)
        train_images = []
        texts = file.split("\n")[:-1]
        for text in texts:
            text = text.split("\t")
            train_images.append(text[0])
        return set(train_images)
    
    def create_features(self,photos,filename):
        #loading all features
        file_path = self.config.preprocessed_data_path + "/"+ filename
        train_data_features = {}
        all_features = load(open(file_path,"rb"))
        #selecting only needed features
        for ph in photos:
            if ph in all_features.keys():
                train_data_features[ph] = all_features[ph]
            else:
                train_data_features[ph] = []
        return train_data_features
    

    def clean_final_data(self,data_captions, data_images, data_features):
        invalid_ids = []  # Store invalid image IDs

        for img_id, _ in data_captions.items():
            image_feature = data_features[img_id]  # Extract image feature vector
            image_feature = np.array(image_feature)

            if image_feature.shape != (512,):  # Check if the shape is incorrect
                # print(img_id, image_feature.shape)
                invalid_ids.append(img_id)  # Collect invalid IDs

        # Delete all invalid IDs **after** iteration
        for img_id in invalid_ids:
            del data_captions[img_id] 
            del data_features[img_id]
            data_images.discard(img_id)

        return data_captions, data_features, data_images
    
    def get_vectorizer(self, train_data_captions):

        """Here data required is in list format. So converting data in required format only i.e list"""
        all_desc = []

        for key in train_data_captions.keys():
            all_desc = all_desc + train_data_captions[key]
        
        vectorizer = tf.keras.layers.TextVectorization(
                            max_tokens=7151,
                            output_mode="int", 
                            output_sequence_length=self.config.SEQ_LENGTH
                    )
        # print(all_desc)
        vectorizer.adapt(all_desc)
        
        return vectorizer
    

    def vectorize_data(self,vectorizer,data):
        tokenized_data = {img_id: vectorizer([f"<start> {cap} <end>"]) for img_id, caps in data.items() for cap in caps}
        return tokenized_data
        
    
    def save_training_validation_data(self,data_caption, data_image, data_feature, data):
        
        if data == "train":
            path = self.config.training_data + "/" + "train_data.pkl"
        else:
            path = self.config.validation_data + "/" + "validation_data.pkl"

        with open(path, "wb") as file:
            pickle.dump({"caption_data": data_caption, "image_data": data_image, "feature_data": data_feature}, file)