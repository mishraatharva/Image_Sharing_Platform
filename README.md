## Image Sharing Plateform Project:
  [![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white)]
  [![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?logo=tensorflow&logoColor=white)]
  ![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white)
  ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white)
  [![Flask](https://img.shields.io/badge/-Flask-FF4B4B)]
  [![mongodb](https://img.shields.io/badge/-mongodb-3776AB?logo=python&logoColor=white)]
  [![VScode](https://img.shields.io/badge/-VScode-3776AB?logo=python&logoColor=white)]

- Simple image sharing plateform project which basically utilizes concept of Image Captioning.

### Key Features:
- User can signup by providing required data.
- Login again using gmail and password
- All data in stored in mongodb
- User will login and upload a picture of his choice and caption for uploaded image can be created automatically.

### Overview:

This repository contains code for an image caption generation system using deep learning techniques. The system leverages a pretrained VGG16 model for feature extraction and a custom captioning model which was trained using LSTM for generating captions. The model is trained on the Flickr8k dataset using an lstm to generate caption.

### How It Works:

1. Image Processing with CNNs: The CNN component of the model processes the input images, extracting high-level features that represent the visual content.
2. Caption Generation with LSTMs: These features are then passed to the LSTM network, which generates a natural language description of the image.
3. End-to-End Learning: The model is trained end-to-end, ensuring that the captions generated are closely aligned with the visual content.


## About the Dataset

The [Flickr8k dataset](https://www.kaggle.com/adityajn105/flickr8k) is used for training and evaluating the image captioning system. It consists of 8,091 images, each with five captions describing the content of the image. The dataset provides a diverse set of images with multiple captions per image, making it suitable for training caption generation models.

Download the dataset from [Kaggle](https://www.kaggle.com/adityajn105/flickr8k) and organize the files as follows:

- flickr8k
  - Images
    - (image files)
  - captions.txt
