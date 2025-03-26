import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

class ImageCaptionGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_features, tokenized_captions, batch_size=32, shuffle=True):
        """
        Custom Data Generator for Image Captioning using RNN/LSTM.
        """
        # self.config = config
        self.image_features = image_features  # Dict: {image_id -> feature vector}
        self.tokenized_captions = tokenized_captions  # Dict: {image_id -> tokenized captions}
        self.max_seq_length = 32
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_ids = list(self.image_features.keys())  # List of image IDs
        self.data = self._prepare_data()  # Flattened list of (img_feature, caption input, next word)
        self.on_epoch_end()  # Shuffle data at the start

    def _prepare_data(self):
        """
        Convert (image, caption) pairs into stepwise (feature, partial caption, next word) triplets.
        """
        data = []
        for img_id in self.image_ids:
            img_feature = self.image_features[img_id]  # Precomputed CNN feature
            for caption in self.tokenized_captions[img_id]:  # Each image has multiple captions
                for i in range(1, len(caption)):  # Create step-wise sequences
                    X2_seq = caption[:i]  # Input sequence
                    y_word = caption[i]   # Next word to predict
                    data.append((img_feature, X2_seq, y_word))
        print(len(data))
        return data

    def __len__(self):
        """Returns the total number of batches per epoch."""
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        """Generates a batch of data."""
        batch_data = self.data[index * self.batch_size : (index + 1) * self.batch_size]
        return self.__data_generation(batch_data)

    def __data_generation(self, batch_data):
        """Generates one batch of data."""
        X1, X2, Y = [], [], []
        
        for img_feature, X2_seq, y_word in batch_data:
            X1.append(img_feature)   # Image feature (CNN output)
            X2.append(X2_seq)        # Caption input sequence
            Y.append(y_word)         # Target next word
        
        
        X1 = np.array(X1, dtype=np.float32)  
        X2_padded = pad_sequences(X2, maxlen=self.max_seq_length, padding='post')
        Y = np.array(Y)

        if X1.ndim > 2:
            X1 = X1.reshape(len(X1), -1)
    
        return [X1, X2_padded], np.array(Y)

    def on_epoch_end(self):
        """Shuffle data at the end of each epoch."""
        if self.shuffle:
            np.random.shuffle(self.data)
