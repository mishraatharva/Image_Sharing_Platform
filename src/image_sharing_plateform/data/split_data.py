import numpy as np

import numpy as np

class TrainTestSplit():
    def __init__(self):
        pass

    def train_val_split(caption_data, train_size=0.8, shuffle=True):

       # 1. Get the list of all image names
        all_images = list(caption_data.keys())

        # 2. Shuffle if necessary
        if shuffle:
            np.random.shuffle(all_images)

        # 3. Split into training and validation sets
        train_size = int(len(caption_data) * train_size)

        training_data = {
            img_name: caption_data[img_name] for img_name in all_images[:train_size]
        }
        validation_data = {
            img_name: caption_data[img_name] for img_name in all_images[train_size:]
        }

        # 4. Return the splits
        return training_data, validation_data