from src.image_sharing_plateform.config.configuration import ML_FlowConfigurationManager
from src.image_sharing_plateform.model.cnn_lstm_model.model_evaluation import vgg_lstm_mlflow
import os
import pickle
from src.image_sharing_plateform.constants import *
from tensorflow.keras.models import load_model


class mlflow_pipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            mlflow_manager = ML_FlowConfigurationManager(PARAMS_FILE_PATH)
            mlflow_config = mlflow_manager.get_mlflow_config()
            
            model_path = mlflow_config.trained_model_path
            model_path  = os.path.join(model_path,"trained_model.h5")
            print(model_path)
            
            history_path = mlflow_config.history_path
            history_path  = os.path.join(history_path,"history.pkl")
            print(history_path)
            
            with open(history_path, "rb") as f:
                history = pickle.load(f)

            vgg_lstm_mlflow(history,model_path,history_path)
        
        except Exception as e:
            raise e