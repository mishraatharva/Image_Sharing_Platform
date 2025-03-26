import mlflow
import os
import tensorflow as tf
import pickle  # To save the history object


def vgg_lstm_mlflow(history,model_path,history_path):
    # Extract final training and validation loss
    final_train_loss = history["loss"][-1]
    final_val_loss = history["val_loss"][-1]

    # Extract accuracy if available
    final_train_acc = history.get("accuracy", [None])[-1]
    final_val_acc = history.get("val_accuracy", [None])[-1]

    # Log parameters, metrics, and artifacts in MLflow
    with mlflow.start_run():
        mlflow.log_param("epochs", 10)
        mlflow.log_metric("train_loss", final_train_loss)
        mlflow.log_metric("val_loss", final_val_loss)

        if final_train_acc is not None:
            mlflow.log_metric("train_accuracy", final_train_acc)
        if final_val_acc is not None:
            mlflow.log_metric("val_accuracy", final_val_acc)
        
        # Log artifacts (history and model)
        mlflow.log_artifact(history_path)
        mlflow.log_artifact(model_path)

        print("Training metrics and history saved. Run `mlflow ui` to view logs.")



# def vgg_lstm_train_val_graph(history)