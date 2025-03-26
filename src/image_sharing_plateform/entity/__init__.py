from dataclasses import dataclass
from pathlib import Path


from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExtractedFeatureConfig:
    image_data_path: Path
    extracted_feature_path: Path
    resize_img_height: int
    resize_img_width: int


@dataclass(frozen=True)
class DataTransformationConfig:
    original_image_data_path: Path
    original_caption_data_path: Path
    preprocessed_data_path : Path
    training_data : Path
    validation_data : Path
    vectorize_path : Path
    SEQ_LENGTH : int


@dataclass(frozen=True)
class ModelTrainingConfig:
    train_data_path : Path
    validation_data_path : Path
    trained_model_path : Path
    history_path : Path
    CreateSqueezeModel_config: dict
    CreateLSTMSequence_config: dict


@dataclass(frozen=True)
class ModelPredictionConfig:
    trained_model_path : Path
    vectorizer_path : Path


@dataclass(frozen=True)
class MLFLOW_CONFIG:
    trained_model_path : Path
    history_path : Path