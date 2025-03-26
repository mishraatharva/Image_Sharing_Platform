import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


project_name = "image_sharing_plateform"

list_of_files = [

    f"src/{project_name}/data/__init__.py",
    f"src/{project_name}/data/load_data.py",
    f"src/{project_name}/data/process_data.py",
    f"src/{project_name}/data/split_data.py",

    f"src/{project_name}/feature/__init__.py",
    f"src/{project_name}/feature/feature_selection.py",
    
    f"src/{project_name}/model/__init__.py",
    f"src/{project_name}/model/model_training.py",
    f"src/{project_name}/model/predict_model.py",
    f"src/{project_name}/model/model_evaluation.py",

    f"src/{project_name}/test/test_data_processing.py",
    f"src/{project_name}/test/test_model.py",
    f"src/{project_name}/test/test_visualization.py",
    
    f"src/{project_name}/web_app/__init__.py",
    f"src/{project_name}/web_app/signup_form.py",
    f"src/{project_name}/web_app/login_forms.py",
    f"src/{project_name}/web_app/templates",


    "app.py",
    "main.py",
    "requirements.txt",
    "setup.py",
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory:{filedir} for the file {filename}")

    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath,'w') as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} is already exists")



