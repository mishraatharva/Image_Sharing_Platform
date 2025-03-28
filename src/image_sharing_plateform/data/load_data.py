import os
import yaml
from ensure import ensure_annotations
from pathlib import Path
from typing import Any
import yaml
from types import SimpleNamespace
import string
from pickle import dump, load


#############################################################################################################
# GENERAL UTILITY FUNCTIONS
#############################################################################################################

def dict_to_namespace(d):
    """Recursively converts a dictionary into a SimpleNamespace object."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d

def load_doc(filename):
    """
        For loading the document file and reading the contents inside the file into a string.
    """
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def read_yaml(path_to_yaml: Path):
    try:
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            return dict_to_namespace(content)  # Convert dict to namespace
    except FileNotFoundError:
        print(f"File not found: {path_to_yaml}")


def create_directories(path_to_directories: list, verbose=True):
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)

#############################################################################################################
# DATA TRANSFORMATION UTILITY FUNCTIONS
#############################################################################################################

def create_photos(filename): 
    file = load_doc(filename)
    photos = file.split("\n")[:-1]
    return photos

def create_captions(filename):
    file = load_doc(filename)
    train_captions = {}
    texts = file.split("\n")
    for text in texts:
        text = text.split("\t")
        if text[0] not in train_captions:
            train_captions[text[0]] = []
        else:
            train_captions[text[0]].append(text[1])
    return train_captions

def create_features(photos):
    #loading all features
    train_data_features = {}
    all_features = load(open(r"U:\nlp_project\Image_Sharing_Plateform\data\processed\extracted_features.p","rb"))
    #selecting only needed features
    for ph in photos:
        if ph in all_features.keys():
            train_data_features[ph] = all_features[ph]
        else:
            train_data_features[ph] = []
    return train_data_features

def save_data(data, path_to_save):
    pass

def dict_to_list(descriptions):
    all_desc = []
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

def max_len(captions):
    return max([len(stmt) for stmt in captions])