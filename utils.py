import os
import torch
from joblib import dump, load
MODEL_SAVE_FOLDER = './models'


def _prove_directory_exists(directory):
    check_array = directory.split('/')
    # if name contains subdirectory we should prove that it is exist
    os.makedirs(os.path.join(*check_array[:-1]), exist_ok=True)


def save_net_model(model, name):
    save_path = os.path.join(MODEL_SAVE_FOLDER, name)
    _prove_directory_exists(save_path)
    torch.save(model.state_dict(), save_path)


def load_net_model(model, name):
    load_path = os.path.join(MODEL_SAVE_FOLDER, name)
    model.load_state_dict(torch.load(load_path))
    model.eval()
    return model


def save_ml(model, name):
    save_path = os.path.join(MODEL_SAVE_FOLDER, name)
    _prove_directory_exists(save_path)
    dump(model, save_path)


def load_ml(name):
    return load(name)
