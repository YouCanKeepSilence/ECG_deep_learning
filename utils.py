import os
import torch
from joblib import dump, load

import models

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
    load_path = os.path.join(MODEL_SAVE_FOLDER, name)
    return load(load_path)


def create_model_by_name(name, weights_path):
    if name.startswith('VGG_'):
        net = models.get_vgg(name.split('_')[-1], batch_norm=True)
        return load_net_model(net, weights_path)
    elif name.startswith('VGGLikeCNN'):
        pooling = 'max' if len(name.split('_')) == 1 else 'avg'
        net = models.VGGLikeCNN(pooling=pooling)
        return load_net_model(net, weights_path)
    elif name.startswith('CNN'):
        pooling = 'max' if len(name.split('_')) == 1 else 'avg'
        net = models.CNN(pooling=pooling)
        return load_net_model(net, weights_path)
    elif name in ['MLP']:
        net = getattr(models, name)()
        return load_net_model(net, weights_path)
    elif name in ['RF', 'SVM', 'XGBoost']:
        return load_ml(weights_path)
    else:
        raise Exception(f'Unknown model type {name}')

