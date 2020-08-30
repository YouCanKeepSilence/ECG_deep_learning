import datetime
import os
from collections import namedtuple

import torch
import matplotlib.pyplot as plt
from joblib import dump, load

import dataset
import models

MODEL_SAVE_FOLDER = './models'


def _prove_directory_exists(directory):
    check_array = directory.split('/')
    # if name contains subdirectory we should prove that it is exist
    os.makedirs(os.path.join(*check_array[:-1]), exist_ok=True)


def draw(args):
    reference_path = os.path.join(args.base_path, 'REFERENCE.csv')
    df = dataset.Loader(args.base_path, reference_path).load_as_df_for_net(normalize=True)
    sample = df.iloc[1]['ecg']
    labels = list(range(sample.shape[1]))
    for i in range(sample.shape[0]):
        plt.plot(labels, sample[i], label=f'sensor_{i}')
    plt.grid()
    plt.legend(ncol=3, loc='best')
    plt.savefig('ecg_example.png')
    plt.show()


def write_checkpoint(writer, e, epochs, i, iteration_per_epochs, acc, loss, val_acc, val_loss):
    in_epoch_progress = round(i / iteration_per_epochs, 2)
    full_progress = round((e * iteration_per_epochs + i) / (epochs * iteration_per_epochs), 2)
    print(f'{datetime.datetime.now()}\n '
          f'Epoch: {e}/{epochs}, Iteration: {i}/{iteration_per_epochs}, '
          f'In epoch progress ({in_epoch_progress * 100} %)\n'
          f'Full progress {full_progress * 100} %\n'
          f'Loss: {loss.item()} , Acc: {acc} \n'
          f'Val Loss: {val_loss}, Val Acc: {val_acc} \n'
          )
    step_label = e * iteration_per_epochs + i
    writer.add_scalar('Train/Acc', acc, step_label)
    writer.add_scalar('Train/Loss', loss.item(), step_label)
    writer.add_scalar('Val/Loss', val_loss, step_label)
    writer.add_scalar('Val/Acc', val_acc, step_label)
    return step_label


def save_net_model(model, name):
    save_path = os.path.join(MODEL_SAVE_FOLDER, name)
    _prove_directory_exists(save_path)
    torch.save(model.state_dict(), save_path)


def load_net_model(model, name):
    load_path = os.path.join(MODEL_SAVE_FOLDER, name)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(load_path))
    else:
        model.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
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


if __name__ == '__main__':
    shadow_args = namedtuple('s_args', ['base_path'])
    shadow_args.base_path = './TrainingSet1'
    draw(shadow_args)

