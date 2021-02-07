import datetime
import logging
import os
from scipy import signal
from collections import namedtuple

import torch
import matplotlib.pyplot as plt
from joblib import dump, load

import dataset
import models

MODEL_SAVE_FOLDER = './models'


DIAGNOSIS_MAP = {
    '0': 'Normal',  # Здоров
    '1': 'Atrial fibrillation (AF)',  # Мерцательная аритмия
    '2': 'First - degree atrioventricular block (I - AVB)',  # Атриовентрикулярная блокада первой степени
    '3': 'Left bundle branch block (LBBB)',  # Блокада левой ножки пучка Гиса
    '4': 'Right bundle branch block (RBBB)',  # Блокада правой ножки пучка Гиса
    '5': 'Premature atrial contraction (PAC)',  # Преждевременное сокращение предсердий
    '6': 'Premature ventricular contraction (PVC)',  # Преждевременное сокращение желудочков
    '7': 'ST - segment depression (STD)',  # ? ST - депрессия сегмента
    '8': 'ST - segment elevated (STE)'  # ? ST - сегмент повышенный
}

GENDER_MAP = {
    '0': 'Male',  # Мужской
    '1': 'Female'  # Женский
}


def init_logger(is_colab=False):
    if is_colab:
        # Because it work only this way in colab
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)  # was INFO

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)  # was INFO

        fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', None)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    else:
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)


# Filter whole ecg with butterworth bandpass filter
def filter_preprocess_full_ecg(ecg, frequency, borders=(3, 30)):
    for i in range(ecg.shape[0]):
        ecg[i, :] = filter_preprocess_single_ecg_lead(ecg[i, :], frequency, borders)
    return ecg


def filter_preprocess_single_ecg_lead(ecg_lead, frequency, borders=(1, 40)):
    # Filters the single lead of ecg with a butterworth bandpass filter with cutoff frequencies fc=[a, b]
    f0 = 2 * float(borders[0]) / float(frequency)
    f1 = 2 * float(borders[1]) / float(frequency)
    b, a = signal.butter(2, [f0, f1], btype='bandpass')
    return signal.filtfilt(b, a, ecg_lead)


def get_diagnosis_str(diagnosis_label):
    return DIAGNOSIS_MAP[str(diagnosis_label)]


def get_gender_str(gender_str):
    return GENDER_MAP[str(gender_str)]


def _prove_directory_exists(directory):
    check_array = directory.split('/')
    # if name contains subdirectory we should prove that it is exist
    os.makedirs(os.path.join(*check_array[:-1]), exist_ok=True)


def draw_ecg(ecg_sample):
    sensors_count, ecg_len = ecg_sample.shape
    labels = list(range(ecg_len))
    for i in range(sensors_count):
        plt.plot(labels, ecg_sample[i], label=f'sensor_{i}')
    plt.grid()
    plt.legend(ncol=3, loc='best')
    plt.savefig('ecg_example.png')
    plt.show()


def draw_signal(signal_12_lead, preprocessing_func=None):
    fig, ax = plt.subplots(signal_12_lead.shape[0], 1, figsize=(24, 0.8 * signal_12_lead.shape[0]), sharex=True)
    x_ticks = list(range(signal_12_lead.shape[1]))
    for i in range(signal_12_lead.shape[0]):
        to_draw = signal_12_lead[i, :]
        if preprocessing_func:
            to_draw = preprocessing_func(to_draw)
        # lc = LineCollection(signal_12_lead[i, :], cmap='inferno_r')
        ax[i].plot(x_ticks, to_draw)
        ax[i].set_title(f'Lead {i + 1}')
        ax[i].grid()

    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    plt.grid()
    plt.show()


def draw(args):
    reference_path = os.path.join(args.base_path, 'REFERENCE.csv')
    df = dataset.Loader(args.base_path, reference_path).load_as_df_for_net(normalize=True)
    sample = df.iloc[1]['ecg']
    draw_ecg(sample)


def write_checkpoint(writer, e, epochs, i, iteration_per_epochs, acc, loss, val_acc, val_loss):
    in_epoch_progress = round(i / iteration_per_epochs, 2)
    full_progress = round((e * iteration_per_epochs + i) / (epochs * iteration_per_epochs), 2)
    logging.info(f'Epoch: {e}/{epochs}, Iteration: {i}/{iteration_per_epochs}, '
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

