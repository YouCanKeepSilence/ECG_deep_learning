import base64
import datetime
import logging
import os
import uuid
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from captum.attr import Saliency
from matplotlib.collections import LineCollection
from torch.utils.data import DataLoader

import dataset
import utils

REPORT_BASE_PATH = './Reports'
BASE_DATA_PATH = './TrainingSet1'
SLICES_COUNT = 1
SLICES_LEN = 2500  # 5 secs at 500Hz
ECG_FREQUENCY = 500.0


# --- done ---
# Обучить модель на сглаженных данных signal_preprocessing, borders=(3, 30) (3, 50)
# Сгладить "важность" атрибутов (или сегментов) -> попробовать тем же фильтром что и для сигнала
# Добавить нормировку и метки с полом и возрастом
# Добавить на график инфу диагноз, правильный диагноз, пол, возраст и их важность ( в процентах ) - несколько штук или все
# Рефакторинг, добавить создание отчетов для всех пациентов (сгруппировать по диагнозам и по правильным/неправильным)

# --- todo ---
# confusion matrix + accuracy
# try another xai methods
# create fair validate dataset (50 samples from each type)
# EDA (get from Evgeniy)
# DenseNet (interesting) /ResNet/LSTM mb try it
# (much later) calculate f1 at test subsamples


def generate_html_report(
        encoded_ecg_image: bytes, non_ecg_data: np.array, non_ecg_interpretability: np.array,
        diagnosis_label: int, net_diagnosis: int, net_confidence: float
) -> str:
    """
    Generate importance report with ecg map as html string
    :param encoded_ecg_image: Importance map of ecg
    :param non_ecg_data: array of additional patient info [age, gender]
    :param non_ecg_interpretability: importance of additional info
    :param diagnosis_label: number of truth diagnosis
    :param net_diagnosis: number of predicted diagnosis
    :param net_confidence: network confidence in prediction
    :return: generated html string
    """
    gender_importance = round(non_ecg_interpretability[1] * 100, 2)
    age_importance = round(non_ecg_interpretability[0] * 100, 2)
    diagnosis_confidence = round(net_confidence * 100, 2)

    net_diagnosis_str = f'Net diagnosis: {utils.get_diagnosis_str(net_diagnosis)} ({diagnosis_confidence} %)'
    if str(net_diagnosis) != str(diagnosis_label):
        net_diagnosis_str = f'<span style="color: red">{net_diagnosis_str}</span>'

    html_str = f'<h3> Patient info </h3>' \
               f'Age: {non_ecg_data[0].astype(int)} ({age_importance} %)<br>' \
               f'Gender: {utils.get_gender_str(non_ecg_data[1].astype(int))} ({gender_importance} %)<br>' \
               f'Diagnosis: {utils.get_diagnosis_str(int(diagnosis_label))}<br>' \
               f'{net_diagnosis_str}<br>' \
               f'<h3> ECG data </h3> ' \
               f"<img src='data:image/png;base64,{encoded_ecg_image.decode('utf-8')}'>"

    return html_str


def draw_signal_with_interpretability(
        ecg_data: np.ndarray, ecg_interpretability: np.ndarray,
        non_ecg_data: np.ndarray, non_ecg_interpretability: np.ndarray,
        *, need_to_draw=False
) -> (bytes, np.ndarray):
    """
    Create importance map as image from original data and interpretability data (from different algorithms)
    :param ecg_data: 12-lead ecg signal (legacy, not used now)
    :param ecg_interpretability: importance of ecg points
    :param non_ecg_data: array of additional patient info [age, gender]
    :param non_ecg_interpretability: importance of additional info
    :param need_to_draw: if set to True this method will also call .draw method of plot lib
    :return: encoded importance map and importance of additional info (normalized)
    """
    signals_count, signal_len = ecg_data.shape
    fig, axs = plt.subplots(
        nrows=signals_count, ncols=1, figsize=(20, 2.5 * signals_count), sharex=True
    )

    ticks = np.linspace(0, signal_len / ECG_FREQUENCY, signal_len)

    full_interpretability = np.concatenate((non_ecg_interpretability, ecg_interpretability.reshape(-1)))
    norm_func = plt.Normalize(full_interpretability.min(), full_interpretability.max())

    for i in range(signals_count):
        current_lead = ecg_data[i, :]
        current_interpretability = ecg_interpretability[i, :]
        # Преобразуем данные в формат точек x-y
        points_array = np.array([ticks, current_lead])
        # И преобразуем из формата [[x1, x2, x3], [y1, y2, y3]]
        # В -> [[[x1, y1], [x2, y2], [x3, y3]]] (понадобится для построения сегментов)
        points = points_array.T.reshape(-1, 1, 2)
        # Преобразуем в сегменты вида [[[x1, y1], [x2, y2]], [[x2, y2], [x3, y3]]]
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # Нужна для построения корректного color-map (нормируем значения на промежуток [0, 1]

        # Рисуем LineCollection (т.к. график превратился в набор линий)
        lc = LineCollection(segments, cmap='plasma', norm=norm_func)
        # Используем значения interpretability как colormap
        lc.set_array(current_interpretability)
        # Просто устанавливаем ширину линий
        lc.set_linewidth(3.0)

        axs[i].add_collection(lc)
        color_bar = fig.colorbar(lc, ax=axs[i])
         # fig.colorbar(lc, cax=axs[i, 1])
        color_bar.set_label(f'Lead {i + 1} cmap')

        # Устанавливаем границы, т.к. для LineCollection они не устанавливаются по умолчанию
        axs[i].set_xlim(ticks.min(), ticks.max())
        # Добавляем небольшой оффсет, чтоб график имел отступ от края
        axs[i].set_ylim(current_lead.min() - 0.05, current_lead.max() + 0.05)

        # Нам не нужны тики, т.к. иначе фон становится черным, к тому же они не репрезентативны
        axs[i].get_yaxis().set_ticks([])
        axs[i].get_xaxis().set_ticks([])
        # Устаналвиваем цвет фона в серый, т.к. при выбранной color-map'e удобнее смотреть на сером
        axs[i].set_facecolor('grey')
        # axs[i].grid()

    # logging.info(f'Non ecg_data: {non_ecg_data}. Interpreb: {norm_func(non_ecg_interpretability)}.')
    # plt.subplots_adjust(wspace=0.05, hspace=0.1)
    plt.tight_layout()
    if need_to_draw:
        plt.show()
    tmp_file = BytesIO()
    fig.savefig(tmp_file, format='png')
    encoded_image = base64.b64encode(tmp_file.getvalue())
    plt.close(fig)

    return encoded_image, norm_func(non_ecg_interpretability)


def create_reports_directories():
    for key, value in utils.DIAGNOSIS_MAP.items():
        # May be later I'll add diagnosis description in folder name
        success_directory_path = os.path.join(REPORT_BASE_PATH, f'{key}', 'success')
        failed_directory_path = os.path.join(REPORT_BASE_PATH, f'{key}', 'failed')

        os.makedirs(success_directory_path, exist_ok=True)
        os.makedirs(failed_directory_path, exist_ok=True)


def get_saliency_interpretability(
        model: nn.Module, non_ecg: torch.Tensor, ecg: torch.Tensor, diagnosis: torch.Tensor
) -> (np.ndarray, np.ndarray):
    # Required for interpretability (at least for Saliency)
    non_ecg.requires_grad = True
    ecg.requires_grad = True

    # Create interpretability method
    saliency = Saliency(model)

    # Get interpretation
    non_ecg_grads, ecg_grads = saliency.attribute((non_ecg, ecg), target=diagnosis.item())
    # Preprocess interpretation (because they are tensors)
    # .squeeze убирает все axis с размерностью 1, в данном случае мы избавляемся от батча
    ecg_grads_numpy = ecg_grads.squeeze().cpu().detach().numpy()
    non_ecg_grads_numpy = non_ecg_grads.squeeze().cpu().detach().numpy()
    return non_ecg_grads_numpy, ecg_grads_numpy


def launch_xai(model):
    # Set model to .eval mode
    model.eval()

    # Create directories to save generated reports
    create_reports_directories()

    # Prepare model and dataset
    reference_path = f'{BASE_DATA_PATH}/REFERENCE.csv'
    data = dataset.Loader(BASE_DATA_PATH, reference_path).load_as_df_for_net(normalize=True)
    df = dataset.ECGDataset(data, slices_count=SLICES_COUNT, slice_len=SLICES_LEN, random_state=42)
    loader = DataLoader(df, batch_size=1, num_workers=8, shuffle=False)
    loader_iter = iter(loader)

    df_len = len(df)
    logging.info(f'Dataset length is {df_len}')

    # Calculate importance and create reports (TODO move to function)
    for i in range(df_len):
        non_ecg, ecg, diagnosis = next(loader_iter)
        # Save data to keep original data (both to evaluate and to print in report)
        non_ecg_numpy = non_ecg.squeeze().cpu().detach().numpy()
        ecg_numpy = ecg.squeeze().cpu().detach().numpy()

        # Get interpreb
        non_ecg_grads_numpy, ecg_grads_numpy = get_saliency_interpretability(
            model, non_ecg, ecg, diagnosis
        )

        # Get original model prediction with confidence
        out = F.softmax(model(non_ecg, ecg), 1)
        confidence, pred = torch.max(out.data, 1)
        net_diagnosis = pred.squeeze().detach().item()
        net_confidence = confidence.squeeze().detach().item()

        # Create interpretability map and normalize it for additional fields
        img, mapped_non_ecg_importance = draw_signal_with_interpretability(
            ecg_numpy, ecg_grads_numpy, non_ecg_numpy, non_ecg_grads_numpy
        )

        # Create report
        html = generate_html_report(
            img, non_ecg_numpy, mapped_non_ecg_importance, diagnosis.item(), net_diagnosis, net_confidence,
        )

        # Save report to concrete folder
        diagnosis = diagnosis.item()
        last_folder_name = 'success' if net_diagnosis == diagnosis else 'failed'
        save_path = os.path.join(REPORT_BASE_PATH, str(diagnosis), last_folder_name, f'{uuid.uuid4()}.html')
        logging.info(f'Saving {i} to {save_path}')
        with open(save_path, 'w') as file:
            file.write(html)


if __name__ == '__main__':
    utils.init_logger()
    _model = utils.create_model_by_name('CNN_a', 'CNN_a/CNN_a_with_preprocessing.pth')
    launch_xai(_model)
