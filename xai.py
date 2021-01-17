import torch
from captum.attr import Occlusion, DeepLift, Saliency, GradientShap
from scipy import signal
from torch.utils.data import DataLoader
import torch.nn.functional as F

import dataset
import models
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import utils
import base64
from io import BytesIO

from plotly.subplots import make_subplots
import plotly.graph_objects as go


BASE_PATH = './TrainingSet1'
SLICES_COUNT = 10
SLICES_LEN = 2500  # 5 secs at 500Hz
ECG_FREQUENCY = 500.0


# Как будто просто "сглаживает" графики.
def signal_preprocessing(ecg_lead, sample_freq=ECG_FREQUENCY, borders=(1, 40)):
    # Filters the signal with a butterworth bandpass filter with cutoff frequencies fc=[a, b]
    f0 = 2 * float(borders[0]) / float(sample_freq)
    f1 = 2 * float(borders[1]) / float(sample_freq)
    b, a = signal.butter(2, [f0, f1], btype='bandpass')
    return signal.filtfilt(b, a, ecg_lead)

def graw_plotly_signal(signal_12_lead, interpretability, non_ecg_data, non_ecg_interpretability):
    signals_count, signal_len = signal_12_lead.shape
    fig = make_subplots(rows=signals_count, cols=1)
    x_ticks = list(range(signal_len))
    for i in range(signals_count):
        current_lead = signal_12_lead[i, :]
        fig.add_trace(
            go.Scatter(x=x_ticks, y=current_lead, name=f'Lead {i + 1}'),
            row=(i + 1), col=1
        )

    fig.show()


def generate_html_report(encoded_image, non_ecg_data, non_ecg_interpretability,
                         diagnosis_label, net_diagnosis, net_confidence, is_wrong=False, file_name='test_report'):
    gender_importance = round(non_ecg_interpretability[1] * 100, 2)
    age_importance = round(non_ecg_interpretability[0] * 100, 2)
    diagnosis_confidence = round(net_confidence * 100, 2)
    net_diagnosis_str = f'Net diagnosis: {utils.get_diagnosis_str(net_diagnosis)} ({diagnosis_confidence} %)'
    if is_wrong:
        net_diagnosis_str = f'<div style="color: red">{net_diagnosis_str}</div>'
    html_str = f'<h3> Patient info </h3>' \
               f'Age: {non_ecg_data[0].astype(int)} ({age_importance} %)<br>' \
               f'Gender: {utils.get_gender_str(non_ecg_data[1].astype(int))} ({gender_importance} %)<br>' \
               f'Diagnosis: {utils.get_diagnosis_str(int(diagnosis_label))}<br>' \
               f'{net_diagnosis_str}<br>' \
               f'<h3> ECG data </h3> ' \
               f"<img src='data:image/png;base64,{encoded_image}'>"
    with open(f'{file_name}.html', 'w') as file:
        file.write(html_str)


def draw_signal_with_interpretability(signal_12_lead, interpretability, non_ecg_data, non_ecg_interpretability):
    signals_count, signal_len = signal_12_lead.shape
    fig, axs = plt.subplots(nrows=signals_count, ncols=1,
                           figsize=(20, 2.5 * signals_count), sharex=True)
                           # gridspec_kw={"width_ratios": [1, 0.025]})
    ticks = np.linspace(0, signal_len / ECG_FREQUENCY, signal_len)
    min_interpreb = np.concatenate((non_ecg_interpretability, interpretability.reshape(-1))).min()
    max_interpreb = np.concatenate((non_ecg_interpretability, interpretability.reshape(-1))).max()
    norm_func = plt.Normalize(min_interpreb, max_interpreb)
    for i in range(signals_count):
        current_lead = signal_12_lead[i, :]
        current_interpretability = interpretability[i, :]
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
        axs[i].set_ylim(current_lead.min() - 0.05, current_lead.max() + 0.05)

        axs[i].get_yaxis().set_ticks([])
        axs[i].get_xaxis().set_ticks([])
        axs[i].set_facecolor('grey')
        # axs[i].grid()

    print(f'Non ecg_data: {non_ecg_data}. Interpreb: {norm_func(non_ecg_interpretability)}.')
    # plt.subplots_adjust(wspace=0.05, hspace=0.1)
    plt.grid()
    # plt.show()
    tmp_file = BytesIO()
    fig.savefig(tmp_file, format='png')
    encoded_image = base64.b64encode(tmp_file.getvalue()).decode('utf-8')
    return encoded_image, norm_func(non_ecg_interpretability)

# --- done ---
# Обучить модель на сглаженных данных signal_preprocessing, borders=(3, 30) (3, 50)
# Сгладить "важность" атрибутов (или сегментов) -> попробовать тем же фильтром что и для сигнала
# Добавить нормировку и метки с полом и возрастом

# --- todo ---
# Добавить на график инфу диагноз, правильный диагноз, пол, возраст и их важность ( в процентах ) - несколько штук или все
# calculate f1 at test subsamples


def launch_xai(model):
    reference_path = f'{BASE_PATH}/REFERENCE.csv'
    data = dataset.Loader(BASE_PATH, reference_path).load_as_df_for_net(normalize=True)
    df = dataset.ECGDataset(data, slices_count=SLICES_COUNT, slice_len=SLICES_LEN, random_state=42)
    loader = DataLoader(df, batch_size=1, shuffle=False)
    loader_iter = iter(loader)
    for i in range(10):
        non_ecg, ecg, label = next(loader_iter)
        non_ecg_numpy = non_ecg.squeeze().cpu().detach().numpy()
        ecg_numpy = ecg.squeeze().cpu().detach().numpy()
        saliency = Saliency(model)
        non_ecg.requires_grad = True
        ecg.requires_grad = True
        non_ecg_grads, ecg_grads = saliency.attribute((non_ecg, ecg), target=label)
        model.eval()
        out = F.softmax(model(non_ecg, ecg), 1)
        confidence, pred = torch.max(out.data, 1)
        net_diagnosis = pred.squeeze().detach().item()
        net_confidence = confidence.squeeze().detach().item()
        # .squeeze убирает все axis с размерностью 1, в данном случае мы избавляемся от батча
        ecg_grads_numpy = ecg_grads.squeeze().cpu().detach().numpy()
        non_ecg_grads_numpy = non_ecg_grads.squeeze().cpu().detach().numpy()
        # ecg_numpy = ecg.squeeze().cpu().detach().numpy()
        img, mapped_non_ecg_importance = draw_signal_with_interpretability(
            ecg_numpy, ecg_grads_numpy, non_ecg_numpy, non_ecg_grads_numpy
        )
        generate_html_report(
            img, non_ecg_numpy, mapped_non_ecg_importance,
            label, net_diagnosis, net_confidence,
            is_wrong=label != pred,
            file_name=f'report_{i}'
        )
        # graw_plotly_signal(ecg_numpy, ecg_grads_numpy, non_ecg_numpy, non_ecg_grads_numpy)


if __name__ == '__main__':
    _model = utils.create_model_by_name('CNN_a', 'CNN_a/CNN_a_with_preprocessing.pth')
    _model.eval()
    launch_xai(_model)