from captum.attr import Occlusion, DeepLift, Saliency, GradientShap
from scipy import signal
from torch.utils.data import DataLoader

import dataset
import models
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import utils

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


def draw_signal_with_interpretability(signal_12_lead, interpretability, non_ecg_data, non_ecg_interpretability):
    signals_count, signal_len = signal_12_lead.shape
    fig, ax = plt.subplots(nrows=signals_count, ncols=2,
                           figsize=(24, 3 * signals_count), sharex=True,
                           gridspec_kw={"width_ratios": [1, 0.025]})
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
        lc = LineCollection(segments, cmap='inferno_r', norm=norm_func)
        # Используем значения interpretability как colormap
        lc.set_array(current_interpretability)
        # Просто устанавливаем ширину линий
        lc.set_linewidth(3.0)

        ax[i, 0].add_collection(lc)
        color_bar = fig.colorbar(lc, cax=ax[i, 1])
        color_bar.set_label(f'Lead {i + 1} cmap')

        # Устанавливаем границы, т.к. для LineCollection они не устанавливаются по умолчанию
        ax[i, 0].set_xlim(ticks.min(), ticks.max())
        ax[i, 0].set_ylim(current_lead.min(), current_lead.max())

        ax[i, 0].get_yaxis().set_ticks([])
        ax[i, 0].get_xaxis().set_ticks([])
        ax[i, 0].grid()

    print(f'Non ecg_data: {non_ecg_data}. Interpreb: {norm_func(non_ecg_interpretability)}.'
          f' MinMax intepreb: {norm_func(np.array([min_interpreb, max_interpreb]))}')
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    plt.grid()
    plt.show()


# Обучить модель на сглаженных данных signal_preprocessing, borders=(3, 30) (3, 50)
# Сгладить "важность" атрибутов (или сегментов) -> попробовать тем же фильтром что и для сигнала
# Добавить нормировку и метки с полом и возрастом

def launch_xai(model):
    reference_path = f'{BASE_PATH}/REFERENCE.csv'
    data = dataset.Loader(BASE_PATH, reference_path).load_as_df_for_net(normalize=True)
    df = dataset.ECGDataset(data, slices_count=SLICES_COUNT, slice_len=SLICES_LEN, random_state=42)
    loader = DataLoader(df, batch_size=1, shuffle=False)
    loader_iter = iter(loader)
    non_ecg, ecg, label = next(loader_iter)
    non_ecg_numpy = non_ecg.squeeze().cpu().detach().numpy()
    ecg_numpy = ecg.squeeze().cpu().detach().numpy()
    # utils.draw_signal(ecg_numpy)
    # utils.draw_signal(ecg_numpy, preprocessing_func=signal_preprocessing)
    saliency = Saliency(model)
    non_ecg.requires_grad = True
    ecg.requires_grad = True
    non_ecg_grads, ecg_grads = saliency.attribute((non_ecg, ecg), target=label)
    # .squeeze убирает все axis с размерностью 1, в данном случае мы избавляемся от батча
    ecg_grads_numpy = ecg_grads.squeeze().cpu().detach().numpy()
    non_ecg_grads_numpy = non_ecg_grads.squeeze().cpu().detach().numpy()
    # ecg_numpy = ecg.squeeze().cpu().detach().numpy()
    draw_signal_with_interpretability(ecg_numpy, ecg_grads_numpy, non_ecg_numpy, non_ecg_grads_numpy)


if __name__ == '__main__':
    _model = utils.create_model_by_name('CNN_a', 'CNN_a/CNN_a_with_preprocessing.pth')
    _model.eval()
    launch_xai(_model)