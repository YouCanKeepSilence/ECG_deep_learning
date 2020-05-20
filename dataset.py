import glob
import os

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.utils.data as data_utils
from keras.preprocessing import sequence
from torchvision.transforms import transforms


class Loader:
    def __init__(self, record_base_path, reference_path,
                 start_value=4000, end_value=6000,
                 median_length=5000, pad_value=0):
        self.record_base_path = record_base_path
        self.start_value = start_value
        self.end_value = end_value
        self.median_length = median_length
        self.pad_value = pad_value
        self.padded_len = median_length * 12
        self.reference = pd.read_csv(reference_path)

    def _prepare_input(self, flatten=False):
        record_paths = glob.glob(os.path.join(self.record_base_path, '*.mat'))
        res = []
        for record_path in record_paths:
            name = record_path.split('/')[-1].split('.')[0]
            answer = self.reference[self.reference['Recording'] == name].iloc[0]
            mat = sio.loadmat(record_path)
            ecg = np.array(mat['ECG']['data'][0, 0])
            if self.end_value >= ecg.shape[1] >= self.start_value:
                if flatten:
                    padded = sequence.pad_sequences(ecg, dtype=np.float32, maxlen=self.median_length, value=self.pad_value,
                                                    padding='post', truncating='post').flatten()
                    res.append([mat['ECG']['sex'][0, 0][0], mat['ECG']['age'][0, 0][0][0], answer['First_label'], *padded])
                else:
                    padded = sequence.pad_sequences(ecg, dtype=np.float32, maxlen=self.median_length, value=self.pad_value,
                                                    padding='post', truncating='post')
                    res.append(
                        (mat['ECG']['sex'][0, 0][0], mat['ECG']['age'][0, 0][0][0], answer['First_label'], padded))
        return res

    def load_as_df_for_net(self, normalize=True):
        data = self._prepare_input(flatten=False)
        names_array = ['gender', 'age', 'label', 'ecg']
        data = pd.DataFrame.from_records(data, index=list(range(0, len(data))), columns=names_array)
        if normalize:
            data.dropna(axis=0, inplace=True)
            data.drop(data[data['age'] == -1].index, inplace=True)
            data['age'] = data['age'].astype(np.float32)
            data.at[data['gender'] == 'Male', 'gender'] = 0
            data.at[data['gender'] == 'Female', 'gender'] = 1
            data['gender'] = data['gender'].astype(np.float32)
        return data

    def load_as_x_y_for_ml(self, normalize=True, *, augmentation_multiplier=0, augmentation_slice_size=2500):
        data = self._prepare_input(flatten=True)
        names_array = ['gender', 'age', 'label', *[f'c_{i}' for i in range(self.padded_len)]]
        data = pd.DataFrame.from_records(data, index=list(range(0, len(data))), columns=names_array)
        if normalize:
            data.dropna(axis=0, inplace=True)
            data.drop(data[data['age'] == -1].index, inplace=True)
            data['age'] = data['age'].astype(np.float32)
            data = self._normalize_data(data, 'label')

        check_na_series = data.isnull().sum()
        if len(check_na_series[check_na_series > 0].index) != 0:
            raise Exception('There are some NA values in Dataset')

        if augmentation_multiplier > 0:
            data = self._augmentation(data, augmentation_multiplier, augmentation_slice_size)
        x = data.drop('label', axis=1).to_numpy(dtype=np.float32)
        y = data['label'].to_numpy(dtype=np.long) - 1
        return x, y

    def _augmentation(self, df, multiplier, slice_size):
        """
        Make augmentation of df by slicing n = multiplier times with slice window size = slice_size
        :param df: source dataset
        :param multiplier: count of repeat
        :param slice_size: window size
        :return: Augmented dataset
        """

        old_batch = self.median_length
        sensors_count = 12
        non_sliceable_columns = df.iloc[:, 0:3]
        sliceable_columns = df.iloc[:, 3:]

        new_data = pd.DataFrame()
        final_measures_names = ['c_{}'.format(i + 1) for i in range(slice_size * sensors_count)]
        for i in range(multiplier):
            low_border = np.random.randint(0, old_batch - slice_size)
            indexes = []
            for k in range(sensors_count):
                indexes.extend([k * old_batch + low_border + j for j in range(slice_size)])

            sliced = sliceable_columns.iloc[:, indexes]
            new_measure_names = {x: y for x, y in zip(sliced.columns, final_measures_names)}
            sliced = sliced.rename(columns=new_measure_names)
            sliced = pd.concat([non_sliceable_columns, sliced], axis=1, sort=False)
            new_data = pd.concat([new_data, sliced], axis=0, sort=False, ignore_index=True)

        return new_data

    @staticmethod
    def _normalize_data(df, target_column_name, columns_to_remove=None):
        """
        Make data normalization. For numeric features normalize them regarding mean by std.
        For categorical features uses OneHot encoding or binarization.
        Also removes useless columns
        :param df: Source dataset
        :param target_column_name: target column (which will be predicted)
        :param columns_to_remove: columns which should be removed
        :return: Normalized dataset
        """
        # Remove useless columns
        if columns_to_remove:
            df.drop(columns_to_remove, axis=1, inplace=True)

        numerical_names = [name for name in df.columns if df[name].dtype.name != 'object']
        categorical_names = [name for name in df.columns if df[name].dtype.name == 'object']

        # We don't want to do anything with target column, so just remove it from all lists
        if target_column_name in numerical_names:
            numerical_names.remove(target_column_name)

        if target_column_name in categorical_names:
            categorical_names.remove(target_column_name)

        # Make vectorization for all categorical features
        for categorical_name in categorical_names:
            unique_values = df[categorical_name].unique()
            if len(unique_values) == 2:
                # if this make binarization
                for idx, value in enumerate(unique_values):
                    df.at[df[categorical_name] == value, categorical_name] = idx

                categorical_names.remove(categorical_name)
                numerical_names.append(categorical_name)
            else:
                # else we should make onehot encoding, but in our dataset no such columns
                raise Exception(f'Not implemented OneHot Encoding for categorical column {categorical_name}')

        # Numeric columns normalization
        columns_to_normalize = df.columns.difference([*categorical_names]).to_list()
        # Optimize normalization, to fit in memory
        normalize_batch = 2500
        for i in range(0, len(columns_to_normalize), normalize_batch):
            sliced = columns_to_normalize[i:i + normalize_batch]
            df[sliced] = (df[sliced] - df[sliced].mean(axis=0)) / df[sliced].std(axis=0)

        return df


class ECGDataset(data_utils.Dataset):
    def __init__(self, data, slices_count=10, slice_len=2500,
                 sensors_count=12,
                 sensor_length=5000,
                 random_state=42,
                 sensors_transform=None,
                 non_sensors_transform=None):
        super().__init__()
        np.random.seed(random_state)
        self.slices_count = slices_count
        self.slice_starts = np.random.random_integers(0, sensor_length - slice_len, slices_count)
        self.sensors_transform = sensors_transform
        self.non_sensors_transform = non_sensors_transform
        self.slice_len = slice_len
        self.data = data
        self.data_len = len(data)
        self.ecg_shape = (sensors_count, slice_len)
        self.non_ecg_numpy = np.stack(self.data[['age', 'gender']].to_numpy())
        self.labels_numpy = self.data['label'].to_numpy() - 1
        self.ecg_numpy = np.stack(self.data['ecg'])

    def __len__(self):
        return len(self.data) * self.slices_count

    def __getitem__(self, idx):
        current_slice_idx = int(np.floor(idx / self.data_len))
        offset = current_slice_idx * self.data_len
        slice_starter = self.slice_starts[current_slice_idx]
        slice_ender = slice_starter + self.slice_len
        current_idx = idx - offset
        label = self.labels_numpy[current_idx]
        non_ecg_tensor = self.non_ecg_numpy[current_idx]
        ecg_tensor = self.ecg_numpy[current_idx, :, slice_starter: slice_ender]
        if self.sensors_transform:
            # TODO transform
            pass
        if self.non_sensors_transform:
            pass
        return non_ecg_tensor, ecg_tensor, label


