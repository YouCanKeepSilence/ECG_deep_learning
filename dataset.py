import glob
import os

import numpy as np
import pandas as pd
import scipy.io as sio
import torch.utils.data as data_utils


class Loader:
    def __init__(self, record_base_path, reference_path):
        self.record_base_path = record_base_path
        self.max_len_per_sensor = None
        self.reference = pd.read_csv(reference_path)

    @property
    def padded_len(self):
        return self.max_len_per_sensor * 12

    def _prepare_input(self, flatten=False):
        record_paths = glob.glob(os.path.join(self.record_base_path, '*.mat'))
        res = []
        # We will calculate max element due processing
        self.max_len_per_sensor = 0

        for record_path in record_paths:
            name = record_path.split('/')[-1].split('.')[0]
            answer = self.reference[self.reference['Recording'] == name].iloc[0]
            mat = sio.loadmat(record_path)

            ecg = np.array(mat['ECG']['data'][0, 0]).astype(np.float32)
            if ecg.shape[-1] > self.max_len_per_sensor:
                self.max_len_per_sensor = ecg.shape[-1]

            sex = mat['ECG']['sex'][0, 0][0]
            age = mat['ECG']['age'][0, 0][0][0]
            if flatten:

                res.append([sex, age, answer['First_label'], *ecg.flatten()])
            else:
                res.append((sex, age, answer['First_label'], ecg))

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

    def load_as_x_y_for_ml(self, normalize=True, *,
                           augmentation_multiplier=0, augmentation_slice_size=2500, check_to_na=False):
        data = self._prepare_input(flatten=True)
        names_array = ['gender', 'age', 'label', *[f'c_{i}' for i in range(self.padded_len)]]
        data = pd.DataFrame.from_records(data, index=list(range(0, len(data))), columns=names_array)
        if normalize:
            data.dropna(axis=0, inplace=True)
            data.drop(data[data['age'] == -1].index, inplace=True)
            data['age'] = data['age'].astype(np.float32)
            data = self._normalize_data(data, 'label')
        if check_to_na:
            check_na_series = data.isnull().sum()
            if len(check_na_series[check_na_series > 0].index) != 0:
                raise Exception('There are some NA values in Dataset')

        # For NN we use ECGDataset to augmentation
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

        sensors_count = 12
        non_sliceable_end = 3
        non_sliceable_columns = df.iloc[:, :non_sliceable_end]
        non_sliceable_columns_names = list(non_sliceable_columns.columns)

        def get_record_slice(rec):
            non_ecg = rec.iloc[:non_sliceable_end]
            ecg = rec.iloc[non_sliceable_end:]
            ecg_len = len(ecg)
            # Original per sensor length
            per_sensor_len = ecg_len / 12
            slice_start = np.random.randint(0, per_sensor_len - slice_size)
            indexes = []
            for k in range(sensors_count):
                indexes.extend([k * per_sensor_len + slice_start + j for j in range(slice_size)])

            return non_ecg.append(ecg[indexes], ignore_index=True)

        final_measures_names = ['c_{}'.format(i + 1) for i in range(slice_size * sensors_count)]
        new_names = [*non_sliceable_columns_names, *final_measures_names]

        new_data = df.apply(get_record_slice, axis=1).set_axis(new_names, axis=1)
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
        # Optimize normalization, to fit in memory
        normalize_batch = 1000
        for i in range(0, len(numerical_names), normalize_batch):
            sliced = numerical_names[i:i + normalize_batch]
            df[sliced] = (df[sliced] - df[sliced].mean(axis=0)) / df[sliced].std(axis=0)

        return df


class ECGDataset(data_utils.Dataset):
    def __init__(self, data, slices_count=10, slice_len=1000,
                 sensors_count=12,
                 random_state=42,
                 sensors_transform=None,
                 non_sensors_transform=None):
        super().__init__()
        np.random.seed(random_state)
        self.slices_count = slices_count
        self.sensors_transform = sensors_transform
        self.non_sensors_transform = non_sensors_transform
        self.slice_len = slice_len
        self.data = data
        self.data_len = len(data)
        self.non_ecg_numpy = self.data[['age', 'gender']].to_numpy()
        self.labels_numpy = self.data['label'].to_numpy() - 1

    def __len__(self):
        # We just show that ds is larger on slices_count times
        return len(self.data) * self.slices_count

    def __getitem__(self, idx):
        current_idx = idx % self.data_len

        current_rec = self.data.iloc[current_idx]
        current_ecg = current_rec['ecg']

        # at ever __getitem__ call we get RANDOM slice for only current record
        current_ecg_len = current_ecg.shape[-1]
        slice_start = np.random.randint(0, current_ecg_len - self.slice_len)
        slice_end = slice_start + self.slice_len

        label = self.labels_numpy[current_idx]
        non_ecg_tensor = self.non_ecg_numpy[current_idx]
        ecg_tensor = current_ecg[:, slice_start: slice_end]
        if self.sensors_transform:
            # TODO transform
            pass
        if self.non_sensors_transform:
            pass
        return non_ecg_tensor, ecg_tensor, label


