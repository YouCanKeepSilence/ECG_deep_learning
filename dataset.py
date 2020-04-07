import pandas as pd
import numpy as np
import torch
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split


def get_prepared_dataset(file_path, names, target_column, columns_to_delete, *,
                         augmentation_multiplier=0, augmentation_slice_size=2500):
    """
    Возращает готовый к работе датасет
    :param file_path: Путь до csv с данными
    :param names: Имена столбцов
    :param target_column: Целевой столбец (то, что будет предсказываться)
    :param columns_to_delete: Столбцы, которые нужно удалить
    :param augmentation_multiplier: Кол-во повторений в аугментации
    :param augmentation_slice_size: Размер "окна" в аугментации
    :return: X, y
    """
    df = _load_raw_data(file_path, names)
    df = _normalize_data(df, target_column, columns_to_delete)
    if augmentation_multiplier > 0:
        df = _augmentation(df, augmentation_multiplier, augmentation_slice_size)
    check_na_series = df.isnull().sum()
    if len(check_na_series[check_na_series > 0].index) != 0:
        raise Exception('There are some NA values in Dataset')
    X = df.drop(target_column, axis=1).to_numpy(dtype=np.float32)
    y = df[target_column].to_numpy(dtype=np.long)
    y -= 1
    return X, y


def split(x, y, test_size, random_state=42, batch_size=10000):
    """
    Просто обертка для разбития на тестовую и тренировочную выборку
    :param x:
    :param y:
    :param test_size: Размер в процентах тестовой выборки
    :param random_state:
    :param batch_size: Размер батча
    :return: Разбитые данные X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    # train = data_utils.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    # test = data_utils.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    # train_loader = data_utils.DataLoader(train, batch_size=batch_size)
    # test_loader = data_utils.DataLoader(test, batch_size=batch_size)
    return [(torch.from_numpy(X_train), torch.from_numpy(y_train))], [(torch.from_numpy(X_test), torch.from_numpy(y_test))] # train_loader, test_loader


def _load_raw_data(file_path, names):
    """
    Загружает данные из csv в pd.DataFrame
    :param file_path: Путь до csv
    :param names: Имена столбцов
    :return: pd.DataFrame
    """
    df = pd.read_csv(file_path, names=names)
    return df


def _augmentation(df, multiplier, slice_size):
    """
    Аугментирует датасет используя рандомный срез заданной длины, выбирает рандомно заданное число раз
    :param df: Исходный датасет
    :param multiplier: Кол-во повторений
    :param slice_size: Размер "окна" среза
    :return: Аугментированный датасет
    """

    old_batch = 5000
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


def _normalize_data(df, target_column_name, columns_to_remove):
    """
    Нормализует данные. Для числовых признаков нормализует их относительно мат ожидания
    Для категориальных признаков используется OneHot encoding или бинаризация
    Удаляет ненужные столбцы
    :param df: Исходный датасет
    :param target_column_name: Имя целевого столба (то, что будет предсказываться)
    :param columns_to_remove: Колонки, которые необходимо удалить
    :return: Нормализованный датасет
    """
    # Удаляем ненужные столбцы
    df.drop(columns_to_remove, axis=1, inplace=True)

    numerical_names = [name for name in df.columns if df[name].dtype.name != 'object']
    categorical_names = [name for name in df.columns if df[name].dtype.name == 'object']

    # Не хотим учитывать целевой как численный или категориальный
    if target_column_name in numerical_names:
        numerical_names.remove(target_column_name)

    if target_column_name in categorical_names:
        categorical_names.remove(target_column_name)


    # Векторизация категориальных признаков, за исключением целевого
    for categorical_name in categorical_names:
        unique_values = df[categorical_name].unique()
        if len(unique_values) == 2:
            # Если так, то проводим бинаризацию
            for idx, value in enumerate(unique_values):
                df.at[df[categorical_name] == value, categorical_name] = idx

            categorical_names.remove(categorical_name)
            numerical_names.append(categorical_name)
        else:
            # Иначе onehot encoding, но тут нет необходимости, т.к. из категориальных только пол
            raise Exception(f'Not implemented OneHot Encoding for categorical column {categorical_name}')


    # Нормировка числовых переменных
    numerical_data = df[numerical_names]
    mean = numerical_data.mean(axis=0)
    std = numerical_data.std(axis=0)
    numerical_data = (numerical_data - mean) / std
    df[numerical_names] = numerical_data

    return df

