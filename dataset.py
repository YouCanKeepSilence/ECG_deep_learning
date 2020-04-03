import pandas as pd


def get_prepared_dataset(file_path, names, target_column, columns_to_delete, *,
                         augmentation_multiplier=0, augmentation_slice_size=2500):
    df = _load_raw_data(file_path, names)
    df = _normalize_data(df, target_column, columns_to_delete)
    # TODO add augmentation
    # TODO add test/train split
    return df


def _load_raw_data(file_path, names):
    df = pd.read_csv(file_path, names=names)
    return df


def _normalize_data(df, target_column_name, columns_to_remove):
    numerical_names = [name for name in df.columns if df.columns[name].type not in ['object', target_column_name]]
    categorical_names = [name for name in df.columns if df.columns[name].type in ['object', target_column_name]]
