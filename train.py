from dataset import get_prepared_dataset

file_path = '/Users/silence/Workbench/ml/ml_labs/LegacyData/test.csv'
columns_count = 60000
target_column = 'first_label'
names_array = ['gender', 'age', 'first_label', 'second_label', 'third_label', *['c_{}'.format(i + 1) for i in range(columns_count)]]

if __name__ == '__main__':
    df = get_prepared_dataset(file_path, names_array, target_column,
                              ['second_label', 'third_label'], augmentation_multiplier=2)
    X = df.drop(target_column, axis=1).to_numpy()
    Y = df[target_column].to_numpy()
    Y -= 1