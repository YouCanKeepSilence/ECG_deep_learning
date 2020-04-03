file_path = '/Users/silence/Workbench/ml/ml_labs/LegacyData/half.csv'
columns_count = 60000
names_array = ['gender', 'age', 'first_label', 'second_label', 'third_label', *['c_{}'.format(i + 1) for i in range(columns_count)]]