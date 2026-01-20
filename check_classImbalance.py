from data import DatasetLoader

dl = DatasetLoader()
train_df, _, _ = dl.get_dataframes()

print(train_df['instrument_family'].value_counts())
