import os
import pandas as pd
from parse_json import parse_json_file
from tqdm import tqdm

class DatasetLoader:
    def create_dataframe(self,data_dir, image_dir):
        sample_ids, metadata = parse_json_file(data_dir)
        generated_ids = [f.replace('.png', '') for f in os.listdir(image_dir) if f.endswith('.png')]
        dataset_records = []
        print(f"Processing records for {data_dir}...")
        for image_id in tqdm(generated_ids):
            if image_id in metadata:
                dataset_records.append({
                    'image_id': image_id,
                    'instrument_family': metadata[image_id]['instrument_family'],
                    'instrument_family_str': metadata[image_id]['instrument_family_str']
                })
        df=pd.DataFrame(dataset_records)
        return df

    def get_dataframes(seld):
        l=DatasetLoader()
        train_df = l.create_dataframe('nsynth-train', 'train_images')
        valid_df = l.create_dataframe('nsynth-valid', 'valid_images')
        test_df = l.create_dataframe('nsynth-test', 'test_images')
    
        return train_df,valid_df,test_df