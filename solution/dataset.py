import pandas
import os


def preprocess_df(full_df): 
    full_df['path'] = os.path.join(DATASET_PATH, 'train') + '/' + full_df['object_id'].astype(str) + '/' + full_df['img_name']
    # full_df['embedding_path'] = full_df['path']+'.embedding'
    # full_df['group'] = full_df['group'].replace('ДПИ',  'Декоративно-прикладное искусство')
    return full_df

DATASET_PATH = os.getenv("DATASET_PATH") or '../mincult-train/'
df = pandas.read_csv(os.path.join(DATASET_PATH, 'train.csv'), sep=';')
df = preprocess_df(df)
