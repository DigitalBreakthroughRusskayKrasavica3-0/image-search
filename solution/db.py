import numpy as np
import clickhouse_connect 
from PIL import Image
import model


class DB:
    def __init__(self): 
        self.client = clickhouse_connect.get_client()
        self.client.command('''
        CREATE TABLE IF NOT EXISTS museum_items (
            object_id Integer, 
            img_name String, 
            name Nullable(String), 
            description Nullable(String), 
            group String,
            image_embedding Array(Float32)
        ) ENGINE MergeTree ORDER BY object_id
        ''')
    
    def insert_museum_items(self, df):
        for index, row in df.iterrows(): 
            image = Image.open(row.path)
            emb = model.large_clip.encode_images(image).tolist()
            row = row.replace({np.nan: None}).to_dict()
            data = [[row['object_id'], row['img_name'], row['name'], row['description'], row['group'], emb]]
            # cols = ['object_id', 'img_name', 'name', 'description', 'group', 'image_embedding']
            self.client.insert('museum_items', data, column_names='*')

        print('inserted', self.client.command('select count(*) from museum_items'), ' museum items')

    def search_similar(self, img: str | Image.Image):
        if isinstance(img, str): 
            img = Image.open(img) 
        emb = model.large_clip.encode_images(img).tolist()
        parameters = {'query_embedding': emb}
        res = self.client.query('''
            SELECT object_id, img_name, L2Distance(image_embedding, {query_embedding:Array(Float32)}) as dist 
            FROM museum_items 
            ORDER BY dist ASC
            LIMIT 10
        ''', parameters=parameters)
        print(res.result_rows)
        return []


if __name__ == '__main__': 
    db = DB()
    N_TO_INSERT = 100
    # db.insert_museum_items(model.df[:100])
    db.search_similar('../sword.jpg')