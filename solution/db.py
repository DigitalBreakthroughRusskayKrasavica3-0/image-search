import sys
import os
import numpy as np
import os
import clickhouse_connect 
from PIL import Image
import dataset
import ai_model

# factor out dataset.py from ai_model.py

class DB:
    def __init__(self, model): 
        self.model = model
        self.client = clickhouse_connect.get_client(
            host=os.getenv('CLICKHOUSE_HOST') or None, 
            port=os.getenv('CLICKHOUSE_PORT') or None,
            username=os.getenv('CLICKHOUSE_USERNAME') or None, 
            password=os.getenv('CLICKHOUSE_PASSWORD') or None,
        )
        print('connected to clickhouse')
        self.client.command('''
        CREATE TABLE IF NOT EXISTS museum_imgs (
            object_id Integer, 
            img_name String, 
            name Nullable(String), 
            description Nullable(String), 
            group String,
            image_embedding Array(Float32),
        ) ENGINE MergeTree ORDER BY object_id 
        ''') # TODO: add unique constraint
    
    def insert_museum_imgs(self, df):
        data = [] 
        for index, row in df.iterrows(): 
            image = Image.open(row.path)
            emb = self.model.encode_images(image).tolist()
            row = row.replace({np.nan: None}).to_dict()
            data.append([row['object_id'], row['img_name'], row['name'], row['description'], row['group'], emb])
        self.client.insert('museum_imgs', data, column_names='*')

        print('inserted', self.client.command('select count(*) from museum_imgs'), 'museum images')

    def search_similar(self, img: str | Image.Image):
        if isinstance(img, str): 
            img = Image.open(img) 
        emb = self.model.encode_images(img).tolist()
        return self._search_by_embedding(emb)

    def search_similar_by_text(self, text: str):
        emb = self.model.encode_texts([text])[0].tolist()
        return self._search_by_embedding(emb)

    def _search_by_embedding(self, emb): 
        parameters = {'query_embedding': emb}
        res = self.client.query('''
            SELECT object_id, group, img_name, L2Distance(image_embedding, {query_embedding:Array(Float32)}) as dist 
            FROM museum_imgs 
            ORDER BY dist ASC
            LIMIT 10
        ''', parameters=parameters)
        return [(str(row[0]), row[1], os.path.join(str(row[0]), row[2]), row[3]) for row in res.result_rows]


if __name__ == '__main__': 
    df = dataset.df
    if len(sys.argv) > 1: 
        df = df[:int(sys.argv[1])]
    db = DB(ai_model.large_clip)
    db.insert_museum_imgs(df)