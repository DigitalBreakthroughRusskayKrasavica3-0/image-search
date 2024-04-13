import pandas 
import json
import sys
from PIL import Image
import clip
import torch
from scipy.spatial import distance
import model

indexed_df = model.df[:100]

def find_best(img): 
    if isinstance(img, str): 
        img = Image.open(img)
    emb = model.large_clip.encode_images([img])[0].tolist()
    distances = {} 
    for index, row in df.iterrows(): 
        with open(row.embedding_path, 'r') as f:
            other_emb = json.loads(f.readline()) 
        distances[index] = distance.cosine(emb, other_emb)
    dists = sorted(list(distances.items()), key=lambda a: a[1])[:10] 
    res = []
    for i, dist in dists: 
        res.append((df.iloc[i].path))#, dist))
    return res

if __name__ == '__main__': 
    print(find_best(sys.argv[1]))
