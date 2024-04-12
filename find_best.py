import pandas 
import json
import sys
from PIL import Image
import clip
import torch
from scipy.spatial import distance

df = pandas.read_csv('./mincult-train/train.csv', sep=';')
df = df[:30]
df['path'] = './mincult-train/train/'+ df['object_id'].astype(str) + '/' + df['img_name']
df['embedding_path'] = df['path']+'.embedding'

from sentence_transformers import SentenceTransformer, util

# We use the original clip-ViT-B-32 for encoding images
img_model = SentenceTransformer('clip-ViT-B-32')

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"using {device}")
# device = torch.device(device)
# model, preprocess = clip.load("ViT-L/14")
# model.to(device)

# def get_embedding(img): 
#     if isinstance(img, str):
#         img = Image.open(img)
#     image = preprocess(img).unsqueeze(0).to(device)
#     with torch.no_grad():
#         return model.encode_image(image)[0].tolist()

def find_best(img): 
    if isinstance(img, str): 
        img = Image.open(img)
    emb = img_model.encode([img])[0].tolist()
    distances = {} 
    for index, row in df.iterrows(): 
        with open(row.embedding_path, 'r') as f:
            other_emb = json.loads(f.readline()) 
            dist = distance.cosine(emb, other_emb)
            distances[index] = dist
    dists = sorted(list(distances.items()), key=lambda a: a[1])[:10] 
    res = []
    for i, dist in dists: 
        res.append((df.iloc[i].path))#, dist))
    print(res)
    return res[0], res[1:]

if __name__ == '__main__': 
    print(find_best(sys.argv[1]))
