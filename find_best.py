import pandas 
import sys
from PIL import Image
import clip
import torch
from scipy.spatial import distance

df = pandas.read_csv('./mincult-train/train.csv', sep=';')
df = df[:30]
df['path'] = './mincult-train/train/'+ df['object_id'].astype(str) + '/' + df['img_name']
df['embedding_path'] = df['path']+'.embedding'

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device}")
device = torch.device(device)
model, preprocess = clip.load("ViT-L/14")
model.to(device)

def get_embedding(image_path): 
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        return model.encode_image(image)[0].tolist()

def find_best(image_path): 
    emb = get_embedding(image_path) 
    distances = {} 
    for index, row in df.iterrows(): 
        with open(row.embedding_path, 'r') as f:
            other_emb = eval(f.readline()) # TODO: parse correctly
            dist = distance.cosine(emb, other_emb)
            distances[index] = dist
    dists = sorted(list(distances.items()), key=lambda a: a[1])[:10] 
    res = []
    for i, dist in dists: 
        res.append((df.iloc[i].path, dist))
    return res

if __name__ == '__main__': 
    print(find_best(sys.argv[1]))
