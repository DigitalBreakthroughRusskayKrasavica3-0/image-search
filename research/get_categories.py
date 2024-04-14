import csv
import sys
from scipy.spatial import distance
import clip
import torch
from PIL import Image
# import translators as ts

# from find_best import model, device, preprocess

from sentence_transformers import SentenceTransformer, util
from find_best import img_model

# We use the original clip-ViT-B-32 for encoding images
# img_model = SentenceTransformer('clip-ViT-B-32')

# Our text embedding model is aligned to the img_model and maps 50+
# languages to the same vector space
text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')



data_path = './mincult-train/train.csv'
# categories parsing
categories = []
# translations = {}
with open(data_path, encoding='utf-8', newline='') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=';')
    for row in reader:
        if row['group'] not in categories:
            # category = ts.translate_text(row['group'], translator='bing', from_language='ru')
            categories.append(row['group'])
            # translations[category] = row['group']

def get_categories(img: str | Image.Image):
    if isinstance(img, str): 
        img = Image.open(img)
    # Download the dataset
    # data_path = 'data/train.csv'
    # image_path = 'data/3850376.jpg'


    # Vectors creating
    # image_input = preprocess(img).unsqueeze(0).to(device)
    # text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in categories]).to(device)

    img_embedding = img_model.encode([img])[0]
    texts = [f"это музейный экспонат категории {c}" for c in categories]

    text_embeddings = text_model.encode(texts)
    dists = {} 
    for i, t in enumerate(texts): 
        dists[t] = distance.cosine(text_embeddings[i], img_embedding)
    best = sorted(list(dists.items()),key=lambda a: a[1])[:5]
    return best

    # for text, scores in zip(texts, cos_sim):
    #     max_img_idx = torch.argmax(scores)
    #     print("Text:", text)
    #     print("Score:", scores[max_img_idx] )

    # Pick the top 5 most similar labels for the image
    # image_features /= image_features.norm(dim=-1, keepdim=True)
    # text_features /= text_features.norm(dim=-1, keepdim=True)
    # similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    # values, indices = similarity[0].topk(15)

    # res = [(categories[index], 100 * value.item()) for value, index in zip(values, indices)]
    # print(res) 
    return

if __name__ == '__main__': 
    get_categories(sys.argv[1])