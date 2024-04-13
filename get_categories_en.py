import csv
import clip
import torch
from PIL import Image
import translators as ts

from find_best_en import model, device, preprocess

data_path = './mincult-train/train.csv'

# categories parsing
categories = []
translations = {}
with open(data_path, encoding='utf-8', newline='') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=';')
    for row in reader:
        if row['group'] not in translations.values():
            category = ts.translate_text(row['group'], translator='bing', from_language='ru')
            categories.append(category)
            translations[category] = row['group']

# ЛОЖКУ ОПРЕДЕЛЯЕТ КАК ОРУЖИЕ
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in categories]).to(device)
text_features = model.encode_text(text_inputs)
text_features /= text_features.norm(dim=-1, keepdim=True)

def get_categories(img: str | Image.Image):
    if isinstance(img, str): 
        img = Image.open(img)
    # Download the dataset
    # data_path = 'data/train.csv'
    # image_path = 'data/3850376.jpg'


    # Vectors creating
    image_input = preprocess(img).unsqueeze(0).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(15)

    res = [(translations[categories[index]], 100 * value.item()) for value, index in zip(values, indices)]
    print(res) 
    return [pred[0] for pred in res]
