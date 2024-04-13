import abc
import torch
import pandas 
from PIL import Image
import model

class CategoryGetter: 
    def __init__(self, full_df, pref, other_thresh, model: CategoryModel): 
        self.model = model
        self.categories = full_df['group'].unique().tolist()
        self.categories.remove('Прочие')
        self.categories.append('Меч')
        # print(self.categories)
        self.pref = pref
        self.other_thresh = other_thresh
        self.texts = [pref+c for c in self.categories]
        self.text_features = self.model.encode_texts(self.texts)
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def get_categories(self, img: str | Image.Image):
        if isinstance(img, str): 
            img = Image.open(img)
        image_features = self.model.encode_images(img)
        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = enumerate((100.0 * image_features @ self.text_features.T).softmax(dim=-1).tolist())
        # print(similarity)
        res = [[self.categories[i], conf] for i, conf in sorted(similarity, key=lambda a: a[1], reverse=True)]
        # print(res)
        if res[0][0] == 'Меч':
            res[0][0] = 'Оружие'
        if res[0][1] < self.other_thresh:
            res[0][0] = 'Прочие'
        # if res[0][0] == 'Печатная продукция': 
        #     res = res[1:]
        return res

category_getter = CategoryGetter(model.df, "экспонат музея категории ", 0.3, model.large_clip)

if __name__ == '__main__': 
    category_getter.get_categories(sys.argv[1])