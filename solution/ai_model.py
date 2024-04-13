import os
from multilingual_clip import pt_multilingual_clip
import transformers
import open_clip
import abc
import torch

class CategoryModel(abc.ABC): 
    @abc.abstractmethod
    def encode_images(self, img) -> torch.Tensor: 
        pass
    @abc.abstractmethod
    def encode_texts(self, texts) -> torch.Tensor: 
        pass 

class LargeClipCat(CategoryModel):
    def __init__(self):
        self.model = pt_multilingual_clip.MultilingualCLIP.from_pretrained('M-CLIP/XLM-Roberta-Large-Vit-B-16Plus')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('M-CLIP/XLM-Roberta-Large-Vit-B-16Plus')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.img_model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
        self.img_model.to(self.device)

    def encode_images(self, img): 
        image = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.img_model.encode_image(image)[0]
    def encode_texts(self, texts): 
        return self.model.forward(texts, self.tokenizer)

large_clip = LargeClipCat() 