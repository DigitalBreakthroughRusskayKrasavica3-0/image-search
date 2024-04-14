from datetime import datetime

from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, BlipProcessor, BlipForConditionalGeneration
from translate import Translator


def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start = datetime.now()
        res = func(*args, **kwargs)
        print(datetime.now() - start)
        return res

    return wrapper


# @measure_execution_time
# fast and 50/50 result
def get_img_description_blip(img: str | Image.Image) -> str:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    if isinstance(img, str):
        img = Image.open(img).convert('RGB')

    # conditional image captioning
    text = "An exhibit of "
    inputs = processor(img, text, return_tensors="pt")

    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)


# @measure_execution_time
# fast and 50/50 result (but better than first)
def get_img_desc_base_git(img: str | Image.Image):
    processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

    if isinstance(img, str):
        img = Image.open(img)

    pixel_values = processor(images=img, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption


# @measure_execution_time
# good result and slow
def get_img_desc_large_git(img: str | Image.Image):
    processor = AutoProcessor.from_pretrained("alexgk/git-large-coco")
    model = AutoModelForCausalLM.from_pretrained("alexgk/git-large-coco")

    if isinstance(img, str):
        img = Image.open(img)

    pixel_values = processor(images=img, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_caption


# @measure_execution_time
# better than the first but slower and faster than large blip
def get_img_desc_blip_mocha(img: str | Image.Image):
    processor = BlipProcessor.from_pretrained("moranyanuka/blip-image-captioning-large-mocha")
    model = BlipForConditionalGeneration.from_pretrained("moranyanuka/blip-image-captioning-large-mocha")

    if isinstance(img, str):
        img = Image.open(img).convert('RGB')

    # conditional image captioning
    text = "an exhibit of"
    inputs = processor(img, text, return_tensors="pt")

    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)


def get_img_description_ru(img: str | Image.Image, get_img_desc_func=get_img_desc_blip_mocha) -> str:
    en_desc = get_img_desc_func(img)
    print('hello', en_desc)
    translator = Translator(to_lang="ru")

    ru_desc = translator.translate(en_desc)
    return ru_desc
