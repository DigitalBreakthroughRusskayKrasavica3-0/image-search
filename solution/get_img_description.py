from datetime import datetime
import re

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, BlipProcessor, BlipForConditionalGeneration, pipeline, \
    LlavaForConditionalGeneration
from translate import Translator


from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from tinyllava.model import *

from tinyllava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from tinyllava.conversation import conv_templates, SeparatorStyle
from tinyllava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

from PIL import Image


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


def eval_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = TinyLlavaLlamaForCausalLM.from_pretrained(args.model_path)

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()

    vision_tower.to(device='mps', dtype=torch.float32)

    image_processor = vision_tower.image_processor
    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    # if 'phi' in model_name.lower() or '3.1b' in model_name.lower():
    #     conv_mode = "phi"
    # if "llama-2" in model_name.lower():
    #     conv_mode = "llava_llama_2"
    # elif "v1" in model_name.lower():
    #     conv_mode = "llava_v1"
    # elif "mpt" in model_name.lower():
    #     conv_mode = "mpt"
    # else:
    #     conv_mode = "llava_v0"
    #
    # if args.conv_mode is not None and conv_mode != args.conv_mode:
    #     print(
    #         "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
    #             conv_mode, args.conv_mode, args.conv_mode
    #         )
    #     )
    # else:
    # args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    images = [args.image_file]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    # input_token_len = input_ids.shape[1]
    # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    # if n_diff_input_output > 0:
    #     print(
    #         f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
    #     )
    outputs = tokenizer.batch_decode(
        output_ids, skip_special_tokens=True
    )[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    return outputs


def get_img_desc_llava(img: str | Image.Image):
    model_path = "bczhou/TinyLLaVA-1.5B"
    prompt = "What is displayed in the image?"

    if isinstance(img, str):
        img = Image.open(img).convert("RGB")

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": "phi",
        "image_file": img,
        "sep": ",",
        "temperature": 0.6, # TODO: set this
        "top_p": None,
        "num_beams": 1,
        "load_8bit": None,
        "load_4bit": None,
        "device": "mps",
        "max_new_tokens": 512
    })()

    return eval_model(args)


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

    translator = Translator(to_lang="ru")

    ru_desc = translator.translate(en_desc)
    return ru_desc
