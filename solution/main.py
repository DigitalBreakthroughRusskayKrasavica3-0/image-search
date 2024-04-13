import json

from PIL import Image
import tempfile
import gradio as gr

import get_img_description
import os
import db 

import ai_model
import dataset
from categories_getter import category_getter

DB = db.DB(model=ai_model.large_clip)

def get_images(image: Image.Image) -> tuple[str, str, list[str]]:
    fetched_images = DB.search_similar(image)
    images = [os.path.join(dataset.DATASET_PATH, 'train', img_path) for _, img_path in fetched_images]
    print(images)
    return fetched_images[0][0], images[0], images[1:]


def get_categories(image: Image.Image) -> list[str]:
    cats = category_getter.get_categories(image)
    print(cats)
    cats = [f'{cat} ({round(100*conf, 2)}%)' for cat, conf in cats]
    return cats


def get_description(image: Image) -> str:
    desc = get_img_description.get_img_description_ru(image, get_img_description.get_img_desc_large_git)
    return desc[0].upper() + desc[1:]


def get_json_dump(object_id: str, category: str, description: str) -> str:
    temp_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False)
    category = category[:category.rfind('(')].strip()
    json.dump(
        {
            'similar_object_id': object_id,  
            'category': category, 
            'description': description,
        },
        temp_file,
        separators=(', ', ': '),
        ensure_ascii=False,
    )
    return temp_file.name


def change_outputs_visible() -> list:
    global is_visible
    is_visible = not is_visible
    print(is_visible)
    return [gr.update(visible=is_visible) for _ in range(8)] + [None]


css = '''
.big textarea {font-size: 24px}
.big {font-size: 24px}
#logo {width: auto !important; height; auto !important; max-width: 20px;}
.header_image {overflow: scroll !important;}
'''


with gr.Blocks(css=css) as demo:
    is_visible = False
    image_id = gr.State()

    # header
    with gr.Row():
        logo = gr.Image(
            'static/logo.png',
            container=False,
            show_download_button=False,
            elem_id='logo'
        )
        with gr.Column():
            title = gr.HTML('<h1>Хакатон "Цифровой Прорыв ЮФО"</h1>')
            subtitle = gr.HTML('<h1>Команда "Русская Красавица 3.0"</h1>')

    # download chapter
    with gr.Row():
        input_image = gr.Image(
            height='500px',
            label='Загрузите изображение',
            type='pil',
            elem_classes='header_image'
        )
        best_image = gr.Image(
            interactive=False,
            sources=[],
            height='500px',
            label='Лучшее совпадение по изображению',
            type='filepath',
            elem_classes='header_image'
        )

    # description chapter
    with gr.Row():
        description = gr.Text(
            info='Описание изображения',
            elem_classes='big',
            show_label=False,
            scale=5,
            visible=False
        )
        json_btn = gr.DownloadButton(
            'Скачать в JSON',
            elem_classes='big',
            variant='primary',
            visible=False
        )

    # categories chapter
    with gr.Row():
        best_category = gr.Text(
            interactive=False,
            info='Лучшее совпадение по категории (в скобках указаны проценты уверенности)',
            elem_classes='big',
            show_label=False,
            scale=3,
            visible=False
        )
        categories = [gr.Text(
            info='Похожая категория',
            elem_classes='big',
            show_label=False,
            visible=False
        ) for _ in range(4)]

    # images gallery chapter
    images = gr.Gallery(
        label='Похожие изображения',
        columns=5,
        rows=2,
        elem_id='gallery',
        visible=False
    )

    # input image events
    input_image.upload(
        fn=get_images,
        inputs=input_image,
        outputs=[image_id, best_image, images]
    )
    input_image.upload(
        fn=get_categories,
        inputs=input_image,
        outputs=[best_category, *categories]
    )
    input_image.upload(
        fn=get_description,
        inputs=input_image,
        outputs=[description]
    )

    # change output visible
    input_image.upload(
        fn=change_outputs_visible,
        outputs=[best_category, description, json_btn, *categories, images]
    )
    input_image.clear(
        fn=change_outputs_visible,
        outputs=[best_category, description, json_btn, *categories, images, best_image]
    )

    # JSON button event (download JSON)
    json_btn.click(
        fn=get_json_dump,
        inputs=[image_id, best_category, description],
        outputs=[json_btn]
    )


if __name__ == '__main__':
    demo.launch(share=True, server_port=8042)
