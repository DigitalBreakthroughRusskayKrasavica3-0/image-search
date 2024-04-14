import json

from PIL import Image
import tempfile
import gradio as gr

import get_img_description
import os
import db 

import ai_model
import dataset
# from categories_getter import category_getter

DB = db.DB(model=ai_model.large_clip)

def img_get_images_and_categories(image: Image.Image):
    fetched_images = DB.search_similar(image)
    return _convert_to_ui(fetched_images)

IMAGES_DUPLICATES_THRESHOLD = 7 # 0.1
def _convert_to_ui(fetched_images): 
    print(fetched_images[0][3])
    if fetched_images[0][3] < IMAGES_DUPLICATES_THRESHOLD: 
        gr.Warning('Загруженное изображение очень похоже на изображение экспоната, уже введённого в базу данных. Возможно, это дубликат!')
    images = [os.path.join(dataset.DATASET_PATH, 'train', img_path) for _, _, img_path, _ in fetched_images]
    return fetched_images[0][0], images[0], images[1:], fetched_images[0][1]

def desc_get_images_and_categories(description: str):
    print(description)
    fetched_images = DB.search_similar_by_text(description)
    return _convert_to_ui(fetched_images)


def img_get_description(image: Image.Image) -> str:
    desc = get_img_description.get_img_description_ru(image, get_img_description.get_img_desc_large_git)
    print(desc)
    return desc[0].upper() + desc[1:]


def desc_get_description(description: str) -> str:
    return '' # todo: think about this

def get_json_dump(image: str, category: str, description: str) -> str:
    temp_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False)
    json.dump(
        {'image_id': image, 'category': category, 'description': description},
        temp_file,
        separators=(', ', ': '),
        ensure_ascii=False
    )
    return temp_file.name


def set_outputs_visible() -> list:
    return [gr.update(visible=True) for _ in range(4)]


def set_outputs_invisible() -> list:
    return [gr.update(visible=False) for _ in range(4)] + [None]


css = '''
.big_font, .big_font textarea {font-size: 24px}
#logo {width: auto !important; height; auto !important; max-width: 20px;}
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
            title = gr.HTML('<h1 style="text-align: center">Цифровой Прорыв (Южный Федеральный Округ)<h1>')
            subtitle = gr.HTML('<h1 style="text-align: center">Команда "Русская Красавица 3.0"</h1>')

    # input chapter
    with gr.Row():
        with gr.Tab('Изображение') as img_tab:
            input_image = gr.Image(
                label='Загрузите изображение',
                height=500,
                type='pil'
            )
        with gr.Tab('Описание') as desc_tab:
            input_description = gr.Text(
                label='Загрузите описание'
            )
            send_description_btn = gr.Button(
                'Отправить описание',
                variant='primary'
            )
        best_image = gr.Image(
            interactive=False,
            height=550,
            label='Лучшее совпадение по изображению',
            type='filepath'
        )

    # description chapter
    with gr.Row():
        description = gr.Text(
            info='Описание изображения',
            elem_classes='big_font',
            show_label=False,
            scale=5,
            visible=False
        )
        json_btn = gr.DownloadButton(
            label='Скачать в JSON',
            elem_classes='big_font',
            variant='primary',
            visible=False
        )

    # categories chapter
    with gr.Row():
        best_category = gr.Text(
            interactive=False,
            info='Лучшее совпадение по категории',
            elem_classes='big_font',
            show_label=False,
            scale=3,
            visible=False
        )
        # categories = [gr.Text(
        #     info='Похожая категория',
        #     elem_classes='big_font',
        #     show_label=False,
        #     visible=False
        # ) for _ in range(4)]

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
        fn=img_get_images_and_categories,
        inputs=input_image,
        outputs=[image_id, best_image, images, best_category]#, *categories]
    )
    input_image.upload(
        fn=img_get_description,
        inputs=input_image,
        outputs=[description]
    )

    # input description events
    send_description_btn.click(
        fn=desc_get_images_and_categories,
        inputs=input_description,
        outputs=[image_id, best_image, images, best_category]#, *categories]
    )
    send_description_btn.click(
        fn=desc_get_description,
        inputs=input_description,
        outputs=[description]
    )

    # input tab change event
    img_tab.select(
        fn=lambda: None,
        outputs=input_description
    )
    img_tab.select(
        fn=set_outputs_invisible,
        outputs=[best_category, description, json_btn, images, best_image]
    )
    desc_tab.select(
        fn=lambda: None,
        outputs=input_image
    )
    desc_tab.select(
        fn=set_outputs_invisible,
        outputs=[best_category, description, json_btn, images, best_image]
    )

    # change output visible
    input_image.upload(
        fn=set_outputs_visible,
        outputs=[best_category, description, json_btn, images]
    )
    input_image.clear(
        fn=set_outputs_invisible,
        outputs=[best_category, description, json_btn, images]
    )
    input_image.clear(
        fn=lambda: None,
        outputs=[best_image]
    )
    send_description_btn.click(
        fn=set_outputs_visible,
        outputs=[best_category, description, json_btn, images]
    )

    # JSON button event (download JSON)
    json_btn.click(
        fn=get_json_dump,
        inputs=[image_id, best_category, description],
        outputs=[json_btn]
    )


if __name__ == '__main__':
    demo.launch(share=True, server_port=8042)

