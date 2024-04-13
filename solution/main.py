from PIL.Image import Image
import gradio as gr

import get_img_description
import os
import db 
import ai_model
import get_categories

DB = db.DB(ai_model.large_clip) 

def get_images(image: Image) -> tuple[str, list[str]]:
    images = [os.path.join(ai_model.DATASET_PATH, 'train', img_path) for img_path in DB.search_similar(image)]
    print(images)
    return images[0], images[1:]

def get_categories_texts(image: Image) -> tuple[str, list[str]]:
    cats = get_categories.category_getter.get_categories(image)
    print(cats)
    cats = [f'{cat} ({round(100*conf, 2)}%)' for cat, conf in cats]
    return cats[0], *cats[1:]

def get_description(image: Image) -> str: 
    desc =  get_img_description.get_img_description_ru(image, get_img_description.get_img_desc_large_git)
    print(desc)
    return desc[0].upper() + desc[1:]

css = '''
.big textarea {font-size: 24px}
'''

with gr.Blocks(css=css) as demo:
    with gr.Row():
        input_image = gr.Image(height='500px', label='Загрузите изображение', type='pil')
        best_image = gr.Image(height='500px', label='Лучшее совпадение по изображению')
    btn = gr.Button('Поехали!')
    description = gr.Text(info='Описание изображения', elem_classes='big', show_label=False)
    with gr.Row():
        best_category = gr.Text(info='Лучшее совпадение по категории', elem_classes='big', show_label=False, scale=3)
        categories = [gr.Text(info='Похожая категория', elem_classes='big', show_label=False) for _ in range(4)]
    images = gr.Gallery(label='Похожие изображения', columns=5)
    btn.click(
        fn=get_images,
        inputs=input_image,
        outputs=[best_image, images]
    )
    btn.click(
        fn=get_categories_texts,
        inputs=input_image,
        outputs=[best_category, *categories]
    )
    btn.click(
        fn=get_description,
        inputs=input_image,
        outputs=[description]
    )

if __name__ == '__main__':
    demo.launch(share=True)
