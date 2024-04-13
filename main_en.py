from PIL.Image import Image
import gradio as gr

import get_categories_en
import find_best_en


def get_images(image: Image) -> tuple[str, list[str]]:
    images = find_best_en.find_best(image)
    return images[0], images[1:]

def get_categories(image: Image) -> tuple[str, list[str]]:
    cats = get_categories_en.get_categories(image)
    return cats[0], *cats[1:]


with gr.Blocks() as demo:
    with gr.Row(equal_height=True):
        input_image = gr.Image(label='Загрузите изображение', type='pil')
        best_image = gr.Image(label='Лучшее совпадение')
    btn = gr.Button('Поехали!')
    with gr.Row():
        best_category = gr.Text(show_label=False, scale=5)
        categories = [gr.Text(show_label=False) for _ in range(4)]
    images = gr.Gallery(label='Похожие изображения', columns=5)
    btn.click(
        fn=get_images,
        inputs=input_image,
        outputs=[best_image, images]
    )
    btn.click(
        fn=get_categories,
        inputs=input_image,
        outputs=[best_category, *categories]
    )

if __name__ == '__main__':
    demo.launch(share=True)
