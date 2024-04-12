import gradio as gr
import find_best 
import get_categories

with gr.Blocks() as demo:
    with gr.Row():
        input_image = gr.Image(type='pil')
        best_image = gr.Image()
    btn = gr.Button('Let\'s go')
    gallary = gr.Gallery(columns=5)
    btn.click(fn=find_best.find_best, inputs=input_image, outputs=[best_image, gallary])
    btn.click(fn=get_categories.get_categories, inputs=input_image)

if __name__ == '__main__':
    demo.launch(share=True)