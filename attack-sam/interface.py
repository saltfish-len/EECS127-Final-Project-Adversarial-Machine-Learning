import gradio as gr
import numpy as np
from SamHelper import SamAPI


class Interface:
    def __init__(self, ckpt):
        self.annot_image = None
        self.show_sam_button = None
        self.reset_button = None
        self.sam = SamAPI(checkpoint=ckpt)

        self.display_click = None
        self.image = None

        self.clicks = []

    @staticmethod
    def get_click_coords(evt: gr.SelectData):
        return np.array(evt.index)

    def click_pipline(self, evt: gr.SelectData):
        coord = self.get_click_coords(evt)
        self.clicks.append(coord)

    def show_sam(self, image):
        labels = {f"label_{i}": [click] for i, click in enumerate(self.clicks)}
        self.sam.set_image(image)
        mask = self.sam.predict(labels, {})
        return image, mask

    def reset(self):
        self.clicks = []
    def set_ui(self):
        with gr.Row():
            self.image = gr.Image(interactive=True, height=f"{16 / 9}%", show_label=False)
            self.annot_image = gr.AnnotatedImage(show_label=False)
        with gr.Row():
            self.reset_button = gr.Button("Reset", variant="stop")
            self.show_sam_button = gr.Button("Show SAM", variant="primary")

        self.image.select(self.click_pipline, inputs=[])
        self.show_sam_button.click(self.show_sam, inputs=[self.image], outputs=[self.annot_image])
        self.reset_button.click(self.reset)

if __name__ == "__main__":
    app = Interface("/Users/shizhh/PythonProjects/AnnotateMask/checkpoints/sam_vit_b_01ec64.pth")
    with gr.Blocks() as demo:
        app.set_ui()
    demo.launch()
