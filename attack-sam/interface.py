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
        gr.Info(f"Clicked on {coord[0]}, {coord[1]}")

    def show_sam(self, image):
        labels = {f"label_{i}": [click] for i, click in enumerate(self.clicks)}
        self.sam.set_image(image)
        mask = self.sam.predict(labels, {})
        return (image, mask), self.show_iou()

    def show_iou(self):
        return self.sam.miou
    def reset(self):
        self.clicks = []

    def set_ui(self):
        with gr.Row():
            self.image = gr.Image(interactive=True , show_label=False)
            self.annot_image = gr.AnnotatedImage(show_label=False)
        with gr.Row():
            mIoU = gr.Textbox(label="mIoU", interactive=False)
            epsilon = gr.Slider(0, 1, 0.5, label="Epsilon")
        with gr.Row():
            self.reset_button = gr.Button("Reset", variant="stop")
            self.show_sam_button = gr.Button("Show SAM", variant="primary")

        self.image.select(self.click_pipline, inputs=[])
        self.show_sam_button.click(self.show_sam, inputs=[self.image], outputs=[self.annot_image,mIoU])

        self.reset_button.click(self.reset)
        epsilon.release(self.sam.set_epsilon, inputs=[epsilon])


if __name__ == "__main__":
    app = Interface("/Users/shizhh/PythonProjects/AnnotateMask/checkpoints/sam_vit_b_01ec64.pth")
    with gr.Blocks() as demo:
        app.set_ui()
    demo.queue().launch(show_error=True)
