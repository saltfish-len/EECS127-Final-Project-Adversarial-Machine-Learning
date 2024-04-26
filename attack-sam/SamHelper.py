import torch.cuda
from numpy import ndarray
from segment_anything import SamPredictor, sam_model_registry
import numpy as np


class SamAPI:
    def __init__(self, model_name="vit_b", checkpoint=None):
        self.sam = sam_model_registry[model_name](checkpoint=checkpoint)
        if torch.cuda.is_available():
            self.sam.to(device="cuda")
        self.predictor = SamPredictor(self.sam)

    def set_image(self, image: ndarray):
        self.predictor.set_image(image)

    def predict(self, label_coords: dict[str:list[ndarray]], negative_coords: dict[str:list[ndarray]]) -> list[
        tuple[ndarray, str]]:
        '''
        generate mask from image for each label. When generating i-th mask, the rest of the labels are used as background.
        background is the first label in the label_coords and will not be used as a mask.
        :param label_coords: a dictionary of labels and their coordinates
        :param negative_coords: a dictionary of labels and their coordinates that are not the label
        :return: a tuple of the mask and the label
        '''
        res = []
        for label, coords in label_coords.items():
            point_coords = np.array(coords)
            point_labels = np.ones(len(coords))
            if label in negative_coords.keys() and len(negative_coords[label]) > 0:
                neg_coords = np.array(negative_coords[label])
                point_coords = np.concatenate((point_coords, neg_coords))
                point_labels = np.concatenate((point_labels, np.zeros(len(neg_coords))))
            mask, _, _ = self.predictor.predict(point_coords=point_coords, point_labels=point_labels,
                                                multimask_output=False)
            res.append((mask[0], label))
        return res

if __name__ == "__main__":
    sam = SamAPI(checkpoint="/Users/shizhh/PythonProjects/AnnotateMask/checkpoints/sam_vit_b_01ec64.pth")
    sam.set_image(np.zeros((1024, 1024, 3)))
    print(sam.predict({"label_0": [[100, 100], [200, 200]], "label_1": [[300, 300]]}, {"label_0": [[400, 400]]}))
    print("done")
