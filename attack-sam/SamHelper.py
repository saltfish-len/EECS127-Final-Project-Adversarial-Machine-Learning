import torch.cuda
from numpy import ndarray
from typing import List, Tuple, Dict

from segment_anything import SamPredictor, sam_model_registry
import numpy as np

from SlimSAM.segment_anything.predictor_grad import MySamPredictor

import torch, torchvision
from SlimSAM.segment_anything import SamPredictor


class SamAPI:
    def __init__(self, model_name="vit_b", checkpoint=None):
        self.miou = None
        SlimSAM_model = torch.load("./checkpoints/SlimSAM-50.pth", map_location=torch.device('cpu'))
        SlimSAM_model.image_encoder = SlimSAM_model.image_encoder.module

        def forward(self, x):
            x = self.patch_embed(x)
            if self.pos_embed is not None:
                x = x + self.pos_embed

            for blk in self.blocks:
                x, qkv_emb, mid_emb, x_emb = blk(x)

            x = self.neck(x.permute(0, 3, 1, 2))

            return x

        device = "cpu"
        import types

        funcType = types.MethodType
        SlimSAM_model.image_encoder.forward = funcType(forward, SlimSAM_model.image_encoder)
        SlimSAM_model.to(device)
        SlimSAM_model.eval()
        self.predictor = MySamPredictor(SlimSAM_model)
        self.epsilon = 0.5

    def set_epsilon(self, epsilon: float):
        self.epsilon = epsilon

    def set_image(self, image: ndarray):
        self.predictor.set_image(image)
        self.image = image

    @staticmethod
    def get_mIoU(pred, target):
        pred = pred > 0
        target = target > 0
        intersection = (pred & target).float().sum((1, 2))
        union = (pred | target).float().sum((1, 2))
        iou = intersection / union
        return iou.mean().item()

    def predict(self, label_coords: Dict[str, List[ndarray]], negative_coords: Dict[str, List[ndarray]]) -> List:
        '''
        generate mask from image for each label. When generating i-th mask, the rest of the labels are used as background.
        background is the first label in the label_coords and will not be used as a mask.
        :param label_coords: a dictionary of labels and their coordinates
        :param negative_coords: a dictionary of labels and their coordinates that are not the label
        :return: a tuple of the mask and the label
        '''
        res = []
        iou_l = []
        for label, coords in label_coords.items():
            point_coords = np.array(coords)
            point_labels = np.ones(len(coords))
            if label in negative_coords.keys() and len(negative_coords[label]) > 0:
                neg_coords = np.array(negative_coords[label])
                point_coords = np.concatenate((point_coords, neg_coords))
                point_labels = np.concatenate((point_labels, np.zeros(len(neg_coords))))
            img = self.image

            pred_mask, img_grad = self.predictor.forward(
                image=img.astype(np.uint8),
                point_coords=point_coords,
                point_labels=[1],
            )
            img_FGSM_grad = img_grad.grad.sign()
            self.predictor.set_FGSM_image(img_grad + self.epsilon * img_FGSM_grad)
            fgsm_masks, _, _ = self.predictor.predict(point_coords=point_coords,
                                                      point_labels=[1])
            # print(pred_mask.shape,fgsm_masks.shape)
            res.append((pred_mask.detach().numpy() > 0, f"original_{label}"))
            res.append((fgsm_masks[0].detach().numpy(), f"FGSM_{label}"))
            iou = self.get_mIoU(pred_mask.unsqueeze(0), fgsm_masks[:1])
            iou_l.append(iou)
        self.miou = np.mean(iou_l)
        return res


if __name__ == "__main__":
    sam = SamAPI(checkpoint="/Users/shizhh/PythonProjects/AnnotateMask/checkpoints/sam_vit_b_01ec64.pth")
    sam.set_image(np.zeros((1024, 1024, 3)))
    print(sam.predict({"label_0": [[100, 100], [200, 200]], "label_1": [[300, 300]]}, {"label_0": [[400, 400]]}))
    print("done")
