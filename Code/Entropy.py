from skimage.filters.rank import entropy
from skimage.morphology import square
import torch
import numpy as np

class Entropy(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        img = sample['image']
        return {**sample, 'image': self.entropy_for_image(img)}

    def entropy_for_image(self, img):
        entr_img0 = entropy(img[0], square(4))
        entr_img1 = entropy(img[1], square(4))
        entr_img2 = entropy(img[2], square(4))
        entropy_img = torch.tensor(np.array([entr_img0, entr_img1, entr_img2]))
        return entropy_img
