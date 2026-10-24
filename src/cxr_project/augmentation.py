"""Explicit image transforms; random operations are training-only."""
from __future__ import annotations
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

class RandomGamma:
    def __init__(self, probability=.3, gamma_range=(.8, 1.2)):
        self.probability = float(probability)
        self.gamma_range = tuple(float(x) for x in gamma_range)
        if not 0 <= self.probability <= 1 or len(self.gamma_range) != 2 or not 0 < self.gamma_range[0] <= self.gamma_range[1]:
            raise ValueError('invalid gamma probability/range')

    def __call__(self, image):
        if torch.rand(()).item() < self.probability:
            gamma = torch.empty(()).uniform_(*self.gamma_range).item()
            return TF.adjust_gamma(image, gamma)
        return image

class CLAHE:
    """Deterministic luminance CLAHE, applied before resize/padding."""
    def __init__(self, clip_limit=2., grid_size=(8, 8)):
        self.clip_limit = float(clip_limit)
        self.grid_size = tuple(int(x) for x in grid_size)
        if self.clip_limit <= 0 or len(self.grid_size) != 2 or min(self.grid_size) < 1:
            raise ValueError('invalid CLAHE clip limit/grid size')
        try:
            import cv2
        except ImportError as error:
            raise ImportError('clahe requires opencv-python-headless') from error

    def __call__(self, image):
        import cv2
        lab = cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.grid_size).apply(lab[:, :, 0])
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
