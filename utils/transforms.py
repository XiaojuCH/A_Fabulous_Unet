import random
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

class JointCompose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class JointResize:
    def __init__(self, size):
        self.size = size
    def __call__(self, img, mask):
        img = F.resize(img, self.size, interpolation=InterpolationMode.BILINEAR)
        mask = F.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)
        return img, mask

class JointRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img, mask):
        if random.random() < self.p:
            img = F.hflip(img)
            mask = F.hflip(mask)
        return img, mask

class JointRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img, mask):
        if random.random() < self.p:
            img = F.vflip(img)
            mask = F.vflip(mask)
        return img, mask