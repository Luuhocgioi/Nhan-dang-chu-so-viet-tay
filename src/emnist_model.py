"""CNN 62 lớp: số, chữ hoa và chữ thường."""
import torch
from torch import nn
import cv2
import numpy as np

CHARACTERS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'


def normalize_glyph(image):
    image = np.asarray(image, dtype=np.uint8)
    ys,xs = np.where(image > max(15,float(image.max())*.15))
    if not len(xs):
        return np.zeros((28,28),np.uint8)
    crop=image[ys.min():ys.max()+1,xs.min():xs.max()+1]
    scale=20/max(crop.shape)
    w,h=max(1,round(crop.shape[1]*scale)),max(1,round(crop.shape[0]*scale))
    small=cv2.resize(crop,(w,h),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
    out=np.zeros((28,28),np.uint8)
    y,x=(28-h)//2,(28-w)//2
    out[y:y+h,x:x+w]=small
    return out


class CharacterCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(1,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((3,3)))
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(1152,256),nn.ReLU(),nn.Dropout(.3),nn.Linear(256,62))

    def forward(self,x):
        return self.head(self.features(x))
