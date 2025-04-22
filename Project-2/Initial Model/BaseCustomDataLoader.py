import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch.utils.data as data
from PIL import Image
from pathlib import Path

class BaseCustomDataLoader(data.DataLoader):
    def __init__(self, rootDir: Path, df: pd.DataFrame, transforms=None, mode=None):
        self.root = rootDir
        self.transform = transforms
        self.imgId = (df["0"] + ".png").values
        self.attrId = df["1"]
        self.mode = mode
        # self.allImgs = []
        # self.allAttrs = []
        # import pdb
        # pdb.set_trace()
        # self.loadImgs()

    def __len__(self):
        return len(self.imgId)

    def __getitem__(self, idx):
        ImgId = self.imgId[idx]
        AttrId = self.attrId[idx]
        fileName = os.path.join(self.root, ImgId)
        img = cv2.imread(fileName)
        if self.transform:
            img = self.transform(img)
        if self.mode == 1:
            target = np.zeros(2, dtype=np.float32)
            target[AttrId] = 1
        if self.mode == 2:
            target = np.zeros(398, dtype=np.float32)
            target[AttrId] = 1
        return img , torch.tensor(target)
