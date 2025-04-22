import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch.utils.data as data
from PIL import Image
from pathlib import Path

class CustomDataLoader(data.DataLoader):
    def __init__(self, rootDir: Path, df: pd.DataFrame, transforms=None, size=128):
        self.root = rootDir
        self.transform = transforms
        self.imgId = (df["id"] + ".png").values
        self.attrId = df["attribute_ids"]
        # self.loadAllImages
        # import pdb
        # pdb.set_trace()
        self.X = size

    def __len__(self):
        return len(self.imgId)

    def __getitem__(self, idx):
        ImgId = self.imgId[idx]
        AttrId = self.attrId[idx]
        fileName = self.root / ImgId
        img = Image.open(fileName)

        if self.transform:
            img = self.transform(img)
        labels = np.zeros(1103)
        for item in AttrId.split():
            labels[int(item)] = 1
        return cv2.resize(img, (self.X, self.X)), torch.tensor(labels)