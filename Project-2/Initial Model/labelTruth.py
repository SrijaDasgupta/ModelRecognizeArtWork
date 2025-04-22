import os
import cv2
import csv
import glob
import random
import pandas as pd
from collections import Counter

allFiles = []
with open('imet-2019-fgvc6/train.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader, None)
    for row in reader:
        allFiles.append(row)
random.shuffle(allFiles)
cltCount = 0
tagCount = 0
cltResult = []
tagResult = []

while True:
    if cltCount + tagCount >= 40000:
        break
    for name, label in allFiles:
        cltArr = [0]*398
        tagArr = [0]*705

        newLabel = [0 if int(item) <= 397 else 1 for item in label.split(' ')]
        for attr in label.split(' '):
            if int(attr) <= 397:
                cltArr[int(attr)] = 1
            else:
                tagArr[int(attr)-398] = 1

        counts = Counter(newLabel)
        if counts[0] == counts[1]:
            continue
        if counts[0] > counts[1] and cltCount < 20000:
            cltCount += 1
            cltResult.append([name, cltArr])
        if counts[1] > counts[0] and tagCount < 20000:
            tagCount += 1
            tagResult.append([name, tagArr])

pd.DataFrame(cltResult).to_csv('cltTruth.csv')
pd.DataFrame(tagResult).to_csv('tagTruth.csv')