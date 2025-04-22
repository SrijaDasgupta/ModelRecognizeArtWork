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
eqCount = 0
result = []

# for name, label in allFiles:
#     label = [0 if int(item) <= 397 else 1 for item in label.split(' ')]
#     counts = Counter(label)
#     if counts[0] == counts[1]:
#         eqCount += 1
#     elif counts[0] > counts[1]:
#         cltCount += 1
#     else:
#         tagCount += 1
# print(eqCount)
# print(cltCount)
# print(tagCount)

import pdb
pdb.set_trace()
while True:
    if cltCount + tagCount >= 20000:
        break
    for name, label in allFiles:
        temp = label
        label = [0 if int(item) <= 397 else 1 for item in label.split(' ')]
        counts = Counter(label)
        if counts[0] == counts[1]:
            continue
        # import pdb
        # pdb.set_trace()
        if counts[0] > counts[1] and cltCount < 10000:
            print("CLT: ", label, counts)
            cltCount += 1
            result.append([name, 0])
        if counts[1] > counts[0] and tagCount < 10000:
            print("TAG: ", label, counts)
            tagCount += 1
            result.append([name, 1])
# import pdb
# pdb.set_trace()
pd.DataFrame(result).to_csv('baseTruth.csv')