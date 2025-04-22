import os
import cv2
import glob
# import pdb
# pdb.set_trace()

allFiles = glob.glob(r"imet-2019-fgvc6\train\*.png")
count = 0
for fileName in allFiles:
    count += 1
    img = cv2.imread(fileName)
    fin = cv2.resize(img, (128, 128))
    assert cv2.imwrite(os.path.join("Resize_128", os.path.basename(fileName)), fin)
    print(count)
