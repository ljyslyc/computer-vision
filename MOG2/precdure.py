import cv2 
import numpy as np  


# 图片全部删去上角标
# 图片裁剪
for i in range(497,611):
    path = '2019 APMCM Problem A Attachment\\0' + str(i) + '.bmp'
    outpath = 'data_cutted\\0' + str(i) + '.bmp'
    img = cv2.imread(path) # 读取RGB彩⾊图⽚
    img = img[200:820,500:1100]
    cv2.imwrite(outpath,img)