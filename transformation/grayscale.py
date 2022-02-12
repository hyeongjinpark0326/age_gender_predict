import cv2
import numpy as np
import os
from cv2 import *

def bgr2gray(bgr_img):
    # BGR 색상값
    b = bgr_img[:, :, 0]
    g = bgr_img[:, :, 1]
    r = bgr_img[:, :, 2]
    result = ((0.299 * r) + (0.587 * g) + (0.114 * b))
    # imshow 는 CV_8UC3 이나 CV_8UC1 형식을 위한 함수이므로 타입변환
    return result.astype(np.uint8)


categories = ["f20"]
img_path = 'C:/Users/admin/Desktop/final/images/manwoman/age_combined_gray_resized_copy_140/30/'
save_path = 'C:/Users/admin/Desktop/gray2/'
# for c in categories:
    # print(c)
for i in os.listdir(img_path):
    #print(c)
    print(i)
        # input_img = cv2.imread(img_path+c+'/'+i, cv2.IMREAD_COLOR)
        # bgr_img = bgr2gray(input_img)
        # h,w = bgr_img.shape
        # img = cv2.resize(bgr_img, dsize=(200, int(h/w*200) ) )


        # #cv2.namedWindow('GrayScale Image')
        # # 지정한윈도우에 이미지를 보여준다.

        # #cv2.imshow("GrayScale Image", img)
        # # 지정한 시간만큼 사용자의 키보드입력을 대기한다. 0으로하면 키보드대기입력을 무한히 대기하도록한다.
        # # cv2.waitKey(0)

        # # 이미지 저장(경로, 이미지)
        # cv2.imwrite(save_path+c+'/'+i, img)