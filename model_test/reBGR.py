
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
import cv2
import os
import numpy as np
from PIL import Image
 
load_path = "C:/Users/admin/Desktop/final/images/age/f50/"
save_path = 'C:/Users/admin/Desktop/save/'
imagePaths = [os.path.join(load_path,file_name) for file_name in os.listdir(load_path)]
count = 0
for i in imagePaths:
    count += 1
    img = Image.open(i)
    img_numpy = np.array(img, 'uint8')
    gray = cv2.cvtColor(img_numpy, cv2.)
    cv2.imwrite("C://Users//jayfl//Desktop//Capston//DataGathering//Gray//User.1." + str(count) + ".jpg" , color)
print("All Done")
