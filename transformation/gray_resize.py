import cv2
import os
import numpy as np
from PIL import Image
 


# path = "C:/Users/admin/Desktop/final/images/combined/train/man/"
# imagePaths = [os.path.join(path,file_name) for file_name in os.listdir(path)]
# for imagePath in imagePaths:
#     img = Image.open(imagePath)
#     img_numpy = np.array(img, 'uint8')
#     gray = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2GRAY)
#     cv2.imwrite("C:/Users/admin/Desktop/final/images/gray/train/man/" + imagePath + ".jpg" , gray)
# print("All Done")


path = "C:/Users/admin/Desktop/dddd/woman/"
for p in os.listdir(path):
    # print(p)
    img = Image.open(os.path.join(path, p))
    img_numpy = np.array(img, 'uint8')
    gray = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2GRAY)
    #gray = cv2.resize(gray, dsize=(224,224))
    cv2.imwrite("C:/Users/admin/Desktop/dddd/man/" + p , gray)
print("All Done")
