import cv2
import os
import shutil

path = 'C:/Users/admin/Desktop/imdb_man_1/'

for img in os.listdir(path):
    #img.split('_')[0]
    #img.split('_')[1]
    # print(img)
    # print(img[-6:-4])
    os.rename(path+img, path+img.split('_')[0]+'_'+'0_'+img.split('_')[1])

