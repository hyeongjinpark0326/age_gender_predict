import cv2
import os
import glob

# for path, dir, file in os.walk('C:/Users/admin/Desktop/age_gender_estimation-master (2)/test/'):
#     print(dir+file)

# folder = ['f20', 'f30', 'f40', 'f50', 'f60', 'm20', 'm30', 'm40', 'm50', 'm60']
# 원본 이미지 경로


path = 'C:/Users/admin/Desktop/wiki_crop/00/'
for i in os.listdir(path):
    # print(i)
    
    # for j in os.listdir(path+i):
    #     print(path+i+j)

    facedata = "C:/Users/admin/Desktop/final/age_gender_estimation-master (2)/haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(facedata)

    img = cv2.imread(path+i)
    print(img)
    minisize = (img.shape[1],img.shape[0])
    miniframe = cv2.resize(img, minisize)
    # miniframe = cv2.resize(img, dsize=(200,200))

    faces = cascade.detectMultiScale(miniframe)

    save_path = 'C:/Users/admin/Desktop/wiki/'

    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))

        sub_face = img[y:y+h, x:x+w]
        fname, ext = os.path.splitext(i)
        cv2.imwrite(save_path+fname+"_cropped_"+ext, sub_face)
