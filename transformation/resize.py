import cv2
import os
import glob

# C:/Users/admin/Desktop/final/images/combined/train
# categories = ['man', 'woman']
#C:/Users/admin/Desktop/preprocessed/20/20

img_path = 'C:/Users/admin/Desktop/AAFD_WIKI_UTK (3)/AAFD_WIKI_UTK/'
save_path = 'C:/Users/admin/Desktop/140_gray/'

for i in os.listdir(img_path):
    # print(i)
    img = cv2.imread(img_path+i)
    # h,w,c = img.shape
    img = cv2.resize(img, dsize=(140,140) ) 
    cv2.imwrite(save_path+i, img)
    


# for c in categories:
#     for i in os.listdir(img_path+c+'/'):
#         # print(i)
#         img = cv2.imread(img_path+c+'/'+i)
#         # h,w,c = img.shape
#         img = cv2.resize(img, dsize=(200,200) ) 
        
#         cv2.imwrite(save_path+c+'/'+i, img)


# img = cv2.imread('f20 (66)_cropped_.jpg')
# h,w,c = img.shape
# img = cv2.resize(img, dsize=(150, int(h/w*150) ) )
# cv2.imwrite('aa.jpg',img)
# # cv2.imshow('img', img)
# # cv2.waitKey(0)