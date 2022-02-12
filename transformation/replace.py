# 파일이름에서 _가 2개 포함된 이름을 1개로 바꿈

import os, shutil



image_dir = "C:/Users/admin/Desktop/last_images/wiki/"
save_dir = "C:/Users/admin/Desktop/last_images/wiki_man/"
for file in os.listdir(image_dir):
    print(file.split('_')[0])
    os.rename(image_dir+file, image_dir+file.split('_')[0]+'_'+file.split('_')[2])