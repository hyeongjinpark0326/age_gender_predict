# 파일 이름에서 나이를 읽어와 폴더를 옮기는 함수
import os, shutil
img_path    = 'C:/Users/admin/Desktop/imdb_young_ext_200/' 
old_path    = 'C:/Users/admin/Desktop/old/'
young_path  = 'C:/Users/admin/Desktop/imdb_young/'    
    

# for folder in range(1,70):
#     # print(folder)
for file_name in os.listdir(img_path):
    pass
    # print(file_name)
    try:
        birth_year = int(file_name.split('_')[2][0:4])
        age = (2021 - birth_year)
        # print(age)
        if age <= 69:
            # print(img_path+file_name)
            shutil.move(img_path+file_name, young_path+file_name)
        else:
            pass
    except:
        pass

    #print(birth_year)
    # new_name = (2021 - birth_year)
    # if new_name >= 69:
    #     shutil.move(img_path+file_name, young_path)
    # elif os.path.exists(img_path+file_name):
    #     print('중복')
    #     pass
        