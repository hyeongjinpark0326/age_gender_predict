# 파일에서 나이를 계산해서 이름 변경하는 함수
import os, shutil

young_path  = 'C:/Users/admin/Desktop/imdb_young/'  

i = 0
for file_name in os.listdir(young_path):

    birth_year = int(file_name.split('_')[2][0:4])
    new_name = str(2021 - birth_year)
    new_name = (new_name)
    os.rename(young_path+file_name, young_path+new_name+'_'+str(i)+'.jpg')
    i += 1

